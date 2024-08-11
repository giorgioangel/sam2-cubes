# Giorgio Angelotti - 2024

# Train/Fine Tune SAM 2 on Annotated Cubes Dataset

# this file is a readaptation of this: https://github.com/sagieppel/fine-tune-train_segment_anything_2_in_60_lines_of_code/blob/8b6d59d8f764a2a1d9f018ef949571ddcac57c9b/TRAIN.py

import numpy as np
import torch
import nrrd
import cv2
from pathlib import Path
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def create_data(volume, labels):
    lbl = np.zeros((volume.shape[0],volume.shape[1]))

    while np.sum(lbl) == 0:
        idx = np.random.randint(0,3*volume.shape[0]+1)
        
        coordinate = idx // volume.shape[0]
        islice = idx % volume.shape[0]
        
        if coordinate == 0:
            img = volume[islice]
            lbl = labels[islice]
        elif coordinate == 1:
            img = volume[:,islice,:]
            lbl = labels[:,islice,:]
        elif coordinate == 2:
            img = volume[:,:,islice]
            lbl = labels[:,:,islice]
        else:
            img = np.zeros((volume.shape[0],volume.shape[1]))
            lbl = np.zeros((volume.shape[0],volume.shape[1]))

        if np.sum(lbl) == 0:
            continue
        # RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # resize image

        r = np.min([1024 / img.shape[1], 1024 / img.shape[0]])
        img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
        lbl = cv2.resize(lbl, (int(lbl.shape[1] * r), int(lbl.shape[0] * r)),interpolation=cv2.INTER_NEAREST)

        masks = []
        points= []
        unique_labels = np.unique(lbl)[1:]
        for label in unique_labels:
            mask = (lbl == label).astype(np.uint8)
            masks.append(mask)
            coords = np.argwhere(mask > 0)
            yx = np.array(coords[np.random.randint(len(coords))])
            points.append([[yx[1], yx[0]]])

        return img, np.array(masks), np.array(points), np.ones([len(masks),1])


# Read data

cube_folder = Path("/your/custom/path/finished-cubes") # replace with your own PATH

# List all directories in the specified path
directories = [f for f in cube_folder.iterdir() if f.is_dir()]


def read_batch(data):
    directory  = data[np.random.randint(len(data))]
    directory_name = directory.name 
    try:
        # Construct the full paths to the mask and volume files
        mask_file = directory / f"{directory_name}_mask.nrrd"
        volume_file = directory / f"{directory_name}_volume.nrrd"   
        # Check if the files exist
        if mask_file.exists() and volume_file.exists():
            mask, _ = nrrd.read(mask_file)
            volume, _ = nrrd.read(volume_file)
            return create_data(volume, mask)
        else:
            print(f"Files not found in {directory_name}:")
            if not mask_file.exists():
                print(f"  Missing: {mask_file}")
            if not volume_file.exists():
                print(f"  Missing: {volume_file}")
            return 0,0,0,0
    except:
        print(directory_name)

# Load model

sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt" # "sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml" # "sam2_hiera_l.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
predictor = SAM2ImagePredictor(sam2_model)

# Set training parameters
predictor.model.sam_mask_decoder.train(True)
predictor.model.sam_prompt_encoder.train(True)
optimizer=torch.optim.AdamW(params=predictor.model.parameters(),lr=1e-5,weight_decay=4e-5)
scaler = torch.cuda.amp.GradScaler() # mixed precision

# Training loop

for itr in range(100000):
    with torch.cuda.amp.autocast(): # cast to mix precision
        #with torch.cuda.amp.autocast():
            image,mask,input_point, input_label = read_batch(directories) # load data batch
            if mask.shape[0]==0: continue # ignore empty batches
            predictor.set_image(image) # apply SAM image encodet to the image

            # prompt encoding

            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(input_point, input_label, box=None, mask_logits=None, normalize_coords=True)
            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels),boxes=None,masks=None,)

            # mask decoder

            batched_mode = unnorm_coords.shape[0] > 1 # multi object prediction
            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),sparse_prompt_embeddings=sparse_embeddings,dense_prompt_embeddings=dense_embeddings,multimask_output=True,repeat_image=batched_mode,high_res_features=high_res_features,)
            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])# Upscale the masks to the original image resolution

            # Segmentaion Loss caclulation

            gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
            prd_mask = torch.sigmoid(prd_masks[:, 0])
            seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean()

            # Score loss calculation (intersection over union) IOU

            inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
            iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
            score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
            loss=seg_loss+score_loss*0.05  # mix losses

            # apply back propogation

            predictor.model.zero_grad() # empty gradient
            scaler.scale(loss).backward()  # Backpropogate
            scaler.step(optimizer)
            scaler.update() # Mix precision

            if itr%1000==0: torch.save(predictor.model.state_dict(), "model.torch") # save model

            # Display results

            if itr==0: mean_iou=0
            mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
            print("step)",itr, "Accuracy(IOU)=",mean_iou)