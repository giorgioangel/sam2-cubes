# SAM 2 for Annotated Instances in Cubes (Vesuvius Challenge)

This repository contains a fork of the **SAM 2 (Segment Anything Model 2)** project, with added functionality for fine-tuning the model on instance-annotated cubes from the Vesuvius Challenge. The fine-tuning process allows the model to better segment instances within 3D volumetric data, enhancing its performance for this specific task.

## Overview

**SAM 2** is a versatile segmentation model that includes various modules for tasks like feature extraction, prompt processing, mask prediction, and mask propagation in videos. This repository extends SAM 2 to work with 3D instance-annotated cubes by fine-tuning the model on this data.

The instance-annotated cubes used for fine-tuning are available [here](https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/seg-volumetric-labels/instance-annotated-cubes/).

## Setup

### 1. Install SAM 2

First, follow the [official SAM 2 README](https://github.com/facebookresearch/segment-anything-2) to install the base SAM 2 model.
There are a couple more python packages that you might need, like `pynrrd`.


### 2. Prepare the Annotated Cubes

Download the instance-annotated cubes from the Vesuvius Challenge and organize them into a directory, such as `finished-cubes/` or `instance-annotated-cubes/`. Update the path in the `train.py` script accordingly:

```python
cube_folder = Path("/your/custom/path/finished-cubes")  # Update with your path
```

## Fine-Tuning Process

### 1. Fine-Tuning the Model

The `train.py` script is designed to fine-tune the `prompt-encoding` and `mask-decoder` modules of SAM 2 on the annotated cubes. Here’s how it works:

- **Data Preparation**: The 3D cubes are sliced into 2D images along the axes.
- **Sampling Points**: A single point is randomly sampled within an instance in each slice.
- **Mask Prediction**: SAM 2's image predictor, given the raw image slice and the sampled point, generates a mask for the pointed instance.
- **Module Fine-Tuning**: The prompt-encoding and mask-decoder modules are fine-tuned based on these predictions.

My checkpoint reached 0.65 IoU on the training set, haven't checked on the validation set. You can surely do better.

### 2. Generating New Instance-Annotated Cubes

After fine-tuning, you can generate new instance-annotated cubes using the `inference_cube.ipynb` notebook:

- **Loading Weights**: Load the original SAM 2 weights and the fine-tuned weights.
- **Frame Slicing**: The `read_image` function saves a `.jpg` image for each frame of a cube (z-slices).
- **Mask Propagation**: Initialize the SAM video predictor with masks for the first and last frames, then propagate the mask through the other frames to create a fully annotated cube.

## References and Additional Resources

This fine-tuning approach is inspired by [this SAM 2 fork](https://github.com/sagieppel/fine-tune-train_segment_anything_2_in_60_lines_of_code) and the accompanying [Medium blog post](https://medium.com/@sagieppel/train-fine-tune-segment-anything-2-sam-2-in-60-lines-of-code-928dd29a63b3) by the fork's author.

For further guidance, check out the [official SAM 2 tutorials](https://github.com/facebookresearch/segment-anything-2/tree/main/notebooks) provided by Meta.

---

Feel free to customize this further to match your style or add more details as needed. This version should be more structured and easier to follow for anyone looking to understand and use your project.

# SAM 2: Segment Anything in Images and Videos

**[AI at Meta, FAIR](https://ai.meta.com/research/)**

[Nikhila Ravi](https://nikhilaravi.com/), [Valentin Gabeur](https://gabeur.github.io/), [Yuan-Ting Hu](https://scholar.google.com/citations?user=E8DVVYQAAAAJ&hl=en), [Ronghang Hu](https://ronghanghu.com/), [Chaitanya Ryali](https://scholar.google.com/citations?user=4LWx24UAAAAJ&hl=en), [Tengyu Ma](https://scholar.google.com/citations?user=VeTSl0wAAAAJ&hl=en), [Haitham Khedr](https://hkhedr.com/), [Roman Rädle](https://scholar.google.de/citations?user=Tpt57v0AAAAJ&hl=en), [Chloe Rolland](https://scholar.google.com/citations?hl=fr&user=n-SnMhoAAAAJ), [Laura Gustafson](https://scholar.google.com/citations?user=c8IpF9gAAAAJ&hl=en), [Eric Mintun](https://ericmintun.github.io/), [Junting Pan](https://junting.github.io/), [Kalyan Vasudev Alwala](https://scholar.google.co.in/citations?user=m34oaWEAAAAJ&hl=en), [Nicolas Carion](https://www.nicolascarion.com/), [Chao-Yuan Wu](https://chaoyuan.org/), [Ross Girshick](https://www.rossgirshick.info/), [Piotr Dollár](https://pdollar.github.io/), [Christoph Feichtenhofer](https://feichtenhofer.github.io/)

[[`Paper`](https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/)] [[`Project`](https://ai.meta.com/sam2)] [[`Demo`](https://sam2.metademolab.com/)] [[`Dataset`](https://ai.meta.com/datasets/segment-anything-video)] [[`Blog`](https://ai.meta.com/blog/segment-anything-2)] [[`BibTeX`](#citing-sam-2)]

![SAM 2 architecture](assets/model_diagram.png?raw=true)

**Segment Anything Model 2 (SAM 2)** is a foundation model towards solving promptable visual segmentation in images and videos. We extend SAM to video by considering images as a video with a single frame. The model design is a simple transformer architecture with streaming memory for real-time video processing. We build a model-in-the-loop data engine, which improves model and data via user interaction, to collect [**our SA-V dataset**](https://ai.meta.com/datasets/segment-anything-video), the largest video segmentation dataset to date. SAM 2 trained on our data provides strong performance across a wide range of tasks and visual domains.

![SA-V dataset](assets/sa_v_dataset.jpg?raw=true)

## Installation

SAM 2 needs to be installed first before use. The code requires `python>=3.10`, as well as `torch>=2.3.1` and `torchvision>=0.18.1`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. You can install SAM 2 on a GPU machine using:

```bash
git clone https://github.com/facebookresearch/segment-anything-2.git

cd segment-anything-2; pip install -e .
```

To use the SAM 2 predictor and run the example notebooks, `jupyter` and `matplotlib` are required and can be installed by:

```bash
pip install -e ".[demo]"
```

Note:
1. It's recommended to create a new Python environment for this installation and install PyTorch 2.3.1 (or higher) via `pip` following https://pytorch.org/. If you have a PyTorch version lower than 2.3.1 in your current environment, the installation command above will try to upgrade it to the latest PyTorch version using `pip`.
2. The step above requires compiling a custom CUDA kernel with the `nvcc` compiler. If it isn't already available on your machine, please install the [CUDA toolkits](https://developer.nvidia.com/cuda-toolkit-archive) with a version that matches your PyTorch CUDA version.

Please see [`INSTALL.md`](./INSTALL.md) for FAQs on potential issues and solutions.

## Getting Started

### Download Checkpoints

First, we need to download a model checkpoint. All the model checkpoints can be downloaded by running:

```bash
cd checkpoints
./download_ckpts.sh
```

or individually from:

- [sam2_hiera_tiny.pt](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt)
- [sam2_hiera_small.pt](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt)
- [sam2_hiera_base_plus.pt](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt)
- [sam2_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt)

Then SAM 2 can be used in a few lines as follows for image and video prediction.

### Image prediction

SAM 2 has all the capabilities of [SAM](https://github.com/facebookresearch/segment-anything) on static images, and we provide image prediction APIs that closely resemble SAM for image use cases. The `SAM2ImagePredictor` class has an easy interface for image prompting.

```python
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(<your_image>)
    masks, _, _ = predictor.predict(<input_prompts>)
```

Please refer to the examples in [image_predictor_example.ipynb](./notebooks/image_predictor_example.ipynb) (also in Colab [here](https://colab.research.google.com/github/facebookresearch/segment-anything-2/blob/main/notebooks/image_predictor_example.ipynb)) for static image use cases.

SAM 2 also supports automatic mask generation on images just like SAM. Please see [automatic_mask_generator_example.ipynb](./notebooks/automatic_mask_generator_example.ipynb) (also in Colab [here](https://colab.research.google.com/github/facebookresearch/segment-anything-2/blob/main/notebooks/automatic_mask_generator_example.ipynb)) for automatic mask generation in images.

### Video prediction

For promptable segmentation and tracking in videos, we provide a video predictor with APIs for example to add prompts and propagate masklets throughout a video. SAM 2 supports video inference on multiple objects and uses an inference state to keep track of the interactions in each video.

```python
import torch
from sam2.build_sam import build_sam2_video_predictor

checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint)

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    state = predictor.init_state(<your_video>)

    # add new prompts and instantly get the output on the same frame
    frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, <your_prompts>):

    # propagate the prompts to get masklets throughout the video
    for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
        ...
```

Please refer to the examples in [video_predictor_example.ipynb](./notebooks/video_predictor_example.ipynb) (also in Colab [here](https://colab.research.google.com/github/facebookresearch/segment-anything-2/blob/main/notebooks/video_predictor_example.ipynb)) for details on how to add click or box prompts, make refinements, and track multiple objects in videos.

## Load from 🤗 Hugging Face

Alternatively, models can also be loaded from [Hugging Face](https://huggingface.co/models?search=facebook/sam2) (requires `pip install huggingface_hub`).

For image prediction:

```python
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor

predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(<your_image>)
    masks, _, _ = predictor.predict(<input_prompts>)
```

For video prediction:

```python
import torch
from sam2.sam2_video_predictor import SAM2VideoPredictor

predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large")

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    state = predictor.init_state(<your_video>)

    # add new prompts and instantly get the output on the same frame
    frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, <your_prompts>):

    # propagate the prompts to get masklets throughout the video
    for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
        ...
```

## Model Description

|      **Model**       | **Size (M)** |    **Speed (FPS)**     | **SA-V test (J&F)** | **MOSE val (J&F)** | **LVOS v2 (J&F)** |
| :------------------: | :----------: | :--------------------: | :-----------------: | :----------------: | :---------------: |
|   sam2_hiera_tiny    |     38.9     |          47.2          |        75.0         |        70.9        |       75.3        |
|   sam2_hiera_small   |      46      | 43.3 (53.0 compiled\*) |        74.9         |        71.5        |       76.4        |
| sam2_hiera_base_plus |     80.8     | 34.8 (43.8 compiled\*) |        74.7         |        72.8        |       75.8        |
|   sam2_hiera_large   |    224.4     | 24.2 (30.2 compiled\*) |        76.0         |        74.6        |       79.8        |

\* Compile the model by setting `compile_image_encoder: True` in the config.

## Segment Anything Video Dataset

See [sav_dataset/README.md](sav_dataset/README.md) for details.

## License

The models are licensed under the [Apache 2.0 license](./LICENSE). Please refer to our research paper for more details on the models.

## Contributing

See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

## Contributors

The SAM 2 project was made possible with the help of many contributors (alphabetical):

Karen Bergan, Daniel Bolya, Alex Bosenberg, Kai Brown, Vispi Cassod, Christopher Chedeau, Ida Cheng, Luc Dahlin, Shoubhik Debnath, Rene Martinez Doehner, Grant Gardner, Sahir Gomez, Rishi Godugu, Baishan Guo, Caleb Ho, Andrew Huang, Somya Jain, Bob Kamma, Amanda Kallet, Jake Kinney, Alexander Kirillov, Shiva Koduvayur, Devansh Kukreja, Robert Kuo, Aohan Lin, Parth Malani, Jitendra Malik, Mallika Malhotra, Miguel Martin, Alexander Miller, Sasha Mitts, William Ngan, George Orlin, Joelle Pineau, Kate Saenko, Rodrick Shepard, Azita Shokrpour, David Soofian, Jonathan Torres, Jenny Truong, Sagar Vaze, Meng Wang, Claudette Ward, Pengchuan Zhang.

Third-party code: we use a GPU-based connected component algorithm adapted from [`cc_torch`](https://github.com/zsef123/Connected_components_PyTorch) (with its license in [`LICENSE_cctorch`](./LICENSE_cctorch)) as an optional post-processing step for the mask predictions.

## Citing SAM 2

If you use SAM 2 or the SA-V dataset in your research, please use the following BibTeX entry.

```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  url={https://arxiv.org/abs/2408.00714},
  year={2024}
}
```
