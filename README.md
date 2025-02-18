# Implementation of Diffusion Probabilistic Model Made Slim, CVPR 2023
- This is unofficial implementation repository.
  - [Paper Link](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_Diffusion_Probabilistic_Model_Made_Slim_CVPR_2023_paper.pdf)
- This paper adds and modifies wavelet layer and the number of channels for each layer so that most code of this repository is similar to [Stable Diffusion](https://github.com/CompVis/stable-diffusion).  
- For clarity and better readibility, I modified variable names and integrated modules.

# Requirements
- ...
# Usage
- ...
## Training
- **Note** : If you are using V100 or T4 or some old one, bf16 is not supported. So, you must set right configuration for your hardware.
  - Google's TPU is compatible with bf16 training.
- ...
## Inference
- ...

## Serving
- ...

## To do
- All UNet model parameters must be included in just one dot-dict configuration object.
  - Enhance readibility.