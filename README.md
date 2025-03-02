# Implementation of Diffusion Probabilistic Model Made Slim, CVPR 2023
- This is unofficial implementation repository.
  - [Paper Link](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_Diffusion_Probabilistic_Model_Made_Slim_CVPR_2023_paper.pdf) and [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- This paper adds and modifies wavelet layer and the number of channels for each layer so that some code of this repository is similar to [Stable Diffusion](https://github.com/CompVis/stable-diffusion).  
- For clarity and better readibility, I modified variable names and integrated modules.
- Since num_timesteps_cond is always set to '1', all related legacy code blocks are removed.
- Redundant or legacy parts are removed.

# Requirements
- Python version : > 3.9
- PyTorch version 2.0 or later is required.
- The whole traininng pipeline is written under Pytorch Lightning 2.5.0
# Usage
- ...
## Training
### Stage 1 : Train AutoEncoder
```sh
python main.py --stage autoencoder
```
### Stage 2 : Train Diffusion Model
```sh
python main.py --stage diffusion --autoencoder_ckpt [AUTOENCODER PATH]
```
  
## Inference
- ...

## Serving
- ...

## To do
- Add overall logging parts !! (Critical Features)
  - autoencoder
  - diffusion
- Replace the UNet's Down & Up sample blocks with WaveletLayers.
- Add distillation parts for Diffusion Model
- Check hyperparameters 
  - ema update period
  - validation period
- Check all Device Parts
