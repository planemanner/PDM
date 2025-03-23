# Implementation of Stable Diffusion and One Step Diffusion Model
- This is unofficial implementation repository for Stable Diffusion (2021) and One Step Diffusion (ICLR 2025).
  - [One Step Diffusion](https://arxiv.org/pdf/2410.12557)
- This paper adds and modifies wavelet layer and the number of channels for each layer so that some code of this repository is similar to [Stable Diffusion](https://github.com/CompVis/stable-diffusion).  
- For clarity and better readibility, I modified variable names and integrated modules.
- Since num_timesteps_cond is always set to '1', all related legacy code blocks are removed.
- Redundant or legacy parts are removed.
# Note
- In current version, the autoencoder is not VQ-VAE. It is conventional variational auto-encoder.
  - So, unlike original Stable Diffusion model, the autoencoder is trained without EMA update. 
  - The Wavelet layers will be applied soon.

- Current version is very incomfortable to use most features.
  - Most configuration files are integrated for convenience and reduction of confusion.
  - Since this is the first project using pytorch lightning, many parts are not graceful.
  - So, some migrations are needed

- Current version's prompt for generation is "Binary Mask".
# Update
- Configuration 정리
  - Dataclass 활용해서 정리함 
  - 실험 관리를 용이하게 하기 위함.
- CLI 를 기반으로 실험을 편하게 할 수 있도록 Interface 개선

# Requirements
- Python version : > 3.9
- PyTorch version 2.0 or later is required.
- The whole traininng pipeline is written under Pytorch Lightning 2.5.0
# Usage
## Training
- First, train auto-encoder
- Second, train unet with flow-matching method or ddpm(ddim)


## To do
- Update quantitative metrics for validation. (e.g., FID)
- Add additional analysis tools with callback function (e.g., MLFlow)
- Code base looks spagetti. So, I'll refactor configuration and related part for convenience in terms of usage and maintenance.
- Implement a method to empower condition in generation process.