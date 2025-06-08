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
- This implementation is based on the xformers lib of Meta AI. So, you must use Linux OS to for this repo.

- If you want to see the part of implementation for [ICLR 2025](https://arxiv.org/pdf/2410.12557), please see [ShortCutSampler](samplers/shortcut.py) and [Diffusion Training Step](diffusion.py)
```python
# diffusion.py [Line 60-79]
d, d_idx = self.sampler.sample_d(len(images), self.device)
t = self.sampler.sample_t(len(images), d_idx, self.device)

d0 = torch.zeros_like(d, device=self.device)
x0 = torch.randn_like(z).to(self.device)
x_t = (1-t)[:, None, None, None] * x0 + t[:, None, None, None] * z
flow_labels = z - x0

with torch.no_grad():
    x_t_plus_d, t_plus_d, s_first = self.sampler.shortcut_step(self.unet, x_t, t, d, conds)
    _, _, s_second = self.sampler.shortcut_step(self.unet, x_t_plus_d, t_plus_d, d, conds)
    s_second = torch.clip(s_second, -4, 4)
    s_target = 0.5 * (s_first + s_second)

x_t_plus_d0, _, s_0 = self.sampler.shortcut_step(self.unet, x_t, t, d0, conds)
_, _, self_consistency = self.sampler.shortcut_step(self.unet, x_t_plus_d0, t, 2 * d, conds)
loss = F.mse_loss(s_0, flow_labels) + F.mse_loss(self_consistency, s_target)
self.log('TRAIN_LOSS', loss.item(), prog_bar=True, on_step=True, on_epoch=True)
return loss
```

- Current version's prompt for generation is "Binary Mask".

# Requirements
- Python version : > 3.9
- PyTorch version 2.0 or later is required.
- The whole traininng pipeline is written under Pytorch Lightning 2.5.0
# Usage
## Training
- First, train auto-encoder
  - Before run training script ```train_stage_1.sh``` in ```scripts``` folder, you must carefully setup the config file in ```configs/diffusion_cfg.py``` and ```configs/data_cfg.py```
    - ```configs/data_cfg.py``` has the train & test directory path arugments and data augmentation related things.
    - Since this repository is based on PyTorch Lightning modules, training setups like DDP and FSDP depend on Lightning's API and structure. Please check the ```train_stage_1.sh``` & ```trainer.py``` before running.

- Second, train unet with flow-matching method or ddpm(ddim)
  - Use train_stage_2.sh
    - Similarily, you must check the data_cfg.py before running this script.

## To do
- Update quantitative metrics for validation. (e.g., FID)
- Add additional analysis tools with callback function (e.g., MLFlow)
- Code base looks spagetti. So, I'll refactor configuration and related part for convenience in terms of usage and maintenance.
- Implement a method to empower condition in generation process.