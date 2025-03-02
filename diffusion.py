from typing import Optional, Union
import lightning as L
from PIL import Image
from common_utils import get_module, disable_model_training, EMAModel
import torch
from torch.nn import functional as F
from typing import List
from contextlib import contextmanager

class StableDiffusion(L.LightningModule):
    def __init__(self, unet_cfg, vae_cfg, sampler_cfg, conditioner_cfg=None):
        super().__init__()
        
        self.unet = get_module(unet_cfg, 'unet')
        self.vae_model = get_module(vae_cfg, 'vae') # The vae model should be already trained on image data.
        self.conditioner = get_module(conditioner_cfg, 'conditioner') # If it is None, this Diffusion model is to be trained without any condition
        self.sampler = get_module(sampler_cfg, 'sampler')

        disable_model_training(self.conditioner)
        disable_model_training(self.vae_model)

        self.n_timesteps = sampler_cfg.timesteps
        
        if unet_cfg.mode == 'train':
            self.ema = EMAModel(self.unet)

    def training_step(self, batch, batch_idx):
        # images : List of float tensors
        # sketches : List of PIL images.
        images, sketches = batch
        conds = self.conditioner(sketches)
        z = self.vae_model.encode(images).sample()
        t = torch.randint(0, self.n_timesteps, (len(images), ), device=self.device).long()
        noisy_inputs, labels = self.sampler.q_sample(z, t)
        pred_noises = self.unet(noisy_inputs, t, conds)
        loss = F.mse_loss(pred_noises, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        # Get MSE LOSS 
        images, sketches = batch
        decoded = self.generate_sketch2image(sketches)
        # 복원된 images 와 reference images 사이의 차이 기록

    @torch.no_grad()
    def generate_sketch2image(self, prompts: List[Image.Image]) -> Image.Image:
        conds = self.conditioner(prompts)
        denoised_z = self.sampler.sampling(self.unet, conds.shape, conds, n_steps=200)
        decoded = self.vae.decode(denoised_z)
        return decoded

    def configure_optimizers(self):
        lr = 2e-06 # Generally used.
        params = self.unet.parameters()
        opt = torch.optim.AdamW(params, lr=lr)
        return opt
    
    def on_train_batch_end(self):
        # model ema update
        pass