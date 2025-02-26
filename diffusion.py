from typing import Any, Optional, Union
import lightning as L
from PIL import Image
from common_utils import get_module, disable_model_training
import torch

class StableDiffusion(L.LightningModule):
    def __init__(self, unet_cfg, vae_cfg, sampler_cfg, conditioner_cfg=None):
        super().__init__()
        
        self.unet = get_module(unet_cfg, 'unet')
        self.vae_model = get_module(vae_cfg, 'vae') # The vae model should be already trained on image data.
        self.conditioner = get_module(conditioner_cfg, 'conditioner') # If it is None, this Diffusion model is to be trained without any condition
        self.sampler = get_module(sampler_cfg, 'sampler')

        disable_model_training(self.conditioner)
        disable_model_training(self.vae_model)

    def training_step(self, batch):
        pass
    
    def validation_step(self, *args: Any, **kwargs: Any):
        pass

    def forward(self, x):
        pass

    @torch.no_grad()
    def generate(self, x: Image.Image=None, cond: Optional[Union[str, Image.Image]]=None) -> Image.Image:
        """
        :x : input image
        :cond : 현재는 text Only
        학습 해왔던 이미지 사이즈 활용
        condition 도 마찬가지
        """
        posterior = self.vae.encode(x)
        z = posterior.sample()
        denoised = self.sampler(self.unet, z, cond)
        decoded = self.vae.decode(denoised)
        return decoded