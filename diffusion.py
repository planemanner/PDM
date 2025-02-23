from torch import nn
import lightning as L
from typing import List, Union, Optional
from common_utils import get_module


class StableDiffusion(L.LightningModule):
    def __init__(self, unet_cfg, sampler_cfg, vae_cfg, conditioner_cfg=None):
        super().__init__()
        # Device 관련된건 나중에
        # 기본적으로 CLIP 기반을 가정
        self.unet = get_module(unet_cfg)
        self.vae = get_module(vae_cfg)
        self.sampler = get_module(sampler_cfg)
        self.conditioner = get_module(conditioner_cfg)
    
    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, *args, **kwargs):
        return super().validation_step(*args, **kwargs)
    
    def generate(self, x, cond=None):
        pass

    def log_images(self):
        pass

    def progressive_sampling(self):
        pass

    