from typing import Any
import pytorch_lightning as pl


class Diffusion(pl.LightningModule):
    def __init__(self, unet_model, vae_model):
        super().__init__()
        self.unet = unet_model
        self.vae_model = vae_model
    
    def training_step(self, batch):
        pass
    
    def validation_step(self, *args: Any, **kwargs: Any):
        pass

    def generate(self, x, cond=None):
        # cond can be image or text
        pass

