from torch import nn
from typing import Optional
from samplers.ddim import DDIMSampler
from samplers.ddpm import DDPMSampler
from models.unet import UNetModel
from models.autoencoder import AutoEncoder
from models.conditioner import ImagePrompter
import torch
from typing import List
from PIL import Image
from torchvision.utils import make_grid
import numpy as np

def get_sampler(cfg):
    if cfg.sampler_type == 'ddim':
        return DDIMSampler(cfg)
    elif cfg.sampler_type == 'ddpm':
        return DDPMSampler(cfg)
    else:
        raise NotImplementedError('You must check the sampler name in architecture configuration file')

def get_unet(cfg) -> UNetModel:
    return UNetModel(cfg)

def get_conditioner(cfg):
    if cfg.prompt_type == 'image':
        return ImagePrompter(cfg)
    else:
        raise NotImplementedError('Now, image prompt-type is only implemented')

def get_vae(cfg) -> AutoEncoder:
    return AutoEncoder(cfg)

def get_module(cfg, module_type):
    # this function must return right object if corresponding cfg variable is given 
    # For example, if cfg is sampler cfg, this function should return sampler object.
    # cfg is Dotdict-like type.

    if module_type == 'sampler':
        return get_sampler(cfg)
    elif module_type == 'unet':
        return get_unet(cfg)
    elif module_type == 'conditioner':
        return get_conditioner(cfg)
    elif module_type == 'vae':
        return get_vae(cfg)
    else:
        raise NotImplementedError("Check the 'module_type' argument. ")

def disable_model_training(model: Optional[nn.Module] = None) -> None:
    if model is None:
        return None
    
    for param in model.parameters():
        param.requires_grad = False
    model.eval()


class EMAModel(nn.Module):
    def __init__(self, model: nn.Module, decay=0.9999):
        super().__init__()
        if not (0.0 < decay < 1.0):
            raise ValueError('Momentum must be between 0 and 1')
        
        self.ema_params = {}
        self.global_iter = 0
        self.base_decay = decay

        for name, p in model.named_parameters():
            if p.requires_grad:
                ema_name = f"ema_{name}"
                self.ema_params[ema_name] = nn.Parameter(p.clone().detach(), requires_grad=False)

    @torch.no_grad()
    def forward(self, model):
        # update ema model

        decay = min(self.base_decay, (1 + self.global_iter) / (10 + self.global_iter))
        
        for name, p in model.named_parameters():
            if p.requires_grad:
                ema_name = f"ema_{name}"
                # New Weight : m * w_{prev} + (1-m) * a_{cur}
                self.ema_params[ema_name].mul_(decay).add_((1.0 - decay) * p) 

    @torch.no_grad()
    def copy2current(self, model:nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad:
                ema_name = f'ema_{name}'
                p.data.copy_(self.ema_params[ema_name].data)

def min_max_normalize(tensor: torch.Tensor, eps: float = 1e-7):
    """
    Min-Max Normalize a batch of images in (B, C, H, W) format.

    Args:
        tensor (torch.Tensor): Input tensor of shape (B, C, H, W).
        eps (float): Small value to prevent division by zero.

    Returns:
        torch.Tensor: Normalized tensor with values in range [0, 1].
    """
    B, C, H, W = tensor.shape  # (Batch, Channel, Height, Width)

    # Compute min and max per batch and channel
    min_vals = tensor.amin(dim=(2, 3), keepdim=True)  # Shape: (B, C, 1, 1)
    max_vals = tensor.amax(dim=(2, 3), keepdim=True)  # Shape: (B, C, 1, 1)

    # Normalize: (x - min) / (max - min)
    normalized = (tensor - min_vals) / (max_vals - min_vals + eps)

    return normalized

def tensor2images(tensor_images: torch.FloatTensor) -> List[Image.Image]:
    # tensor_images shape : (b, c, h, w)
    normalized_tensors = min_max_normalize(tensor_images)
    scaled_tensors = normalized_tensors * 255
    return scaled_tensors

def save_grid(tensor_images: torch.Tensor, save_path, nrow=4) -> None:
    grid_image = make_grid(tensor_images, nrow=nrow)
    grid_image = Image.fromarray(grid_image.cpu().detach().numpy().astype(np.uint8))
    grid_image.save(save_path)