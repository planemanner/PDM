from torch import nn
from typing import Optional
from samplers.ddim import DDIMSampler
from samplers.ddpm import DDPMSampler
from samplers.shortcut import ShortcutFlowSampler
from models.unet import UNetModel
from models.autoencoder import AutoEncoder
from models.conditioner import ImagePrompter, TextPrompter
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
    elif cfg.sampler_type == 'shortcut':
        return ShortcutFlowSampler(cfg)
    else:
        raise NotImplementedError('You must check the sampler name in architecture configuration file')

def get_unet(cfg) -> UNetModel:
    return UNetModel(cfg)

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

def get_conditioner(context_cfg):
    if context_cfg.context_type == "text":
        # CLIP Text Encoder
        return TextPrompter(context_cfg)
    
    elif context_cfg.context_type == "mask":
        # CLIP Image Encoder
        return ImagePrompter(context_cfg)
    else:
        raise ValueError("Please give correct context type. Currently, context can be one of types along 'text' and 'mask' ")

def disable_model_training(model: Optional[nn.Module] = None) -> None:
    if model is None:
        return None
    
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

def create_zero_images_batch(n, width, height, channels=3):
    # (n, height, width, channels) 형태의 배열 생성
    arrays = np.zeros((n, height, width, channels), dtype=np.uint8)
    
    # 각 배열을 PIL Image로 변환
    images = [Image.fromarray(array) for array in arrays]
    return images

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
        print("Current Decay : ", decay)
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

def denorm(x):
    return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)
    
def tensor2images(tensor_images: torch.FloatTensor) -> List[Image.Image]:
    # tensor_images shape : (b, c, h, w)
    normalized_tensors = denorm(tensor_images)
    scaled_tensors = normalized_tensors * 255
    return scaled_tensors

def save_grid(tensor_images: torch.Tensor, save_dir, nrow=4) -> None:
    import os
    tensor_images = tensor_images.permute(0, 2, 3, 1)
    
    for i, tensor_image in enumerate(tensor_images):
        grid_image = Image.fromarray(tensor_image.cpu().detach().numpy().astype(np.uint8))
        grid_image.save(os.path.join(save_dir, f"{i}_th_generation.png"))