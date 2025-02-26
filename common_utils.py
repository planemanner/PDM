from torch import nn
from typing import Optional
from PDM.samplers.sampler_legacy.ddim import DDIMSampler
from PDM.samplers.sampler_legacy.ddpm import DDPMSampler
from models.unet import UNetModel
from models.autoencoder import AutoEncoder

def get_module(cfg, module_type):
    # this function must return right object if corresponding cfg variable is given 
    # For example, if cfg is sampler cfg, this function should return sampler object.
    # cfg is Dotdict-like type.

    if module_type == 'sampler':
        if cfg['sampler_type'] == 'ddim':
            return DDIMSampler(cfg)
        elif cfg['sampler_type'] == 'ddpm':
            return DDPMSampler(cfg)
        else:
            raise NotImplementedError('You must check the sampler name in architecture configuration file')
    elif module_type == 'unet':
        return UNetModel(cfg)
    elif module_type == 'conditioner':
        # Image conditioner 정리되면 넣기. 그전엔 그냥 CLIP.
        pass
    elif module_type == 'vae':
        return AutoEncoder(cfg)
    else:
        raise NotImplementedError("Check the 'module_type' argument. ")
    return

def disable_model_training(model: Optional[nn.Module] = None) -> None:
    if model is None:
        return None
    
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    