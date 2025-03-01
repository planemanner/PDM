from torch import nn
from typing import Optional
from samplers.ddim import DDIMSampler
from samplers.ddpm import DDPMSampler
from models.unet import UNetModel
from models.autoencoder import AutoEncoder
from models.conditioner import ImagePrompter

def get_sampler(cfg):
    if cfg.sampler_type == 'ddim':
        return DDIMSampler(cfg)
    elif cfg.sampler_type == 'ddpm':
        return DDPMSampler(cfg)
    else:
        raise NotImplementedError('You must check the sampler name in architecture configuration file')

def get_unet(cfg):
    return UNetModel(cfg)

def get_conditioner(cfg):
    if cfg.prompt_type == 'image':
        return ImagePrompter(cfg)
    else:
        raise NotImplementedError('Now, image prompt-type is only implemented')

def get_vae(cfg):
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

class DotDict(dict):
    def __init__(self, d: dict={}):
        super().__init__()
        for key, value in d.items():
            self[key] = DotDict(value) if type(value) is dict else value

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(key)
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__