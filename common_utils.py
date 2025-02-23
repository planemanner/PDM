from typing import Optional
from models.unet import UNetModel
from samplers.ddim import DDIMSampler
from models.autoencoder import AutoEncoder

MODULE_LIST = ['unet', 'sampler', 'vae', 'conditioner']

def get_module(cfg, module_type):
    """
    module_type 은 unet, sampler, vae
    """
    assert module_type in MODULE_LIST

    if module_type == 'unet':
        return UNetModel(cfg)
    
    elif module_type == 'sampler':
        return DDIMSampler(cfg)
    
    elif module_type == 'vae':
        return AutoEncoder(cfg)
    
    elif module_type == 'conditioner':
        # 만들어 놓고 !
        # 일단 CLIP 가정
        return 
    else:
        raise ValueError("You must check the 'module_type' argument.")