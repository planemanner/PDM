from torch import nn
from typing import Optional
from samplers.ddim import DDIMSampler
from samplers.ddpm import DDPMSampler
from models.unet import UNetModel
from models.autoencoder import AutoEncoder
from models.conditioner import ImagePrompter
import torch

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
                self.ema_params[ema_name].mul_(decay).add_(1.0 - decay * p) 

    @torch.no_grad()
    def copy2current(self, model:nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad:
                ema_name = f'ema_{name}'
                p.data.copy_(self.ema_params[ema_name].data)
                
"""
import torch
from torch import nn

class LitEma(nn.Module):
    def __init__(self, model, decay=0.9999, use_num_upates=True):
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')

        self.m_name2s_name = {}
        self.register_buffer('decay', torch.tensor(decay, dtype=torch.float32))
        self.register_buffer('num_updates', torch.tensor(0,dtype=torch.int) if use_num_upates
                             else torch.tensor(-1,dtype=torch.int))

        for name, p in model.named_parameters():
            if p.requires_grad:
                #remove as '.'-character is not allowed in buffers
                s_name = name.replace('.','')
                self.m_name2s_name.update({name:s_name})
                self.register_buffer(s_name,p.clone().detach().data)

        self.collected_params = []

    def forward(self,model):
        decay = self.decay

        if self.num_updates >= 0:
            self.num_updates += 1
            decay = min(self.decay,(1 + self.num_updates) / (10 + self.num_updates))

        one_minus_decay = 1.0 - decay

        with torch.no_grad():
            m_param = dict(model.named_parameters())
            shadow_params = dict(self.named_buffers())

            for key in m_param:
                if m_param[key].requires_grad:
                    sname = self.m_name2s_name[key]
                    shadow_params[sname] = shadow_params[sname].type_as(m_param[key])
                    shadow_params[sname].sub_(one_minus_decay * (shadow_params[sname] - m_param[key]))
                else:
                    assert not key in self.m_name2s_name

    def copy_to(self, model):
        m_param = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())
        for key in m_param:
            if m_param[key].requires_grad:
                m_param[key].data.copy_(shadow_params[self.m_name2s_name[key]].data)
            else:
                assert not key in self.m_name2s_name

    def store(self, parameters):
        
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)


"""