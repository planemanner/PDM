import pytorch_lightning as pl
import torch
from torch import nn
import numpy as np 
from typing import TypedDict, List

class BatchDict(TypedDict):
    images: List[torch.FloatTensor]
    texts : List[str]

def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta, verbose=True):
    # select alphas for computing the variance schedule
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())

    # according the the formula provided in https://arxiv.org/abs/2010.02502
    sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    if verbose:
        print(f'Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}')
        print(f'For the chosen value of eta, which is {eta}, '
              f'this results in the following sigma_t schedule for ddim sampler {sigmas}')
    return sigmas, alphas, alphas_prev


def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=True):
    if ddim_discr_method == 'uniform':
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == 'quad':
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
    else:
        raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

    # assert ddim_timesteps.shape[0] == num_ddim_timesteps
    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    steps_out = ddim_timesteps + 1
    if verbose:
        print(f'Selected timesteps for ddim sampler: {steps_out}')
    return steps_out

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
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def extract_into_tensor(alpha_like:torch.FloatTensor, t: torch.IntTensor, x_dim):
    # alpha_like : sqrt_recip_alphas_cumprod or similar things...
    # t : time step
    b, *_ = t.shape
    out = alpha_like.gather(-1, t)
    # 아래 연산은 unsqueeze 와 같음.
    return out.reshape(b, *((1,) * (x_dim - 1)))

class DiffusionWrapper(pl.LightningModule):
    def __init__(self, model, conditioning_key):
        super().__init__()
        self.diffusion_model = model
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm']

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None):
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t)
        elif self.conditioning_key == 'crossattn':
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc)
        elif self.conditioning_key == 'hybrid':
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc)
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        else:
            raise NotImplementedError()

        return out
    
def make_beta_schedule(n_timestep, linear_start=1e-4, linear_end=2e-2) -> torch.FloatTensor:
    # if schedule == "linear":
    #     # 왜 torch 로 생성했다가 numpy 로 return 해 ?
    # numpy 와 rounding error 정도의 차이밖에 없음. 소수점 4째자리 이상
    betas = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    return betas
    # elif schedule == "cosine":
    #     timesteps = (
    #             torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
    #     )
    #     alphas = timesteps / (1 + cosine_s) * np.pi / 2
    #     alphas = torch.cos(alphas).pow(2)
    #     alphas = alphas / alphas[0]
    #     betas = 1 - alphas[1:] / alphas[:-1]
    #     betas = np.clip(betas, a_min=0, a_max=0.999)

    # elif schedule == "sqrt_linear":
    #     betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    # elif schedule == "sqrt":
    #     betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    # else:
    #     raise ValueError(f"schedule '{schedule}' unknown.")
    # return betas.numpy()


def make_ddim_timesteps(ddim_discr_method, n_ddim_timesteps, n_ddpm_timesteps, verbose=True):
    if ddim_discr_method == 'uniform':
        c = n_ddpm_timesteps // n_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, n_ddpm_timesteps, c)))

    elif ddim_discr_method == 'quad':
        ddim_timesteps = ((np.linspace(0, np.sqrt(n_ddpm_timesteps * .8), n_ddim_timesteps)) ** 2).astype(int)
    else:
        raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

    # assert ddim_timesteps.shape[0] == num_ddim_timesteps
    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    steps_out = ddim_timesteps + 1
    if verbose:
        print(f'Selected timesteps for ddim sampler: {steps_out}')
    return steps_out


def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta, verbose=True):
    # select alphas for computing the variance schedule
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())

    # according the the formula provided in https://arxiv.org/abs/2010.02502
    sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    if verbose:
        print(f'Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}')
        print(f'For the chosen value of eta, which is {eta}, '
              f'this results in the following sigma_t schedule for ddim sampler {sigmas}')
    return sigmas, alphas, alphas_prev


betas = make_beta_schedule(10)
print(betas)
# alphas = 1 - betas
# alphas_cumprod = torch.cat((torch.Tensor([1.0,]), torch.cumprod(alphas, dim=0)[:-1]))
# alphas_cumprod_np = np.cumprod(alphas.numpy(), axis=0)
# alphas_cumprod_prev = np.append(1., alphas_cumprod_np)
# n, = alphas_cumprod.shape
# print(n)
# print(torch.maximum(alphas_cumprod, torch.tensor([1.0])))
# print(np.maximum(alphas_cumprod_np, 1))
# print(alphas_cumprod_prev)