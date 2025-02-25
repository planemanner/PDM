import torch
from torch import nn

class DDIMSampler(nn.Module):
    def __init__(self, n_steps:int=1000, linear_start: float=1e-4, linear_end: float=2e-2):
        super().__init__()
        # The sampler_cfg should have the same hyperparameters shared with DDPM.
        self.register_schedule(n_steps, linear_start, linear_end)

    def register_schedule(self, n_steps, linear_start=1e-4, linear_end=2e-2):
        betas = torch.linspace(linear_start**0.5, linear_end**0.5, n_steps) ** 2
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat((torch.Tensor([1.0]), alphas_cumprod[:-1]))
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1-alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
    def pred_x0(self, model, xt, t, context=None):
        pred_eps = model(xt, t, context)
        x0_hat = (xt - self.sqrt_one_minus_alphas_cumprod[t] * pred_eps) / self.alphas_cumprod[t]
        return x0_hat
    
    def get_xt(self, x0: torch.Tensor, t: int, value_clip:bool=True):
        eps = torch.randn_like(x0, device=x0.device)
        return self.sqrt_alphas_cumprod[t] * x0 + self.sqrt_one_minus_alphas_cumprod[t] * eps

    def from_xt_to_xt_1(self, xt: torch.Tensor, t:int, ) -> torch.Tensor:
        pass

    def sampling(self, model, latent_var=None, condition=None):
        """
        model : UNet
        input_image : 
        Deterministic Sampling 
        """
        pass

if __name__ == "__main__":

    x0 = torch.randn(4, 3, 12, 12)
    sampler = DDIMSampler()
    v = sampler.get_xt(x0, 2)
    print(v.shape)