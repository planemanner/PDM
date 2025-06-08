from torch import nn
import torch
from typing import Tuple
from tqdm import tqdm

class DDPMSampler(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg 
        self.v_posterior = cfg.v_posterior # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
        self.register_precomputed_values()

    def register_precomputed_values(self):
        betas = torch.linspace(self.cfg.linear_start ** 0.5, self.cfg.linear_end ** 0.5, self.cfg.n_steps) **2
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        # for time step 't-1' case
        alphas_cumprod_prev = torch.cat((torch.Tensor([1.0]), alphas_cumprod[:-1]))

        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        lvlb_weights = betas ** 2 / (2 * posterior_variance * torch.sqrt(alphas) * (1 - alphas_cumprod))                
        lvlb_weights[0] = lvlb_weights[1]

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('one_over_alphas_cumprod', 1 / alphas_cumprod)
        self.register_buffer('one_over_alphas', 1 / alphas)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_alphas_cumprod_prev', torch.sqrt(alphas_cumprod_prev))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod_prev', torch.sqrt(1.0 - alphas_cumprod_prev))
        self.register_buffer('lvlb_weights', lvlb_weights)
        
    def q_sample(self, x0: torch.FloatTensor, t: int) -> torch.Tensor:
        """
        Mathematical Expression 
        => x_{t} ~ q(x_{t}|x_{0}) = N(x_{t};\sqrt{\bar{\alpha}_{t}}x_{0}, \sqrt{1-\bar{\alpha}_{t}}I)
        What this function do ?
        sampling a x_{t} from the sampling distribution
        """
        eps = torch.randn_like(x0, device=x0.device)
        
        return self.sqrt_alphas_cumprod[t][:, None, None, None] * x0 + self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None] * eps, eps
    
    def p_sample(self, model, xt, t, cond=None) -> torch.FloatTensor:
        """
        Mathematical Expression
        => x_{t-1} = \frac{1}{\bar{\alpha}_{t}}(x_{t} - \frac{1-\alpha_{t}}{\sqrt{1-\bar{\alpha}_{t}}}\epsilon_{\theta}(x_{t}, t, c)) + \sigma_{t}z
        z ~ N(0, I)
        \sigma_{t}^{2} = \beta_{t} or \frac{1 - \bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t}}\beta_{t}
        Since experimental results show similar performance, for simplicity, I chosen \beta_{t} as \sigma_{t}^{2}
        Transition Expression
        x_{t} -> x_{t-1}
        """
        pred_eps = model(xt, t, cond)
        noise = torch.randn_like(xt, device=xt.device)
        return self.one_over_alphas[t][:, None, None, None] * (xt - self.betas[t][:, None, None, None] / self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None] * pred_eps) + torch.sqrt(self.betas[t][:, None, None, None]) * noise
    
    def sampling(self, model, 
                 image_shape: Tuple[int], 
                 cond=None, clamp:bool=True):
        xt = torch.randn(image_shape, device=self.device)

        for step_i in tqdm(range(self.cfg.n_steps, -1, -1), desc="Sampling..."):
            xt = self.p_sample(model, xt, step_i, cond)
            if clamp:
                xt = torch.clamp(xt, -1.0, 1.0)
        
        return xt