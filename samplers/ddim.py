from .ddpm import DDPMSampler
import torch
from tqdm import tqdm

class DDIMSampler(DDPMSampler):
    def __init__(self, cfg):
        # This sampler is only used in inference stage.
        super().__init__(cfg)
        self.eta = 0.0

    def p_sample(self, model, xt, t, t_prev,cond=None) -> torch.FloatTensor:
        # Deterministic Sampling Only.
        pred_eps = model(xt, t, cond)
        pred_x0  = (xt - self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None] * pred_eps) / self.sqrt_alphas_cumprod[t][:, None, None, None]

        alpha_cumprod_t = self.alphas_cumprod[t][:, None, None, None]
        alpha_cumprod_t_prev = self.alphas_cumprod[t_prev][:, None, None, None]
        
        sigma_t = self.eta * torch.sqrt((1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))
        
        if self.eta > 0.0:
            noise = torch.randn_like(xt)
        
        c1 = torch.sqrt(alpha_cumprod_t_prev)
        c2 = torch.sqrt(1 - alpha_cumprod_t_prev - sigma_t**2)
        if self.eta > 0.0:
            xt_prev = c1 * pred_x0 + c2 * pred_eps + sigma_t * noise
        else:
            xt_prev = c1 * pred_x0 + c2 * pred_eps

        return xt_prev
    
    def sampling(self, model, latent_shape, cond, n_steps=None, clamp:bool=False):
        if n_steps is None:
            n_steps = self.cfg.n_steps
            Warning('You are trying to do sampling with full steps. You cannot benefit of DDIM sampling')

        if n_steps < self.cfg.n_steps:
            skip = self.cfg.n_steps // n_steps
            timesteps = list(range(0, self.cfg.n_steps, skip))
            timesteps.reverse()
        else:
            timesteps = list(range(self.cfg.n_steps - 1, -1, -1))
            
        xt = torch.randn(latent_shape, device=cond.device)
        
        bsz = len(xt)
        
        for i, step_t in enumerate(tqdm(timesteps, desc='DDIM Sampling...')):
            step_t = torch.full((bsz, ), step_t, device=cond.device, dtype=torch.long)
            t_prev = timesteps[i+1] if i < len(timesteps)-1 else 0
            t_prev_tensor = torch.full((bsz,), t_prev, device=cond.device, dtype=torch.long)
            xt = self.p_sample(model, xt, step_t, t_prev_tensor, cond)
            
            if clamp:
                xt = torch.clamp(xt, -1.0, 1.0)
                
        return xt