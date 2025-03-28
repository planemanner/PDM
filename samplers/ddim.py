import torch
from tqdm import tqdm
from typing import Optional

from .ddpm import DDPMSampler

class DDIMSampler(DDPMSampler):
    def __init__(self, cfg):
        # This sampler is only used in inference stage.
        super().__init__(cfg)
        self.eta = 0.0

    def p_sample(self, model, xt, t, t_prev, cond=None) -> torch.FloatTensor:
        # Deterministic Sampling Only.
        pred_eps = model(xt, t, context=cond)
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
    
    def sampling(self, model, 
                 latent_shape, 
                 cond: torch.FloatTensor, 
                 uncond: Optional[torch.FloatTensor]=None,
                 n_steps: Optional[int]=None, 
                 clamp:bool = False,
                 cfg_weight: float=7.5):
        """
        latent_shape : (b, c, h, w)

        """
        if n_steps is None:
            n_steps = self.cfg.n_steps
            Warning('You are trying to do sampling with full steps. You cannot benefit of DDIM sampling')

        if n_steps < self.cfg.n_steps:
            skip = self.cfg.n_steps // n_steps
            timesteps = list(range(0, self.cfg.n_steps, skip))
            timesteps.reverse()
        else:
            timesteps = list(range(self.cfg.n_steps - 1, -1, -1))
        
        bsz = latent_shape[0]
        latent_shape = list(latent_shape)

        if uncond is not None:
            latent_shape[0] *= 2
            xt = torch.randn(latent_shape, device=cond.device)
            combined_cond = torch.cat([cond, uncond])

        else:
            xt = torch.randn(latent_shape, device=cond.device)
            combined_cond = cond

        for i, step_t in enumerate(tqdm(timesteps, desc='DDIM Sampling...')):
            if uncond is not None:
                step_t = torch.full((2 * bsz,), step_t, device=cond.device, dtype=torch.long)
                t_prev = timesteps[i+1] if i < len(timesteps)-1 else 0
                t_prev_tensor = torch.full((2 * bsz,), t_prev, device=cond.device, dtype=torch.long)
                xt = self.p_sample(model, xt, step_t, t_prev_tensor, combined_cond)
                xt, uncond_xt = xt.chunk(2)
                guided_xt = uncond_xt + cfg_weight * (xt - uncond_xt)
                xt = torch.cat([guided_xt, uncond_xt])
            else:
                step_t = torch.full((bsz, ), step_t, device=cond.device, dtype=torch.long)
                t_prev = timesteps[i+1] if i < len(timesteps)-1 else 0
                t_prev_tensor = torch.full((bsz,), t_prev, device=cond.device, dtype=torch.long)
                xt = self.p_sample(model, xt, step_t, t_prev_tensor, combined_cond)
            
            if clamp:
                xt = torch.clamp(xt, -1.0, 1.0)
        if uncond is not None:
            return xt[:bsz]
        else:
            return xt