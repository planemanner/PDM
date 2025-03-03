from .ddpm import DDPMSampler
import torch
from tqdm import tqdm

class DDIMSampler(DDPMSampler):
    def __init__(self, cfg):
        # This sampler is only used in inference stage.
        super().__init__(cfg)
        
    def p_sample(self, model, xt, t, cond=None) -> torch.FloatTensor:
        # Deterministic Sampling Only.
        pred_eps = model(xt, t, cond)
        pred_x0  = (xt - self.sqrt_one_minus_alphas_cumprod[t] * pred_eps) / self.sqrt_alphas_cumprod[t]
        xt_minus_one = self.alphas_cumprod_prev[t] * pred_x0 + self.sqrt_one_minus_alphas_cumprod_prev[t] * pred_eps
        return xt_minus_one
    
    def sampling(self, model, image_shape, cond, n_steps=None, clamp:bool=True):
        if n_steps is None:
            n_steps = self.cfg.n_steps
            Warning('You are trying to do sampling with full steps. You cannot benefit of DDIM sampling')

        xt = torch.randn(image_shape, device=self.device)

        for step_i in tqdm(range(n_steps, -1, -1), desc='DDIM Sampling...'):
            xt = self.p_sample(model, xt, step_i, cond)
            if clamp:
                xt = torch.clamp(xt, -1.0, 1.0)
        return xt