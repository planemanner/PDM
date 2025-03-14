"""
This is the implementation of the shortcut sampler.
ICLR 2025, https://arxiv.org/pdf/2410.12557
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional

"""
1. ||S_{\theta}(x_{t}, t, d)||=1 인지 Check 필요
2. d 에 대한 명확한 정의 필요
3. t, d 값을 sampling 할 때 무슨 관계가 있는 지 확인 필요
- We find that sampling t ∼ U(0,1) uniformly is the simplest and works as well as any other sampling scheme
위 맥락에 따라, t 는 Uniform Distribution 으로 샘플링 해야 함.
4.  d ∈ (1/128,1/64 ... 1/2,1). During each training step, we sample xt, t, and a random d <1,then take two sequential steps with the shortcut model.
This creates log2(128) + 1 = 8 possible shortcut lengths according to d ∈ (1/128,1/64 ... 1/2,1).
During each training step, we sample xt, t, and a random d <1,then take two sequential steps with the shortcut model.
The concatenation of these two steps is then used as the target to train the model at 2d.

!! Note that the second step is queried at x′ t+d under the denoising ODE and not the empirical data pairing, i.e. it is constructed by adding the predicted first shortcut to xt
When d is at the smallest value (e.g. 1/128), we instead query the model at d = 0.
"""
class ShortcutFlowSampler(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 가능한 shortcut 길이 정의
        self.shortcut_lengths = [1.0 / (2**i) for i in range(cfg.shortcut_length, -1, -1)]  # [1/128, 1/64, ..., 1/2, 1]
        self.min_shortcut_length = self.shortcut_lengths[0]  # 1/128
        self.max_steps = int(1 / self.shortcut_lengths[0])

    def sample_t(self, batch_size: int, d_idx, device: torch.device) -> torch.Tensor:
        d_section = torch.pow(2, d_idx)
        t = torch.zeros(batch_size, device=device)
        for i in range(batch_size):
            t[i] = torch.randint(low=0, high=d_section.item(), size=(1,), device=device).item()
        t = t / d_section
        return t
    
    def sample_d(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, int]:
        # You must sample d before sampling t
        d_idx = torch.randint(0, len(self.shortcut_lengths), (batch_size,), device=device)
        d = torch.tensor([self.shortcut_lengths[idx.item()] for idx in d_idx], device=device)
        
        return d, d_idx
    
    def shortcut_step(self, model: nn.Module, 
                      x_t: torch.FloatTensor, 
                      t: torch.IntTensor, 
                      d: torch.FloatTensor,
                      prompts: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.IntTensor]:
        model_output = model(x_t, t, d, prompts) # s_{\theta}(x_{t}, t, d)
        """
        Refer to Heun Method.
        """
        x_t_plus_d = x_t + 0.5 * d[:, None, None, None] * model_output
        return x_t_plus_d, t+d, model_output
    
    @torch.no_grad()
    def sampling(self, 
                 model: nn.Module,
                 latent_shape: Tuple[int, int, int, int],
                 prompts: torch.FloatTensor, 
                 n_steps: int = 64) -> torch.Tensor:

        assert n_steps <= self.max_steps
        bsz = latent_shape[0]
        device = prompts.device
        x_t = torch.randn(*latent_shape, device=device)
        t = torch.zeros(bsz, device=device)
        d = torch.tensor([self.shortcut_lengths[0]], device=device).expand(bsz)

        for i in range(n_steps):
            
            x_t, t, model_output = self.shortcut_step(model, x_t, t, d, prompts)
        
        return x_t
