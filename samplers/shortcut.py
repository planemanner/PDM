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
    def __init__(self):
        super().__init__()
        # 가능한 shortcut 길이 정의
        self.shortcut_lengths = [1.0 / (2**i) for i in range(7, -1, -1)]  # [1/128, 1/64, ..., 1/2, 1]
        self.min_shortcut_length = self.shortcut_lengths[0]  # 1/128
    
    def sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Uniform distribution에서 t 샘플링"""
        return torch.rand(batch_size, device=device)
    
    def sample_d(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, int]:
        """Shortcut 길이 d 샘플링"""
        d_idx = torch.randint(0, len(self.shortcut_lengths), (batch_size,), device=device)
        d = torch.tensor([self.shortcut_lengths[idx.item()] for idx in d_idx], device=device)
        return d, d_idx
    
    def shortcut_step(self, 
                     model: nn.Module, 
                     x_t: torch.Tensor, 
                     t: torch.Tensor, 
                     d: torch.Tensor) -> torch.Tensor:
        """
        단일 shortcut step 수행
        Args:
            model: Flow 모델
            x_t: 입력 텐서
            t: 현재 시간 스텝
            d: shortcut 길이
        Returns:
            x_{t±d}: shortcut step 결과
        """
        # 모델 출력 정규화를 위한 스케일 계산
        model_output = model(x_t, t, d)
        norm = torch.norm(model_output, dim=1, keepdim=True)
        normalized_output = model_output / (norm + 1e-8)
        
        return normalized_output
    
    def forward(self, 
                model: nn.Module, 
                x_t: torch.Tensor, 
                training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Shortcut Flow Sampling/Training step
        Args:
            model: Flow 모델
            x_t: 입력 텐서
            training: 학습 모드 여부
        Returns:
            output: 모델 출력
            loss: 학습 시 손실값 (training=True인 경우)
        """
        batch_size = x_t.shape[0]
        device = x_t.device
        
        # t ~ U(0,1) 샘플링
        t = self.sample_t(batch_size, device)
        
        if training:
            # 학습 모드
            # d와 2d에 대한 shortcut step 수행
            d, d_idx = self.sample_d(batch_size, device)
            double_d_idx = torch.clamp(d_idx + 1, max=len(self.shortcut_lengths) - 1)
            double_d = torch.tensor([self.shortcut_lengths[idx.item()] 
                                   for idx in double_d_idx], device=device)
            
            # 첫 번째 shortcut step
            x_t_d = self.shortcut_step(model, x_t, t, d)
            
            # t+d에서의 두 번째 step을 위한 시간 계산
            t_plus_d = torch.clamp(t + d, max=1.0)
            
            # 두 번째 shortcut step
            x_t_2d = self.shortcut_step(model, x_t_d, t_plus_d, d)
            
            # 직접 2d step의 결과 계산 (target)
            target = self.shortcut_step(model, x_t, t, double_d)
            
            # MSE Loss 계산
            loss = torch.mean((x_t_2d - target) ** 2)
            
            return x_t_2d, loss
            
        else:
            # 추론 모드
            # 가장 효율적인 shortcut 길이 선택
            d = torch.tensor([self.shortcut_lengths[-1]], device=device).expand(batch_size)  # 시작은 가장 큰 d
            
            # Shortcut step 수행
            output = self.shortcut_step(model, x_t, t, d)
            
            return output, torch.tensor(0.0, device=device)
    
    @torch.no_grad()
    def sample(self, 
               model: nn.Module, 
               x_init: torch.Tensor, 
               steps: int = 10) -> torch.Tensor:
        """
        Shortcut Flow를 사용한 샘플링
        Args:
            model: Flow 모델
            x_init: 초기 입력
            steps: 샘플링 스텝 수
        Returns:
            생성된 샘플
        """
        x_t = x_init
        batch_size = x_init.shape[0]
        device = x_init.device
        
        for i in range(steps):
            # 현재 진행 상황에 따라 shortcut 길이 조정
            progress = i / steps
            if progress < 0.25:  # 초기 25%
                d_idx = -1  # 가장 큰 shortcut (d=1)
            elif progress < 0.5:  # 25%-50%
                d_idx = -2  # d=1/2
            elif progress < 0.75:  # 50%-75%
                d_idx = -3  # d=1/4
            else:  # 마지막 25%
                d_idx = 0  # 가장 작은 shortcut (d=1/128)
            
            d = torch.tensor([self.shortcut_lengths[d_idx]], device=device).expand(batch_size)
            t = self.sample_t(batch_size, device)
            
            x_t = self.shortcut_step(model, x_t, t, d)
        
        return x_t
