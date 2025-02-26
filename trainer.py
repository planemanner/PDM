"""
Trainer Interface 는 VQ-VAE 와 Diffusion 을 모두 포괄할 수 있도록 구성하기
기본적으로 Multi-Node DDP 로 구성
https://www.restack.io/p/pytorch-lightning-answer-vs-huggingface-trainer-cat-ai
"""
import lightning as L
from diffusion import StableDiffusion

# AutoEncoder Trainer

# UNet Diffuser Trainer

# Adapter Trainer : to be done after figuring out all codes are good.

