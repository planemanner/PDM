"""
Trainer Interface 는 VQ-VAE 와 Diffusion 을 모두 포괄할 수 있도록 구성하기
기본적으로 Multi-Node DDP 로 구성
https://www.restack.io/p/pytorch-lightning-answer-vs-huggingface-trainer-cat-ai
"""
import lightning as L
from diffusion import StableDiffusion
from configs import trainer_cfg, unet_cfg, sampler_cfg, conditioner_cfg, autoencoder_cfg, data_cfg
from data.dataset import Sketch2ImageDataModule, VAEDataModule
from models.autoencoder import AutoEncoder
from lightning.pytorch.callbacks import ModelCheckpoint
import os


def train_diffusion(save_dir, seed: int=42, save_period: int=50):
    L.seed_everything(seed)
    model = StableDiffusion(unet_cfg=unet_cfg.unet_config,
                            vae_cfg=autoencoder_cfg.autoencoder_config,
                            sampler_cfg=sampler_cfg.sampler_config,
                            conditioner_cfg=conditioner_cfg.conditioner_config)
    
    data_module = Sketch2ImageDataModule(data_cfg.diffusion_data_config)

    trainer = L.Trainer(**trainer_cfg.trainer_diffusion_config)
    trainer.fit(model, data_module)

def train_autoencoder(save_dir, seed: int=42, save_period: int=50):
    L.seed_everything(seed)
    autoencoder = AutoEncoder(autoencoder_cfg.autoencoder_config)
    data_module = VAEDataModule(data_cfg.autoencoder_data_config)
    
    trainer = L.Trainer(**trainer_cfg.trainer_autoencoder_config)
    trainer.fit(autoencoder, data_module)

