"""
Trainer Interface 는 VQ-VAE 와 Diffusion 을 모두 포괄할 수 있도록 구성하기
기본적으로 Multi-Node DDP 로 구성
https://www.restack.io/p/pytorch-lightning-answer-vs-huggingface-trainer-cat-ai
"""
import lightning as L
from diffusion import StableDiffusion
from configs import data_cfg, diffusion_cfg
from data.dataset import Sketch2ImageDataModule, VAEDataModule
from models.autoencoder import AutoEncoder
from lightning.pytorch.callbacks import ModelCheckpoint

def train_diffusion(save_dir, seed: int=42, save_period: int=50):
    L.seed_everything(seed)
    model = StableDiffusion(unet_cfg=diffusion_cfg.unet_config,
                            vae_cfg=diffusion_cfg.ae_config,
                            sampler_cfg=diffusion_cfg.sampler_config,
                            conditioner_cfg=diffusion_cfg.conditioner_config,
                            save_dir=save_dir,
                            save_period=save_period)
    
    data_module = Sketch2ImageDataModule(data_cfg.diffusion_data_config)

    trainer = L.Trainer(**diffusion_cfg.trainer_diff_cfg)
    trainer.fit(model, data_module)

def train_autoencoder(save_dir, seed: int=42, save_period: int=50):
    L.seed_everything(seed)
    autoencoder = AutoEncoder(diffusion_cfg.ae_config, save_dir=save_dir, save_period=save_period)
    data_module = VAEDataModule(data_cfg.autoencoder_data_config)
    
    trainer = L.Trainer(**diffusion_cfg.trainer_ae_cfg)
    trainer.fit(autoencoder, data_module)

