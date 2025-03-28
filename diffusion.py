import lightning as L
from PIL import Image
from common_utils import disable_model_training, tensor2images, save_grid, get_conditioner, create_zero_images_batch
import torch
from torch.nn import functional as F
from typing import List
import os
from samplers.shortcut import ShortcutFlowSampler
from samplers.ddim import DDIMSampler
from configs.diffusion_cfg import FlowConfig, DDConfig
from models.unet import UNetModel
from models.autoencoder import AutoEncoder

class StableDiffusion(L.LightningModule):
    def __init__(self, diff_cfg):
        super().__init__()
        
        self.unet = UNetModel(diff_cfg.unet)
        self.vae = AutoEncoder(diff_cfg.ae)
        self.unet_input_channels = diff_cfg.unet.in_channels

        vae_state_dict = torch.load(diff_cfg.ae_ckpt_path, map_location="cpu")
        if "state_dict" in vae_state_dict:
            vae_state_dict = vae_state_dict["state_dict"]        
        self.vae.load_state_dict(vae_state_dict)
        self.conditioner = get_conditioner(diff_cfg.context)

        if diff_cfg.sampler_type == "flow-matching":
            self.sampler = ShortcutFlowSampler(FlowConfig)
            self.n_timesteps = FlowConfig.n_steps
        else:
            self.sampler = DDIMSampler(DDConfig)
            self.n_timesteps = DDConfig.n_steps

        disable_model_training(self.conditioner)
        disable_model_training(self.vae)

        self.ckpt_save_dir = diff_cfg.diffusion_ckpt_save_dir
        self.lr = diff_cfg.lr
        self.sample_save_dir = diff_cfg.sample_save_dir
        self.save_period = diff_cfg.save_period
        self.isflowmodel = True if isinstance(self.sampler, ShortcutFlowSampler) else False 

    def training_step(self, batch, batch_idx):
        # images : List of float tensors
        # sketches : List of PIL images.
        images, sketches = batch      
        conds = self.conditioner(sketches)
        z = self.vae.encode(images).sample()

        if not self.isflowmodel:
            # DDPM or DDIM    
            t = torch.randint(0, self.n_timesteps, (len(images), ), device=self.device).long()
            noisy_inputs, labels = self.sampler.q_sample(z, t)
            
            pred_noises = self.unet(noisy_inputs, t, conds)
            loss = F.mse_loss(pred_noises, labels)
            self.log('TRAIN_LOSS', loss.item(), prog_bar=True, on_step=True, on_epoch=True)
            
            return loss
        else:
            d, d_idx = self.sampler.sample_d(len(images), self.device)
            t = self.sampler.sample_t(len(images), d_idx, self.device)
            
            d0 = torch.zeros_like(d, device=self.device)
            x0 = torch.randn_like(z).to(self.device)
            x_t = (1-t)[:, None, None, None] * x0 + t[:, None, None, None] * z
            flow_labels = z - x0

            with torch.no_grad():
                x_t_plus_d, t_plus_d, s_first = self.sampler.shortcut_step(self.unet, x_t, t, d, conds)
                _, _, s_second = self.sampler.shortcut_step(self.unet, x_t_plus_d, t_plus_d, d, conds)
                s_second = torch.clip(s_second, -4, 4)
                s_target = 0.5 * (s_first + s_second)
            
            x_t_plus_d0, _, s_0 = self.sampler.shortcut_step(self.unet, x_t, t, d0, conds)
            _, _, self_consistency = self.sampler.shortcut_step(self.unet, x_t_plus_d0, t, 2 * d, conds)
            loss = F.mse_loss(s_0, flow_labels) + F.mse_loss(self_consistency, s_target)
            self.log('TRAIN_LOSS', loss.item(), prog_bar=True, on_step=True, on_epoch=True)
            return loss

    def validation_step(self, batch, batch_idx):
        images, sketches = batch
        decoded = self.generate_sketch2image(sketches)

        if batch_idx == 0 and self.global_rank == 0:
            save_dir = os.path.join(self.sample_save_dir, str(self.current_epoch).zfill(3))
            os.makedirs(save_dir, exist_ok=True)
            save_grid(decoded, save_dir)

    @torch.no_grad()
    def generate_sketch2image(self, prompts: List[Image.Image], 
                              latent_shape: List[int]=[32, 32],
                              cfg_weight: float=7.5,
                              n_steps: int = 128,
                              use_cfg: bool=True) -> torch.IntTensor:
        
        conds = self.conditioner(prompts, latent_shape)

        if use_cfg:
            w, h = prompts[0].size
            uncond_prompts = create_zero_images_batch(len(prompts), w, h)
            unconds = self.conditioner(uncond_prompts, latent_shape)
        else:
            unconds = None

        latent_shape = [len(conds), self.unet_input_channels] + latent_shape 
        denoised_z = self.sampler.sampling(self.unet, 
                                           latent_shape, 
                                           conds,
                                           unconds, 
                                           cfg_weight=cfg_weight,
                                           n_steps=n_steps)
        
        decoded = self.vae.decode(denoised_z) # Tensor Images
        decoded = tensor2images(decoded) # 8-bit tensors
        
        return decoded

    def configure_optimizers(self):
        lr = self.lr # Generally used.
        params = self.unet.parameters()
        opt = torch.optim.AdamW(params, lr=lr)
        return opt
    
    def ema_update(self):
        # This is required to enhance sampling quality. 
        pass

    def on_train_epoch_end(self, *args, **kwargs):
        if (self.current_epoch + 1) % self.save_period == 0 or self.current_epoch == 0:
            ckpt_path = f"{self.ckpt_save_dir}/diffusion_epoch_{(self.current_epoch+1):03d}.ckpt"
            self.trainer.save_checkpoint(ckpt_path)
        