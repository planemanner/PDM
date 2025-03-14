from typing import Optional, Union
import lightning as L
from PIL import Image
from common_utils import get_module, disable_model_training, tensor2images, save_grid
import torch
from torch.nn import functional as F
from typing import List
import os
from samplers.shortcut import ShortcutFlowSampler
from configs.diffusion_cfg import flow_sampler_config

class StableDiffusion(L.LightningModule):
    def __init__(self, unet_cfg, vae_cfg, sampler_cfg, conditioner_cfg, save_dir:str, save_period:int):
        super().__init__()
        
        self.unet = get_module(unet_cfg, 'unet')
        self.vae_model = get_module(vae_cfg, 'vae') # The vae model should be already trained on image data.
        vae_state_dict = torch.load(vae_cfg.common.ckpt_path, map_location="cpu")
        if "state_dict" in vae_state_dict:
            vae_state_dict = vae_state_dict["state_dict"]        
        self.vae_model.load_state_dict(vae_state_dict)
        self.conditioner = get_module(conditioner_cfg, 'conditioner') # If it is None, this Diffusion model is to be trained without any condition

        if unet_cfg.use_step_embed:
            self.sampler = ShortcutFlowSampler(flow_sampler_config)
        else:
            self.sampler = get_module(sampler_cfg, 'sampler')

        disable_model_training(self.conditioner)
        disable_model_training(self.vae_model)

        self.n_timesteps = sampler_cfg.n_steps
        """
        if unet_cfg.mode == 'train':
            self.ema = EMAModel(self.unet)
        """
        self.ckpt_save_dir = save_dir
        self.lr = unet_cfg.lr
        self.sample_save_dir = unet_cfg.sample_save_dir
        self.save_period = save_period
        self.isflowmodel = True if isinstance(self.sampler, ShortcutFlowSampler) else False 

    def training_step(self, batch, batch_idx):
        # images : List of float tensors
        # sketches : List of PIL images.
        images, sketches = batch      
        conds = self.conditioner(sketches)
        z = self.vae_model.encode(images).sample()

        if not self.isflowmodel:    
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
                x_t_plus_d, _, s_first = self.sampler.shortcut_step(self.unet, x_t, t, d, conds)
                _, _, s_second = self.sampler.shortcut_step(self.unet, x_t_plus_d, t, d, conds)
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
    def generate_sketch2image(self, prompts: List[Image.Image], latent_shape: List[int]=[4, 32, 32]) -> torch.IntTensor:
        
        conds = self.conditioner(prompts)
        latent_shape = [len(conds)] + latent_shape 
        denoised_z = self.sampler.sampling(self.unet, latent_shape, conds, n_steps=64)
        decoded = self.vae_model.decode(denoised_z) # Tensor Images
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
        