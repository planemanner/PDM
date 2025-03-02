import torch
from torch import nn
import numpy as np
from functools import partial
from einops import rearrange, repeat
from contextlib import contextmanager
from torchvision.utils import make_grid
from tqdm import tqdm
from utils import extract_into_tensor, noise_like, make_beta_schedule, BatchDict

class DDPMSampler(nn.Module):
    # classic DDPM with Gaussian diffusion, in image space
    # 이건 Sampler 뿐만 아니라 그냥 Diffusion 모델 자체라고 보면 됨
    # 따라서 이것과 Sampler 를 분리해야함
    def __init__(self, sampler_cfg):
        self.sampler_cfg = sampler_cfg
        self.register_schedule(n_steps=self.sampler_cfg.timesteps, 
                               linear_start=self.sampler_cfg.linear_start, 
                               linear_end=self.sampler_cfg.linear_end)
        
    def register_schedule(self, n_steps=1000, linear_start=1e-4, linear_end=2e-2):
        betas = make_beta_schedule(n_steps, linear_start=linear_start, linear_end=linear_end)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0) # \bar{\alpha}
        alphas_cumprod_prev = torch.cat((torch.Tensor([1.0]), alphas_cumprod[:-1]))
        # timesteps, = betas.shape
        self.n_timesteps = int(n_steps)
        self.linear_start = linear_start
        self.linear_end = linear_end

        assert alphas_cumprod.shape[0] == self.n_timesteps, 'alphas have to be defined for each timestep'
        self.register_buffer('alphas', alphas)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.sampler_cfg.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.sampler_cfg.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', posterior_variance)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', torch.log(torch.maximum(posterior_variance, torch.tensor([1e-20])))) 
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        lvlb_weights = betas ** 2 / (2 * posterior_variance * torch.sqrt(alphas) * (1 - alphas_cumprod))
        lvlb_weights[0] = lvlb_weights[1] # ?
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    def q_mean_variance(self, x_start, t):
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, xt, t, clip_denoised: bool):
        model_out = model(xt, t) # eps
        # Get x0 from given x_t and t
        x_recon = self.predict_start_from_noise(xt, t=t, noise=model_out)

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=xt, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        x_shape = x.shape
        b, device = x_shape[0], x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x_shape, device, repeat_noise)
        # no noise when t == 0
        # 아래는 unsqueeze operation 임
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x_shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device) # x_T
        intermediates = [img]
    
        for i in tqdm(range(self.n_timesteps, -1, -1), desc='Sampling t', total=self.n_timesteps):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t,
                                clip_denoised=self.sampler_cfg.clip_denoised)
            
            if i % self.sampler_cfg.log_every_t == 0 or i == self.n_timesteps - 1:
                intermediates.append(img)

        if return_intermediates:
            return img, intermediates
        return img

    def xT_to_x0(self, input_shape, model, context=None, return_intermediate:bool=False):
        if return_intermediate:
            intermediate = []

        t_step_img = torch.randn(input_shape, device=model.device)

        for t in tqdm(range(self.n_timesteps, -1, -1), desc='Sampling...', total=self.n_timesteps):
            if return_intermediate:
                intermediate.append(t_step_img)
            pred_noise = model(t_step_img, t, context)
            t_step_img = self.xt_to_xt_minus_one(t_step_img, t, pred_noise)
        return t_step_img

    def xt_to_xt_minus_one(self, xt, t, predicted_noise):
        noise = torch.randn_like(xt)
        return (1 / torch.sqrt(self.alphas[t])) * (xt - self.betas[t] / self.sqrt_one_minus_alphas_cumprod[t] * predicted_noise) + torch.sqrt(self.betas[t]) * noise
    
    # def x0_to_xt(self, x_start: torch.FloatTensor, t: torch.IntTensor, noise: torch.FloatTensor=None):
    #     # forward process
    #     noise = torch.randn_like(x_start) if noise is None else noise

    #     return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.ndim) * x_start +
    #             extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.ndim) * noise)
    
    def _get_rows_from_list(self, samples):
        # Do not use einops operation during training because it always occurs a bottleneck.
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, input_key: str='image'):
        log = dict()
        x = self.get_input(batch, input_key)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]
        log["inputs"] = x

        # get diffusion row
        diffusion_row = list()
        x_start = x[:n_row]

        for t in range(self.num_timesteps):
            if t % self.sampler_cfg.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                t = t.to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)

        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, denoise_row = self.sample(batch_size=N, return_intermediates=True)

            log["samples"] = samples
            log["denoise_row"] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log
    
    # def get_input(self, batch: BatchDict, image_key='images'):
    #     imgs = batch[image_key]
    #     assert imgs.ndim == 4, "You must give the input batch having shape : (b, c, h, w)"
    #     return imgs

    # def configure_optimizers(self):
    #     lr = self.learning_rate
    #     params = list(self.model.parameters())
    #     if self.learn_logvar:
    #         params = params + [self.logvar]
    #     opt = torch.optim.AdamW(params, lr=lr)
    #     return opt
    
    # @contextmanager
    # def ema_scope(self, context=None):
    #     if self.use_ema:
    #         self.model_ema.store(self.model.parameters())
    #         self.model_ema.copy_to(self.model)
    #         if context is not None:
    #             print(f"{context}: Switched to EMA weights")
    #     try:
    #         yield None
    #     finally:
    #         if self.use_ema:
    #             self.model_ema.restore(self.model.parameters())
    #             if context is not None:
    #                 print(f"{context}: Restored training weights")

    # def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
    #     sd = torch.load(path, map_location="cpu")
    #     if "state_dict" in list(sd.keys()):
    #         sd = sd["state_dict"]
    #     keys = list(sd.keys())
    #     for k in keys:
    #         for ik in ignore_keys:
    #             if k.startswith(ik):
    #                 print("Deleting key {} from state_dict.".format(k))
    #                 del sd[k]
    #     missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
    #         sd, strict=False)
    #     print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
    #     if len(missing) > 0:
    #         print(f"Missing Keys: {missing}")
    #     if len(unexpected) > 0:
    #         print(f"Unexpected Keys: {unexpected}")    

    # def training_step(self, batch, batch_idx):
    #     loss, loss_dict = self.shared_step(batch)

    #     self.log_dict(loss_dict, prog_bar=True,
    #                   logger=True, on_step=True, on_epoch=True)

    #     self.log("global_step", self.global_step,
    #              prog_bar=True, logger=True, on_step=True, on_epoch=False)

    #     if self.use_scheduler:
    #         lr = self.optimizers().param_groups[0]['lr']
    #         self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

    #     return loss

    # @torch.no_grad()
    # def validation_step(self, batch, batch_idx):
    #     _, loss_dict_no_ema = self.shared_step(batch)
    #     with self.ema_scope():
    #         _, loss_dict_ema = self.shared_step(batch)
    #         loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
    #     self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
    #     self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    # def on_train_batch_end(self, *args, **kwargs):
    #     if self.use_ema:
    #         self.model_ema(self.model)    
    # def shared_step(self, batch):
    #     x = self.get_input(batch, self.first_stage_key)
    #     loss, loss_dict = self(x)
    #     return loss, loss_dict
    # def p_losses(self, x_start, t, noise=None):
    #     # Noise 를 인위적으로 주지 않는 경우 Zero Mean Standard Gaussian Noise 생성
    #     noise = torch.randn_like(x_start) if noise is None else noise
    #     # Forward Process
    #     x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
    #     # Noise Prediction. model_out : \epsilon
    #     model_out = self.model(x_noisy, t)
    #     loss_dict = {}
    #     # self.parametrization 은 prediction target 을 뭐로 할 지 에 대한 결정을 나타냄. x0 보다 eps 가 성능적으로 나음
    #     target = noise
    #     # batch 방향만 남김 
    #     loss = self.loss_fn(model_out, target).mean(dim=[1, 2, 3])
    #     log_prefix = 'train' if self.training else 'val'
    #     loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
    #     loss_simple = loss.mean()
    #     # t 는 batch size 길이 만큼의 time step 
    #     # lvlb 는 전체 time step 의 길이와 동일하므로 t로 indexing 하면 해당 time step 에 대한 계수값들을 얻을 수 있음. 
    #     loss_vlb = (self.lvlb_weights[t] * loss).mean()
    #     loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})
    #     # 
    #     loss = self.l_simple_weight * loss_simple + self.original_elbo_weight * loss_vlb
    #     loss_dict.update({f'{log_prefix}/loss': loss})

    #     return loss, loss_dict

    # def forward(self, x, *args, **kwargs):
    #     # b, c, h, w, device, img_size, = *x.shape, x.device, self.sampler_cfg.image_size
    #     # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
    #     t = torch.randint(0, self.num_timesteps, (len(x),), device=self.device).long()
    #     # => randomly generate time steps 
    #     return self.p_losses(x, t, *args, **kwargs)    

if __name__ == "__main__":
    a = torch.full((10,), 1)
    print(a)