from dotdict import DotDict

# Following configuration values are referenced from Stable Diffusion Repo
autoencoder_cfg = {"encoder": {'in_channels': 3,
                               'middle_channels': 128,
                               'temb_ch': 128,
                               'n_res_blocks': 2,
                               'ch_mult': (1, 1, 2, 2, 4, 4),
                               'attn_resolutions': [16, 8],
                               'dropout': 0.0,
                               'resamp_with_conv': True,
                               'resolution': 256,
                               'z_channels': 64,
                               'double_z': True, 
                               'user_linear_attn': False,
                               'attn_type': 'vanilla'},

                   "decoder": {'out_channels': 3,
                               'middle_channels': 128,
                               'n_res_blocks': 2,
                               'temb_ch': 0,
                               'ch_mult': (1, 1, 2, 2, 4, 4),
                               'attn_resolutions': [16, 8],
                               'dropout': 0.0,
                               'resamp_with_conv': True,
                               'resolution': 256,
                               'z_channels': 64,
                               'tanh_out': True,
                               'give_pre_end' : False,
                               'user_linear_attn': False,
                               'attn_type': 'vanilla'},

                    "common": {"embed_dim": 64,
                               "ckpt_path": "",
                               "base_lr": 4.5e-6},
                    "loss_fn": {"disc_start": 50001,
                                "kl_weight": 1e-6,
                                "disc_weight": 0.5},
                               }

unet_cfg = {"middle_channels":128,
            "temb_ch": 128,
            "in_channels": 3, # color
            "out_channels": 3,
            "ch_mult": (1, 1, 2, 2, 4, 4),
            "use_timestep": True,
            "resolution": 256,
            "n_res_blocks": 2,
            "dropout": 0.0,
            "attn_type": 'vanilla',
            "attn_resolutions": [16, 8],
            "resamp_with_conv": True}

ddpm_cfg = {"clip_denoised": True,
            "log_every_t": 0, 
            "image_size": 256,
            "in_channels": 3,
            "use_ema": True,
            "v_posterior": 0.0,
            "original_elbo_weight": 0.0,
            "l_simple_weight": 1.0,
            "timesteps": 1000,
            "linear_start": 1e-4,
            "linear_end": 2e-2}

lddpm_cfg = {"clip_denoised": True,
            "log_every_t": 0, 
            "image_size": 256,
            "in_channels": 3,
            "use_ema": True,
            "v_posterior": 0.0,
            "original_elbo_weight": 0.0,
            "l_simple_weight": 1.0,
            "timesteps": 1000,
            "linear_start": 1e-4,
            "linear_end": 2e-2, 
            # conditions : ['concat', 'crossattn', 'none']
            "condition": 'none',
            "n_cond_time_steps": 1,
            "scale_factor": 1.0}

pipeline_cfg = {"sampler": 'lddim'}