from lightning.pytorch.strategies import FSDPStrategy, DDPStrategy
from .dotdict import DotDict

trainer_strategy = DDPStrategy(find_unused_parameters=True)
latent_dim = 4
image_size = 256
context_dim = 768 # CLIP 
image_channels = 3
attn_resolutions = [16, 8]
ae_middle_channels = 128
ae_ch_mult = (1, 2, 4 , 4)
ae_n_res_block = 2

trainer_diff_cfg = DotDict({"accelerator": "gpu", # cpu, gpu, tpu, auto
                    "devices": [0], # 'auto', List[int], int, "0, 1", -1 ...
                    "precision": "32", # int or str like this example.
                    "strategy": trainer_strategy,
                    "max_epochs": 500, 
                    "limit_val_batches":1
                    })

trainer_ae_cfg = DotDict({"accelerator": "gpu", # cpu, gpu, tpu, auto
                  "devices": [0], # 'auto', List[int], int, "0, 1", -1 ...
                  "precision": "32", # int or str like this example.
                  "strategy": trainer_strategy,
                  "max_epochs": 500,
                  "limit_val_batches":1
                  })

conditioner_config = DotDict({"prompt_type": "image",
                      "model_name": "",
                      })

unet_config = DotDict({"middle_channels":192,
                       "in_channels": latent_dim, 
                       "out_channels": latent_dim,
                       "ch_mult": (1, 2, 2, 4, 4, 4),
                       "use_timestep": True,
                       "resolution": image_size // 8,
                       "n_res_blocks": 2,
                       "dropout": 0.0,
                       "attn_type": 'vanilla',
                       "attn_resolutions": attn_resolutions,
                       "resamp_with_conv": True,
                       "transformer_depth": 1,
                       "context_dim": context_dim,
                       "use_spatial_transformer": True,
                       "num_head_channels": 32,
                       "resblock_updown": False,
                       "lr": 1e-4,
                       "mode": "train",
                       "sample_save_dir": "",
                       "use_step_embed": False})

sampler_config = DotDict({"sampler_type": "ddim",
                          "clip_denoised": True,
                          "log_every_t": 0, 
                          "image_size": image_size,
                          "in_channels": image_channels,
                          "use_ema": True,
                          "v_posterior": 0.0,
                          "original_elbo_weight": 0.0,
                          "l_simple_weight": 1.0,
                          "n_steps": 1000,
                          "linear_start": 1e-4,
                          "linear_end": 2e-2})

flow_sampler_config = DotDict({"shortcut_length": 7})

ae_config = DotDict({"vae_type": "kl_vae",
             "encoder": {'in_channels': image_channels,
                         'middle_channels': ae_middle_channels,
                         'temb_ch': 0,
                         'n_res_blocks': ae_n_res_block,
                         'ch_mult': ae_ch_mult,
                         'attn_resolutions': attn_resolutions,
                         'dropout': 0.0,
                         'resamp_with_conv': True,
                         'resolution': image_size,
                         'z_channels': latent_dim,
                         'double_z': True, 
                         'user_linear_attn': False,
                         'attn_type': 'vanilla'},

            "decoder": {'out_channels': image_channels,
                        'middle_channels': ae_middle_channels,
                        'n_res_blocks': ae_n_res_block,
                        'temb_ch': 0,
                        'ch_mult': ae_ch_mult,
                        'attn_resolutions': attn_resolutions,
                        'dropout': 0.0,
                        'resamp_with_conv': True,
                        'resolution': image_size,
                        'z_channels': latent_dim,
                        'tanh_out': True,
                        'give_pre_end' : False,
                        'user_linear_attn': False,
                        'attn_type': 'vanilla'},

                        "common": {"embed_dim": latent_dim,
                                   "ckpt_path": "",
                                   "lr": 4.5e-6},
                                
                        "loss_fn": {"disc_start": 50001,
                                    "kl_weight": 1e-6,
                                    "disc_weight": 0.5},
                                    })