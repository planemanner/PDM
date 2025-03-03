from .dotdict import DotDict

autoencoder_config = {"vae_type": "kl_vae",
                      "encoder": {'in_channels': 3,
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

autoencoder_config = DotDict(autoencoder_config)