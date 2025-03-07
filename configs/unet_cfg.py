# Following configuration values are referenced from Stable Diffusion Repo
from .dotdict import DotDict
# use_scale_shift_norm: true

unet_config = {"middle_channels":192,
               "in_channels": 64,
               "out_channels": 64,
               "ch_mult": (1, 2, 2, 4, 4, 4),
               "use_timestep": True,
               "resolution": 32,
               "n_res_blocks": 2,
               "dropout": 0.0,
               "attn_type": 'vanilla',
               "attn_resolutions": [16, 8],
               "resamp_with_conv": True,
               "transformer_depth": 1,
               "context_dim": 512,
               "use_spatial_transformer": True,
               "num_head_channels": 32,
               "resblock_updown": True,
               "lr":1e-4,
               "mode": "train",
               "sample_save_dir": ""}

unet_config = DotDict(unet_config)