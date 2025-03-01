# Following configuration values are referenced from Stable Diffusion Repo
unet_config = {"middle_channels":32,
            "in_channels": 3, # color
            "out_channels": 3,
            "ch_mult": (1, 1, 2, 2, 4, 4),
            "use_timestep": True,
            "resolution": 256,
            "n_res_blocks": 2,
            "dropout": 0.0,
            "attn_type": 'vanilla',
            "attn_resolutions": [16, 8],
            "resamp_with_conv": True,
            "transformer_depth": 1,
            "context_dim": 128,
            "use_spatial_transformer": True,
            "num_head_channels": 32,
            "resblock_updown": False}