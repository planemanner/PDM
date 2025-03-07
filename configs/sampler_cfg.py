from .dotdict import DotDict

sampler_config = {"sampler_type": "ddim",
                  "clip_denoised": True,
                  "log_every_t": 0, 
                  "image_size": 256,
                  "in_channels": 3,
                  "use_ema": True,
                  "v_posterior": 0.0,
                  "original_elbo_weight": 0.0,
                  "l_simple_weight": 1.0,
                  "n_steps": 1000,
                  "linear_start": 1e-4,
                  "linear_end": 2e-2}

sampler_config = DotDict(sampler_config)