from .dotdict import DotDict

diffusion_data_config = DotDict({
    "train": {
        "img_dir": "/aidata01/core_impingement/data/StableDiffusion/Split/train/img",
        "sketch_dir": "/aidata01/core_impingement/data/StableDiffusion/Split/train/cathode",
        "bsz": 4,
        "transform": {"resize_height": 256,
                      "resize_width": 256,
                      "hflip": 0.5,
                      "vflip": 0.5,
                      "rot90": 0.5},

        "normalize_mean": [0.5, 0.5, 0.5],
        "normalize_std": [0.5, 0.5, 0.5],

    },
    "test": {
        "img_dir": "/aidata01/core_impingement/data/StableDiffusion/Split/test/img",
        "sketch_dir": "/aidata01/core_impingement/data/StableDiffusion/Split/test/cathode",
        "bsz": 4,
        "normalize_mean": [0.5, 0.5, 0.5],
        "normalize_std": [0.5, 0.5, 0.5],
        "transform": {"resize_height": 256,
                      "resize_width": 256},        
    }

})

autoencoder_data_config = DotDict({
    "train": {
        "img_dir": "/aidata01/core_impingement/data/StableDiffusion/Split/train/img",
        "bsz": 8,
        "transform": {"resize_height": 256,
                      "resize_width": 256,
                      "hflip": 0.5,
                      "vflip": 0.5,
                      "rot90": 0.5},

        "normalize_mean": [0.5, 0.5, 0.5],
        "normalize_std": [0.5, 0.5, 0.5],
    },

    "test": {
        "img_dir": "/aidata01/core_impingement/data/StableDiffusion/Split/test/img",
        "bsz": 8,
        "normalize_mean": [0.5, 0.5, 0.5],
        "normalize_std": [0.5, 0.5, 0.5],
        "transform": {"resize_height": 256,
                      "resize_width": 256},        
    }

}
)