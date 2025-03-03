from .dotdict import DotDict

diffusion_data_config = DotDict({
    "train": {
        "img_dir": "",
        "sketch_dir": "",
        "bsz": 32,
        "transform": {"resize_height": 512,
                      "resize_width": 512,
                      "hflip": 0.5,
                      "vflip": 0.5,
                      "rot90": 0.5},

        "normalize_mean": [],
        "normalize_std": [],

    },
    "test": {
        "img_dir": "",
        "sketch_dir": "",
        "bsz": 32,
        "normalize_mean": [],
        "normalize_std": [],
        "transform": {"resize_height": 512,
                      "resize_width": 512},        
    }

})

autoencoder_data_config = DotDict({
    "train": {
        "img_dir": "",
        "bsz": 32,
        "transform": {"resize_height": 512,
                      "resize_width": 512,
                      "hflip": 0.5,
                      "vflip": 0.5,
                      "rot90": 0.5},

        "normalize_mean": [],
        "normalize_std": [],

    },
    "test": {
        "img_dir": "",
        "bsz": 32,
        "normalize_mean": [],
        "normalize_std": [],
        "transform": {"resize_height": 512,
                      "resize_width": 512},        
    }

}
)