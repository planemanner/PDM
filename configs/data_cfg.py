from .dotdict import DotDict

diffusion_data_config = DotDict({
    "train": {
        "img_dir": "",
        "sketch_dir": "",
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
        "img_dir": "",
        "sketch_dir": "",
        "bsz": 8,
        "normalize_mean": [0.5, 0.5, 0.5],
        "normalize_std": [0.5, 0.5, 0.5],
        "transform": {"resize_height": 256,
                      "resize_width": 256},        
    }

})

autoencoder_data_config = DotDict({
    "train": {
        "img_dir": "",
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
        "img_dir": "",
        "bsz": 8,
        "normalize_mean": [0.5, 0.5, 0.5],
        "normalize_std": [0.5, 0.5, 0.5],
        "transform": {"resize_height": 256,
                      "resize_width": 256},        
    }

}
)