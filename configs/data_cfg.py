from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class TransformConfig:
    resize_height: int
    resize_width: int
    hflip: Optional[float] = None
    vflip: Optional[float] = None
    rot90: Optional[float] = None
    cond_drop_rate: float = 0.0

@dataclass
class DatasetConfig:
    img_dir: str
    prompt_dir: str
    bsz: int
    normalize_mean: List[float]
    normalize_std: List[float]
    transform: TransformConfig

@dataclass
class DatasetConfig:
    img_dir: str
    bsz: int
    normalize_mean: List[float]
    normalize_std: List[float]
    transform: TransformConfig

@dataclass
class AutoencoderDataConfig:
    train: DatasetConfig
    test: DatasetConfig

@dataclass
class DiffusionDataConfig:
    train: DatasetConfig
    test: DatasetConfig

if __name__ == "__main__":
    diffusion_data_config = DiffusionDataConfig(
        train=DatasetConfig(
            img_dir="",
            prompt_dir="",
            bsz=4,
            normalize_mean=[0.5, 0.5, 0.5],
            normalize_std=[0.5, 0.5, 0.5],
            transform=TransformConfig(
                resize_height=256,
                resize_width=256,
                hflip=0.5,
                vflip=0.5,
                rot90=0.5,
                cond_drop_rate=0.15
            )
        ),
        test=DatasetConfig(
            img_dir="",
            sketch_dir="",
            bsz=4,
            normalize_mean=[0.5, 0.5, 0.5],
            normalize_std=[0.5, 0.5, 0.5],
            transform=TransformConfig(
                resize_height=256,
                resize_width=256
            )
        )
    )

    autoencoder_data_config = AutoencoderDataConfig(
        train=DatasetConfig(
            img_dir="",
            bsz=8,
            normalize_mean=[0.5, 0.5, 0.5],
            normalize_std=[0.5, 0.5, 0.5],
            transform=TransformConfig(
                resize_height=256,
                resize_width=256,
                hflip=0.5,
                vflip=0.5,
                rot90=0.5
            )
        ),
        test=DatasetConfig(
            img_dir="",
            bsz=8,
            normalize_mean=[0.5, 0.5, 0.5],
            normalize_std=[0.5, 0.5, 0.5],
            transform=TransformConfig(
                resize_height=256,
                resize_width=256
            )
        )
    )