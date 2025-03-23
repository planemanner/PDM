from dataclasses import dataclass, field
from typing import Tuple, List

IMAGE_SIZE = 256
ATTN_RESOL = (16, 8)
CONTEXT_DIM = 768 # CLIP 768
LATENT_DIM = 4
AE_CH_MULT = (1, 2, 4, 4)
AE_N_RES_BLOCKS = 2
AE_MIDDLE_CH = 128
IMG_CHANNELS = 3

@dataclass
class UNetConfig:
    middle_channels: int = 192
    in_channels: int = LATENT_DIM
    out_channels: int = LATENT_DIM
    ch_mult: Tuple[int] = (1, 2, 2, 4, 4, 4)
    resolution: int = IMAGE_SIZE // 8
    dropout : float = 0.0
    attn_type: str = 'vanilla'
    resample_with_conv: bool = True
    n_res_blocks : int = 2
    use_time_step: bool = True
    attn_resolution : Tuple[int] = ATTN_RESOL
    transformer_depth: int = 1
    context_dim : int = CONTEXT_DIM
    use_spatial_transformer : bool = True
    num_head_channels : int = 32
    resblock_updwon: bool = False

@dataclass
class AeEncoderConfig:
    in_channels : int = IMG_CHANNELS
    middle_channels : int = AE_MIDDLE_CH
    temb_ch : int = 0
    n_res_blocks : int = AE_N_RES_BLOCKS
    ch_mult : Tuple[int] = AE_CH_MULT
    attn_resolution : Tuple[int] = ATTN_RESOL
    dropout : float = 0.0
    resamp_with_conv: bool = True
    resolution: int = IMAGE_SIZE
    z_channels : int = LATENT_DIM
    double_z: bool = True
    user_linear_attn: bool = False
    attn_type : str = "vanilla"

@dataclass
class AeDecoderConfig:
    out_channels : int = IMG_CHANNELS
    middle_channels : int = AE_MIDDLE_CH
    temb_ch : int = 0
    n_res_blocks : int = AE_N_RES_BLOCKS
    ch_mult : Tuple[int] = AE_CH_MULT
    attn_resolution : Tuple[int] = ATTN_RESOL
    dropout : float = 0.0
    resamp_with_conv: bool = True
    resolution: int = IMAGE_SIZE
    z_channels : int = LATENT_DIM
    tanh_out : bool = True
    give_pre_end: bool = False
    double_z: bool = True
    user_linear_attn: bool = False
    attn_type : str = "vanilla"

@dataclass
class ContextConfig:
    context_type: str = "mask"
    model_name : str = "openai/clip-vit-base-patch32"

@dataclass
class AutoEncoderConfig:
    vae_type : str = "kl_vae"
    lr: float = 4.5e-6
    ckpt_path : str = ""
    embed_dim : int = 4
    disc_weight : float = 0.5
    disc_start: int = 50001
    kl_weight: float = 1e-6
    encoder : AeEncoderConfig = field(default_factory=AeEncoderConfig)
    decoder : AeDecoderConfig = field(default_factory=AeDecoderConfig)
    ckpt_save_dir : str = ""
    save_period : int = 50

@dataclass
class FlowConfig:
    shortcut_length: int = 7
    n_steps: int = 128

@dataclass
class DDConfig:
    image_size : int = IMAGE_SIZE
    in_channels: int = IMG_CHANNELS
    n_steps: int = 1000
    linear_start: float = 1e-4
    linear_end: float = 2e-2
    v_posterior: float = 0.0
    DD_TYPE :str = "ddim"

@dataclass
class DiffusionConfig:
    unet: UNetConfig = field(default_factory=UNetConfig)
    sample_save_dir: str = ""
    diffusion_ckpt_save_dir: str = ""
    lr: float = 1e-4
    ae : AutoEncoderConfig = field(default_factory=AutoEncoderConfig)
    ae_ckpt_path : str = ""
    sampler_type : str = "flow-matching"
    context : ContextConfig = field(default_factory=ContextConfig)
    save_period: int = 50


if __name__ == "__main__":
    cfg = DiffusionConfig()
    cfg.sampler_type = 'ddim'
    print(cfg.sampler_type)