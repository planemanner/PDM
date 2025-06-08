"""
Trainer Interface 는 VQ-VAE 와 Diffusion 을 모두 포괄할 수 있도록 구성하기
기본적으로 Multi-Node DDP 로 구성
https://www.restack.io/p/pytorch-lightning-answer-vs-huggingface-trainer-cat-ai
"""
import lightning as L
from diffusion import StableDiffusion
from configs.data_cfg import AutoencoderDataConfig, DiffusionDataConfig, DatasetConfig, TransformConfig
from configs.diffusion_cfg import DiffusionConfig, AutoEncoderConfig
from data.dataset import Sketch2ImageDataModule, VAEDataModule
from models.autoencoder import AutoEncoder
from lightning.pytorch.strategies import FSDPStrategy, DDPStrategy

def train_diffusion(args):
    L.seed_everything(args.seed)
    diff_cfg = DiffusionConfig()
    img_size = diff_cfg.ae.encoder.resolution
    diff_data_cfg = DiffusionDataConfig(train=DatasetConfig(
        img_dir=f"{args.train_dir}/img",
        prompt_dir=f"{args.train_dir}/mask",
        bsz=args.bsz,
        normalize_mean=[0.5, 0.5, 0.5],
        normalize_std=[0.5, 0.5, 0.5],
        transform=TransformConfig(
            resize_height=img_size,
            resize_width=img_size,
            hflip=0.5,
            vflip=0.5,
            rot90=0.5
        )
    ),
    test=DatasetConfig(
        img_dir=f"{args.test_dir}/img",
        prompt_dir=f"{args.test_dir}/mask",
        bsz=args.bsz,
        normalize_mean=[0.5, 0.5, 0.5],
        normalize_std=[0.5, 0.5, 0.5],
        transform=TransformConfig(
            resize_height=img_size,
            resize_width=img_size
        )
    ))

    diff_cfg.sampler_type = args.sampler_type
    diff_cfg.lr = args.lr

    model = StableDiffusion(diff_cfg)
    
    data_module = Sketch2ImageDataModule(diff_data_cfg)
    trainer_strategy = DDPStrategy(find_unused_parameters=True)

    trainer_cfg = {"accelerator": "gpu", # cpu, gpu, tpu, auto
                   "devices": args.gpus, # 'auto', List[int], int, "0, 1", -1 ...
                   "precision": args.precision, # int or str like this example.
                   "strategy": trainer_strategy,
                   "max_epochs": args.epochs, 
                   "limit_val_batches":1}
    trainer = L.Trainer(**trainer_cfg)
    trainer.fit(model, data_module)

def train_autoencoder(args):
    L.seed_everything(args.seed)
    ae_cfg = AutoEncoderConfig()
    img_size = ae_cfg.encoder.resolution

    data_cfg = AutoencoderDataConfig(
        train=DatasetConfig(
            img_dir=f"{args.train_dir}/img",
            bsz=args.bsz,
            normalize_mean=[0.5, 0.5, 0.5],
            normalize_std=[0.5, 0.5, 0.5],
            transform=TransformConfig(
                resize_height=img_size,
                resize_width=img_size,
                hflip=0.5,
                vflip=0.5,
                rot90=0.5
            )
        ),
        test=DatasetConfig(
            img_dir=f"{args.test_dir}/img",
            bsz=args.bsz,
            normalize_mean=[0.5, 0.5, 0.5],
            normalize_std=[0.5, 0.5, 0.5],
            transform=TransformConfig(
                resize_height=img_size,
                resize_width=img_size
            )
        )
    )
    ae_cfg.lr = args.lr
    data_cfg.train.bsz = args.bsz
    data_cfg.test.bsz = args.bsz

    autoencoder = AutoEncoder(ae_cfg)
    data_module = VAEDataModule(data_cfg)
    trainer_strategy = DDPStrategy(find_unused_parameters=True)
    trainer_cfg = {"accelerator": "gpu", # cpu, gpu, tpu, auto
                   "devices": args.gpus, # 'auto', List[int], int, "0, 1", -1 ...
                   "precision": args.precision, # int or str like this example.
                   "strategy": trainer_strategy,
                   "max_epochs": args.epochs, 
                   "limit_val_batches":1}
    
    trainer = L.Trainer(**trainer_cfg)
    trainer.fit(autoencoder, data_module)

