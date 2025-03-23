import argparse
import os
from trainer import train_diffusion, train_autoencoder


def main():
    parser = argparse.ArgumentParser(description="Training script for AutoEncoder and Diffusion models")
    parser.add_argument('--stage', type=str, required=True, choices=['autoencoder', 'diffusion'],
                        help="Specify the training stage: 'autoencoder' or 'diffusion'")
    parser.add_argument('--autoencoder_ckpt', type=str, default='path/to/autoencoder/checkpoint.ckpt',
                        help="Path to the AutoEncoder checkpoint file")
    parser.add_argument('--mode', type=str, choices=["train", "test"], default="train")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--precision', type=str, default="32")
    parser.add_argument('--gpus', type=int, nargs='+')
    parser.add_argument('--sampler_type', type=str, default="flow-matching")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--bsz', type=int, default=8)
    
    parser.add_argument('--train_dir', type=str, default="")
    parser.add_argument('--test_dir', type=str, default="")
    args = parser.parse_args()

    if args.stage == 'autoencoder':
        train_autoencoder(args)
    elif args.stage == 'diffusion':
        train_diffusion(args)
    else:
        raise ValueError("Invalid stage specified. Choose either 'autoencoder' or 'diffusion'.")

if __name__ == "__main__":
    main()