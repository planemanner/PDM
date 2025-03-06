import argparse
import os
from trainer import train_diffusion, train_autoencoder

def main():
    parser = argparse.ArgumentParser(description="Training script for AutoEncoder and Diffusion models")
    parser.add_argument('--stage', type=str, required=True, choices=['autoencoder', 'diffusion'],
                        help="Specify the training stage: 'autoencoder' or 'diffusion'")
    parser.add_argument('--autoencoder_ckpt', type=str, default='path/to/autoencoder/checkpoint.ckpt',
                        help="Path to the AutoEncoder checkpoint file")
    parser.add_argument('--seed', type=int, default=42)    
    parser.add_argument('--save_dir', type=str, default="")
    
    args = parser.parse_args()

    if args.stage == 'autoencoder':
        train_autoencoder(args.save_dir, args.seed)
    elif args.stage == 'diffusion':
        if not os.path.exists(args.autoencoder_ckpt):
            # This if senetence's goal is just check if there exists checkpoint or not.
            # Please manually check if the checkpoint is valid or not
            raise FileNotFoundError(f"AutoEncoder checkpoint not found at {args.autoencoder_ckpt}. Please train the AutoEncoder first.")
        train_diffusion(args.save_dir, args.seed)
    else:
        raise ValueError("Invalid stage specified. Choose either 'autoencoder' or 'diffusion'.")

if __name__ == "__main__":
    main()

