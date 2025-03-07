import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from models.autoencoder import AutoEncoder
from PIL import Image
import os
import argparse
from configs.dotdict import DotDict
from torchvision import transforms
from configs.autoencoder_cfg import autoencoder_config

def load_model(config, checkpoint_path, device):
    """Load a trained AutoEncoder model"""
    cfg = DotDict(config)
    model = AutoEncoder(cfg)
    
    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model

def load_images(image_dir, image_size=256, n_images=8):
    """Load images from directory"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    images = []
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    assert len(image_paths) >= n_images, f"Not enough images in {image_dir}"
    image_paths = image_paths[:n_images]
    
    print(f"Loading {n_images} images from {image_dir}")

    for path in image_paths:  # Limit to 8 images for visualization
        img = Image.open(path).convert('RGB')
        img_tensor = transform(img)
        images.append(img_tensor)
    
    return torch.stack(images)

def visualize_reconstructions(model: AutoEncoder, images, output_path=None, device="cuda"):
    """Generate and visualize reconstructions"""
    images = images.to(device)
    
    with torch.no_grad():
        reconstructions, _ = model(images)
    
    # Denormalize
    def denorm(x):
        return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)
    
    # Create comparison grid
    n_samples = images.shape[0]
    comparison = torch.cat([denorm(images), denorm(reconstructions)])
    comparison = make_grid(comparison, nrow=n_samples)
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.imshow(comparison.cpu().permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.title("Original Images (Top) vs Reconstructions (Bottom)")
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    
    plt.show()
    
    # Compute reconstruction metrics
    mse = F.mse_loss(denorm(images), denorm(reconstructions)).item()
    print(f"Mean Squared Error: {mse:.4f}")
    
    return reconstructions


def main():
    parser = argparse.ArgumentParser(description="Visualize AutoEncoder reconstructions")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory with test images")
    parser.add_argument("--output", type=str, default="some_dir", help="Output visualization path")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on")
    parser.add_argument("--image_size", type=int, default=256, help="Size to resize images to")
    parser.add_argument("--n_images", type=int, default=8, help="Number of images to visualize")
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(autoencoder_config, args.checkpoint, device)
    
    # Load images
    images = load_images(args.image_dir, args.image_size, n_images=args.n_images)
    
    # Visualize
    visualize_reconstructions(model, images, args.output, device)
    
if __name__ == "__main__":
    main()