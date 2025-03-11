import torch
import numpy as np
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader
from torchvision.models import inception_v3
from torchvision.transforms import functional as F
from torchvision import transforms
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance as FID

def calculate_fid(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)
    
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def get_inception_features(images: torch.Tensor, model: torch.nn.Module, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        images = images.to(device)
        features = model(images)
    return features.cpu().numpy()

def compute_fid_score(real_loader: DataLoader, fake_loader: DataLoader, device: torch.device) -> float:
    inception = inception_v3(pretrained=True, transform_input=False).to(device)
    inception.fc = torch.nn.Identity()  # Remove the final classification layer

    real_features = []
    fake_features = []

    for real_images, fake_images in tqdm(zip(real_loader, fake_loader)):
        real_features.append(get_inception_features(real_images[0], inception, device))
        fake_features.append(get_inception_features(fake_images[0], inception, device))

    real_features = np.concatenate(real_features, axis=0)
    fake_features = np.concatenate(fake_features, axis=0)

    fid_score = calculate_fid(real_features, fake_features)
    return fid_score

# Example usage
if __name__ == "__main__":
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import ToTensor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tf = transforms.Compose([transforms.Resize([299, 299]), ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                  std=[0.229, 0.224, 0.225])])
    real_dataset = CIFAR10(root="./data", train=True, download=True, transform=tf)
    fake_dataset = CIFAR10(root="./data", train=True, download=True, transform=tf)

    real_loader = DataLoader(real_dataset, batch_size=32, shuffle=False)
    fake_loader = DataLoader(fake_dataset, batch_size=32, shuffle=False)

    # fid_score = compute_fid_score(real_loader, fake_loader, device)
    # print(f"FID Score: {fid_score}")

    FID_OBJ = FID().to(device)
    min_size = min(len(real_loader), len(fake_loader))  # 작은 쪽에 맞추기
    real_iter = iter(real_loader)
    fake_iter = iter(fake_loader)
    for i in range(min_size):
        real_data = next(real_iter)[0]
        fake_data = next(fake_iter)[0]
        FID_OBJ.update((real_data * 255).clamp(0, 255).byte().to(device), real=True)
        FID_OBJ.update((fake_data* 255).clamp(0, 255).byte().to(device), real=False)
    fid_score = FID_OBJ.compute()
    print(fid_score)