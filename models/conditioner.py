from torch import nn
import torch
from abc import ABC, abstractmethod
from transformers import CLIPModel, CLIPImageProcessor
from PIL import Image
from typing import List, Tuple
import requests
from torch.nn import functional as F

from .utils import freeze_model

class Conditioner(nn.Module, ABC):
    def __init__(self, condition_model):
        super().__init__()
        self.cond_model = condition_model

    @abstractmethod
    def forward(self, **kwargs) -> Tuple[torch.FloatTensor, Image.Image]:
        pass

class ImagePrompter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # self.processor = CLIPProcessor.from_pretrained(cfg.model_name)
        self.processor = CLIPImageProcessor.from_pretrained(cfg.model_name)
        self.model = CLIPModel.from_pretrained(cfg.model_name)
        self.model.eval()
        freeze_model(self.model)

    @torch.no_grad()
    def forward(self, prompt_images: List[Image.Image], latent_size: List[int]=(32, 32)) -> Tuple[torch.FloatTensor, Image.Image]:
        # 이미지를 전처리하여 모델 입력 형식으로 변환
        inputs = self.processor(images=prompt_images, return_tensors="pt")
        device = self.model.device  # Assuming self.model.device is set correctly
        inputs = {key: value.to(device) for key, value in inputs.items()}        
        # 전처리된 이미지를 사용하여 이미지 특징 추출
        vision_outputs = self.model.vision_model(**inputs)
        last_hidden_states = vision_outputs.last_hidden_state  # (B, N, D)

        # CLS 토큰 제외
        patch_embeddings = last_hidden_states[:, 1:, :]  # (B, N-1, D)
        
        # N-1을 (H, W)로 변환
        B, N, C = patch_embeddings.shape  # N = H * W, C = Channel
        H = W = int(N ** 0.5)  # 일반적으로 H=W

        # (B, N, C) → (B, C, H, W)
        feature_map = patch_embeddings.permute(0, 2, 1).reshape(B, C, H, W)
        feature_map = F.interpolate(feature_map, latent_size)
        return feature_map  # (B, C, H, W)

if __name__ == "__main__":
    cfg = {"model_name": "openai/clip-vit-base-patch32"}
    prompter = ImagePrompter(cfg)
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    copied_images = [image, image, image, image]

    v, img = prompter(copied_images)
    print(v.shape)
