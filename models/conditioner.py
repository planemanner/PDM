from torch import nn
import torch
from abc import ABC, abstractmethod
from transformers import CLIPModel, CLIPImageProcessor, CLIPTokenizer
from PIL import Image
from typing import List, Tuple
from torch.nn import functional as F

from .utils import freeze_model

class Conditioner(nn.Module, ABC):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(self, **kwargs) -> Tuple[torch.FloatTensor, Image.Image]:
        pass

class TextPrompter(Conditioner):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(cfg.model_name)
        self.model = CLIPModel.from_pretrained(cfg.model_name)
        freeze_model(self.model)

    @torch.no_grad()
    def forward(self, prompt_texts: List[str]) -> Tuple[torch.FloatTensor, Image.Image]:
        device = self.model.device  # Assuming self.cond_model.device is set correctly
        inputs = self.clip_tokenizer(text=prompt_texts, return_tensors="pt", padding=True, trunction=True, max_length=self.cfg.token_max_length).to(device)
        text_outputs = self.model.text_model(**inputs)
        
        last_hidden_states = text_outputs.last_hidden_state  # (B, N, D)
        return last_hidden_states

class ImagePrompter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # self.processor = CLIPProcessor.from_pretrained(cfg.model_name)
        self.processor = CLIPImageProcessor.from_pretrained(cfg.model_name)
        self.model = CLIPModel.from_pretrained(cfg.model_name)
        self.model.eval()
        freeze_model(self.model)

    @torch.no_grad()
    def forward(self, prompt_images: List[Image.Image], latent_size: Tuple[int]=(32, 32)) -> torch.FloatTensor:
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
    pass
    # from dataclasses import dataclass
    # from typing import Tuple, List
    # @dataclass
    # class Config:
    #     model_name: str = "openai/clip-vit-base-patch32"
    #     token_max_length: int = 128
    # cfg = Config()
    
    # text_prompter = TextPrompter(cfg)
    # prompt_texts = ["A photo of a cat", "A photo of a dog"]
    # text_embeddings = text_prompter(prompt_texts)
    # print(text_embeddings.shape)  # Should be (B, 1, D)
    # prompter = ImagePrompter(cfg)
    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)

    # copied_images = [image, image, image, image]

    # v, img = prompter(copied_images)
    # print(v.shape)
