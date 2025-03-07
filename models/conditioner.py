from torch import nn
import torch
from abc import ABC, abstractmethod
from transformers import CLIPProcessor, CLIPModel, CLIPImageProcessor
from PIL import Image
from typing import List, Tuple
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
    def forward(self, prompt_images: List[Image.Image]) -> Tuple[torch.FloatTensor, Image.Image]:
        # 이미지를 전처리하여 모델 입력 형식으로 변환
        inputs = self.processor(images=prompt_images, return_tensors="pt", device=self.model.device)
        device = self.model.device  # Assuming self.model.device is set correctly
        inputs = {key: value.to(device) for key, value in inputs.items()}
        # 전처리된 이미지를 사용하여 이미지 특징 추출
        outputs = self.model.get_image_features(**inputs)
        return outputs

if __name__ == "__main__":
    model_name = "openai/clip-vit-base-patch32"
    processor = CLIPImageProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    # prompter = ImagePrompter(cfg)
    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)

    # copied_images = [image, image, image, image]

    # v, img = prompter(copied_images)
    # print(v.shape)
