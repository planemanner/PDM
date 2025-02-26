import lightning as L
import albumentations as A
from glob import glob
import os
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
import numpy as np
from typing import Tuple
from transformers import PreTrainedTokenizer

def load_img_list(root_dir):
    data_list = []
    fmts = ["jpg", "JPG", "jpeg", "JPEG", "png", "PNG"]
    for fmt in fmts:
        data_list.extend(glob(os.path.join(root_dir, f"*.{fmt}")))
    return data_list

class VaeDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.img_list = load_img_list(data_dir)
        self.transform = transform

    def __getitem__(self, idx: int) -> Image.Image:
        img = Image.open(self.img_list[idx])
        if self.transform:
            img = self.transform(image=np.array(img))['image']
        return Image.fromarray(img)
    
    def __len__(self):
        return len(self.img_list)

class ImageTextPairDataset(Dataset):
    def __init__(self, metadata_file: str, img_root_dir: str, transform=None):
        # LAION-400M
        # Transform is albumentations
        self.metadata = pd.read_csv(metadata_file)
        self.transform = transform
        self.img_root_dir = img_root_dir
    
    def __getitem__(self, idx: int) -> Tuple[Image.Image, str]:
        meta_data = self.metadata.iloc[idx]
        img_path = os.path.join(self.img_root_dir, meta_data['file_path'])
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(image=np.array(img))['image']
        
        caption = meta_data['caption']

        return Image.fromarray(img), caption

    def __len__(self):
        return len(self.metadata)
    
class ImageTextCollater:
    def __init__(self, tokenizer: PreTrainedTokenizer, token_max_length:int=128):
        self.tokenizer = tokenizer
        self.token_max_length = token_max_length

    def __call__(self, batch):
        images, captions = batch
        tknized_captions = self.tokenizer(captions, padding=True, truncation=True, return_tensors='pt', max_length=self.token_max_length)
        return images, tknized_captions

class ImageTextDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()


class ImageDataModule(L.LightningDataModule):
    def __init__(self, img_dir):
        super().__init__()
        self.img_dir = img_dir
        
    def prepare_data(self) -> None:

        pass