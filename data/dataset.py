import lightning as L
import albumentations as A
from glob import glob
import os
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
import numpy as np
from typing import Tuple, List
from transformers import PreTrainedTokenizer
import torch
from albumentations.pytorch import ToTensorV2
import random 

from .data_utils import get_tf_for_sketch, get_tf_for_ae, load_img_list, check_pair

class VAEDataset(Dataset):
    def __init__(self, data_dir, transform, normalize_mean, normalize_std):
        self.img_list = load_img_list(data_dir)
        self.transform = transform
        self.normalization = A.Normalize(mean=normalize_mean, std=normalize_std)
        self.totensor = ToTensorV2()

    def __getitem__(self, idx: int) -> torch.FloatTensor:
        img = Image.open(self.img_list[idx]).convert('RGB')
        img = self.transform(image=np.array(img))['image']
        img = self.totensor(image=self.normalization(image=img)['image'])['image']

        return img
    
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
    
class Sketch2ImageDataset(Dataset):
    def __init__(self, img_dir, sketch_dir, 
                 transform : A.BasicTransform, 
                 normalize_mean: List[float], 
                 normalize_std: List[float],
                 cond_drop_rate: float=0.0):
        
        # image and sketch must consist of a pair.
        self.img_list = load_img_list(img_dir)
        self.sketch_list = load_img_list(sketch_dir)
        self.sketch_dir = sketch_dir
        self.cond_drop_rate = cond_drop_rate

        if not check_pair(self.img_list, self.sketch_list):
            raise FileNotFoundError('Image and sketch lists are not matched.')

        self.tf = transform
        self.normalization = A.Normalize(mean=normalize_mean, std=normalize_std)
        self.totensor = ToTensorV2()

    def __getitem__(self, idx) -> Tuple[torch.FloatTensor, np.ndarray]:
        # I do not recommend you to use opencv to read image. 
        # Since there is some unrecognizable error (it is hard to find some point to debug), imread function raises an assertion error (Sometimes.).
        img_path = self.img_list[idx]
        file_name = os.path.basename(img_path)
        img = Image.open(img_path).convert('RGB')
        sketch = Image.open(os.path.join(self.sketch_dir, file_name)).convert('RGB')
        
        if self.tf:
            transformed = self.tf(image=np.array(img), mask=np.array(sketch)) # Tensor in case of training
            img = transformed['image']
            img = self.totensor(image=self.normalization(image=img)['image'])['image']
            sketch = transformed['mask']
            # for classifier free guidance
            bsz = len(sketch)
            n_drop = int(self.cond_drop_rate * bsz)
            maskout_ids = random.sample([i for i in range(bsz)], n_drop)
            if len(maskout_ids) > 0:
                sketch[maskout_ids, ...] = 0.0
        return img, sketch

    def __len__(self):
        return len(self.img_list)
        
class ImageTextCollater:
    def __init__(self, tokenizer: PreTrainedTokenizer, token_max_length:int=128):
        self.tokenizer = tokenizer
        self.token_max_length = token_max_length

    def __call__(self, batch):
        images, captions = batch
        tknized_captions = self.tokenizer(captions, padding=True, truncation=True, return_tensors='pt', max_length=self.token_max_length)
        return images, tknized_captions


class Sketch2ImageDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def prepare_data(self) -> None:
        pass

    def setup(self, stage:str=None):
        if stage == 'fit':
            tf = get_tf_for_sketch(self.config.train, mode='train')
            self.train_set = Sketch2ImageDataset(self.config.train.img_dir, 
                                                 self.config.train.sketch_dir,
                                                 transform=tf,
                                                 normalize_mean=self.config.train.normalize_mean,
                                                 normalize_std=self.config.train.normalize_std
                                                 )
                                                 
            tf = get_tf_for_sketch(self.config.test, mode='test')
            self.val_set = Sketch2ImageDataset(self.config.test.img_dir, 
                                               self.config.test.sketch_dir,
                                               transform=tf,
                                               normalize_mean=self.config.test.normalize_mean,
                                               normalize_std=self.config.test.normalize_std
                                               )
        if stage == 'test':
            tf = get_tf_for_sketch(self.config.test, mode='test')
            self.test_set = Sketch2ImageDataset(self.config.test.img_dir, 
                                                self.config.test.sketch_dir,
                                                transform=tf,
                                                normalize_mean=self.config.test.normalize_mean,
                                                normalize_std=self.config.test.normalize_std
                                                )
    
    def train_dataloader(self):
        # num_workers, shuffle, and sampler are automatically set up by internal ways.
        return DataLoader(self.train_set, batch_size=self.config.train.bsz)
    
    def val_dataloader(self):
        # num_workers, shuffle, and sampler are automatically set up by internal ways.
        return DataLoader(self.val_set, batch_size=self.config.test.bsz)
    

class VAEDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def prepare_data(self):
        pass
    
    def setup(self, stage: str):
        if stage == 'fit':
            tf = get_tf_for_ae(self.config.train, mode='train')
            self.train_set = VAEDataset(self.config.train.img_dir, transform=tf,
                                        normalize_mean=self.config.train.normalize_mean,
                                        normalize_std=self.config.train.normalize_std)
            
            tf = get_tf_for_ae(self.config.test, mode='test')
            self.val_set = VAEDataset(self.config.test.img_dir, transform=tf,
                                        normalize_mean=self.config.test.normalize_mean,
                                        normalize_std=self.config.test.normalize_std)
        if stage == 'test':
            tf = get_tf_for_ae(self.config.test, mode='test')
            self.test_set = VAEDataset(self.config.test.img_dir, transform=tf,
                                        normalize_mean=self.config.test.normalize_mean,
                                        normalize_std=self.config.test.normalize_std)
    
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.config.train.bsz)
    
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.config.test.bsz)