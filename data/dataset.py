import lightning as L
import albumentations as A
from glob import glob
import os
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
import numpy as np
from typing import Tuple, Optional, List
from transformers import PreTrainedTokenizer
from data_utils import get_tf_for_sketch, get_tf_for_ae
import torch

def load_img_list(root_dir):
    data_list = []
    fmts = ["jpg", "JPG", "jpeg", "JPEG", "png", "PNG"]
    for fmt in fmts:
        data_list.extend(glob(os.path.join(root_dir, f"*.{fmt}")))
    return data_list

def check_pair(list_1:List[str], list_2:List[str]):
    # This function returns True if list_1 and list_2 have same filename list.
    assert len(list_1) == len(list_2), "Two lists have the different lengths of list"

    hashtbl = set()

    for file_path in list_1:
        hashtbl.add(os.path.splitext(os.path.basename(file_path))[0])
    
    for file_path in list_2:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        if file_name not in hashtbl:
            return False     
    return True

class VAEDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.img_list = load_img_list(data_dir)
        self.transform = transform

    def __getitem__(self, idx: int) -> torch.FloatTensor:
        img = Image.open(self.img_list[idx])
        if self.transform:
            img = self.transform(image=np.array(img))['image']
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
    def __init__(self, img_dir, sketch_dir, target_transform : A.BasicTransform, sketch_transform: A.BasicTransform):
        # image and sketch must consist of a pair.
        self.img_list = load_img_list(img_dir)
        self.sketch_list = load_img_list(sketch_dir)
        self.sketch_dir = sketch_dir
        if not check_pair(self.img_list, self.sketch_list):
            raise FileNotFoundError('Image and sketch lists are not matched.')

        self.tgt_tf = target_transform
        self.sk_tf = sketch_transform

    def __getitem__(self, idx) -> Tuple[torch.FloatTensor, Image.Image]:
        # I do not recommend you to use opencv to read image. 
        # Since there is some unrecognizable error (it is hard to find some point to debug), imread function raises an assertion error (Sometimes.).
        img_path = self.img_list[idx]
        file_name = os.path.basename(img_path)
        img = Image.open(img_path)
        sketch = Image.open(os.path.join(self.sketch_dir, file_name))
        
        if self.transform:
            img = self.transform(image=img)['image'] # Tensor
            sketch = self.transform(image=sketch)['image'] # PIL Image

        return img, Image.fromarray(sketch)

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
            img_tf, sketch_tf = get_tf_for_sketch(self.config.train)
            self.train_set = Sketch2ImageDataset(self.config.train.img_dir, 
                                                 self.config.train.sketch_dir,
                                                 target_transform=img_tf,
                                                 sketch_transform=sketch_tf
                                                 )

        if stage == 'test':
            img_tf, sketch_tf = get_tf_for_sketch(self.config.test)
            self.test_set = Sketch2ImageDataset(self.config.test.img_dir, 
                                                self.config.test.sketch_dir,
                                                target_transform=img_tf,
                                                sketch_transform=sketch_tf)
    
    def train_dataloader(self):
        # num_workers, shuffle, and sampler are automatically set up by internal ways.
        return DataLoader(self.train_set, batch_size=self.config.train.bsz)
    
    def test_dataloader(self):
        # num_workers, shuffle, and sampler are automatically set up by internal ways.
        return DataLoader(self.test_set, batch_size=self.config.test.bsz)
    

class VAEDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def prepare_data(self):
        pass
    
    def setup(self, stage: str):
        if stage == 'fit':
            img_tf = get_tf_for_ae(self.config)
            self.train_set = VAEDataset(self.config.train.img_dir, transform=img_tf)

        if stage == 'test':
            img_tf = get_tf_for_ae(self.config)
            self.test_set = VAEDataset(self.config.test.img_dir, transform=img_tf)
    
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.config.train.bsz)
    
    def train_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.config.test.bsz)    