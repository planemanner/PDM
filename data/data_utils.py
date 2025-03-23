import albumentations as A
from albumentations.pytorch import ToTensorV2
from glob import glob
import os
from typing import List


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

def get_tf_for_sketch(config, mode='train') -> A.BasicTransform:
    # prompt can be 'sketch', 'text', and e.t.c....

    resize = A.Resize(height=config.transform.resize_height, width=config.transform.resize_width)

    if mode != 'test':
        hflip = A.HorizontalFlip(p=config.transform.hflip)
        vflip = A.VerticalFlip(p=config.transform.vflip)
        rot90 = A.RandomRotate90(p=config.transform.rot90)
        # normalize rescale the image value range from [0, 255] to [0, 1]
        tf = A.Compose([resize, hflip, vflip, rot90])
        return tf
    else:
        # test
        tf = A.Compose([resize])
        return tf

def get_tf_for_ae(config, mode='train') -> A.BasicTransform:
    resize = A.Resize(height=config.transform.resize_height, width=config.transform.resize_width)

    if mode != 'test':
        random_resized_crop = A.RandomResizedCrop(height=config.transform.resize_height, width=config.transform.resize_width, scale=(0.8, 1.0))
        hflip = A.HorizontalFlip(p=config.transform.hflip)
        vflip = A.VerticalFlip(p=config.transform.vflip)
        color_jitter = A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
        rot90 = A.RandomRotate90(p=config.transform.rot90)
        
        tf = A.Compose([random_resized_crop, hflip, vflip, color_jitter, 
                            rot90])
        return tf
    else:
        # test
        tf = A.Compose([resize])
        return tf


