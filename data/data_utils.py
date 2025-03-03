import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_tf_for_sketch(config) -> A.BasicTransform:
    # prompt can be 'sketch', 'text', and e.t.c....

    resize = A.Resize(height=config.transform.resize_height, width=config.transform.resize_width)

    if config.mode != 'test':
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

def get_tf_for_ae(config) -> A.BasicTransform:
    resize = A.Resize(height=config.transform.resize_height, width=config.transform.resize_width)

    if config.mode != 'test':
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


