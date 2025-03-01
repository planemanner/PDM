import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_tf_for_sketch(config) -> A.BasicTransform:
    # prompt can be 'sketch', 'text', and e.t.c....

    resize = A.Resize(height=config.resize_height, width=config.resize_width)
    normalization = A.Normalize(mean=config.normalize.mean, std=config.normalize.std)
    totensor = ToTensorV2()

    if config.mode != 'test':
        hflip = A.HorizontalFlip(p=config.hflip.p)
        vflip = A.VerticalFlip(p=config.vflip.p)
        rot90 = A.RandomRotate90(p=config.rot90.p)
        # normalize rescale the image value range from [0, 255] to [0, 1]
        img_tf = A.Compose([resize, hflip, vflip, rot90, normalization, totensor])
        sketch_tf = A.Compose([resize, hflip, vflip, rot90])
        return img_tf, sketch_tf
    else:
        # test
        img_tf = A.Compose([resize, normalization, totensor])
        sketch_tf = A.Compose([resize])
        return img_tf, sketch_tf

def get_tf_for_ae(config) -> A.BasicTransform:
    resize = A.Resize(height=config.resize_height, width=config.resize_width)
    normalization = A.Normalize(mean=config.normalize.mean, std=config.normalize.std)
    totensor = ToTensorV2()

    if config.mode != 'test':
        random_resized_crop = A.RandomResizedCrop(height=config.resize_height, width=config.resize_width, scale=(0.8, 1.0))
        hflip = A.HorizontalFlip(p=config.hflip.p)
        vflip = A.VerticalFlip(p=config.vflip.p)
        color_jitter = A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
        rot90 = A.RandomRotate90(p=config.rot90.p)
        
        img_tf = A.Compose([random_resized_crop, hflip, vflip, color_jitter, 
                            rot90, normalization, totensor])
        return img_tf
    else:
        # test
        img_tf = A.Compose([resize, normalization, totensor])
        return img_tf


