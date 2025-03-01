import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transform(config):
    resize = A.Resize(height=config.resize_height, width=config.resize_width)
    normalization = A.Normalize(mean=config.normalize.mean, std=config.normalize.std)
    totensor = ToTensorV2()

    if config.mode != 'test':
        hflip = A.HorizontalFlip(p=config.hflip.p)
        vflip = A.VerticalFlip(p=config.vflip.p)
        rot90 = A.RandomRotate90(p=config.rot90.p)
        # normalize rescale the image value range from [0, 255] to [0, 1]
        return A.Compose([resize, hflip, vflip, rot90, normalization, totensor])
    else:
        # test
        return A.Compose([resize, normalization, totensor])


