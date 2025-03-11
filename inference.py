from diffusion import StableDiffusion
from configs.diffusion_cfg import unet_config, ae_config, sampler_config, conditioner_config
import argparse
import torch
from PIL import Image
from data.dataset import load_img_list
import os
import numpy as np
from typing import List
from matplotlib import pyplot as plt

def paste_images(images:List[Image.Image], subtitles: List[str], save_path: str):
    fig, axes = plt.subplots(nrows=1, ncols=len(images), figsize=(12, 4))

    for ax, img, subtitle in zip(axes, images, subtitles):
        ax.imshow(np.array(img))
        ax.set_title(subtitle, fontsize=12, pad=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default="")
    parser.add_argument('--save_dir', type=str, default="")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--test_dir', type=str, default="The directory path of prompt images")
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--bsz', type=int, default=8)

    args = parser.parse_args()

    ckpt = torch.load(args.ckpt_path)

    diffuser = StableDiffusion(unet_cfg=unet_config, 
                               vae_cfg=ae_config, 
                               sampler_cfg=sampler_config,
                               conditioner_cfg=conditioner_config,
                               save_dir=args.save_dir,
                               )
    
    diffuser.load_state_dict(ckpt['state_dict'])
    diffuser.eval()

    diffuser.to(args.device)

    mask_list = load_img_list(args.test_dir)
    subtitles = ['PROMPT', 'GENERATED', 'ORIGINAL']
    hint = 'MASK_HINT'
    with torch.no_grad():
        batch_buffer = []
        path_buffer = []

        while mask_list:
            mask_path = mask_list.pop()
            batch_buffer.append(Image.open(mask_path).convert('RGB'))
            path_buffer.append(mask_path)
            if len(batch_buffer) == args.bsz or len(mask_list) == 0:
                decoded = diffuser.generate_sketch2image(batch_buffer)
                decoded = decoded.permute(0, 2, 3, 1)
                for mask_p, decode in zip(path_buffer, decoded):
                    _decode = Image.fromarray(decode.cpu().detach().numpy().astype(np.uint8))
                    orig_img = Image.open(mask_p.replace(hint, "img")).convert('RGB')
                    prompt = Image.open(mask_p).convert('RGB')
                    save_path = os.path.join(args.save_dir, os.path.basename(mask_p))
                    paste_images([prompt, _decode, orig_img], subtitles, save_path)
                batch_buffer = []
                path_buffer = []