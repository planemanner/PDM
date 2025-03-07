#!/bin/bash

autoencoder_ckpt=/data/smddls77/StableDiffusion/epoch=499.ckpt
save_dir=/data/smddls77/StableDiffusion/UNet

python ../main.py --stage diffusion \
                  --autoencoder_ckpt $autoencoder_ckpt \
                  --save_dir $save_dir