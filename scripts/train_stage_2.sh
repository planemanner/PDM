#!/bin/bash

autoencoder_ckpt=/path/to/autoencoder/checkpoint.ckpt
save_dir=/path/to/save/diffusion/model

python ../main.py --stage diffusion \
                  --autoencoder_ckpt $autoencoder_ckpt \
                  --save_dir $save_dir