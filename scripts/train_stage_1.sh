#!/bin/bash

save_dir=/my_path
seed=42

python ../main.py --stage autoencoder \
                  --save_dir $save_dir
                  --seed $seed