#!/bin/bash

seed=42
bsz=64
sampler_type="flow-matching"
gpus=(0 1 2 3)
epochs=100
seed=42
python ../main.py --stage autoencoder \
                  --bsz $bsz \
                  --sampler_type $sampler_type \
                  --seed $seed \
                  --gpus $gpus \
                  --epochs $epochs \
                  --seed $seed 