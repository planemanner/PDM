# gpu for training : https://lightning.ai/docs/pytorch/stable/accelerators/gpu_basic.html#multi-gpu
# FSDP : https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/fsdp.html
# Trainer config : https://lightning.ai/docs/pytorch/stable/common/trainer.html
trainer_diffusion_config = {"accelerator": "gpu", # cpu, gpu, tpu, auto
                  "devices": [0, 1], # 'auto', List[int], int, "0, 1", -1 ...
                  "strategy": "auto", # auto, "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_2_offload", "deepspeed_stage_3", "fsdp"
                  "precision": "16-mixed" # int or str like this example.
                  }

trainer_autoencoder_config = {"accelerator": "gpu", # cpu, gpu, tpu, auto
                  "devices": [0, 1], # 'auto', List[int], int, "0, 1", -1 ...
                  "strategy": "auto", # auto, "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_2_offload", "deepspeed_stage_3", "fsdp"
                  "precision": "16-mixed" # int or str like this example.
                  }
"""
Note followings
1. When saving a model using DeepSpeed and Stage 3, model states and optimizer states will be saved in 
separate sharded states (based on the world size). 
2. FSDP trades off speed for memory efficient training due to communication overheads. 

"""