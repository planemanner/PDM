# Implementation of Stable Diffusion and One Step Diffusion Model
- This is unofficial implementation repository for Stable Diffusion (2021) and One Step Diffusion (ICLR 2025).
  - [One Step Diffusion](https://arxiv.org/pdf/2410.12557)
- This paper adds and modifies wavelet layer and the number of channels for each layer so that some code of this repository is similar to [Stable Diffusion](https://github.com/CompVis/stable-diffusion).  
- For clarity and better readibility, I modified variable names and integrated modules.
- Since num_timesteps_cond is always set to '1', all related legacy code blocks are removed.
- Redundant or legacy parts are removed.
# Note
- In current version, the autoencoder is not VQ-VAE. It is conventional variational auto-encoder.
  - So, unlike original Stable Diffusion model, the autoencoder is trained without EMA update. 
  - The Wavelet layers will be applied soon.

- Current version is very incomfortable to use most features.
  - Most configuration files are integrated for convenience and reduction of confusion.
  - Since this is the first project using pytorch lightning, many parts are not graceful.
  - So, some migrations are needed

# Requirements
- Python version : > 3.9
- PyTorch version 2.0 or later is required.
- The whole traininng pipeline is written under Pytorch Lightning 2.5.0
# Usage
- ...
## Training
### Stage 1 : Train AutoEncoder
```sh
python main.py --stage autoencoder
```
### Stage 2 : Train Diffusion Model
```sh
python main.py --stage diffusion --autoencoder_ckpt [AUTOENCODER PATH]
```

## Inference
- Inference 코드를 작성하여 모델을 사용하여 이미지를 생성합니다.
- 예시:
```python
from models.autoencoder import AutoEncoder
from models.unet import UNetModel
from samplers.ddpm import DDPMSampler
from configs import autoencoder_cfg, unet_cfg, sampler_cfg
from common_utils import DotDict

# Load models
autoencoder = AutoEncoder(DotDict(autoencoder_cfg.autoencoder_config))
unet = UNetModel(DotDict(unet_cfg.unet_config))
sampler = DDPMSampler(DotDict(sampler_cfg.sampler_config))

# Load checkpoints
autoencoder.load_state_dict(torch.load('path/to/autoencoder/checkpoint.ckpt'))
unet.load_state_dict(torch.load('path/to/unet/checkpoint.ckpt'))

# Generate images
generated_images = sampler.sampling(unet, image_shape=(batch_size, 3, 256, 256))
```

## Serving
- Serving 코드를 작성하여 모델을 배포합니다.
- 예시:
```python
from flask import Flask, request, jsonify
from models.autoencoder import AutoEncoder
from models.unet import UNetModel
from samplers.ddpm import DDPMSampler
from configs import autoencoder_cfg, unet_cfg, sampler_cfg
from common_utils import DotDict

app = Flask(__name__)

# Load models
autoencoder = AutoEncoder(DotDict(autoencoder_cfg.autoencoder_config))
unet = UNetModel(DotDict(unet_cfg.unet_config))
sampler = DDPMSampler(DotDict(sampler_cfg.sampler_config))

# Load checkpoints
autoencoder.load_state_dict(torch.load('path/to/autoencoder/checkpoint.ckpt'))
unet.load_state_dict(torch.load('path/to/unet/checkpoint.ckpt'))

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    batch_size = data.get('batch_size', 1)
    generated_images = sampler.sampling(unet, image_shape=(batch_size, 3, 256, 256))
    return jsonify({'generated_images': generated_images.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## Convert sharded checkpoint to consolidated checkpoint
```bash
cd lightning_logs/version_0/checkpoints
python -m lightning.pytorch.utilities.consolidate_checkpoint epoch=0-step=3.ckpt
```

## To do
- Update quantitative metrics for validation. (e.g., FID)
- Add additional analysis tools with callback function (e.g., MLFlow)
- Code base looks spagetti. So, I'll refactor configuration and related part for convenience in terms of usage and maintenance.