# Implementation of Diffusion Probabilistic Model Made Slim, CVPR 2023
- This is unofficial implementation repository.
  - [Paper Link](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_Diffusion_Probabilistic_Model_Made_Slim_CVPR_2023_paper.pdf) and [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
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
# To do
- Replace the UNet's Down & Up sample blocks with WaveletLayers.
- Add distillation parts for Diffusion Model
- Check the configuration combinations
- Check whether this code base is actually working or not.
  - Loading models and pipelines on GPUs
  - Sampling
  - Multi-GPUs & Multi-Node
- Update quantitative metrics for validation. (e.g., FID)
- Add additional analysis tools with callback function (e.g., MLFlow)
- ICLR 2025 Paper 내용을 고려하여 UNet 구조를 약간 변경 및 Sampler 구현
# Requirements
- Python version : > 3.9
- PyTorch version 2.0 or later is required.
- The whole traininng pipeline is written under Pytorch Lightning 2.5.0
# Usage
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

### 출력 결과

추론 과정은 각 입력 스케치에 대해 다음과 같은 결과를 생성합니다:

1. 입력 스케치 이미지
2. 모델이 생성한 이미지
3. 원본 참조 이미지 (있는 경우)

이 세 이미지는 하나의 파일로 병합되어 `--save_dir` 경로에 저장됩니다. 각 결과 이미지에는 'PROMPT', 'GENERATED', 'ORIGINAL'이라는 제목이 표시됩니다.

### 디렉토리 구조

테스트 디렉토리는 다음과 같은 구조를 가져야 합니다:

```
test_dir/
  ├── MASK_HINT/  # 입력 스케치 이미지
  │    ├── image1.png
  │    ├── image2.png
  │    └── ...
  └── img/        # 원본 참조 이미지 (선택 사항)
       ├── image1.png
       ├── image2.png
       └── ...
```

### 코드 예시

모델을 프로그래밍 방식으로 사용하려면 다음 코드를 참조하세요:

```python
from diffusion import StableDiffusion
import torch
from PIL import Image

# 모델 로드
diffuser = StableDiffusion(unet_cfg=unet_config, 
                          vae_cfg=ae_config, 
                          sampler_cfg=sampler_config,
                          conditioner_cfg=conditioner_config)

# 체크포인트 로드
ckpt = torch.load("path/to/checkpoint.ckpt")
diffuser.load_state_dict(ckpt['state_dict'])
diffuser.eval()
diffuser.to("cuda")

# 이미지 생성
sketch_image = Image.open("path/to/sketch.png").convert('RGB')
with torch.no_grad():
    generated_image = diffuser.generate_sketch2image([sketch_image])
```

### 주의사항

- 입력 스케치 이미지는 RGB 형식이어야 합니다.
- 최상의 결과를 위해 학습 데이터와 유사한 스타일의 스케치를 사용하세요.
- 대용량 이미지를 처리할 때는 배치 크기(`--bsz`)를 조정하여 메모리 사용량을 관리하세요.