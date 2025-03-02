from dotdict import DotDict

conditioner_config = {"prompt_type": "image",
                      "model_name": "openai/clip-vit-base-patch32"
                      }

conditioner_config = DotDict(conditioner_config)