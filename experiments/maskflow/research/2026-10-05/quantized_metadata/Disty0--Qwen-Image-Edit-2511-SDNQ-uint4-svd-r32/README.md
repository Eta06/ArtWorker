---
license: apache-2.0
base_model:
- Qwen/Qwen-Image-Edit-2511
base_model_relation: quantized
library_name: diffusers
tags:
- sdnq
- qwen_image
- 4-bit
---
4 bit (UINT4 with SVD rank 32) quantization of [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) using [SDNQ](https://github.com/Disty0/sdnq).  

Usage:  
```
pip install sdnq
```

```py
import torch
import diffusers
from sdnq import SDNQConfig # import sdnq to register it into diffusers and transformers
from sdnq.common import use_torch_compile as triton_is_available
from sdnq.loader import apply_sdnq_options_to_model

pipe = diffusers.QwenImageEditPlusPipeline.from_pretrained("Disty0/Qwen-Image-Edit-2511-SDNQ-uint4-svd-r32", torch_dtype=torch.bfloat16)

# Enable INT8 MatMul for AMD, Intel ARC and Nvidia GPUs:
if triton_is_available and (torch.cuda.is_available() or torch.xpu.is_available()):
    pipe.transformer = apply_sdnq_options_to_model(pipe.transformer, use_quantized_matmul=True)
    pipe.text_encoder = apply_sdnq_options_to_model(pipe.text_encoder, use_quantized_matmul=True)
    # pipe.transformer = torch.compile(pipe.transformer) # optional for faster speeds

pipe.enable_model_cpu_offload()
pipe.set_progress_bar_config(disable=None)

image1 = Image.open("input1.png")
image2 = Image.open("input2.png")
prompt = "The magician bear is on the left, the alchemist bear is on the right, facing each other in the central park square."
inputs = {
    "image": [image1, image2],
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 40,
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
}
with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_image.save("qwen-image-edit-2511-sdnq-uint4-svd-r32.png")
```