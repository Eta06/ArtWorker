---
license: cc-by-nc-sa-4.0
language:
- en
- zh
library_name: diffusers
pipeline_tag: image-to-image
quantized_by: Abhishek Dujari
base_model:
- Qwen/Qwen-Image-Edit-2511
base_model_relation: quantized
---

This is an NF4 quantized model of Qwen-image-edit-2511 so it can run on GPUs using less than 20GB VRAM. You can run it on lower VRAM like 16GB. 
There were other NF4 models but they made the mistake of blindly quantizing all layers in the transformer. 
This one does not. We retain some layers at full precision in order to ensure that we get quality output.

You can use the original Qwen-Image-Edit parameters.


Model tested: Working perfectly even with 10 steps.
Contact: support@[JustLab.ai](https://justlab.ai) for commercial support, modifications and licensing.


### sample code

```python
import os
from PIL import Image
import torch

from diffusers import QwenImageEditPlusPipeline

model_path = "ovedrive/Qwen-Image-Edit-2511-4bit"
pipeline = QwenImageEditPlusPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
print("pipeline loaded") # not true but whatever. do not move to cuda

pipeline.set_progress_bar_config(disable=None)
pipeline.enable_model_cpu_offload() #if you have enough VRAM replace this line with `pipeline.to("cuda")` which is 20GB VRAM
image = Image.open("./example.png").convert("RGB")
prompt = "Remove the lady head with white hair"
inputs = {
    "image": image,
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 20, # even 10 steps should be enough in many cases
}

with torch.inference_mode():
  output = pipeline(**inputs)

output_image = output.images[0]
output_image.save("output_image_edit.png")
print("image saved at", os.path.abspath("output_image_edit.png"))
```



## FAQ
1. Does it support comfyui? Someone else has done an integration here https://github.com/mengqin/ComfyUI-UnetBnbModelLoader
2. Is there a GGUF etc ? Not here. The motivation for this is to use Diffusers and get the best performance without relying on 3rd party toolkits and transformation wich break the pipeline.
3. How to use speed up lora? See 1 above. I believe the same lora(s) should work in the standard pipeline. In the past and aroudn the first release of Qwen-Image a bug in `diffusers` prevented lora from attaching.
4. Can you support model XYZ? I only work with open source models that use a standard OSS licensing. Any custom license claiming to be FOSS will not be entertained. Alibaba/Qwen has been most consistent and generous and I am not affiliated with anyone.
5. How many steps? You have to test. With speed up lora, users have reported 4-5 steps, I supppose that is minimum a model can do because convergence cannot happen in 1-2 steps. On a consumer GPU like 4090 you could get 2 seconds which was my goal, you only get 2 seconds on the big model with H200 class of gpus. My reccomendation is to skip lora since this model and other like this are already small. Instead use an upscaler.
6. Is this a merge/lora? No, its a quantization using BNB which allows to target specific layers. There are other quant mechanisms and I test them against a specific moodel to see where it could work best. bnb just happens to be my favorite.
7. is it really nf4? You are correct. It is not 100% nf4 that woudl result in very  bad quality. It is mixed precision which is totally normal. Not all layers need to be at full precision. That was my theory without writinga whole paper on it.

Qwen family is the most open and easy to work with model. While it's training data is limited, it also makes it safe for general use and as a lab subject. It has a very good quality.
Qwen-Image also works really well with other models for e.g. the outputs can be upscaled to amazing quality using ERSGAN models, specially if you use Qwen-Edit-2511 4bit.
BNB is another FOSS toolkit and the huggingface diffusers/transformers library is something I wanted to learn by making useful things. Some might say BNB is outdated, but its flexible.

There are limitations to BNB quanitzed models like its speicifc to Nvidia GPUs and it may not work with finetuning.


The original license and attributions are below.

## License Agreement

Qwen-Image is licensed under Apache 2.0. 

## Citation

We kindly encourage citation of our work if you find it useful.

```bibtex
@misc{wu2025qwenimagetechnicalreport,
      title={Qwen-Image Technical Report}, 
      author={Chenfei Wu and Jiahao Li and Jingren Zhou and Junyang Lin and Kaiyuan Gao and Kun Yan and Sheng-ming Yin and Shuai Bai and Xiao Xu and Yilei Chen and Yuxiang Chen and Zecheng Tang and Zekai Zhang and Zhengyi Wang and An Yang and Bowen Yu and Chen Cheng and Dayiheng Liu and Deqing Li and Hang Zhang and Hao Meng and Hu Wei and Jingyuan Ni and Kai Chen and Kuan Cao and Liang Peng and Lin Qu and Minggang Wu and Peng Wang and Shuting Yu and Tingkun Wen and Wensen Feng and Xiaoxiao Xu and Yi Wang and Yichang Zhang and Yongqiang Zhu and Yujia Wu and Yuxuan Cai and Zenan Liu},
      year={2025},
      eprint={2508.02324},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.02324}, 
}
```