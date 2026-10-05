---
license: apache-2.0
tags:
- mlx
- mflux
base_model: Qwen/qwen-image-edit-2511
pipeline_tag: image-to-image
---

# mlx-community/qwen-image-edit-2511-8bit

This repository contains the 8-bit quantized weights for `Qwen/qwen-image-edit-2511`.

**Quantized and Contributed by:** [@lpalbou](https://huggingface.co/lpalbou)

## Usage

You can run this model locally on Apple Silicon using `mflux`:

```bash
pip install mflux

mflux-generate-controlnet --model mlx-community/qwen-image-edit-2511-8bit --image-path path_to_image.png --prompt "Your edit prompt here"
```