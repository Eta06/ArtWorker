---
license: apache-2.0
base_model: Qwen/Qwen-Image-Edit-2511
base_model_relation: quantized
pipeline_tag: image-to-image
library_name: mflux
tags:
  - mflux
  - mlx
  - apple-silicon
  - quantized
  - image-to-image
  - qwen-image
---

# Qwen-Image Edit 2511 · MFlux Q4

This is a **Q4** **MFlux** version of [**Qwen-Image-Edit-2511**](https://huggingface.co/Qwen/Qwen-Image-Edit-2511).

[**MFlux**](https://github.com/filipstrand/mflux) runs the latest state-of-the-art generative image models locally on your Mac in native MLX. MFlux is open source and free.

> [!NOTE]
> Community conversion by [MFlux-Community](https://huggingface.co/mflux-community) -- not an official release from the original model's authors. The original model's license applies unchanged, see [License](#license).

### MFlux versions of Qwen-Image-Edit-2511
|   QUANT   | GB  | Repo |
| :--: | :-: | --- |
| BF16 | 56.6 | [mflux-community/qwen-image-edit-2511-mflux-bf16](https://huggingface.co/mflux-community/qwen-image-edit-2511-mflux-bf16) |
| Q8 | 37.5 | [mflux-community/qwen-image-edit-2511-mflux-q8](https://huggingface.co/mflux-community/qwen-image-edit-2511-mflux-q8) |
| Q6 | 32.4 | [mflux-community/qwen-image-edit-2511-mflux-q6](https://huggingface.co/mflux-community/qwen-image-edit-2511-mflux-q6) |
| Q5 | 29.8 | [mflux-community/qwen-image-edit-2511-mflux-q5](https://huggingface.co/mflux-community/qwen-image-edit-2511-mflux-q5) |
| **Q4** | 29.0 | this repo |
| Q3 | 24.7 | [mflux-community/qwen-image-edit-2511-mflux-q3](https://huggingface.co/mflux-community/qwen-image-edit-2511-mflux-q3) |

Converted on 2026-08-21.

Standard MFlux quantization uses MLX affine quantization (4-bit) across the text encoder, transformer and VAE. The quantization level is stored in the weights, so no `--quantize` flag is needed.

## Usage

**Install MFlux**
```bash
uv tool install --upgrade mflux
```

**Edit an image using Qwen-Image-Edit-2511 Q4:**
```bash
mflux-generate-qwen-edit \
  --model mflux-community/qwen-image-edit-2511-mflux-q4 \
  --base-model qwen-image-edit \
  --image-paths input.png \
  --prompt "A puffin standing on a cliff" \
  --width 1024 \
  --height 1024 \
  --seed 42 \
  --output image.png
```

The weights are downloaded from this repo on first use. To keep a local copy instead:
```bash
hf download mflux-community/qwen-image-edit-2511-mflux-q4 --local-dir ./qwen-image-edit-2511-mflux-q4
```
then pass `--model ./qwen-image-edit-2511-mflux-q4`.

## License

This repo inherits the license of the original model: **apache-2.0**. See the [original model card](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) for the full terms and any usage restrictions. The weights here were modified from the original release only by conversion to MLX and quantization.

## Credits

- Original model: [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)
- MLX implementation: [MFlux](https://github.com/filipstrand/mflux) by Filip Strand and contributors
- Conversion: [MFlux-Community](https://huggingface.co/mflux-community)
