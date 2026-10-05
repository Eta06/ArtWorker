---
license: apache-2.0
base_model: Qwen/Qwen-Image-Edit-2511
library_name: mlx
tags:
- mlx
- qwen-image-edit
- image-to-image
- quantized
---

# qwen-image-edit-2511-mlx-int4

Pre-quantized **MLX** weights for **Qwen-Image-Edit-2511**, for the Swift
`xocialize/qwen-image-edit-swift` package's `QwenImageEdit` turbo tier. They let the engine
load the int4 tier **directly** instead of loading bf16 (~40 GB peak) and quantizing
in-memory.

> ⚠️ **Non-standard format — for the MLXEngine Swift loaders only.** These are **not**
> standard `mlx_lm` / `mlx_vlm` / diffusers-layout repos and will not load with generic MLX
> tooling. Tensors use the engine's internal module key names and carry custom quantization
> metadata; they load **only** via `QwenImageEditWeights.loadQuantizedDiT` and
> `QwenVLPromptEncoder.load(quantizedTextModelPath:)`. For a standard, tool-loadable
> conversion see [`mlx-community/qwen-image-edit-2511-8bit`](https://huggingface.co/mlx-community/qwen-image-edit-2511-8bit).

## Files
| file | size | contents |
|---|---|---|
| `qie-2511-dit-int4-mod8.safetensors` | ~15 GB | 20B DiT transformer — attention + feed-forward Linears **int4** (group 64), modulation linears **int8** (conditioning-critical), top-level projections bf16. Per-layer bits + group size travel in the safetensors metadata. |
| `qie-2511-vl7b-int4.safetensors` | ~4 GB | Qwen2.5-VL-7B text model, **int4** (group 64). The (small) vision tower is not included — it loads bf16 from the base snapshot. |

## Measured footprint (Swift MLXEngine turbo tier, 1024², 4-step DMD)
- Load peak **~21 GB** (no bf16 ever materialized) · resident **~21 GB** · inference peak **~25.5 GB** (with bf16 VAE).
- vs ~57 GB bf16 / ~41 GB quantize-after-load peak.

## Provenance / license
Derived from [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)
(Apache-2.0). Quantization: MLX affine, group size 64. The Lightning 4-step DMD LoRA is
applied separately at runtime and is **not** included here.
