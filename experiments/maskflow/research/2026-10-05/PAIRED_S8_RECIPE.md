# Pinned MaskFlow SFT-S + DMD-S8 recipe

The published accelerated scene recipe is **`maskflow-S.safetensors` plus `maskflow-S-tcfg4-step8.safetensors`, both at scale 1, on Qwen-Image-Edit-2511, with 8 steps and text CFG 4**. The DMD file is a residual to the older SFT-S teacher. `latest.safetensors` is a different teacher and cannot replace SFT-S in this recipe. This route reduces denoising evaluations; it retains the same 20B base model and requires more adapter storage than latest alone.

This audit inspected existing pinned source/config/Hub metadata and previously fetched safetensors headers, and evaluated the exact sigma expression with small Torch **CPU** tensors. It did not download tensor payloads, initialize a full model, run GPU inference, modify either inference runner, or assess image quality. `PAIRED_S8_RECIPE.json` contains the machine-readable evidence and hashes. Header and Hub hashes are expected payload identities, not verification of downloaded payloads.

## Immutable source and weight identities

| Component | Pin |
|---|---|
| [MaskFlow implementation](https://github.com/ReyChiaro/MaskFlow/tree/3125f1ecd72f5a4e068c5a1954e9d6a8df38d443) | Git `3125f1ecd72f5a4e068c5a1954e9d6a8df38d443` |
| [MaskFlow adapter repository](https://huggingface.co/ReyChiaro/MaskFlow/tree/f18f0828ab6a40a7f3456feace7a33bebe8c511e) | HF Git `f18f0828ab6a40a7f3456feace7a33bebe8c511e`, public and ungated in saved metadata |
| [Original Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511/tree/6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9) | HF Git `6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9` |
| [Isolated MLX-Gen runtime](https://github.com/lpalbou/mlx-gen/tree/99fb94dd3eaa9dd1931cd3cd8eae1ae3e20f2ef3) | Git `99fb94dd3eaa9dd1931cd3cd8eae1ae3e20f2ef3` |
| Selected mixed-quantized MLX base | `AbstractFramework/qwen-image-edit-2511-4bit`, HF Git `dbc6d597c1316fc34c0f7726904121a284a69c0b` |

The two adapter files total **3,020,141,352 bytes** including headers:

| File | File bytes | Git LFS pointer blob SHA1 | Expected LFS/file SHA256 |
|---|---:|---|---|
| `maskflow-S.safetensors` | 1,510,070,696 | `828663b393bba67cab68efa5b7794d6db74ae189` | `abde09d33238e6b2133a4a7dd86b9830ef93a8d97978e5ba437d60675c75dafd` |
| `maskflow-S-tcfg4-step8.safetensors` | 1,510,070,656 | `707096a0196a1107150f0259855783468e88e66a` | `109580b294c3993984303fa39cb1dc77dfc90fc892075e90d85d1afd079a49e7` |

These identities agree between the saved Hub file metadata and pinned [MANIFEST.json](https://huggingface.co/ReyChiaro/MaskFlow/blob/f18f0828ab6a40a7f3456feace7a33bebe8c511e/MANIFEST.json). The pointer blob SHA1 identifies Git's small LFS pointer; the LFS SHA256 identifies the complete safetensors file. Each pointer is 135 bytes. The tensor payloads remain unverified until streamed full-file SHA256 checks pass.

## Adapter structure and composition

Both older adapters have **960 BF16 tensors, 480 complete A/B pairs, rank 256, alpha 256**, across all 60 transformer blocks. Their key sets, shapes and dtypes match exactly. Each has 754,974,720 parameters / 1,509,949,440 tensor bytes / 1.40625 GiB. Metadata specifies ordinary LoRA, no DoRA or RSLoRA, zero dropout, no bias, and empty per-target rank/alpha patterns. They are not rank-128 adapters.

Targets per block are the eight attention projections: `attn.to_q`, `attn.to_k`, `attn.to_v`, `attn.to_out.0`, `attn.add_q_proj`, `attn.add_k_proj`, `attn.add_v_proj`, `attn.to_add_out`. A shapes are `[256,3072]`; B shapes are `[3072,256]`. Latest instead has 1,200 tensors / 600 pairs and adds `img_mlp.net.2` and `txt_mlp.net.2`. A validator hard-coded to latest's 1,200 keys or 600 targets must be parameterized in a **separate paired runner**.

The exact official [loader](https://github.com/ReyChiaro/MaskFlow/blob/3125f1ecd72f5a4e068c5a1954e9d6a8df38d443/trainer/lora_utils.py) does this:

1. Load SFT-S as `maskflow`; activate it; fuse it into the base with `checkpoint.lora_scale` (default 1.0); unload the adapter object.
2. Load S8 as `dmd`; activate it. The loader does **not** pass `checkpoint.lora_scale` to DMD. Its default PEFT adapter scaling is alpha/r = 256/256 = 1.

For an individual adapted matrix, full-precision algebra at the published scales is:

```text
W_S    = W_2511 + B_S A_S
y      = x W_Sᵀ + x A_Dᵀ B_Dᵀ
W_eff  = W_2511 + B_S A_S + B_D A_D
```

Increasing `checkpoint.lora_scale` changes the SFT term, not both terms. It therefore changes the teacher beneath the fixed DMD residual. Keep both effective scales exactly 1 for the first paired trial.

In the pinned MLX-Gen source, runtime loading of a second LoRA at an already-adapted layer creates a `FusedLoRALinear` that computes one base output plus the **sum of both low-rank residuals**. It does not apply one residual to the other residual's output. This supplies the intended full-precision algebra without baking Q4 weights. The original raw names (`transformer_blocks.0.attn.to_q.lora_A.weight`, etc.) match zero pinned Qwen mapping keys; an in-memory `diffusion_model.` prefix matches **960/960 keys for each**. Preserve the original downloaded files and apply the bridge only in memory.

This is not bitwise parity with the official BF16 recipe: official SFT fusion first rounds the merged BF16 base matrix, while Q4-base runtime residual evaluation retains the Q4 base and rounds intermediate outputs differently. A tiny dual-adapter CPU test should verify summation, transpose, alpha, scale and key coverage before a paired local trial. Quantized-base quality still requires images. Do not multiply the sum of A matrices by the sum of B matrices: that introduces cross terms.

## Eight steps still use CFG

The `tcfg4` filename does not authorize CFG 1 or a single forward. The pinned [manifest](https://huggingface.co/ReyChiaro/MaskFlow/blob/f18f0828ab6a40a7f3456feace7a33bebe8c511e/MANIFEST.json) recommends **8 inference steps, text CFG 4.0**. The inference defaults are mask CFG 1.0, interaction CFG null, and `rescale_cfg=true`.

The [CFG implementation](https://github.com/ReyChiaro/MaskFlow/blob/3125f1ecd72f5a4e068c5a1954e9d6a8df38d443/pipelines/cfg.py) gives coefficients `pm=4, pn=0, nm=-3, nn=0`, so only `pm` and `nm` run. Both retain source and mask conditioning; `nm` uses the negative prompt, which defaults to an empty string.

```text
v = v_nm + 4 (v_pm - v_nm)
v = v × max(||v_pm||₂,1e-6) / max(||v||₂,1e-6)
```

The norms are per packed token over its final feature dimension. This is **16 transformer forwards**, compared with 100 for the standard 50-step CFG4 trial. The 6.25-fold reduction in evaluations is arithmetic, not a measured wall-clock speedup. VLM conditioning, VAE work, loading, Poisson refinement and adapter overhead remain. Keep velocity comparison tracing disabled, since tracing both Poisson trajectories doubles the denoising forwards.

## Exact sigma grid at 512 × 1152

The original 2511 codec has 16 latent channels, 8× spatial reduction and 2×2 packing, so target tokens are `(512/16) × (1152/16) = 2304`. Use target tokens only in the dynamic shift, excluding the source and mask reference tokens.

The [official scheduler](https://github.com/ReyChiaro/MaskFlow/blob/3125f1ecd72f5a4e068c5a1954e9d6a8df38d443/schedulers/flow_matching.py) uses:

```text
mu = 0.5 + (0.9 - 0.5) × (2304 - 256) / (8192 - 256)
   = 0.603225806451613
t  = float32(linspace(1, 1/8, 8))
s  = exp(mu) / (exp(mu) + (1/t - 1))
```

It appends one terminal zero. The CPU expression produces these float32 sigmas:

```text
1.0000000000000000
0.9275154471397400
0.8457746505737305
0.7528836131095886
0.6463939547538757
0.5230836272239685
0.3786254525184631
0.2070689797401428
0.0000000000000000
```

Little-endian float32 byte SHA256: `612b8a4243bdb13090747e0aa340216b4c7527e0cc7c64f6f6d77ca5bf9c7947`. Torch CPU 2.14.0 and NumPy 2.5.3 evaluated the saved source expression without importing the scheduler module or any MLX model. An integration gate should also compare the eventual paired runtime sampler to the official CPU scheduler. Do not use MLX-Gen's stock `sigma_shift_terminal=0.02` stretch; it changes this trained schedule substantially, including the final nonzero sigma.

The usual MaskFlow updates remain: CFG prediction, x0 reconstruction, SoftPoisson refinement, velocity reconstruction, Euler update and projection onto the source/noise bridge. Gamma is 1 (`background_noise_power=1.0`). Default SoftPoisson remains enabled at every step with lambda_e=lambda_s=1, 50 Jacobi iterations, momentum 0.1. Eight steps do not imply eight Poisson iterations.

## Geometry, codec and preservation requirements

The base is specifically **Qwen-Image-Edit-2511**, including `zero_cond_t=true`: generated tokens use the current sigma/time; source and mask references use zero for image modulation; text uses target time. DiT is 60 layers, 24 heads × 128 dimensions, packed input width 64, output latent channels 16, joint text width 3584. Shape compatibility alone does not validate these timestep semantics. The existing Qwen-Image-2.1 codec and noise buffers must not be reused: that codec is RGBA / 64 channels / 16× reduction with different normalization.

For the current ArtWorker comparison, the exact inputs are RGB 512 × 1152, the 512-square source is at `[0,320,512,832]`, unknown top/bottom pixels are explicitly black, and white marks the edit mask. Source and softened mask are both VAE and VLM conditions, with generated, source, mask token order. All three full-resolution token grids contribute 2304 tokens each, i.e. 6912 image tokens plus text.

The official default aspect-ratio buckets include 9:16 but not 4:9, so force `preprocessing.max_resolution=589824` and `preprocessing.aspect_ratios=["4:9"]` for the exact 512 × 1152 geometry. Avoid an input preprocessor that silently crops or scales this canvas.

Mask dilation 25 followed by Gaussian blur 25 / sigma25 allows a roughly 24-pixel softened band **inside** the square's top and bottom edges. Native final pixel blend selects the preprocessed **BF16 source** where the processed mask is zero; it does not hard-preserve the entire original 512-square or guarantee original uint8 byte equality after BF16 rounding and image saving. Save the native soft-blended output and a separate exact-source hard-paste composite. Do not present the composite as evidence that the model itself preserved the source boundary. The exact hard-paste source is `source_512.png`, not the original 720-square JPEG crop.

## Execution preparation and remaining gates

This pair occupies **2.8125 GiB of BF16 tensor storage**, 0.52734375 GiB more than latest's 2.28515625 GiB. With the selected mixed-quantized base, storage-derived static totals are approximately 19.84 GiB before dropping VL and 15.34 GiB after dropping its 4.498 GiB text/vision stack. They are estimates, not observed peak or resident memory. Runtime residuals avoid baking a new Q8 base but need both adapter payloads resident. Stage VL conditioning first, release VL, record MLX active/cache/peak memory, and use the root's serialized bounded GPU queue.

The following is a **prepared public download command only**, not executed in this audit. Downloading requires approximately 3.021 GB plus verification/headroom and produces a separate paired directory:

```bash
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
  /Users/emir/Documents/ChatGPT/ArtWorker/experiments/qwen/.venv/bin/hf download \
  ReyChiaro/MaskFlow maskflow-S.safetensors maskflow-S-tcfg4-step8.safetensors \
  --revision f18f0828ab6a40a7f3456feace7a33bebe8c511e \
  --local-dir /Users/emir/Documents/ChatGPT/ArtWorker/experiments/maskflow/models/paired_s8 \
  --max-workers 2
```

An official CUDA reference uses a complete local Diffusers **original 2511** snapshot, not the MLX mixed-quantized saved-module directory. Once those local payloads are hash-verified, the published implementation can be configured as follows; paths and prompt are explicit placeholders and this command is not an executed inference result:

```bash
python inference.py \
  pipeline.pretrained_model=/absolute/pinned-diffusers-qwen-image-edit-2511 \
  input.source=/Users/emir/Documents/ChatGPT/ArtWorker/experiments/evaluation/inputs/track3/canvas.png \
  input.mask=/Users/emir/Documents/ChatGPT/ArtWorker/experiments/evaluation/inputs/track3/mask.png \
  'input.prompt=<the frozen comparison prompt>' \
  preprocessing.max_resolution=589824 'preprocessing.aspect_ratios=["4:9"]' \
  checkpoint.sft_path=/absolute/pinned-maskflow-pair \
  checkpoint.sft_weight_name=maskflow-S.safetensors \
  checkpoint.sft_adapter_name=maskflow checkpoint.lora_scale=1.0 \
  checkpoint.dmd_path=/absolute/pinned-maskflow-pair \
  checkpoint.dmd_weight_name=maskflow-S-tcfg4-step8.safetensors \
  checkpoint.dmd_adapter_name=dmd \
  runtime.num_inference_steps=8 runtime.text_cfg_scale=4.0 \
  runtime.mask_cfg_scale=1.0 runtime.interaction_cfg_scale=null \
  runtime.dtype=bfloat16 runtime.device=cuda runtime.seed=42 \
  output.path=/absolute/output/paired_s8_native.png
```

A local MLX trial needs a new paired runner/manifest and separate CPU gates, including 960/960 coverage **for each adapter**, dual residual summation and scaled alpha, the 8-step schedule above, 2511 transformer/zero_cond_t parity, mask/codec/Poisson math, exact source geometry, payload checks and bounded memory. Explicitly reject a mismatched pair: the official loader checks that SFT is present, but does not validate teacher/residual variant identity. Its default inference config is SFT-only with DMD null and 50 steps, so both the DMD and step overrides are necessary. The model must receive shifted sigma without applying a second shift; the final zero is terminal only. The existing standard-latest 50-step runner must stay frozen. This route is ready to implement after that choice; it is not yet payload-verified, inference-ready or quality-approved. MaskFlow code and adapters are MIT; the Qwen base/selected derivative remain subject to their Apache 2.0 terms.
