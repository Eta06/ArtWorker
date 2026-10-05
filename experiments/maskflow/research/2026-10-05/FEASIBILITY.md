# MaskFlow teacher feasibility — 2026-10-05

MaskFlow is a released, task-trained regional-editing teacher. It cannot be reproduced by loading its adapter into our current Qwen-Image-2.1 loop. The safest unmodified reference run is the pinned official CUDA implementation; a local mixed-quantized 2511 route is feasible to prepare with a separate MLX runtime and parity-tested sampler.

## Pinned evidence

- Official code: [ReyChiaro/MaskFlow](https://github.com/ReyChiaro/MaskFlow/tree/3125f1ecd72f5a4e068c5a1954e9d6a8df38d443), commit `3125f1ecd72f5a4e068c5a1954e9d6a8df38d443`, committed 2026-09-28.
- [Paper v3](https://arxiv.org/html/2608.06929v3), revised 2026-09-28: mask-aware probability paths/objective and training/sampling Soft-Poisson refinement. This is trained behavior, separate from our current post-chain hint localization.
- [Adapter repository](https://huggingface.co/ReyChiaro/MaskFlow/tree/f18f0828ab6a40a7f3456feace7a33bebe8c511e), revision `f18f0828ab6a40a7f3456feace7a33bebe8c511e`.
- Base: [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511/tree/6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9), revision `6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9`. Qwen2.1 is not an interchangeable base:2511 RGB VAE has16 channels at8× reduction packed2×2→64; current2.1 RGBA VAE has64 channels at16× reduction, different architecture and mean/std. Equal packed64 width is misleading.

Only source/config/model metadata and HTTP-206 adapter header ranges were downloaded. No tensor payloads, base weights, GPU execution, runtime changes or existing runner changes.

## Exact adapter recipe and headers

`latest.safetensors` is the current general-scene SFT adapter: 50 steps, text CFG4, LoRA scale1. Its file is 2,453,820,032 bytes; expected payload/file SHA256 `773e815a209ebf948dbe0cf76492dd658cc16030f8119471cc553edb208de7b0` has not been verified by a payload download. Header: 1,200 BF16 tensors / 600 A-B pairs / 1,226,833,920 parameters / 2.285156GiB tensor storage. Rank256, alpha256, no DoRA/RSLoRA, so alpha/r=1. Targets eight attention projections plus `img_mlp.net.2` and `txt_mlp.net.2` across 60 blocks. Keys are plain `transformer_blocks.0.attn.to_q.lora_A.weight`, etc.

Older S and SEC SFT adapters are 1,510,070,656–696 bytes and have matching 8-/16-step DMD residual adapters of 1,510,070,656 bytes each. SFT must be combined with the matching same-variant residual; latest+old DMD is not the published recipe. The accelerated variants retain the 20B base and recommended CFG4; they reduce inference evaluations, not parameter size. Our range audit covered latest, S, S8 and S16; SEC metadata was checked but its header was not fetched.

## What the official sampler actually does

Pinned `configs/pipeline/qwenimage_maskflow.yaml`, `pipelines/qwenimage/qwenimage_maskflow.py`, `schedulers/{flow_matching,mask_flow}.py`, `pipelines/cfg.py` and `pipelines/maskflow_utils.py` are saved locally.

- White mask=edit; source and softened mask are both VAE image conditions and VLM conditions. Denoiser token order is target, source, mask. At512×1152 this is 3×2304=6912 image tokens, plus text.
- Mask dilation kernel25, Gaussian kernel25/sigma25. Default source-preservation domain is where the processed mask is exactly zero; for this top/bottom outpaint mask, the processing can edit/blend roughly24 rows inside each original-source boundary. An exact512-square hardpaste would be an additional presentation step.
- Two text-CFG branches at4, with source+mask retained in both, followed by per-token positive-vector-norm rescaling. This is100 transformer evaluations for50 steps, not50.
- Exponential dynamic-shift scheduler, mu based on target tokens only; no `shift_terminal=0.02` stretch. At2304 tokens, mu=0.60322580645 and final nonzero sigma≈0.035964545 for50 steps; the appended terminal sigma is0. Current mflux default last nonzero sigma≈0.02 differs.
- Each prediction is converted to a clean latent, unpacked to16 channels at144×64, refined with a four-neighbor weighted Soft-Poisson solve (lambda_e=lambda_s=1;50 Jacobi iterations; momentum0.1), repacked and converted back to velocity. Then Euler integrates and the noisy-source flow bridge projects the preserved background. Final decode uses the softened-mask pixel blend.
- Soft-Poisson is a small latent-grid operation; its port is straightforward relative to full2511 denoiser/conditioning parity. A periodic `roll` neighbor implementation would be incorrect: the official boundary uses zero-padded four-neighbor sums and variable edge degree.

## Local runtime and resource findings

Installed mflux is pinned to `25661d8e853996fec9134998b995edf97369bfbd`. Its2511 alias resolves to2509 and its old Qwen transformer lacks2511 `zero_cond_t` reference-timestep modulation. Shapes alone cannot prove compatibility. Current LoRA mapping matches zero raw MaskFlow keys; an in-memory `diffusion_model.` prefix bridge maps all1,200 latest keys and all960 keys in each audited older adapter. A tiny CPU Q4-linear/rank256 LoRA arithmetic check passed with exact NumPy parity; that does not validate the full transformer. Current mflux baking upgrades LoRA-targeted Q4 layers toQ8; runtime/no-bake adapters avoid this extra quantization change.

Public [AbstractFramework/qwen-image-edit-2511-4bit](https://huggingface.co/AbstractFramework/qwen-image-edit-2511-4bit/tree/dbc6d597c1316fc34c0f7726904121a284a69c0b) is mixed DiT Q4/Q8 with quantized VL language/vision, BF16 VAE. Metadata totals: DiT13,199,230,548B + VL4,829,929,297B + VAE253,586,147B =18,282,745,992B /17.027GiB. Adding latest BF16 adapter gives19.312GiB static before buffers/activations/OS. Evicting the4.498GiB VL stack after both CFG embeddings are produced leaves about14.815GiB static. These are storage-based estimates, not measured inference peaks.

A newer isolated runtime is available: [MLX-Gen v0.38.0](https://github.com/lpalbou/mlx-gen/tree/99fb94dd3eaa9dd1931cd3cd8eae1ae3e20f2ef3), commit `99fb94dd3eaa9dd1931cd3cd8eae1ae3e20f2ef3`. Direct source inspection confirms2511 `qwen_edit_plus=True`, `zero_cond_t=True`, target timestep t versus reference timestep0 with per-token modulation; text remains at target time. Its declared MLX range includes our0.32.3. The `mlxgen` import facade aliases `mflux`, so a separate venv/checkpoint checkout is necessary to avoid mixing with the pinned2.1 runtime. Its `QwenImageEdit` constructor has no `bake_lora` argument; its adapter path loads runtime LoRA without baking. Do not pass the current mflux `--no-bake-lora` flag blindly. The stock edit command remains ordinary Qwen editing, not MaskFlow; a separate official-recipe wrapper is required. Verified constructor:

```python
from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit
from mflux.models.common.config.model_config import ModelConfig
model = QwenImageEdit(
    model_config=ModelConfig.from_name("qwen-image-edit-2511"),
    model_path="<local AbstractFramework prequantized directory>",
    quantize=None, lora_paths=None,
)
```

Custom bridge loads normalized MaskFlow keys afterwards. Transformer inputs include `t`, `config`, packed `hidden_states`, text embeddings/mask and `cond_image_grid`; source and mask are two reference grids following the target. RoPE is rectangular-grid based; arbitrary sparse IDs are currently unused. CLI entrypoints are `mlxgen` and legacy `mflux-generate-qwen-edit`, but neither stock mask option supplies the MaskFlow recipe. Source links: [model config](https://github.com/lpalbou/mlx-gen/blob/99fb94dd3eaa9dd1931cd3cd8eae1ae3e20f2ef3/src/mflux/models/common/config/model_config.py), [transformer](https://github.com/lpalbou/mlx-gen/blob/99fb94dd3eaa9dd1931cd3cd8eae1ae3e20f2ef3/src/mflux/models/qwen/model/qwen_transformer/qwen_transformer.py), [edit API](https://github.com/lpalbou/mlx-gen/blob/99fb94dd3eaa9dd1931cd3cd8eae1ae3e20f2ef3/src/mflux/models/qwen/variants/edit/qwen_image_edit.py), [packaging](https://github.com/lpalbou/mlx-gen/blob/99fb94dd3eaa9dd1931cd3cd8eae1ae3e20f2ef3/pyproject.toml). Current venv satisfies active version constraints but lacks `av`/`ftfy`; Diffusers is absent. Any install belongs in a new isolated environment.


The larger [mflux-community Q4](https://huggingface.co/mflux-community/qwen-image-edit-2511-mflux-q4/tree/720ad94d982b3dc22f9122ee96af31221d542d47) totals26.959GiB: its VL shards contain no quantization scales/biases and consume14.430GiB, despite the card's broad quantization claim. It is the poorer36GiB choice.

Other public4-bit2511 bases exist as BNB NF4, SDNQ and GGUF. These are different runtime formats and cannot be directly substituted into mflux. Two BNB candidates also advertise CC-BY-NC-SA4.0; the mixed MLX base above is Apache2.0 and avoids that added restriction. The Swift xocialize int4 base has custom loader/module names and is another possible route, with greater wrapper work.

Mac observed by the read-only audit: M4 Max36GiB, Metal recommended28.08GiB,147.75GiB disk available; swap was15.86/17GiB during the snapshot. Serial, staged inference and measured memory gates remain necessary; current swap is not a stable estimate of the next run.

## Proposed CPU gates before any local full-weight run

1. Pinned2511 config and mixed-quantized saved-module schema; every LoRA pair dimension and scale,100% key coverage.
2. Tiny seeded2511 Transformer/blocks against official PyTorch: target timestep t; source/mask timestep0; text modulation at t; correct image-token order/RoPE and negative-prompt masks.
3. Checkerboard source/mask latent packing/unpacking,8×VAE downsample and2×patch packing, soft-mask morphology/nearest sampling and three image grids.
4. Exact50-step sigma grid and two-branch CFG rescale; same explicit saved noise arrays across backends (same numeric seed alone does not guarantee same RNG output).
5. NumPy/MLX/PyTorch Soft-Poisson parity including image edges/corners, processed-mask all-zero/all-one cases, momentum,50 iterations, BF16 cast points, clean-to-velocity inverse and source projection after Euler.
6. Sigma0 preservation in processed-mask-zero latent entries and final pixel blend preservation in processed-mask-zero pixels. Report original-square hardpaste separately.
7. After mechanics pass, bounded staged load and modest-resolution inference with peak MLX/RSS/swap recorded; then exact512×1152 fixed-input visual comparison. Mechanical parity does not prove rail alignment.

## Safest unmodified teacher route

Use the pinned official repo in a separate Linux/CUDA environment with an80GB GPU and at least64GB host RAM. Official base weights total57,699,249,798B /53.737GiB plus latest2.454GB; disk reservation≥65GB for one base/adapter copy, more if duplicating caches. Official loader materializes one component's full CPU weights at a time; the transformer alone is40.861GB. The official entrypoint loads all components on the requested GPU, so36GiB unified memory cannot hold this BF16 recipe. A80GB-GPU inference estimate is60–70GiB including workspaces; this is not measured here. A48GB route needs explicit component offloading absent from the basic official entrypoint.

Pin base and adapter snapshots to local paths before execution; the raw repo identifiers alone track moving revisions. Then run official `inference.py` with latest,50 steps,CFG4,maskCFG1,scale1. Override spatial buckets to `preprocessing.max_resolution=589824` and `preprocessing.aspect_ratios=["4:9"]` to keep our512×1152 source/mask uncropped and unresized. Default9:16 bucketing changes the comparison geometry. This is an inference preparation recommendation; no cloud job was created.

MaskFlow code/adapters and the inspected MLX-Gen runtime: MIT. Qwen2511 and the selected mixed-MLX derivative: Apache2.0. Dataset terms remain separate if we later train/distill.

## Prepared official CUDA command (not executed)

Run from a separate checkout of the pinned official MaskFlow commit after dependencies and pinned local snapshots exist. Replace paths with that host's copies. The existing shared source canvas and edit mask must both stay512×1152.

```bash
uv run python inference.py \
  input.source=/absolute/path/to/track3/canvas.png \
  input.mask=/absolute/path/to/track3/mask.png \
  'input.prompt=Extend this album cover vertically above and below its existing central square to fill the entire portrait canvas. Seamlessly continue its existing colors, lighting, textures, scenery, and artistic style into the empty top and bottom regions. Keep the central artwork unchanged. Make one coherent expanded composition. Do not add new people, duplicated subjects, extra objects, text, letters, logos, frames, or borders. A vertical cinematic music album photograph viewed from above: one man sitting on dark asphalt beside the open door of a red sports car, with a metal roadside guardrail and water beyond it. Continue the existing road texture, water, vegetation and perspective above and below the central artwork. Maintain the muted twilight lighting and realistic photographic grain. Preserve the central scene. No additional cars, people, logos or lettering.' \
  pipeline.pretrained_model=/absolute/path/to/pinned-Qwen-Image-Edit-2511 \
  checkpoint.sft_path=/absolute/path/to/latest.safetensors \
  checkpoint.sft_weight_name=null \
  checkpoint.lora_scale=1.0 \
  runtime.device=cuda runtime.dtype=bfloat16 runtime.seed=42 \
  runtime.num_inference_steps=50 runtime.text_cfg_scale=4.0 runtime.mask_cfg_scale=1.0 \
  preprocessing.max_resolution=589824 'preprocessing.aspect_ratios=["4:9"]' \
  output.path=outputs/track3-maskflow-latest.png
```

This is the exact trained recipe with explicit shared-target preprocessing; the command includes the full frozen localized40 prompt for a matched-input comparison. Model RNG streams still differ unless the same explicit initial noise is supplied. It uses the published defaults for Soft-Poisson, noisy-source path, mask morphology and final soft pixel blend. No CUDA environment was created or used here.
