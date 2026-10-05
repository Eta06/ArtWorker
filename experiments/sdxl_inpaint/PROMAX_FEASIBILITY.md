# Pinned ProMax outpainting feasibility

Prepared 2026-10-05. This is a metadata and source compatibility audit, not a successful inference or quality result. No full weight payloads were downloaded and no GPU work was launched by this preparation. The adjacent `promax_download_manifest.json` defines an isolated `.build/models/sdxl-promax-official/` assembly: **12 new files / 9,618,667,533 bytes**, plus **8 locally verified, byte-identical tokenizer files / 3,171,556 bytes**, for **20 files / 9,621,839,089 bytes**.

## Official recipe and exact components

The [pinned author outpainting demo](https://github.com/xinsir6/ControlNetPlus/blob/b48420576eac63c04388cb65fb74513cbd17405a/promax/controlnet_union_test_outpainting.py) combines ordinary **four-channel SDXL base**, **ProMax Union**, and the **madebyollin FP16-fix VAE**, with an **Euler ancestral scheduler**. This is a concrete trained repaint conditioning path with an author-provided outpainting example; no ranking against other models or dedicated outpainting training claim is established by this audit.

| Component | Pinned revision | Published access/license | Selected payload |
|---|---|---|---|
| [xinsir/controlnet-union-sdxl-1.0](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0/tree/801a4a3fa3d4c936f4feea95b98607bc6726f80c) | `801a4a3fa3d4c936f4feea95b98607bc6726f80c` | Public, ungated; Apache 2.0 metadata | `config_promax.json` + `diffusion_pytorch_model_promax.safetensors` |
| [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/462165984030d82259a11f4367a4eed129e94a7b) | `462165984030d82259a11f4367a4eed129e94a7b` | Public, ungated; OpenRAIL++ metadata | Ordinary four-channel FP16 U-Net, two FP16 CLIP encoders, configs and tokenizers |
| [madebyollin/sdxl-vae-fp16-fix](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/tree/207b116dae70ace3637169f1ddd2434b91b3a8cd) | `207b116dae70ace3637169f1ddd2434b91b3a8cd` | Public, ungated; MIT metadata | `config.json` + `diffusion_pytorch_model.safetensors` |

Metadata was read without authentication. This records repository access and published license identifiers, not an interpretation of deployment rights. Manifest entries pin each repo, revision, source path, destination path, size, raw SHA256 and Git blob SHA1. For LFS weights the Git blob belongs to the LFS pointer; verify actual weights with the raw SHA256.

The FP16-fix VAE checkpoint stores **F32** tensors and is **334,643,238 bytes**; the recipe loads/casts it into `torch.float16`. It has no `.fp16` variant in this pinned selection. The ordinary base VAE is omitted because the pipeline receives the explicitly constructed replacement VAE. A separate bounded HTTP Range audit confirmed all **248 tensors are F32**, with **83,653,863 elements**; the **27,778-byte JSON header** has SHA256 `7a41bd91844832b1ebd932e3f2d9293b13922f9d27b56a8ce65a379609a8ea7d`. Both requests returned HTTP 206; only the 27,786-byte prefix was read. Header plus four bytes per element equals the exact published payload size.

## Reuse evidence against the downloaded native-inpaint package

The old package is `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`, revision `115134f363124c53c7d878647567d04daf26e41e`, described by the existing 18-file `download_manifest.json`. All eight `tokenizer/` and `tokenizer_2/` files match the pinned ordinary base in **byte count and Git blob SHA1**. Their current local file contents under `.build/models/sdxl-inpaint-fp16/` were independently hashed and also match. Copy these after rechecking the source hashes.

Both CLIP weight containers and their configs have different published hashes; at metadata preparation their numeric tensor identity had not been established. Differently serialized containers alone do not prove differing neural tensor values. The inpaint VAE also differs in serialized dtype, size/hash and config from the demo FP16-fix VAE. The scheduler and pipeline index differ. Fresh exact author-recipe files remain selected for all those components; the manifest records config-level differences and prior/current hashes. The old **nine-channel inpaint U-Net** is a different architecture from the demo's ordinary **four-channel** U-Net and must not substitute for it.

**Follow-up numeric audit:** after the parent completed the fresh official download, `official_tensor_comparison.json` independently established bitwise equality for all **196 shared CLIP1 F16 tensors**, all **517 shared CLIP2 F16 tensors**, and all **248 VAE tensors after the official F32 payload was cast to F16 using Torch on CPU**. Each old CLIP container additionally stores the I64 `text_model.embeddings.position_ids` buffer. There are no common-tensor mismatches. The streaming audit took 2.157 seconds and constructed no pretrained model or GPU tensors. Its SHA256 is `3d2adc0c15e90fc513ee1fc5849e7b2fac8b93ec07ccd6dfeb542b8aaff4edbd`. The fresh exact official assembly remains selected with no component substitution; all downloader manifests remain frozen.

## Native Union compatibility: exact shapes passed

The installed runtime is `.build/dreamlite-venv/`: Torch **2.14.0**, Diffusers **0.39.0**, Transformers **4.57.3**, Hugging Face Hub **0.36.2**. The installed Union model source exactly matches [Diffusers release commit a3608b5](https://github.com/huggingface/diffusers/blob/a3608b512ed7248499a44c61d954965ed9bdae4d/src/diffusers/models/controlnets/controlnet_union.py), SHA256 `093ad3610fa3b5af98cc2104a09f1868a9fb3a15b0806d792a3c4264ff7a3d59`.

An independent child audit fetched only the eight-byte header length and then a **112,808-byte safetensors header prefix**, with HTTP 206 and valid Content-Range. It read no tensor payload. The JSON header is **112,800 bytes**, SHA256 `e184777c2a254fba920777d5e1bdbe740105301a0a5d629dfc72d12dc9ff2aa8`. CPU-only `torch.device("meta")` construction using the **complete ProMax config** produced **863 state entries** matching all **863 F16 checkpoint entries**, with **zero missing/unexpected keys and zero shape mismatches**. There are **1,256,614,800 elements**; header bytes plus two bytes per element equals the exact published checkpoint size.

Representative matches include task embedding `[8,320]`, latent input `[320,4,3,3]`, RGB control input `[16,3,3,3]`, and fused MHA projection `[960,320]`. Use `num_control_type=8` **and** `conditioning_embedding_out_channels=[16,32,96,256]`; constructor defaults differ. The source config SHA256 is `6653ad6a0ed181f0d4a7225f1b1c037405d98573fbf0862b1ca3ab6c99b52c21`.

This establishes checkpoint key/shape compatibility, not numerical forward parity or end-to-end MPS success. The original and native implementations both retain the original `nn.MultiheadAttention(batch_first=False)` convention. Do not transpose/fix those axes during adaptation. Source comparison found no structural blocker for **one repaint control at scale 1**. Unequal multi-control scaling differs across implementations, so a later combined-control experiment needs separate validation.

## Conditioning and source preservation

The [pinned ProMax config](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0/blob/801a4a3fa3d4c936f4feea95b98607bc6726f80c/config_promax.json) adds eight task slots: pose 0, depth 1, soft edges/scribble 2, canny/lineart/MLSD 3, normals 4, segmentation 5, tile 6, repaint 7. The author's legacy API provides an eight-slot image list and an eight-element one-hot task type. With the [native Union inpaint pipeline](https://github.com/huggingface/diffusers/blob/a3608b512ed7248499a44c61d954965ed9bdae4d/src/diffusers/pipelines/controlnet/pipeline_controlnet_union_inpaint_sd_xl.py), use **`control_image=[control_rgb]`, `control_mode=[7]`**. At the direct model API this becomes a condensed one-image list, `control_type_idx=[7]`, and a `[batch,8]` one-hot tensor; the legacy eight-image list is not the native model API.

Prepare a target-size RGB canvas with the cover at its intended exact rectangle. `image` is this source canvas; `mask_image` is white in unknown areas and black on the source. For `control_image`, retain known source RGB and zero unknown RGB to black; its preprocessing range is **[0,1]**, not the `-1` sentinel used by some other inpaint ControlNets.

The [author's inpaint pipeline](https://github.com/xinsir6/ControlNetPlus/blob/b48420576eac63c04388cb65fb74513cbd17405a/pipeline/pipeline_controlnet_union_inpaint_sd_xl.py) and native four-channel branch apply learned control residuals and replace known-region latents with source latents at every step, using appropriately noised source until the last clean-latent replacement. Decoding is still not an exact pixel copy. No default final RGB paste occurs unless an overlay/crop path is explicitly invoked. Save raw decoded output and a separate exact-source composite; assess the boundary on both. The installed native Union inpaint pipeline explicitly requires a four-channel U-Net and rejects other channel counts. The legacy author pipeline alone contains a nine-channel concatenation branch without the four-channel known-latent replacement; the separate standard SDXL native-inpaint pipeline also has a nine-channel path. Neither substitutes for this official ProMax baseline.

## Isolated load layout and bounded queued trial

Place ordinary base files under `base/`; rename the two ProMax source filenames into `controlnet/config.json` and `controlnet/diffusion_pytorch_model.safetensors`; place the FP16-fix pair under `vae_fix/`. Load only from these verified local folders:

```python
controlnet = ControlNetUnionModel.from_pretrained(control_dir,
    torch_dtype=torch.float16, use_safetensors=True, local_files_only=True)
vae = AutoencoderKL.from_pretrained(vae_dir,
    torch_dtype=torch.float16, use_safetensors=True, local_files_only=True)
scheduler = EulerAncestralDiscreteScheduler.from_pretrained(base_dir,
    subfolder="scheduler", local_files_only=True)
pipeline = StableDiffusionXLControlNetUnionInpaintPipeline.from_pretrained(
    base_dir, controlnet=controlnet, vae=vae, scheduler=scheduler,
    torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    local_files_only=True)
```

The parent controls payload downloads and GPU scheduling. Only after all 20 destination files pass verification, queue one bounded MPS trial with a CPU-seeded generator and no concurrent denoiser. [Official MPS guidance](https://huggingface.co/docs/diffusers/en/optimization/mps) documents `.to("mps")`; practical speed, memory and image quality remain untested for this assembly. A suggested first hard timeout is 900 seconds, with cleanup and failure artifacts preserved.

Use **30 requested steps, CFG 5, strength 0.9999, scale 1, guess mode false, control range [0,1]**, following the author recipe. The inpaint timestep floor gives **29 actual Euler ancestral iterations**; record scheduler timesteps, U-Net and ControlNet calls and CFG branch batch size instead of reporting 30 actual steps. Initial benchmark: track 3, source cover 512×512 in canvas 512×1152 at `[0,320,512,832]`, seed 42, and the same prompt as the competing trials. This benchmark canvas differs from the author's approximate 1024²-area preprocessing, so it is a controlled resolution choice, not a pixel-exact demo reproduction.

Inspect raw and composite outputs at full size: rail/edge continuation through both boundaries, rail endpoint displacement, duplicate structures, upper/lower scene geometry, noise/detail collapse and visible seams. Save seam closeups, finite-output status, source-region MAE, runtime and peak memory. Exact pixel compositing and finite tensors alone do not establish visual acceptance. A later canny+repaint attempt (`control_mode=[3,7]`, equal scales 1) is a separately declared experiment, not part of the initial trial. No phone runtime, spatial-growth compute saving or quality superiority follows from this feasibility audit.
