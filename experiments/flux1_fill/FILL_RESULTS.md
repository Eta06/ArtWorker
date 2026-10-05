# Native FLUX.1 Fill: two valid trials, complete-image quality still failed

Both native Fill 50-step trials **fail visual acceptance**. The first invents lettering and a second car/person scene. The matched exterior-only prompt removes those obvious repetitions, but curves the upper guardrail into a bridge-like structure across the water and switches the lower asphalt to a brown, coarse texture at a hard horizontal boundary. Both raw decodes and exact-source composites were inspected directly, including source joins. A locally connected rail is insufficient to accept the complete result.

## Actual trial

Run: `runs/2026-10-05/track3-50-seed42/`. Native installed MFLUX Fill, public maintainerQ4 checkpoint pinned to`eebfbaa12c95107169452c7d22622e04771192a3`, seed42,50steps,guidance30. Source512-square is centered at`[0,320,512,832]` in a512×1152 target. Same shared source/canvas geometry as prior experiments. This is a **full-canvas teacher**, not center-out compute or an iPhone benchmark.

Actual conditioning is masked-source64+losslessly packed pixel-mask256=`[1,2304,320]`, concatenated to target noise64 for384-channel input every step. T5 contributes512 text tokens; all2304 target tokens remain present from step1. No extra negative branch, adapter, external Canny or known-latent bridge.50 actual transformer calls,115,200 target-token forwards,950 joint-block and1,900 single-block calls. Every logged prediction and latent state is finite.

| Phase | Seconds | Observed MLX peak bytes |
|---|---:|---:|
| Strict checkpoint hash verification |3.336|CPU file hashing|
| Model load |1.268|9,616,159,422|
| Prompt encoding |1.026|11,092,178,102|
| Masked-source conditioning |0.441|10,085,674,952|
| Denoising |339.515|8,352,201,230|
| Raw decode |0.701|7,115,017,648|

Total suite347.407s. Highest recorded phase peak11,092,178,102bytes (~10.33GiB); denoising peak~7.78GiB. Encoders are deleted after encoding; transformer is deleted before decode. Parameter bytes: transformer6,699,518,080, VAE164,624,710, T52,679,114,752, CLIP69,395,952. These are measured local Mac values. They do not establish mobile feasibility.

## Preservation and provenance

The composite pastes only the original resized512-square source. Source pixels are exact and every generated exterior pixel remains identical to raw decode. Raw source reconstruction MAE(all/top16/bottom16/inner) is6.791/7.034/6.559/6.791 on0–255; a small MAE is not a test of the exterior scene or typography. The raw output already contains all the invented content, so it is not introduced by compositing.

- Source PNG SHA256:`ddc835523a27f455b999673ee2460b31a1b6b93b86ef0420e6f37ff8ea0f0157`.
- Initial float32 noise SHA256:`9b0f130d9dee403294a1429d7552add46da89cbb424e620aa3d8cbd80d549a1c`.
- Final float32 latents SHA256:`90278c1ecd6049202d7d5a0ec80dd4149d389f366fcea4f52a1363ab188f12fd`.
- Raw PNG SHA256:`dc83ca70b69957f59e2a020d6fb6a33aaf0a23e0b24e64ad9c6e4f51471e7e15`.
- Composite PNG SHA256:`c1ff9ddbff04f16c1878b1622e28ecc381bd8a413658617b7cb71fc1413b59d7`.
- Frozen runner SHA256:`e63fac021c3e3849baeac7aeb5f41ae44c27a3cbda38a12520963c8955d8b6b3`.

All19 repository files(9,619,362,389bytes) were verified against exact sizes, eight LFS SHA256 and eleven ordinary Git object IDs before launch. Offline header/tokenizer and six meaningful CPU conditioning groups passed without model loading. The real run then verified the same payloads again and loadedQ4 successfully. Runtime/native conditioning source hashes are recorded in its metrics. Numerical/integrity acceptance and execution success do **not** promote the visual result.

The12B model is genuinely trained for inpaint/outpaint; its384-channel native conditioning is a better task match than general editing with an auxiliary partial structural map. That fact did not guarantee a successful local completion. [BFL model card](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev), [official mask/source conditioning](https://github.com/black-forest-labs/flux/blob/802fb4713906133fcbd0d8dc5351620ca4773036/src/flux/sampling.py#L98-L145)

The downloaded checkpoint is a [public MFLUX maintainer conversion](https://huggingface.co/mflux-community/flux-1-dev-fill-mflux-q4/tree/eebfbaa12c95107169452c7d22622e04771192a3), with inherited FLUX.1 dev Non-Commercial terms. This is a research prototype; the download does not establish production licensing. Installed runtime and earlier model/Player files were not changed.

## Executed exterior-only prompt arm

Run: `runs/2026-10-05/track3-50-exterior-only-seed42/`. Parent scheduled this run; this report's author did not launch GPU inference. The frozen runner, actual source/canvas/mask, actual saved Metal noise and all 51 sigmas are exact matches to the first trial. The static-conditioning recorded hash also matches; the static tensor itself was not retained, so that particular comparison is over the recorded hash. Only the text prompt changes:

> Continuous muted lake water, low green roadside shrubs and a thin metal guardrail extending through the upper area. Uninterrupted dark coarse asphalt continuing through the lower area. Matching twilight illumination, photographic grain and high-angle perspective across the whole scene.

Prompt SHA256: `6acce49026289b9a632093b68242b0d33fbe8bb136fa19475553b3dbe8c60cee`. Independent matching audit: `prompt_arms/2026-10-05/exterior-only/prompt_match_independent.json`, SHA256 `41d4bd76869bebbc2c124602e0413fec08695f06914a5bd6061d7f2f86103e50`. CPU random-normal recreation differs slightly from Metal; the equality claim comes from the two actual saved Metal noise arrays, not from cross-backend assumptions.

Actual 50 finite steps, 50 transformer calls, 115,200 target-token forwards, 950 joint-block and 1,900 single-block calls. This remains native full-canvas generation, without growth, a source latent bridge or negative branch. Source pixels are exact after paste; the exterior remains pixel-identical to raw decode.

| Phase | Seconds | Observed MLX peak bytes |
|---|---:|---:|
| Checkpoint hash verification | 3.610 | CPU file hashing |
| Model load | 1.006 | 9,616,159,422 |
| Prompt encoding | 0.657 | 11,092,551,903 |
| Masked-source conditioning | 0.367 | 10,128,112,048 |
| Denoising | 307.361 | 8,352,184,846 |
| Raw decode | 0.699 | 7,115,017,648 |

Total suite 314.918 s; cumulative process RSS peak 10,049,781,760 bytes. The elapsed difference from the first trial does not establish a prompt-caused performance improvement. Raw source MAE all/top16/bottom16/inner is 7.266/7.312/6.903/7.277 on 0–255; none of these values validates exterior continuity.

- Initial noise SHA256: `9b0f130d9dee403294a1429d7552add46da89cbb424e620aa3d8cbd80d549a1c`, exact first-trial match.
- Static conditioning SHA256: `c718c85772b48fc10d768d82064ba998ad40d51fd0de1dd7263702b1736c2edf`, exact first-trial match.
- Final float32 latents SHA256: `0f4ebc923f15b0cb58daa2a7170a9fb1c8d7de0b010d74be1cd576bf38b80347`.
- Raw PNG SHA256: `9960a1ac5cea254a634c4219f7a3561897810185cb33d78cb33bd850b865ee62`.
- Composite PNG SHA256: `67f8432ab359e600174fd1609770d0c4d477ab376f7f8624f413dd653833a83c`.

The absence of new lettering/car/person in this matched output supports trying exterior-focused prompts; it does not establish the cause or solve the geometry/texture failures. A different guidance value, local-context generation or known collar would be a separate experiment.

## Actual source context and next fallback

The original track3 JPEG is 1280×720, but direct inspection shows flat ochre side letterboxing. The central crop `[280,0,1000,720]` is the photographic square; the width does not provide usable extra photographic context. The crop uses all original rows, so there are also no real top/bottom rows to recover. `source_context_audit.json` records the original SHA256 and this correction to an earlier width-only inference.

The official public SDXL inpaint 0.1 model is prepared as a separate native masked-generation baseline: learned 9-channel U-Net input (4 noisy latent + 1 mask + 4 masked-source latent), public FP16 pipeline pinned to `115134f363124c53c7d878647567d04daf26e41e`. All 18 files / 6,941,218,469 bytes passed exact checksum verification and the CPU gate passed. Preparation/download does not establish output quality, and no SDXL GPU run has been launched by this author. [SDXL readiness and comparison limits](../sdxl_inpaint/SDXL_RESULTS.md), [official SDXL inpainting model card](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1).
