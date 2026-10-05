# ProMax native repaint teacher: observed failure

Recorded 2026-10-05. The parent launched the single MPS inference; preparation and independent validation used CPU only. The completed run is `runs/2026-10-05/track3-30-repaint7-seed42/`. **The image fails visual acceptance.** Successful loading, finite computation, source anchoring and exact final compositing do not qualify this result as successful outpainting.

## Actual images

Both the original decoded `raw.png` and exact-source `composite.png` were directly inspected. Almost the entire upper extension is a flat dark panel. The lower extension contains a narrow horizontal extrusion below the source and then another nearly flat dark panel. It does not continue the lake, guardrail or asphalt into a coherent complete scene. The central source remains recognizable in the raw decoding; the separate composite restores its original pixels exactly. Both source boundaries visibly terminate the original scene.

The raw and composite upper rail joins were additionally viewed at nearest-neighbor 6× magnification, and both lower joins at native size. The rail terminates abruptly at the source boundary; the lower scene becomes a shallow flat slab. The dark exterior is present before source pasting. Independent pixel statistics show **99.7687% of upper exterior pixels** and **87.7264% of lower exterior pixels** have all three RGB channels below 32; upper mean RGB is 18.9611 with standard deviation 3.0096. This is nearly flat dark generation, not a decode image containing missing or non-finite pixels.

| Artifact | SHA256 |
|---|---|
| Raw image | `c016ace72a9b91703e8fe06ee3f18e206f09e4615b6149ed25628d3423c57839` |
| Exact-source composite | `246e38b076a8a387423fcaa946a285d2f846e76de724e84b6f5ca9877cc1819a` |
| Source PNG | `ddc835523a27f455b999673ee2460b31a1b6b93b86ef0420e6f37ff8ea0f0157` |
| Initial/control canvas PNG | `448c04c5be9ed18667008c671d302a09a137265e58003644e0516444301b3bd7` |
| Regeneration mask PNG | `c4386579faf335ed709b14b461c6a843d0e58956e25ddfbe8d88e3a317f6dc70` |

The target is 512 × 1152; the original 512-square source occupies `[0,320,512,832]`. White mask pixels regenerate and black mask pixels preserve. Exterior RGB in both the initialization canvas and repaint control is `[0,0,0]`. Control preprocessing is `[0,1]`; masked-source VAE preprocessing independently uses normalized RGB zero outside the known region. These two representations must not be conflated.

## Observed execution and cost

The frozen runner SHA256 is `ecdd952376fbf04569f8ebc37c61e1cde00a367c7de15a93e2c00d753623bd20`. Exact verified local components are ordinary four-channel SDXL base, eight-task ProMax Union and the FP16-fix VAE, all loaded into FP16. ControlNet, UNet and VAE loading reports have no missing, unexpected or mismatched keys. Pinned repository, payload and runtime provenance is retained in `metrics.json` and the verified assembly manifest.

Requested parameters were 30 steps, CFG 5, strength 0.9999, seed 42, single repaint mode 7, conditioning scale 1, guess mode false and control throughout all timesteps. Actual native strength trimming retained **29 iterations**, from timestep 925 through 1. The scheduler's full list starts at 958 and its begin index is 1. There were **29 UNet calls, 29 ControlNet calls and 29 Euler ancestral updates**. Both neural branches used CFG batch 2, yielding **58 branch samples per model**. Real observed residuals and prediction tensors were finite.

| Instrumented phase | Seconds |
|---|---:|
| Total runner elapsed | 45.4385 |
| Verified checkpoint rehash | 3.8354 |
| Model load | 3.2954 |
| Pipeline total | 36.2115 |
| Denoising including observation | 33.5924 |
| UNet calls, summed | 21.9066 |
| ControlNet calls, summed | 11.5050 |
| Raw VAE decode | 0.8421 |

The supervisor's 45.993-second wrapper time additionally includes process overhead. Sampled MPS memory reached **9,548,185,600 allocated bytes** and **14,311,882,752 driver bytes**; these are sampled maxima, not a continuous memory trace. The process RSS field is not a substitute for GPU driver allocation. Timings include synchronization and tensor observation and do not establish iPhone performance.

The native API actually received a one-element RGB condition list and `control_type_idx=[7]`, with a `[2,8]` one-hot repaint tensor. Initial and first model inputs were `[1,4,144,64]` and `[2,4,144,64]`. The mask was `[2,1,144,64]`, with known source rows 40 through 103. Actual prepared controls, mask, posterior source latents, initialization noise and final/raw tensors were saved for independent checking.

The native per-step source replacement observer passed all 29 actual steps: it compares the real solver output with the real source-noise result and the actual callback tensor, without adding an extra correction or RNG draw. The first 28 updates use original source latents noised at the next timestep; the final update uses clean original source latents. The raw source reconstruction MAE is **4.9487/255** over the source, **6.2019** at the top 16 rows, **4.7195** at the bottom 16 rows and **4.9146** in the interior. Exact-source compositing and generated-exterior equality to the raw output both passed.

## Attribution still being audited

This failure is retained; it is not yet attributed solely to the checkpoint. The independent actual-array audit is saved in `runs/2026-10-05/track3-30-repaint7-seed42/independent_validation.json`: all 14 retained arrays are finite and match their recorded hashes. Source/control pixels, mask nearest-neighbor packing, repaint one-hot, raw half-precision decode postprocessing, final known latent equality and source-only compositing all passed.

The actual initialization is exactly FP16 `image_latents + sigma[1] * actual_noise`, using active sigma **9.5435857773** at timestep 925, not full-list sigma 11.47685051. The initial exterior noise standard deviation is **1.0093**; both first neural inputs exactly match division by `sqrt(sigma[1]²+1)` (**9.5958337784**). Unknown mask sites are all 1 and known sites all 0. Final exterior latent values have **0% equality** to original source latents, excluding an accidental exterior copy/clamp.

Independent CPU RNG replay matches all **32 FP16 draws and their generator states**: full source VAE posterior, initial noise, masked-source VAE posterior and 29 ancestral draws. The initial noise SHA256 is `84d0897db02d9376ea1f5ca23d4512a05b12df44de265d16030420ad9041579b`. The last ancestral call still consumes a draw even with zero added-noise coefficient, as implemented by the native scheduler. Full intermediate solver/source tensors were not retained: all 29 replacement equations were checked by the runner against its actual observations, while only the final saved known latent equality can be independently recomputed afterward.

Numerical forward equivalence between the pinned author's Union and the native Union implementation, and exact author preprocessing differences, are still being audited. Exact key/shape loading alone is not numerical equivalence.

The trial deliberately uses the fixed benchmark's source, positive exterior prompt and 512 × 1152 resolution. It is therefore an official-component/parameter trial, **not** a pixel-exact reproduction of the author's example prompt or approximate 1024²-area preprocessing. More materially, the author's [pinned outpainting demo](https://github.com/xinsir6/ControlNetPlus/blob/b48420576eac63c04388cb65fb74513cbd17405a/promax/controlnet_union_test_outpainting.py) passes the complete resized photograph as `image`, including real pixels under its outer regeneration mask; only its repaint control is zeroed there. Our expanded initialization image contains RGB black in unknown regions. With strength below 1, VAE encoding and noising of the complete image can therefore retain real target-area image information in the author's example. That example is not evidence of equally conditioned blank-expanded-canvas performance. Exact differences are being documented in the separate author semantics audit.

A possible pure-noise strength-1 probe changes initialization and restores the first scheduler timestep, giving 30 actual updates. Unlike the nine-channel native SDXL experiment, this four-channel pipeline still encodes source image latents at strength 1 because they are required for per-step source anchoring. Its actual RNG and latent path must be observed; a shared seed alone cannot establish matched noise or isolate black-canvas bias. No such probe has been launched by this report's author.

## Compute scope

This is one **full-canvas** teacher call: all target latent sites are processed on every step. There is no spatial growth, omitted future-token work, streamed expanding generation, model distillation or mobile deployment in this trial. Its dark exterior is a quality failure independent of that compute distinction. Older Qwen, Fill and native SDXL trials remain preserved and are separately reported in their own result files.
