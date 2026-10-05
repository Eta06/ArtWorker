# Boundary sampler controls — 2026-10-05

The soft-boundary and dynamic-context controls do **not** fix the guardrail's direction and join. Dynamic known-context re-encoding improves decoded source-border reconstruction and reduces the final top/bottom luminance seam diagnostic, but its rail still bends into an incompatible far tangent. This is a partial texture/seam improvement on one image and seed, not a seamless outpainting result.

## Matched sampling and arm changes

All four outputs use track3, seed42, Qwen-Image-2.1 Q4 with Viggle v0.2.1 6-step r256, the same prompt, original 512-square reference prefix, absolute noise, absolute target RoPE and six final-canvas sigmas. Active heights are `[640,896,1152,1152,1152,1152]`: 1,280/1,792/2,304/2,304/2,304/2,304 target tokens, 12,288 forwarded in total per arm. Future target tokens are actually absent from transformer input until activation. Frontier insertion keeps prior target state unchanged; inner regions continue refining until the last step.

The final canvas is 512×1152; the original resized source occupies `(0,320,512,832)`. All final composites paste that 512-square source exactly. “Source exact” does not mean the raw VAE reconstruction or the original pre-resize JPEG is pixel-identical. No source feathering, rail warping/painting, optical-flow correction or color correction is used by these sampler arms.

| Arm | Known target source encoding | Source enforcement |
|---|---|---|
| edge-context | Center crop from an opaque RGBA source canvas padded by repeating top/bottom edge rows | Hard, packed rows `20:52` |
| square-soft | Isolated original square encoding | `0.25,0.75,1…1,0.75,0.25` on source rows |
| edge-soft | Same initial target encoding as edge-context | Same soft source-row mask |
| edge-dynamic | Starts as edge-context; generated full-canvas context re-encoded before steps5/6 | Hard, against the currently chosen target source encoding |

The source reference prefix remains the original square tensor in every arm, separate from the known target tensor. It contains 1,024 source-image latent tokens; the joint text/image prefix is 1,197 tokens.

For soft arms, one packed row corresponds to 16 source pixels. The first/last two source rows, 32 pixels per edge, use weights `0.25` then `0.75`; the 448-pixel interior (rows `22:50`) remains hard. Outside source rows the weight is zero. Before/after each Euler step the source bridge is blended as `w*known_sigma + (1-w)*state`, with the same absolute known noise and sigma. Only the hard interior is required to equal its target source tensor at sigma zero. Actual saved arrays confirm the soft boundary differs from that tensor. A matching weighted clean-source enforcement is used for predicted-clean previews.

The dynamic arm performs exactly two VAE decode/encode pairs, before steps5 and6, with **no extra denoiser calls**. The previous full-canvas predicted-clean estimate supplies exterior RGB in normalized `[-1,1]`; the source region is overwritten with the exact normalized original RGBA and whole-canvas alpha is opaque. The full canvas is encoded and packed rows `20:52` become the new known target. No uint8 quantization is applied before encoding. Original prefix/cache/noise and existing target state remain unchanged by the VAE-only context update; the subsequent known-region clamp uses the new source tensor at the current sigma. The final hard-source assertion compares against the last dynamic target encoding, not the isolated reference-prefix encoding.

## Actual final-image findings

The real full composites, raw/composite rail crops and raw/composite lower-boundary crops were opened directly. The unchanged original-source region is visible in the composite rows.

- **square-soft:** its raw rail is smoother through the former source border, but the reconstructed source-side rail drifts. Pasting the original source back exposes a larger positional break. The smoother raw image is not evidence that original-preserving geometry is fixed.
- **edge-soft:** raw top/bottom source reconstruction improves, while the final rail elbow remains. Its final top seam ratio is worse than the hard edge-context control despite lower raw source-border MAE.
- **edge-dynamic:** source-border reconstruction and the asphalt transition improve. The final rail still bends near the source boundary and its generated far segment retains the wrong direction. Vegetation/water texture transitions and the changed upper composition remain visible. The lower tone transition is reduced rather than eliminated.

![Real final composites, one uniform viewing scale](boundary_runs/2026-10-05/comparison/full-comparison.png)

![Real raw/composite guardrail crops, 3x nearest neighbor](boundary_runs/2026-10-05/comparison/rail-raw-composite-3x.png)

![Real raw/composite lower boundary, native crop pixels](boundary_runs/2026-10-05/comparison/bottom-raw-composite-native.png)

Captions are outside the cropped images. Crop montages contain unchanged saved pixels; rail enlargement is nearest neighbor. Separate untouched native-size crops are retained under the comparison directory. Original 512×1152 raw/composite PNGs remain in each arm directory.

## Reconstruction and rail diagnostics

Raw source MAE is measured before hard compositing, in 0–255 units. It diagnoses reconstruction; it is not a geometric or overall quality score.

| Quantity | edge-context | square-soft | edge-soft | edge-dynamic |
|---|---:|---:|---:|---:|
| Raw source top16 MAE | 8.717 | 9.258 | 7.566 | 4.666 |
| Raw source bottom16 MAE | 4.978 | 4.741 | 4.068 | 3.378 |
| Raw source inner MAE | 3.226 | 3.308 | 3.209 | 3.114 |
| All known source latents exact at final sigma | yes | no | no | yes, last dynamic encoding |
| Hard interior latents exact | yes | yes | yes | yes |
| Final pasted source pixels exact | yes | yes | yes | yes |

The separate [soft-suite evaluator](boundary_eval/2026-10-05/soft-suite/REPORT.md) and [dynamic-suite evaluator](boundary_eval/2026-10-05/dynamic-suite/REPORT.md) use a frozen rail-edge corridor and source fit. Selected feature overlays were inspected. Their local diagnostics are:

| Composite diagnostic | edge-context | square-soft | edge-soft | edge-dynamic |
|---|---:|---:|---:|---:|
| Generated/source far-tangent angle difference, degrees | 8.03 | 7.78 | 7.86 | 7.75 |
| Far tangent endpoint signed x error at `y320`, pixels | -10.82 | -11.45 | -11.02 | -11.17 |
| Near join median signed x offset, pixels | -3.60 | -15.68 | -4.35 | -3.64 |
| Top luminance join/nearby-row-change ratio | 2.57 | 2.14 | 2.76 | 1.98 |
| Bottom luminance join/nearby-row-change ratio | 1.73 | 1.65 | 1.61 | 1.40 |

Far fit uses generated rows270–310 and source rows320–370. Near offset uses generated rows312–319 against the source tangent. Positive angle difference means the generated rail is less steep; negative x error means it lies left of the fitted original continuation. Source curvature, competing rail ridges/posts and corridor selection create uncertainty; window sensitivity and ambiguous feature rows are retained in evaluator JSON. The approximately 8-degree far-angle mismatch persists across all arms. A small near offset can be produced by an elbow connecting incompatible tangents, so it does not establish correct continuation. Seam ratios measure luminance changes and do not replace full-image/geometry inspection.

## Timing and memory

These are single-seed Mac M4 Max measurements, not iPhone evidence or a reliable speed ranking. Dynamic ran in a separate process; execution order, compilation/warm-up, memory pressure and decode warm-up differ.

| Measured phase | edge-context | square-soft | edge-soft | edge-dynamic |
|---|---:|---:|---:|---:|
| DiT/core sampling seconds | 27.31 | 29.13 | 30.28 | 27.53 |
| Additional dynamic context wall seconds | 0 | 0 | 0 | 7.31 |
| Of which VAE decode+encode seconds | 0 | 0 | 0 | 6.66 |
| Final decode/artifact seconds | 6.81 | 1.99 | 2.27 | 1.95 |
| Sampling-phase MLX peak GiB | 8.450 | 8.439 | 8.458 | 11.414 |

Core sampling includes layout/insertion, transformer, Euler and clean prediction; it excludes preview capture and all VAE decoding. Dynamic cost is reported separately and included in its `generation_seconds_excluding_load_previews` (39.42s with conditioning and final decode). The dynamic sampling memory peak includes its in-loop VAE operations and should not be represented as pure DiT memory. Final decode time also includes output/NPZ writes; the control's first final decode pays process-order/warm-up costs. Process RSS is a cumulative macOS process maximum, not per-arm memory.

The three-arm process completed in 131.33s; dynamic completed in 59.90s, both exit0. Shared load/conditioning were 13.09/2.67s for the three-arm suite and 14.10/2.63s for dynamic. Launch wrappers had 900s and600s timeouts respectively. Source-context initial encoding and dynamic update timing are retained in metrics; shared phases must not be summed repeatedly as separate per-arm work.

## Reproduction and validation

The hard edge-context control reproduces both the previous [geometry edge-context raw/composite](geometry_runs/2026-10-05/track3/edge-context/metrics.json) PNG SHA256 values exactly. Its composite SHA256 is `03ad495177a35f9ca8e405fc7b5d5a111c4389ee93230ee30381c9b38dcbe2e8`.

[Artifact validation](boundary_runs/2026-10-05/comparison/validation.json) checks each actual saved mask, hard-interior latent equality, all finite latent/decoded arrays, exact raw/composite PNG-to-NPZ agreement, final source equality, shared prefix/noise/sigmas and actual future-token omission. Dynamic encoding inputs have matching hashes, exact normalized original source placement, opaque alpha and correct encoder range; each dynamic known tensor matches its recorded hash. Final assertions use the chosen final known tensor. Soft arms correctly report that the complete source latent is not exact.

The completed three-arm launch used immutable [run_boundary_suite_v1.py](run_boundary_suite_v1.py), SHA256 `08c6f5f06134359b2424c40f2ef9971c0f1f3fdfab43f1ff351df396cbc5e9f1`. Dynamic used [run_boundary_suite.py](run_boundary_suite.py), SHA256 `402b73572e73517176f43e035394d525ee5017f18f68e434cf78fab66d502669`. Both CPU preflights passed before GPU execution.

```sh
experiments/qwen/.venv/bin/python experiments/qwen/run_boundary_suite_v1.py --track track3 --seed 42 --quantize 4 --arms edge-context square-soft edge-soft --output /absolute/path/to/fresh-three-arm-suite
experiments/qwen/.venv/bin/python experiments/qwen/run_boundary_suite.py --track track3 --seed 42 --quantize 4 --arms edge-dynamic --output /absolute/path/to/fresh-dynamic-suite
```

The runners refuse to overwrite an existing `boundary_metrics.json`. [Three-arm metrics](boundary_runs/2026-10-05/track3/boundary_metrics.json), [dynamic metrics](boundary_runs/2026-10-05/track3-dynamic/boundary_metrics.json), [three-arm launch](boundary_runs/2026-10-05/track3/launch.json), [dynamic launch](boundary_runs/2026-10-05/track3-dynamic/launch.json), [three-arm CPU preflight](boundary_runs/2026-10-05/track3/preflight.json) and [dynamic CPU preflight](boundary_runs/2026-10-05/track3-dynamic/preflight.json) retain runtime evidence. Each arm saves final latent/source tensors and float/uint8 raw/composite arrays; dynamic also saves its two normalized context encoding inputs.

All four are diagnostic controls outside the standard aggregate. These runs do not provide a trained boundary model, physical-device benchmark, Player integration, or live UI streaming implementation. Final source preservation and lower reconstruction/seam diagnostics do not make the remaining rail failure acceptable.
