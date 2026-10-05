# Guardrail/source-boundary geometry probe — 2026-10-05

Changing only the known **target** VAE encoding context reduces raw decoded source-border error and the prominent rail zigzag, while keeping the original image/reference prefix fixed. It does **not** completely solve the final guardrail join: a short elbow/slope mismatch and a water/vegetation transition remain visible after the original source is pasted back. This is a controlled partial improvement, not a seamless-geometry result.

## What changed

`run_geometry_trial.py` is a separate runner. Both arms use track3, seed42, Qwen-Image-2.1 Q4 plus Viggle v0.2.1 r256, the same square-source text/image reference prefix, absolute-position noise, final-canvas sigmas, RoPE, fresh per-arm prefix cache and six EARLY-FRONTIER steps. Active target heights are `[640,896,1152,1152,1152,1152]`; each arm forwards 12,288 target tokens in total. New regions use the same previous predicted-clean frontier initializer and existing target states survive insertion unchanged.

The only arm change is the restored known target source encoding:

- **square-known:** source target latents are the isolated 512-square source VAE encoding, exactly as the preceding EARLY-FRONTIER trial.
- **edge-context:** the same square is edge-padded vertically by 320 pixels above/below, encoded as a 512×1152 canvas, then packed latent rows `20:52` are cropped for the known target source. The reference prefix remains the original square encoding.

The installed reference VAE expects RGBA. Edge padding preserves the original RGB and its constant opaque alpha; no original pixel is changed. This padded canvas is used only to obtain target source latents, not as a new image/text reference. Source-prefix and known-target hashes are recorded separately. The variant changes the known target context and its downstream generated surroundings; it does not replace prefix conditioning or add an extra outer-band initializer.

The source is centered at `(0,320,512,832)` in a 512×1152 final canvas. Pixel preservation refers to the shared resized 512-square source. The original JPEGs and preceding outputs are retained unchanged.

## Controlled results

| Measured quantity | square-known | edge-context |
|---|---:|---:|
| Transformer calls | 6 | 6 |
| Target-token forward sum | 12,288 | 12,288 |
| Raw source top16 MAE, 0–255 | 14.334 | 8.717 |
| Raw source bottom16 MAE, 0–255 | 10.053 | 4.978 |
| Raw source interior MAE, 0–255 | 3.315 | 3.226 |
| Raw complete source MAE, 0–255 | 3.870 | 3.452 |
| Core denoising seconds | 39.72 | 33.50 |
| Final VAE decode seconds | 4.62 | 2.10 |
| Denoising MLX peak GiB | 8.450 | 8.458 |
| Final original source pixels exact | yes | yes |
| Final latents and decoded pixels finite | yes | yes |

The square-known final composite is pixel-for-pixel and SHA256-identical to the prior EARLY-FRONTIER final (`f8f108159ff8478cf49103058a72de233667ab697d93f5827e418fe5af3de0cf`). Source-prefix and absolute-noise hashes also match the preceding trial exactly. The two current arms share those hashes; only their known-target hashes differ.

Raw MAE is measured **before** final hard compositing against the original resized source. Lower MAE supports improved border reconstruction, not correct line geometry or a visually seamless output. Final source equality is achieved by pasting the original source exactly, without feathering it; no source blur, warp, rail painting or blending was applied.

## Actual image inspection

Both full final PNGs and 2×/6× raw/composite source-boundary closeups were opened directly.

- In **square-known raw**, the rail develops a conspicuous zigzag near the first known source rows. Hard compositing restores the original inner rail but leaves the generated-side mismatch at the join.
- In **edge-context raw**, that zigzag is visibly reduced and the reconstructed source-border rail is closer to the original.
- In **edge-context composite**, the join is improved but still has a short elbow/sloped transition. The water/vegetation texture and color change also remain apparent at the source boundary. The full-frame upper sky/horizon-like semantic change has not been solved by this context adjustment.
- At the lower source boundary (`y=832`), the edge-context composite has a more noticeable dark horizontal asphalt transition. Root and an independent read-only reviewer inspected both full-size final composites and confirmed this tradeoff. The variant is not uniformly better across the complete image.

This supports target-source VAE border context as a contributor to the defect. It does not prove the remaining generated-side rail continuation will follow the source's tangent perfectly. Any later boundary-collar or geometry-conditioning experiment needs a separate matched control; it was not run here.

![Two real final outputs](geometry_runs/2026-10-05/track3/final-comparison.png)

![Actual 2x raw/composite source-boundary crop](geometry_runs/2026-10-05/track3/boundary-2x.png)

![Actual 6x rail join crop](geometry_runs/2026-10-05/track3/rail-join-6x.png)

Closeup crops are nearest-neighbor enlargements of the saved pixels. Orange source-boundary markers are outside the cropped images, so they do not paint over the rail. The full comparison is a resized montage for viewing; the untouched 512×1152 raw/composite PNGs remain in each arm directory.

## Timing scope and validation

Shared model load was 15.70s, shared conditioning 2.78s; the extra edge-context VAE encoding within conditioning took 0.37s. Core time includes insertion/layout, transformer, Euler update and the per-step predicted-clean estimate. Snapshot capture, all final/preview VAE decode and model load are excluded. The full two-arm process including six preview decodes completed in 112.83s, exit 0, under a 600s process-group timeout. Load MLX peak was 24.50 GiB; conditioning peak 10.84 GiB. These are Mac measurements with sequence/warm-up/system-state caveats; no iPhone or reliable speed claim follows.

CPU preflight checks original opaque RGBA preservation, exact padding and packed center-crop rows, prefix separation, absolute RoPE/noise indexing and previous-state retention without loading weights. An independent read-only code audit found no blocking sampling bug. Runtime checks confirm unchanged reference prefix, input/cache geometry, six calls per arm, chosen known-target latent equality at sigma zero, finite arrays and exact final source pixels. The old baseline reproduced exactly.

```sh
experiments/qwen/.venv/bin/python experiments/qwen/run_geometry_trial.py --track track3 --preflight
experiments/qwen/.venv/bin/python experiments/qwen/run_geometry_trial.py --track track3 --seed 42 --quantize 4
```

The runner refuses to overwrite an existing `geometry_metrics.json`. To reproduce, use a fresh directory with `--output /absolute/path/to/new-suite`.

- [Complete suite metrics](geometry_runs/2026-10-05/track3/geometry_metrics.json).
- [CPU preflight](geometry_runs/2026-10-05/track3/preflight.json).
- [Completed hash/geometry/source artifact validation](geometry_runs/2026-10-05/track3/validation.json).
- [Command, code hash, timeout and exit status](geometry_runs/2026-10-05/track3/launch.json).
- Actual raw/composite/predicted-clean snapshot files: each arm directory below `geometry_runs/2026-10-05/track3/`.

Both arms are marked diagnostic and excluded from the standard aggregate. No new checkpoint download, training, cloud job, Player integration or third geometry GPU test was performed. All model processes from this probe have exited.
