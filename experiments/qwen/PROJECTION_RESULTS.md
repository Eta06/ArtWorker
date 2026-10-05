# Frozen-VAE boundary projection diagnostic — 2026-10-05

The completed pixel-consistency arm reduces source-border reconstruction error and the lower asphalt tone stripe. It **does not meaningfully fix the guardrail's direction**. The second, derivative-guided arm was interrupted by the 600-second process bound before its final image was saved; no quality verdict is available for that arm.

This is a derived post-denoising repair of the exact terminal tensor from `boundary_runs/2026-10-05/track3/edge-context/final_latents.npz`. It is not a new sampling result, source-preservation proof from latent masking, trained model, LanPaint, or DPS. The previous outputs and original cover remain intact.

## Completed arm

The frozen reference Qwen2.1 F32 VAE was loaded alone: 1,350,961,616 parameter bytes. No DiT, text encoder, LoRA, new checkpoint, or transformer forward was used. The local differentiable tile512/overlap64 decoder matches the untouched reference decoder exactly on this actual tensor. Its baseline RGB also matches the retained boundary-suite raw PNG pixel-for-pixel (maximum error and MAE both zero).

Eight Adam updates optimize source-border RGB MSE and a small latent trust penalty. Only six latent rows on either side of each source boundary can move, including known border latents. Final source pixels remain exact through hard compositing. Off-collar latent entries are bitwise unchanged; global VAE attention can still affect distant decoded pixels.

| Diagnostic | Exact baseline | Pixel consistency, 8 iterations |
|---|---:|---:|
| Raw source top16 MAE, 0–255 | 8.717 | 4.236 |
| Raw source bottom16 MAE, 0–255 | 4.978 | 2.818 |
| Raw source top32 MAE, 0–255 | 6.789 | 3.725 |
| Raw source bottom32 MAE, 0–255 | 4.198 | 2.654 |
| Raw source interior MAE, 0–255 | 3.160 | 3.447 |
| One-pixel composite derivative top MAE | 17.283 | 8.978 |
| One-pixel composite derivative bottom MAE | 7.444 | 5.505 |
| Near rail/source-tangent offset | −3.60 px | −4.36 px |
| Far rail/source-tangent difference | 8.03° | 7.88° |
| Far rail extrapolated endpoint difference | −10.82 px | −11.13 px |
| Final resized source pixels exact | yes | yes |

Raw reconstruction and one-pixel derivative values are not geometric correctness scores. The rail diagnostic is also local and subject to source curvature, frozen search-corridor selection, posts and competing ridges; differences smaller than its sensitivity should not be ranked as a win. The independently saved annotation was opened and checked to follow the same upper metal ridge.

The full composite and actual before/after join crops were viewed. The lower dark horizontal transition is less conspicuous, but the rail still connects through an elbow to a differently directed generated segment. Position remains off by approximately four pixels and the far tangent differs by approximately eight degrees. The raw interior reconstruction gets slightly worse, although final hard compositing retains the original source exactly. Far generated-region mean RGB change is only 0.0343/255.

![Actual before/after rail and lower boundary](projection_runs/2026-10-05/track3-exact-v2/consistency-joins-preview.png)

## Runtime limitation and incomplete arm

The consistency optimizer took 296.39 seconds. Its measured MLX peak was **43,593,100,788 bytes = 40.60 GiB**, exceeding physical unified memory and the configured 18-GiB allocator target. That target did not enforce a hard activation-memory cap. The process's cumulative `ru_maxrss` does not independently represent GPU/unified tensor memory. This large backward graph is unsuitable for a phone and caused expensive memory pressure on this Mac.

The derivative arm completed six of eight updates at 548.13 seconds, then the wrapper terminated the process at 600.43 seconds, exit −15. Its incomplete metrics remain labeled running by the interrupted script; `launch.json` supplies the actual timeout state. The old script did not save an intermediate latent state, so no derivative final or recovery tensor exists. This arm must not be marked passed or assigned a visual ranking.

An initial launch also failed before weight loading because exact terminal NPZs use `source_rect_xyxy`, whereas old preview NPZs use `window_xyxy`. The parser now validates both forms and resolves the correct preceding raw-PNG path. CPU metadata tests cover both valid forms and reject wrong IDs, rectangles and windows.

The repaired runner adds optional `--checkpoint-decoder`: local `mx.checkpoint` wrappers recompute decoder mid/up blocks during backward without changing the forward equations. Tiny CPU neural-module forward and input gradients match their uncheckpointed equivalents exactly; real VAE pixel parity is asserted before optimization. **Actual memory/time reduction remains unmeasured until a new GPU trial.** Per-iteration delta and Adam state are now saved atomically so a future interrupted trial retains its latest state. Prior launch code hashes and outputs are preserved.

## Artifacts

- [Completed consistency and interrupted derivative metrics](projection_runs/2026-10-05/track3-exact-v2/projection_metrics.json).
- [Actual process timeout and old runner hash](projection_runs/2026-10-05/track3-exact-v2/launch.json).
- [Completed consistency raw image](projection_runs/2026-10-05/track3-exact-v2/consistency/raw.png).
- [Completed consistency exact-source composite](projection_runs/2026-10-05/track3-exact-v2/consistency/composite.png).
- [Independent rail diagnostic and annotation](projection_runs/2026-10-05/track3-exact-v2/consistency-eval/REPORT.md).
- [Checkpoint/metadata/gradient CPU preflight](projection_runs/2026-10-05/track3-checkpoint-preflight/preflight.json).
- [Method and primary research context](BOUNDARY_PROJECTION_PLAN.md).

These diagnostics are excluded from the standard model aggregate. No Player integration, device inference, model training, source blur, source warp, feathering, or rail painting was performed.
