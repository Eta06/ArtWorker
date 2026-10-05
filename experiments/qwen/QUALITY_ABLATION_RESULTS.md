# Spatial-growth quality ablation — 2026-10-05

The first controlled correction materially improves final outer-band texture without changing the model weights or training. Opening the entire target by step 3 removes the old run's strong outer grid and bottom pseudo-symbols on track3. It does **not** establish a production-quality solution: the frontier variant invents sky/horizon-like content above the water; the source-edge variant changes the coastline and includes questionable upper metal/vegetation forms. A matched second-cover control is complete. It removes duplicated upper title text but both methods still cut off the lower bodies into a blue panel. No broad quality claim follows from these two covers at one seed.

## Matched protocol

`run_spatial_ablation.py` is a separate runner; the original `run_spatial_trial.py`, installed runtime, checkpoints, old outputs and Player source are preserved. Qwen-Image-2.1 Q4 plus Viggle v0.2.1 r256 uses the same track3 source, actual square reference prefix, prompt, seed42 absolute-position noise and six final-canvas sigmas in all four arms. Model load and reference encoding execute once. Each arm is an independent controlled generation; each growth arm is one continuous six-step sampling loop. It does not finish one outpaint then launch another.

Future target tokens are actually absent from forwards until activation. Existing latent states remain unchanged when tokens are inserted; they continue refining in later model steps. Absolute final-canvas rotary coordinates and the fixed source/text prefix cache remain stable. The full source is 512×512 at `(0,320,512,832)` on a 512×1152 target. Pixel equality refers to this resized common source, not the original 720-pixel crop or JPEG bytes.

| Arm | Target height at steps 1–6 | Newly outermost band's model steps | New-token initializer |
|---|---|---:|---|
| FULL | 1152,1152,1152,1152,1152,1152 | 6 | Shared noise at sigma=1 |
| OLD-GROW | 640,640,896,896,1152,1152 | 2 | Original source latent edge rows + shared noise |
| EARLY-SOURCE | 640,896,1152,1152,1152,1152 | 4 | Same original source latent edge rows + shared noise |
| EARLY-FRONTIER | 640,896,1152,1152,1152,1152 | 4 | Previous active predicted-clean boundary rows + shared noise |

Both growth initializers use `(1-sigma)*clean_guess + sigma*same_absolute_noise`. FRONTIER repeats the preceding clean estimate's top/bottom packed-latent rows with each absolute column retained. It is an untrained extrapolation heuristic, not a semantic continuation predictor or a new denoiser call. At the first step sigma=1, all arms initialize their active tokens from the same noise table.

## Actual final-image assessment

All four final composites were opened and inspected, including the two early variants at their full 512×1152 size.

- **FULL:** coherent water, vegetation, road and one existing guardrail; this remains the cleaner geometric reference for this cover.
- **OLD-GROW:** reproduces the original failed final exactly, including the outer repeating grid, invented upper guardrail and lower pseudo-symbols.
- **EARLY-SOURCE:** the gross outer grid and lower pseudo-symbols disappear. Road texture is much more natural; upper vegetation extends into an invented curved coastline with questionable additional forms.
- **EARLY-FRONTIER:** gross grid and lower pseudo-symbols disappear; road/vegetation texture is substantially cleaner. The upper water develops sky/horizon-like content, altering the original photograph's viewing geometry. Texture improvement is not equivalent to semantic fidelity.

FULL and OLD-GROW's new composites are pixel-for-pixel and SHA256-identical to their prior recorded finals. Thus those controls reproduced, and the improvement comes from the tested sampler changes rather than an accidental checkpoint/runtime drift. The EARLY-SOURCE versus OLD-GROW comparison isolates activation timing while retaining the original initializer. The two EARLY arms isolate the tested initializer at the same activation schedule.

This supports **late activation / insufficient correction opportunity as a material contributor** to the old final defect. It does not isolate the number of corrections from activation sigma: opening earlier changes both. The current evidence does not justify saying spatial training is already necessary, nor that the model alone is too weak; it also does not prove an untrained sampler can solve robust high-quality streaming for all images.

![Four real final outputs](quality_runs/2026-10-05/track3/final-comparison.png)

## Second-cover control: track2

FULL and EARLY-FRONTIER were run in a second matched one-load pair with track2's own shared prompt/source/noise and the same six-step sigma schedule. Both final images were opened directly.

- FULL duplicates the original large title above the preserved album square.
- EARLY-FRONTIER leaves the upper extension as clean cyan wall texture and avoids that duplicate title in this seed.
- **Both still fail natural lower-body continuation:** clothing extends a short way, then terminates at a sharp horizontal cyan panel. Therefore the sampler improvement is useful but it is not a robust album-outpaint solution. This second case also demonstrates a limitation of the full-target baseline, beyond late token insertion alone.

![Second-cover final comparison](quality_runs/2026-10-05/track2/final-comparison.png)

| Track2 measured value | FULL | EARLY-FRONTIER |
|---|---:|---:|
| Transformer calls | 6 | 6 |
| Target-token forward sum | 13,824 | 12,288 |
| Core loop seconds | 33.79 | 27.53 |
| Final VAE decode seconds | 3.70 | 1.62 |
| Denoise MLX peak GiB | 9.00 | 8.46 |

Load 15.66s and conditioning 2.45s were shared once; total pair including six preview decodes was 96.25s, exit 0. Timing remains a one-sequence observation with order/warm-up/system-state caveats; no production speed promise follows. Both outputs have finite final latents/pixels and exact resized central source pixels.

Reproduction uses a fresh output directory if retaining these artifacts:

```sh
experiments/qwen/.venv/bin/python experiments/qwen/run_spatial_ablation.py --track track2 --arms full early-frontier --preflight
experiments/qwen/.venv/bin/python experiments/qwen/run_spatial_ablation.py --track track2 --seed 42 --quantize 4 --arms full early-frontier
```

[Track2 suite](quality_runs/2026-10-05/track2/ablation_metrics.json) · [command/exit/code hash](quality_runs/2026-10-05/track2/launch.json).

## Real intermediate predictions

![Source and actual EARLY-FRONTIER previews](quality_runs/2026-10-05/track3/growth-stages-early-frontier.png)

The saved step2/4/6 images are genuine pre-step `z_sigma - sigma*predicted_velocity` clean estimates, with the known source latent restored. Step2 is an early prediction, not a completed smaller image; the center and earlier bands continue refining until the last step. Clean arrays were captured during sampling but VAE-decoded only after every arm's core loop finished. Live UI streaming and its overhead remain unimplemented/unmeasured.

## Compute and timing

| Track3 measured value | FULL | OLD-GROW | EARLY-SOURCE | EARLY-FRONTIER |
|---|---:|---:|---:|---:|
| Transformer calls | 6 | 6 | 6 | 6 |
| Target-token forward sum | 13,824 | 10,752 | 12,288 | 12,288 |
| Target-token reduction versus FULL | — | 22.2% | 11.1% | 11.1% |
| Core loop seconds | 55.44 | 54.83 | 55.05 | 46.22 |
| Final VAE decode seconds | 5.45 | 2.25 | 2.36 | 2.26 |
| Denoise MLX peak GiB | 9.00 | 8.46 | 8.46 | 8.46 |

Core time includes active-layout construction, insertion, transformer, Euler update and clean-estimate calculation on all six steps. Preview capture and every VAE decode are excluded. Shared load took 21.62s and shared conditioning 4.92s. The whole four-arm process including 12 preview decodes took 281.14s and exited 0 within a 600s process-group timeout.

The Mac had substantial concurrent application memory/swap pressure. Execution order was FULL, OLD-GROW, EARLY-SOURCE, EARLY-FRONTIER. Times are a single sequence and are confounded by system load, warm-up and thermal state. **No reliable speed improvement is established by this run**, and its absolute time is not directly compared to the earlier 29.38/24.33s pair. Token counts are exact structural counts, not proof of wall-clock savings. They exclude the shared prefix and do not summarize total attention cost. These are Mac measurements, not iPhone measurements.

## Validation and reproduction

CPU preflight used actual MLX CPU layout arrays without weights. It checked final-canvas RoPE/prefix identity, old-state retention, absolute noise indexing and frontier boundary mapping with distinguishable synthetic values. A separate read-only code audit found no blocking sampling bug. Runtime checks confirmed input/prediction/cache geometry, exactly six transformer calls per arm, finite final latents/pixels, source latent equality at sigma zero and byte-exact 512-square final composites.

```sh
experiments/qwen/.venv/bin/python experiments/qwen/run_spatial_ablation.py --track track3 --preflight
experiments/qwen/.venv/bin/python experiments/qwen/run_spatial_ablation.py --track track3 --seed 42 --quantize 4
```

The runner refuses to overwrite an existing `ablation_metrics.json`. To reproduce in a fresh directory, append `--output /absolute/path/to/new-suite`.

- Track3 suite and complete phase/step records: [ablation_metrics.json](quality_runs/2026-10-05/track3/ablation_metrics.json).
- CPU checks: [preflight.json](quality_runs/2026-10-05/track3/preflight.json).
- Completed artifact/hash/geometry/source validation: [validation.json](quality_runs/2026-10-05/validation.json).
- Exact command, code hash and exit status: [launch.json](quality_runs/2026-10-05/track3/launch.json).
- All raw/final/step2/4/6 artifacts: each arm subdirectory beneath `quality_runs/2026-10-05/track3/`.

All six inference arms passed runtime checks. The two original track3 controls were also confirmed pixel/hash-identical to their older outputs. All GPU work for this ablation is complete; no further process is left running.

These records set `diagnostic_control:true` and `shared_canvas_comparable:false` and remain outside the standard model comparison aggregate. No model training, cloud job, new checkpoint download or Player integration was performed.
