# Viggle v0.3 and official base-tail outpaint controls — 2026-10-05

The new v0.3 adapter and its official nine-step base tail do **not** resolve the remaining guardrail/source-boundary discontinuity on track3. The nine-step arm slightly improves decoded source-border MAE, but direct raw/composite closeups still show the rail elbow, the top water/shore band and the lower asphalt transition. This is a completed negative geometry control, not a seamless result or a model promotion.

## Controlled setup

`run_turbo_tail_suite.py` is a new runner; existing geometry/spatial runners and installed runtime files were preserved. Root downloaded the pinned v0.3 rank256 adapter and launched one supervised GPU suite. The file is 1,359,147,904 bytes with SHA256 `f06c266e04438b5272bdfb99410421d52a65d7a37f6f42aabc3cb1faf0142644`, repository revision `009a44a895ef85f7e643c80fdca9543795248867`.

All arms use track3, seed42, Qwen-Image-2.1 Q4, the identical actual 512-square image/text reference prefix, the same source encoded in an edge-padded 512×1152 canvas for the known target crop, the same absolute-position noise, final-canvas RoPE and EARLY-FRONTIER initializer. The source and target encoding tensors remain separate. Active target heights are 640, 896 and then 1152 for every remaining step; target image input actually grows from 1,280 to 1,792 to 2,304 tokens. Future target tokens are absent before activation. Every existing target state is retained during insertion; inner regions continue refining rather than becoming permanently finalized.

The six-step arms use the official raw nodes `[1,0.9375,0.875,0.75,0.5,0.25]`. The nine-step arm uses `[1,0.9583,0.9167,0.875,0.75,0.5,0.25,1/6,1/12]`. All nodes undergo the same resolution-dependent shift computed from 2,304 final target tokens; terminal zero is appended with no terminal stretch. This follows the [official v0.3 model card](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo/blob/009a44a895ef85f7e643c80fdca9543795248867/README.md), including seven turbo forwards followed by two base forwards, no CFG and unmerged LoRA.

The base model is loaded once. Switching v0.2.1 to v0.3 first unwraps every actual `LoRALinear` to its identical base linear object, then applies the new adapter through the installed loader. Assertions confirm all 227 modules have the same target names/base objects and no adapter fusion. At zero-based step7 of the nine-step arm (human step8), all 227 actual `LoRALinear.scale` values switch from 1 to 0. The existing turbo prefix K/V cache is discarded before that forward; actual extraction steps are `[1,8]`. The ninth step reuses the freshly extracted base prefix.

Scale zero yields base-model outputs, but this runtime still executes the zero-scaled LoRA matrix multiplications. That overhead is explicitly recorded. The nine-step arm also has three extra forwards and a second prefix extraction, so it is not matched six-step compute. A changed schedule also means activation happens at different sigmas; this is the complete official nine-step policy, not an isolated two-tail-step ablation.

## Measured results

| Quantity | v0.2.1 six | v0.3 six | v0.3 nine + base tail |
|---|---:|---:|---:|
| Transformer forwards | 6 | 6 | 9 |
| Target-token forward sum | 12,288 | 12,288 | 19,200 |
| Prefix cache extraction steps | 1 | 1 | 1, 8 |
| Raw source top16 MAE, 0–255 | 8.717 | 8.563 | 7.990 |
| Raw source bottom16 MAE, 0–255 | 4.978 | 4.860 | 4.592 |
| Raw source interior MAE, 0–255 | 3.226 | 3.222 | 3.207 |
| Core denoising seconds | 26.29 | 28.66 | 46.89 |
| Final VAE decode seconds | 4.11 | 1.59 | 1.62 |
| Denoising MLX peak GiB | 8.446 | 8.454 | 9.006 |
| Original resized source pixels exact | yes | yes | yes |
| Final latent/pixel arrays finite | yes | yes | yes |

The v0.2.1 control raw and composite PNG SHA256 hashes exactly reproduce the preceding geometry `edge-context` control. Prefix, known-target and absolute-noise hashes also match that control. All three new arms share those conditioning/noise hashes. Their final composite hashes are:

- v0.2.1 six: `03ad495177a35f9ca8e405fc7b5d5a111c4389ee93230ee30381c9b38dcbe2e8`.
- v0.3 six: `6d78bf8215ba48f4aa7217c5b77ac7109d11b3ef1f6d30c5e9625bea458ee87f`.
- v0.3 nine: `b1dbab5d806baec7282d4db83186c139d37a742d40225629ef1653d61f62d2df`.

MAE is measured before hard compositing against the unchanged resized 512-square source. It measures border reconstruction, not rail geometry. The original source is pasted back exactly; no source feathering, blur, warp, geometry painting or image postprocessing was used to hide joins. The original JPEGs and prior results were retained.

## Direct visual inspection

The actual final comparison and nearest-neighbor raw/composite rail crops plus the lower asphalt crop were opened directly. An independent read-only reviewer also inspected the final comparison and reached the same conclusion.

- The three large compositions remain broadly similar. Water, foliage and asphalt are coherent away from the join.
- The raw rail still bends near the known-source edge. Hard source compositing replaces the inner part with the original, leaving a short direction change on the generated side.
- v0.3 six changes surface/detail balance but does not remove that generated-side geometry mismatch.
- The nine-step tail modestly changes the raw border reconstruction/fine texture. The final rail remains imperfect, and the top water band plus lower asphalt transition persist.
- The top extension still has the previously observed horizon-like composition. Newer weights/extra low-noise steps do not establish correct continuation of the source camera perspective.

![Three actual final composites](turbo_tail_runs/2026-10-05/track3/final-comparison.png)

![Actual 6x raw/composite rail crops](turbo_tail_runs/2026-10-05/track3/rail-join-6x.png)

![Actual 2x lower asphalt crops](turbo_tail_runs/2026-10-05/track3/asphalt-join-2x.png)

The rail crop covers original coordinates `(270,284,374,354)` and is enlarged six times with nearest-neighbor interpolation. Its left column is raw decode, right column is exact-source composite. The asphalt crop covers `(0,804,512,864)` and is enlarged twice. Orange markers lie outside the image pixels at the true source boundaries y320/y832. These are display-only crops of the saved outputs, not generated edits.

## Validation and timing scope

CPU preflight passed actual small-layer zero-scale/base equality, two real loader swaps without fusion, exact base-object restoration, cache transition mock extraction `[1,8]`, source RGBA padding, absolute layouts/RoPE and growth insertion retention. It loaded no checkpoint and used CPU only. The runner was independently audited before launch.

After inference, a separate CPU artifact validation checked actual final NPZ finiteness, full raster IDs, saved sigmas, known-source crop hash at sigma zero, exact source composites, forward counts, cache extraction steps, recorded actual scales and baseline raw/composite hashes. A second independent read-only audit verified the same runtime/output invariants.

The process completed in 127.16s (supervisor 128.03s), exit0, under a 900s process-group timeout. Shared model load was 11.52s and conditioning 2.36s, including 0.33s edge-context VAE encoding. v0.3 adapter replacement cost 0.29s. Shared load MLX peak was 24.50 GiB and conditioning peak 10.84 GiB. Core timing includes insertion/layout, scale/cache transition, transformer, Euler and predicted-clean formation; it excludes load, adapter replacement, final NPZ capture and VAE decoding. Memory peaks are reset by phase; macOS process peak RSS is cumulative. These single-process/seed measurements have execution-order, kernel warm-up and system-state caveats and make no reliable version speed ranking or iPhone performance claim.

```sh
experiments/qwen/.venv/bin/python experiments/qwen/run_turbo_tail_suite.py --track track3 --preflight
experiments/qwen/.venv/bin/python experiments/qwen/run_turbo_tail_suite.py --track track3 --seed 42 --quantize 4
```

The runner refuses to overwrite an existing suite; reproductions need a fresh `--output` directory.

- [Suite metrics](turbo_tail_runs/2026-10-05/track3/turbo_tail_metrics.json).
- [CPU preflight](turbo_tail_runs/2026-10-05/track3/preflight.json).
- [Completed artifact validation](turbo_tail_runs/2026-10-05/track3/validation.json).
- [Command, runner hash, timeout and exit status](turbo_tail_runs/2026-10-05/track3/launch.json).
- [Unmodified geometry control](GEOMETRY_RESULTS.md).

All arms are diagnostic and excluded from the standard aggregate. No training, new base-model download, cloud job, Player integration or live UI streaming was performed by this suite.
