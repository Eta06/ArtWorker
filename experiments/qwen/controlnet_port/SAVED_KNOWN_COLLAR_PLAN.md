# Top32 against the retained localized40 baseline

The [isolated runner](run_known_collar_saved_baseline_trial.py) executes only `top32`. It compares against the already completed [localized40 arm](../controlnet_runs/2026-10-05/track3-localized40-outpaint/tangent-canny/localized-hints/metrics.json). No actual GPU baseline rerun was performed. All original localized and ordinary known-collar files remain unchanged.

The saved baseline contains the exact inference values for the actual square reference latents, prompt embeddings, Boolean image slots, target/padded context129, initial noise and shifted 40-step sigma schedule. F32 storage represents the original BF16 values losslessly. The new runner restores those tensors directly; it performs zero source/control/prompt encodes and zero noise RNG draws. It retains Q4 base, BF16 control, CFG1, seed42 provenance, original outpaint prompt, tangent structural support64, all 32 base/16 control blocks, zero prefix hints and the target known bridge before and after each Euler update.

Only the completed hint field changes: target IDs 640–703, packed rows 20–21, receive scalar 1. All other effective BF16 hint values are exact against the old saved field, including unknown cosine64 support, deeper source/bottom zeros and the zero text/reference prefix. The executable forward and sampling-loop AST match the frozen baseline after ignoring the forward function name and docstring. The final original source pixels are pasted exactly as before.

## Validation

[Strict persisted-input validation](known_collar_saved_baseline.py) checks the completed old suite/arm, all 40 per-step records, recipe/config/revision/code/runtime hashes, source pixels, source/prompt/context/noise tensor hashes, support, mask rows, reference padding, joint hint suffix, saved final known bridge and image hashes. Output and baseline paths must be disjoint before any directory creation or write. Retained baseline files are hashed again before sampling and after decoding, before success.

[CPU test evidence](saved_known_collar_cpu_validation.json) passes actual MLX CPU BF16 restoration and top32 rounding, three independently computed known-bridge mixtures, and 22 rejection cases covering recipe drift, runtime drift, hint/support changes, corrupted saved noise/source context, inconsistent step records, overlapping output paths and late baseline mutation. No real weights were loaded for these checks.

[Fresh preflight-v5](../controlnet_runs/2026-10-05/track3-saved-top32-preflight-v5/preflight.json) is `preflight_files_ready`; all five code-bound CPU gates are current. Its [persisted-input report](../controlnet_runs/2026-10-05/track3-saved-top32-preflight-v5/saved_baseline_validation.json) records each retained file/tensor hash and states `actual_gpu_baseline_rerun_performed: false`. An independent read-only audit verified actual Boolean-slot layout and BF16 restores, exact sampling AST, valid destructors and integrity-guard placement.

## Parent-only GPU launch

```sh
experiments/qwen/.venv/bin/python experiments/qwen/controlnet_port/run_known_collar_saved_baseline_trial.py \
  --collar-arms top32 \
  --saved-baseline experiments/qwen/controlnet_runs/2026-10-05/track3-localized40-outpaint \
  --output experiments/qwen/controlnet_runs/2026-10-05/track3-saved-top32-outpaint
```

The launch requires a fresh output directory and the current saved-input CPU gate. It plans 40 new full-canvas denoiser calls. The old single-arm run took about 435 seconds overall; this saved-input path avoids its encoding work, with no speed claim until measured. Parent manages the serialized GPU slot and timeout.

The retained old final tensor SHA is `df1fa78c76720e09c4d8bbb7f2c3e7f76595a6860efab266b7f25229fa50a979`; raw PNG SHA is `b47c8d1f86b345fa2f1490aed71c973a1890171f6f9a44502e1c1480646561c9`. GPU determinism of a newly repeated baseline was not measured. Large weight payloads were not rehashed; size/header/config/revision provenance is the same as the frozen baseline. These limits remain explicit.

Quality acceptance requires independent visual/profile evaluation. Restoring matched inputs and preserving source pixels establish a controlled comparison; they do not establish improved rail geometry.

## Completed parent-run experiment

Parent executed the isolated saved-top32 command in a serialized GPU slot. The [run](../controlnet_runs/2026-10-05/track3-saved-top32-outpaint/saved_known_collar_reference_metrics.json) completed successfully in 429.40 seconds with exactly 40 new denoiser calls. No actual GPU baseline rerun occurred.

[Independent NumPy/Pillow integrity audit](../controlnet_runs/2026-10-05/track3-saved-top32-outpaint/independent_saved_top32_integrity.json), produced by [this CPU-only script](audit_saved_top32_completion.py), passes actual saved-input equality, the 64-token effective hint difference, all source-reference checks, final sigma-zero known bridge, exact shared source pixels, unchanged raw exterior during final hardpaste, and retained baseline/code/runtime file hashes. The new final latent SHA is `4b93351925d6ad9f80231d892e5efe8294bb4ee7fa62dcb153e34e53eaa5acbf`.

The [new composite](../controlnet_runs/2026-10-05/track3-saved-top32-outpaint/tangent-canny/top32/composite.png) was viewed directly: the scene remains coherent and the upper rail appears closer at the join, while the horizontal water-tone boundary and coarse lower asphalt transition remain visible. Raw top16 source reconstruction MAE drops from 10.486 to 8.857, with bottom16 7.158 to 6.986. These reconstruction values do not establish rail geometry or final quality acceptance; fixed profile evaluation is separate.
