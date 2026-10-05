# Saved top32: checkpointed VAE projection, four updates per arm

Prepared CPU-only. Parent controls GPU execution. The [isolated instrumented runner](run_saved_top32_checkpoint_projection.py) keeps the original decoder, objective, mask, casts and Adam mathematics unchanged. It loads only the frozen F32 Qwen reference VAE. Model weights are never trained, and no transformer sampling is performed.

The [input manifest](projection_inputs/2026-10-05/track3-saved-top32/input_metadata.json) validates the completed saved-top32 result and records an exact copy of its entire final NPZ and original model raw PNG. Every array and dtype is retained, including full IDs, source rectangle, sigma metadata and source reference latents. NPZ file SHA is `16702e506bcf97665c5147a64d0933a835929aaf2a26103794a013791c79697c`; raw PNG SHA is `adf4a1e7fe993bf6163639c1c5a3a5a34e715d3bad11284393c053aebf29c8c1`. Separate provenance metadata adds no latent or pixel changes. [Copy script](controlnet_port/prepare_saved_top32_projection_input.py).

## Candidate

Two independent arms start from that same tensor, with four Adam updates each:

- `consistency`: original source top/bottom 32-row RGB reconstruction MSE plus latent trust0.01.
- `derivative`: the same objective plus generic one-pixel exterior RGB finite-difference continuity, weight1, derived from adjacent source rows.

Only six latent rows on either side of each source boundary may change: rows14–25 and46–57. Learning rate0.015 and elementwise delta cap0.2 retain the original proposal. No rail is drawn, detected or warped. Outside-collar latent values must stay exact; the final original source pixels are pasted exactly. Global VAE attention can still change distant decoded pixels, which are measured separately.

## CPU readiness

- [Original checkpoint harness preflight](projection_runs/2026-10-05/track3-saved-top32-checkpoint4-preflight/preflight.json) passes on the exact new input.
- [Instrumented preflight-v2](projection_runs/2026-10-05/track3-saved-top32-checkpoint4-instrumented-preflight-v2/preflight.json) passes CPU tiler and tiny checkpointed forward/gradient parity, without real weights or GPU work.
- [CPU instrumentation validation](saved_top32_projection_instrumentation_cpu_validation.json) binds current hashes, exact input files/arrays and unchanged mathematical AST. FFT tests verify zero power for a constant image, fourfold power for doubled texture amplitude, and invariance to a constant tone offset. [Test script](test_saved_top32_projection_instrumentation_cpu.py).
- Independent read-only audit passes all17 core AST checks and source/provenance, duplicate-arm, fresh-output and protected-directory guards. Actual selected source hashes are checked before any output write.

Reviewed runner SHA256: `06a58842753de3b17577395fcb5fbb231294f0117b90554697ed15f73e4624d6`.

```sh
experiments/qwen/.venv/bin/python experiments/qwen/run_saved_top32_checkpoint_projection.py \
  --input experiments/qwen/projection_inputs/2026-10-05/track3-saved-top32/final_latents.npz \
  --output experiments/qwen/projection_runs/2026-10-05/track3-saved-top32-checkpoint4-instrumented \
  --methods consistency,derivative --iterations 4 --checkpoint-decoder
```

Parent proposed a900-second serialized supervisor. If interrupted, that supervisor outcome remains authoritative. One latest atomic Adam snapshot per method is updated after every iteration, including input/source/runner/config/method metadata. It is a recoverable state artifact; automatic resume is not implemented.

## Required runtime and quality evidence

The real differentiable/reference VAE forward must agree and the baseline must reconstruct the copied model raw pixels exactly before optimization. The instrumented runner records per-iteration peaks/timing, cumulative baseline/load peak, final-decode and overall observed MLX peaks, state-save overhead and arm/total runtime. It also records unprivileged `pmset -g therm` warning/pressure text. Physical sensor temperatures in Celsius are unavailable through this probe and are explicitly unmeasured.

Saved raw metrics include two-sided16-row RGB/luma profiles, source-band reconstruction/derivative errors, fixed FFT patch powers against the decoded baseline and original source-known patches, and distant raw changes. FFT patches contain object edges, so spectral ratios describe changes rather than prove preserved grain. Frozen external profile evaluators must assess rail position/slope and the small final-row jog without threshold changes. Raw/composite native crops and full images require direct inspection.

Checkpoint memory benefit has not been measured. The previous completed consistency8 arm did not repair rail geometry; its derivative arm timed out without a final image at an observed40.60GiB backward peak. That failed/incomplete result remains intact in [PROJECTION_RESULTS.md](PROJECTION_RESULTS.md). This candidate establishes no improved quality, mobile feasibility or training result before actual execution and review.
