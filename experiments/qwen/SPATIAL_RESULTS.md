# Actual center-out target growth: matched FULL / GROW probe

This is a custom sampler experiment, not a production feature. One model load
runs two independent six-forward generation loops. The GROW loop starts with a
512×640 target and expands its actual target state to 512×896 and 512×1152, two
steps per size. Future target tokens are absent from the model input and
attention until activation. It does not generate a complete final image then
reveal it. Previously active regions continue refining at the global timestep.

`run_spatial_trial.py` uses Qwen-Image-2.1 q4 with the publisher's recommended
Viggle v0.2.1 r256 six-step adapter. FULL and GROW share the actual 512-square
source prefix, prompt, latent source encoding, seed42 noise table, and final-
canvas sigma schedule. Both build a fresh prefix K/V cache once. The fixed
source contributes 1,024 image latent tokens; the measured complete prefix is
1,197 tokens including text. Absolute final-canvas rotary positions are retained
when the active target grows. CPU preflight checked layouts and insertion before
GPU inference; runtime assertions checked input/prediction/cache geometry.

| Measured quantity | FULL | GROW |
|---|---:|---:|
| Transformer forward calls | 6 | 6 |
| Target tokens per step | 2304×6 | 1280×2,1792×2,2304×2 |
| Total target token forwards | 13,824 | 10,752 |
| Core denoise seconds | 29.38 | 24.33 |
| Final decode seconds | 1.65 | 1.60 |
| Preview capture seconds | 0.20 | 0.16 |
| Preview decode seconds | 4.71 | 3.67 |
| Denoise MLX peak GiB | 9.00 | 8.46 |

GROW reduces target-token forwards by 22.2% and measured core denoise time by 17.2%
in this single matched pair. Execution order, kernel warm-up and thermal state
can affect time. FULL runs first; this is not a repeated performance benchmark.
Shared conditioning peaks 10.84 GiB; on-load checkpoint quantization peaks 24.50 GiB.
The entire process, including one load, shared conditioning, both loops, final
decodes and all six preview decodes, took 76.35s and exited 0.

## Actual quality result

The FULL square-source-prefix output continues the water, guardrail and road
coherently with one car/person; boundary seams are visibly reduced compared to
the earlier gray-prefix pipeline. That observation changes both the reference
conditioning and source latent encoding, so it does not isolate either cause.

GROW's final image **fails visual quality**: newly admitted outer bands contain
grid/ringing patterns, an additional guardrail at the upper edge, and small
hallucinated symbols near the bottom. The generated content is not a clean
seamless expansion. The final failed image and its raw decode are retained.
Both arms produce finite pixels and exact original 512-square final composites.

At late insertion, new tokens use
`(1-sigma)*edge-extrapolated source latent + sigma*same-seed noise`.
This is an untrained initialization guess, not the true conditional flow
marginal. It makes token growth executable but does not establish reliable
sampling. The checkpoint supplies one global target sigma; it was not trained
with separate noise ages for mature inner and fresh outer regions.

Actual predicted-clean snapshots at steps 2/4/6 are saved in each arm's
`previews/` directory. They use the pre-step `z_sigma - sigma*v` estimate and
are decoded only after both denoise loops. Their dimensions for GROW are
512×640, 512×896, 512×1152. These are evolving predictions; the earlier inner
region is not claimed complete or frozen. All preview decode time is separate
from the core denoise comparison.

Metrics and artifacts:

- Pair: `spatial_runs/q4/track3/paired_metrics.json`.
- FULL: `../evaluation/results/qwen21-spatial-r256-full-q4/track3/`.
- GROW: `../evaluation/results/qwen21-spatial-r256-grow-q4/track3/`.
- Architecture and training limitations: `TOKEN_GROWTH_NOTE.md`.

Both records explicitly set `diagnostic_control:true` and
`shared_canvas_comparable:false`. The standard comparison has a different
reference prefix, so these probes are excluded from its common grid.
