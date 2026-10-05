# Frozen-VAE source-boundary projection diagnostic

Prepared 2026-10-05. CPU preflight passed. Root launched the serialized VAE-only GPU diagnostic using the exact final NPZ from the new boundary suite: the consistency arm completed, while the derivative arm timed out without a final image. [Actual results and limitation](PROJECTION_RESULTS.md). Its baseline reconstructs the previous raw PNG pixel-for-pixel and the real differentiable tiled decoder matches the reference exactly. This proposal is a **derived post-denoising repair**, not a new outpainting sampler, trained model, LanPaint, or diffusion posterior sampling implementation.

The previous boundary-context test reduced reconstruction error but left a rail elbow and a lower asphalt tone seam. The reference Qwen2.1 VAE has nonlinear residual convolution blocks and global mid-block attention. Thus fixing a rectangular set of latents does not independently fix the corresponding rectangular set of pixels. The installed tiled decode also converts tensors to NumPy, which detaches gradients; the new isolated script duplicates its tile512/overlap64 arithmetic in differentiable MLX.

`run_boundary_projection.py` loads only the existing F32 reference VAE, validates all weight keys/shapes and freezes its weights. It accepts an exact terminal-state NPZ with full absolute IDs and the shared source rectangle, or the retained full-canvas step06 predicted-clean tensor with its window metadata. Exact final states are preferred. At terminal zero sigma, the earlier predicted-clean tensor and final Euler state have the same algebraic definition and source clamp; the harness checks actual baseline decoded pixels against the retained final PNG rather than assuming bitwise floating-point equality. CPU cases cover both input forms, their different original-raw paths, and malformed rectangle/window/IDs rejection.

Two arms share the same starting tensor and budget:

- `consistency`: RGB reconstruction MSE on the first/last 32 source rows plus a small latent trust penalty.
- `derivative`: the same objective plus generic one-pixel RGB finite-difference continuity across both source boundaries. Adjacent source rows provide the local continuation signal; no rail is detected, drawn, or warped.

Only six latent rows on each side of the top/bottom source boundaries may change. Known border latents may move too; final source **pixels** remain exact through the same hard paste used before. All other latent entries must remain bitwise equal. Global VAE attention can still change distant decoded pixels, so the runner measures that change separately. It does not promise unchanged generated pixels outside the collar.

Defaults are eight bounded Adam iterations, learning rate 0.015 and elementwise latent delta cap 0.2. Top and middle decoder tiles contain the two boundaries away from their overlapping rows, so the optimization uses those exact contexts. The final image uses all reference-layout tiles. Output includes raw and exact-source composite PNGs, projected latents, true final comparison, nearest-neighbor rail crop, reconstruction/derivative diagnostics, outside-collar latent checks, original JPEG hash checks and memory/timing scope.

## What a successful result could establish

Reduced reconstruction or derivative loss alone would not establish a correct guardrail. Final raw/composite closeups and full-frame inspection must show improved position and slope without a new seam. A post-denoising pixel-consistency objective can improve local codec coupling and color continuity. Its source constraints do not provide ground truth for distant rail perspective, vegetation, or horizon; those remain the generator's responsibility.

## Primary research context

[PELC / DecFormer, CVPR 2026](https://arxiv.org/html/2512.05198v1) directly studies nonlinear, spatially entangled VAE mask failures. Its learned 7.7M compositor is demonstrated with the FLUX VAE; existing weights are not an established drop-in for this Qwen64-channel/16× codec. It supports the seam diagnosis and a future codec-specific lightweight compositor, not a claim that this gradient harness implements PELC.

[Diffusion Posterior Sampling](https://arxiv.org/abs/2209.14687) incorporates measurement guidance during denoising. This harness omits the denoiser Jacobian and diffusion prior, so a VAE-only repair must not be labeled DPS.

[LanPaint's current implementation](https://github.com/scraed/LanPaint/tree/2d7912f9a5efe5ece8de334c7ca18317b8288c39) supports Qwen2.1 editing/RGBA and repeatedly evaluates the joint denoiser in conditional Langevin inner iterations. That can change generated geometry during sampling. Its Qwen2.1 example uses base-model 20 outer steps and five inner iterations, not our six-step Turbo adapter. The [paper's Eq.12–14](https://arxiv.org/html/2502.03491v3#S4.SS1) gives a low-noise/asymptotic approximation; five inner iterations do not ensure an exact posterior. Turbo compatibility needs a separate measured control.

```sh
experiments/qwen/.venv/bin/python experiments/qwen/run_boundary_projection.py --preflight
# GPU execution must be serialized with the other local model experiments.
experiments/qwen/.venv/bin/python experiments/qwen/run_boundary_projection.py \
  --input experiments/qwen/boundary_runs/2026-10-05/track3/edge-context/final_latents.npz \
  --output /absolute/path/to/fresh-projection-run \
  --methods derivative --checkpoint-decoder
```

CPU evidence: `projection_runs/2026-10-05/track3/preflight.json`. Mock decoder tiled output matches the reference exactly and its finite gradient is nonzero. Syntax compilation passed. No installed runtime, Player, model weight, preceding runner or retained image was modified.
