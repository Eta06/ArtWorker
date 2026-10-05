# Practical boundary fixes for the growing Qwen outpaint loop — 2026-10-05

The strongest inexpensive next experiments isolate boundary conditioning before replacing the model. The existing source-prefix geometry is correct, and `GEOMETRY_RESULTS.md` already reproduces the preceding baseline exactly. Changing the known target VAE encoding context improves raw rail reconstruction but leaves a generated-side elbow and worsens the lower asphalt transition. Lower reconstruction MAE is not sufficient acceptance.

## Ranked controlled experiments

1. **Narrow generated-side latent collar.** Starting from the fixed edge-context control, weakly anchor one or two latent rows outside the source to a context encoding, tapering the weight outward. Test 16/32 pixel widths and a modest weight separately. Keep every final source pixel exact. A wide hard repeated-row collar would force flat continuation and merely relocate the discontinuity; this proposal is a diagnostic local hypothesis, not a published geometric guarantee.
2. **Known boundary encoding conditioned on the evolving surroundings.** Decode an existing predicted-clean active canvas, replace the source pixels exactly, re-encode, and refresh only the known target boundary latent rows at selected late steps. The square reference prefix remains unchanged. This adds VAE work, not a second diffusion generation. It tests whether a fixed edge-padded encoder neighborhood causes the lower seam. It can also destabilize the trajectory; validate against the same seed, sigma nodes, activation schedule and a unchanged control.
3. **Official 9-step adapter/base tail.** Compare a seven-turbo-step arm to a nine-step arm containing the same first seven updates followed by two base updates. This tests whether additional base refinement resolves boundary detail, rather than confounding adapter version and step count. A separate v0.3 six-step comparison tests weight changes alone.
4. **LanPaint-style conditional inner correction.** Unlike a simple hard known-latent replacement, a conditional correction lets the active target and known neighborhood interact before the next Euler update. No new checkpoint is needed. Start with one/two inner iterations at selected mid/late steps; count all transformer calls and extra stochastic noise. This remains one generation trajectory, with future target tokens genuinely absent. See the exact implementation notes below.
5. **Trained structural control.** Extrapolate high-confidence rail tangent/vanishing geometry into a Canny or MLSD control map and use a trained Qwen2.1 control branch with an outpaint mask. This is a larger runtime port, not an immediately interchangeable LoRA. It offers stronger structural constraints but could overfit a single cover. A generic method needs automatic, confidence-gated control extraction and tests on unrelated artwork.

All experiments must inspect raw and exact-source composites at both boundaries and full-frame output. Track2's truncated lower body is a mandatory second case. Preserve baseline artifacts, compare more than one seed before claiming robustness, and do not replace the player background with a research candidate until acceptance.

## Current Viggle instructions and exact local cache behavior

The current [Viggle model card](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo) publishes v0.3 and says its six-step texture is softer than v0.2.1. It retains the raw nodes `[1,0.9375,0.875,0.75,0.5,0.25]`, no CFG, and its shipped scheduler. Additional nodes belong at the high-noise end. Its nine-step recipe uses raw nodes `[1,0.9583,0.9167,0.875,0.75,0.5,0.25,1/6,1/12]`: disable the unmerged LoRA after zero-based step 6 and rebuild text/reference K/V on the first base forward. Re-enable for a later request. These are schedule instructions, not evidence of fixing this rail.

In the installed MLX reference path, `QwenImage21Transformer.forward_reference` uses a caller-owned `cache` list. Empty means full prefix extraction; nonempty means target-only forward with the saved prefix. The known/source prefix is timestep-zero and target-independent, so growth alone does not invalidate it. Changing LoRA scale does invalidate its projected K/V. Clear that list immediately before the first base update; keep the original square source tensor and prompt embeddings; do not re-run the text encoder. LoRA was loaded with `bake_lora=False`; scale-zero is possible on `LoRALinear`/members of `FusedLoRALinear`. Count the extra prefix extraction in cost. A merged checkpoint cannot provide a real base tail without the original base weights.

The raw nine-step nodes should receive the same final-resolution shift as the current six-step function. Do not shift twice or recompute shift from each transient active window. Confirm `shift_terminal=null`, which avoids the base scheduler's inappropriate last-step modification.

## Structural checkpoint: verified availability, substantial port

[Alibaba PAI's Qwen-Image-2.1-Fun-Controlnet-Union card](https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union) describes an approximately 7 GB branch with sixteen injection blocks. The shared input comprises 64 control latent channels, one mask channel and 64 masked-image latent channels. It supports Canny/MLSD and inpainting together. Published examples use 40 steps; compatibility with this six-step Viggle growth loop is untested.

The existing MLX model has no matching control branch. The [current VideoX pipeline](https://github.com/aigc-apps/VideoX-Fun/blob/4b7b6402a1e0f0406bd6801fb66c0a00bd922621/videox_fun/pipeline/pipeline_qwenimage21_control.py) explicitly enables prefix caching only when `control_context is None`. This is more precise than the card's broad caching note: control skips depend on the per-step joint stream, so blindly preserving the current prefix cache would be invalid. A growing implementation must also gather control inputs at the same absolute target IDs and preserve control RoPE, not resize the control image at each activation.

## LanPaint: feasible isolated MLX probe, no new weights

[LanPaint 2.2.0](https://github.com/scraed/LanPaint/releases/tag/2.2.0) and its [official repository](https://github.com/scraed/LanPaint/tree/2d7912f9a5efe5ece8de334c7ca18317b8288c39) add Qwen2.1 masked editing and RGBA support. Alpha is transparency; the inpaint mask is separate. The repository warns that distilled models can perform worse. Its documented Qwen example uses five thinking iterations, which are additional denoiser work. Its GPL-3.0 code is research provenance, not a mobile deployment licence decision.

For our flow state `z=(1-t)y+t*noise`, velocity `v`, and a known-mask `m` equal to one inside the source:

```
a = (1-t)^2 / ((1-t)^2 + t^2)
b = 1-a
q = sqrt(a) + sqrt(b)
u = q*z                         # variance-preserving coordinates
x0 = z - t*v                   # clean model prediction
s_unknown = -(u-x0)
s_known = -(1+lambda)*(u-y) + lambda*(u-x0_big)
s = (1-m)*s_unknown + m*s_known
A = (1+lambda*m)/b
C = (sqrt(a)*(u+s)-u)/b + A*u
```

At no CFG, `x0_big=x0`; do not insert a fictitious negative-prompt prediction. For a fixed coefficient correction interval `h`, the overdamped update is:

```
k  = -expm1(-A*h)/A
k2 = -expm1(-2*A*h)/(2*A)
u_next = exp(-A*h)*u + k*C + sqrt(2*k2)*eta
```

The first inner iteration evaluates `C` once and advances a full interval. A second iteration uses the previous `C` for a half interval, evaluates a new `C` at that half-step state, adds `(C_new-C_old)*h`, then advances the other half interval with `C_new`. Each iteration costs one transformer forward; a final forward at the corrected state produces the outer Euler prediction. This follows the active overdamped implementation, not its commented-out oscillator.

Current defaults use constant correction step size `0.2` (`MinStepFrac=1`) and round the inner count down with remaining variance fraction `b`. The last outer update has no thinking iterations. An isolated conservative probe can choose `0.1`, one/two requested iterations, and selected outer steps; these are deviations to record, not the untouched official workflow. At `t=1`, `a=0` gives no meaningful clean-image conditioning, so skipping that correction is sensible and must be explicit.

Keep the square image reference and its cache fixed. Inner corrections change only current target latents and use the same sampled outer timestep for every inner forward. The prefix never sees the changed target under the causal layout, so the existing prefix cache remains valid. Gather independent correction noise by absolute target IDs with a separate deterministic random stream. Future targets must not be passed to the transformer, including inner calls. The final known clean prediction and sigma-zero latents are restored to the chosen known source; final image pixels are pasted exactly. Extra noise, effective iteration count, correction timestep, all NFE and all target-token forwards must be logged.

The growing-token combination has not been demonstrated by the official workflow. It is an experimental port requiring finite-value checks, numerical CPU comparison and visual acceptance. Do not report the asymptotic paper guarantee as a six-step production guarantee.

No checkpoint was downloaded and no GPU experiment was run for this report. External sources and local reference code were checked on 2026-10-05.

## Isolated probe prepared, CPU validation only

`run_lanpaint_probe.py` leaves the installed runtime and existing runners unchanged. It compares `edge-control`, `think1` and `think2` with the exact previous edge-context source/reference setup. Corrections occur at outer steps 3, 4 and 5, while the final update has zero inner corrections. The selected six sigma nodes and activation schedule remain unchanged. True planned NFE is respectively **6, 9 and 12**, not six for all arms; total target-token forwards are respectively **12,288, 19,200 and 26,112**.

The CPU preflight passed layout/RoPE preservation, insertion retention, original RGBA/padding geometry, and twelve flow/time/iteration cases compared to independent NumPy scalar-coefficient equations. Maximum correction numerical difference was `4.77e-7`; zero-iteration correction is exactly identity. This verifies arithmetic/accounting with a mock affine velocity model, not real-model quality. No weights were loaded by preflight.

```sh
experiments/qwen/.venv/bin/python experiments/qwen/run_lanpaint_probe.py --track track3 --preflight
```

If independently reviewed and selected for a bounded GPU trial, run it in a fresh output directory. The runner records each inner/final forward, active absolute IDs, sigma, cache extraction, separate correction-noise seeds/hashes and all actual NFE. Future tokens are never forwarded, including inner calls. The source prefix is fixed; current known target can move inside the conditional correction but its standard bridge is restored after every outer update and the chosen known target is exact at sigma zero. Final source pixels are pasted exactly. This last restoration follows the existing growing-loop contract and must be distinguished from an untouched official ComfyUI execution.

A separate read-only reviewer compared the implementation to the pinned upstream active overdamped path and found no blocking mathematical or runtime issue. That review confirmed the flow conversion, no-CFG BIG equivalence, second-iteration midpoint/coefficient correction, source-prefix cache validity and accounting. Baseline image/latent hashes still require verification after a real trial; static matching and CPU mock arithmetic do not prove real-model pixel identity. Per-step query-token fields describe one call; summed compute uses every `all_forward_records` entry, with explicit per-step token sums also recorded.
