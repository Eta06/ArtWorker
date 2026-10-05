# Exterior-only boundary compositor probe — 2026-10-05

The generic CPU compositor reduces the photometric hard-paste seam, but it does **not** solve the rail's remaining angle/elbow error. A pure harmonic color correction is a useful candidate for a later finishing stage. Source-derived optical flow barely changes the rail join; bilinear remapping also softens exterior texture. Cubic remapping preserves texture much better, but the geometric defect remains. This is a post-generation diagnostic, not a better model or successful streamed outpainting.

## Scope and controls

`run_boundary_compositor.py` loads only NumPy, Pillow and the already installed OpenCV 4.14.0. No checkpoint, MLX, GPU, network, dependency installation or Player change is involved. For every variant, the original resized 512×512 source is copied exactly into `(0,320,512,832)` of a 512×1152 output. Original JPEGs, shared source PNGs and input model outputs are hash checked and remain unchanged.

All changes are confined to the generated top/bottom exterior within 96 pixels of the source. The rest of the generated exterior remains byte-for-byte identical to the baseline. No original source pixel is blurred, warped or blended. There are no manually placed rail lines, selected object masks or rail coordinates in the algorithm. The same registration/color procedure was run on all three covers. A rail closeup is used only for visual diagnosis.

The inputs are existing real model results, not a new matched model comparison:

- track3: Qwen EARLY-FRONTIER with edge-context known-target encoding, six steps.
- track2: Qwen EARLY-FRONTIER square-known, six steps.
- track1: the earlier Qwen v0.2.1 r128 six-step standard trial.

These inputs differ in sampling/conditioning. Each cover's compositor variants share exactly its own input raw decode and source; compositor differences are controlled within a cover.

## Variants

Each cover has six saved composites: baseline hard paste; harmonic color alone; DIS flow alone; DIS flow plus harmonic color; DIS flow with a capped normal-derivative extrapolation plus color; Farneback flow plus color. A fourth track3 suite repeats the same six variants with cubic instead of bilinear exterior image remapping. In total, 24 composite records were saved; the cubic baseline and pure-color outputs reproduce the corresponding bilinear suite exactly.

Optical flow is computed from the original known source to its raw decoded reconstruction. It is smoothed, capped at six pixels and used in the correct output-to-raw sampling direction. The first six source rows give a weighted boundary displacement. That displacement is extended outside and tapered smoothly to zero over 96 pixels; the tangent variant also extrapolates the observed known-source displacement derivative, with a conservative cap. This estimates codec/source misalignment, not the direction of scene objects beyond the source. Flow cannot observe the true generated exterior correspondence.

The color procedure computes original-minus-decoded residuals from the first four known source rows at each boundary, smooths across x with sigma 6 and caps RGB residuals at ±24. Cosine modes solve a finite-strip harmonic extension with zero residual 96 pixels away. Additive color transfer keeps existing generated texture rather than inventing content, but can create a halo when the residual actually comes from geometric misregistration.

## Track3 evidence

| Variant | Top adjacent-row RGB jump | Bottom adjacent-row RGB jump | Exterior collar horizontal-gradient energy ratio |
|---|---:|---:|---:|
| Baseline | 14.757 | 5.386 | 1.000 |
| Color only | 9.307 | 4.922 | 1.001 |
| DIS + color, bilinear | 8.846 | 4.639 | 0.797 |
| DIS + tangent + color, bilinear | 8.838 | 4.626 | 0.759 |
| Farneback + color, bilinear | 8.863 | 4.779 | 0.860 |
| DIS + color, cubic | 9.556 | 4.897 | 0.999 |
| DIS + tangent + color, cubic | 9.546 | 4.897 | 0.994 |
| Farneback + color, cubic | 9.409 | 4.929 | 1.001 |

RGB jump is the mean absolute difference between adjacent rows on either side of the hard source boundary, on a 0–255 scale. It is affected by real textures/objects as well as seams. Gradient energy is a texture diagnostic, not a geometry or perceptual quality score. Lower jump alone does not establish success.

Direct inspection of full composites, 2× boundary strips and a 6× rail closeup found:

- Color alone makes the water/color transition less abrupt and reduces the lower asphalt tonality jump, but the rail elbow remains at the same position and angle.
- DIS/Farneback registration moves the rail only subtly. In the known source, DIS reduces top16 reconstruction MAE from 8.717 to 8.239 and full-source MAE from 3.452 to 3.368. This modest codec alignment cannot repair the generated rail's scene tangent.
- Bilinear flow unnecessarily softens parts of the generated collar. The tangent variant has more texture loss. It is not a good default.
- Cubic resampling restores exterior texture sharpness relative to bilinear, at the cost of small ringing/overshoot risk. The actual rail elbow is still visible. The flow variants therefore remain diagnostic, not accepted fixes.
- None of these compositors changes the invented distant sky/horizon or vegetation geometry above the source. They cannot substitute for better generation.

![Track3 bilinear compositor outputs](boundary_compositor_runs/2026-10-05/track3/final-comparison.png)

![Track3 top boundary, actual pixels enlarged 2x](boundary_compositor_runs/2026-10-05/track3/top-boundary-comparison-2x.png)

![Track3 cubic rail closeup, actual pixels enlarged 6x](boundary_compositor_runs/2026-10-05/track3-cubic/rail-detail-6x.png)

## Other-cover checks

Track2 color-only top row jump falls from 15.352 to 7.318; bottom falls from 28.290 to 22.703. The bright clothing/source join can look more continuous locally, but the people still stop shortly below the source and turn into the generated blue panel. Every compositor retains that major semantic failure. Flow with bilinear remapping softens this collar too. This is direct evidence that seam polishing does not solve missing body/scene continuation.

Track1 has largely black exterior; all variants retain its original source, Moon, person and typography exactly. Color changes small patches around source-edge content but does not add meaningful scene completion. Its bottom row-jump measurement actually increases slightly under color-only correction, from 4.232 to 4.323. The procedure is not uniformly better on every edge.

![Track2 compositor outputs: lower body failure remains](boundary_compositor_runs/2026-10-05/track2/final-comparison.png)

## Cost, readiness and validation

On this M4 Max Mac with a single OpenCV CPU thread, color-only computation took approximately 5.6–6.0 ms per 512×1152 image, excluding input load and PNG saving. First DIS flow plus warp took about 35 ms with bilinear or 39 ms with cubic; subsequent DIS variants reuse that estimated flow and their recorded 8–13 ms do **not** include another flow estimation. Standalone DIS+color therefore needs the shared flow cost as well. Farneback+color took about 42–47 ms. Each complete six-variant suite including load, full PNG saves and montages took 0.68–0.71 seconds. These are Mac CPU observations, not iPhone, Swift, Core Image or product latency measurements.

Preflight independently checks inverse-warp direction using a known synthetic translation, the analytic constant-mode harmonic solution, original-source equality and the unchanged far exterior. Runtime validation checks every saved composite against its original source and input baseline, and verifies original JPEG/raw/source hashes. All 24 records pass; no original artifact was overwritten. The standard model aggregate is not modified: these outputs are explicitly postprocessed diagnostics.

An independent read-only reviewer checked the inverse-flow mapping, extrapolation sign, DCT/IDCT damping and all 24 saved composites. It found no blocking correctness bug, confirmed exact source/far-exterior preservation and exact cubic baseline/color controls, and directly confirmed that the rail elbow remains. Both reviewers reject the result as a complete geometry fix.

Recommendation: retain **pure harmonic color** as an optional finishing-stage candidate, and avoid default optical flow until a new model output demonstrates that it solves a real misalignment without exterior damage. The current rail geometry is not fixed by this experiment. Nothing has been integrated into Player.

```sh
experiments/qwen/.venv/bin/python experiments/qwen/run_boundary_compositor.py --preflight
experiments/qwen/.venv/bin/python experiments/qwen/run_boundary_compositor.py --track track3 --output /absolute/path/to/fresh-run
experiments/qwen/.venv/bin/python experiments/qwen/run_boundary_compositor.py --track track3 --warp-interpolation cubic --output /absolute/path/to/fresh-cubic-run
```

The runner refuses to overwrite an existing `compositor_metrics.json`. A shared resized source and same-sized raw decode are required. The `--raw` override allows the same generic method to inspect a later model output; this does not require a new GPU run.

- [Track3 complete metrics](boundary_compositor_runs/2026-10-05/track3/compositor_metrics.json).
- [Track3 cubic complete metrics](boundary_compositor_runs/2026-10-05/track3-cubic/compositor_metrics.json).
- [Track2 complete metrics](boundary_compositor_runs/2026-10-05/track2/compositor_metrics.json).
- [Track1 complete metrics](boundary_compositor_runs/2026-10-05/track1/compositor_metrics.json).
- [Saved-pixel and source-hash validation](boundary_compositor_runs/2026-10-05/validation.json).
