# LanPaint-style conditional correction probe — 2026-10-05

Extra conditional corrections change the expanded composition, but this matched probe does **not** solve the source/guardrail join. One correction produces a steeper generated rail with a remaining elbow and visible source-border texture transition. Two corrections break the guardrail into disconnected sections. This is a failed seamless-geometry probe; the one-correction arm is partial diagnostic evidence, not a promoted result.

## Matched experiment

`run_lanpaint_probe.py` is an isolated experimental MLX implementation of the active overdamped path described in [the method research](BOUNDARY_METHOD_RESEARCH.md), using upstream revision `2d7912f9a5efe5ece8de334c7ca18317b8288c39`. This is **not** the official ComfyUI workflow or a general LanPaint benchmark. The probe uses step size `h=0.1` instead of the cited upstream default `0.2`, conditional lambda `5`, CFG `1`, no negative/BIG-guidance model, and corrections only at outer steps 3, 4 and 5. Its known target source is restored to the normal flow bridge after each outer update. No correction occurs at the final outer step.

All three arms use track3, seed42, Qwen-Image-2.1 Q4 plus Viggle v0.2.1 r256, the original square reference prefix, edge-context known target encoding, identical prompt, absolute-position outer noise, absolute RoPE, six sigma nodes, and EARLY-FRONTIER growth `[640,896,1152,1152,1152,1152]`. Future target tokens are genuinely absent from each forward until activated. Newly inserted regions retain the preceding predicted-clean frontier initializer; it is an untrained heuristic. Existing inner regions continue refining at the same global sigma. Corrections use a separate deterministic noise stream and add real active-target denoiser calls at an unchanged outer timestep.

The source is the common resized 512-square image at `(0,320,512,832)` inside a 512×1152 final canvas. Final source preservation means pasting that resized source exactly; it does not mean preserving every original 720-square JPEG pixel before resizing.

| Quantity | edge-control | think1 | think2 |
|---|---:|---:|---:|
| Outer steps | 6 | 6 | 6 |
| Effective inner corrections by outer step | 0,0,0,0,0,0 | 0,0,1,1,1,0 | 0,0,2,2,2,0 |
| Actual transformer calls / NFE | 6 | 9 | 12 |
| Extra inner transformer calls | 0 | 3 | 6 |
| Total target-token forwards | 12,288 | 19,200 | 26,112 |
| Raw source top16 MAE, 0–255 | 8.717 | 8.560 | 8.720 |
| Raw source bottom16 MAE, 0–255 | 4.978 | 4.777 | 4.635 |
| Core denoising seconds | 25.35 | 40.71 | 56.14 |
| Final source pixels exact | yes | yes | yes |
| Finite final latents and decoded pixels | yes | yes | yes |

Forward counts and token sums were recomputed from every recorded forward, rather than inferring cost from the six outer steps. The complete suite uses 27 NFE. The correction arms have respectively 56.25% and 112.5% more target-token forwards than the control.

The control raw **and** composite PNGs are pixel-for-pixel and SHA256-identical to the previous geometry probe's `edge-context` output. The control composite hash remains `03ad495177a35f9ca8e405fc7b5d5a111c4389ee93230ee30381c9b38dcbe2e8`. Reference-prefix, absolute outer-noise and known-target hashes are equal across the three arms, as are prompt, sigmas, growth schedule, model revision and adapter revision. This rules out accidental baseline drift for this comparison.

## Independent actual image inspection

An independent reviewer opened the saved raw finals, final composite montage, selected-feature overlay, untouched rail closeups, and full-width top/bottom source-boundary crops. The crop markers are outside the image pixels. No warp, feathering, rail drawing or compositing trick was added.

- **edge-control** reproduces the previous short rail elbow, water/vegetation transition at the top source border, and lower asphalt tone/texture transition.
- **think1** removes the control's generated upper sky/horizon-like area and fills the upper extension with water and vegetation. The far visible rail is steeper and visually closer to the source direction, but a short offset/elbow still occurs immediately above the original source. The water and vegetation change visibly across `y=320`. The lower asphalt transition remains apparent. The complete image is not seamless.
- **think2** ends the rail connected to the original source and starts a separate higher rail section with its own post. The gap is present in the raw output before hard compositing, so it is a generated structural failure. Its lower source-border MAE is smaller than the control's despite this clearly worse geometry. Low reconstruction MAE does not imply correct outpainting.

![Saved final composites](boundary_eval/2026-10-05/lanpaint/full.png)

![Actual raw and composite rail join, 5x nearest-neighbor crop](lanpaint_runs/2026-10-05/track3/rail-join-raw-composite.png)

![Actual full-width upper source border, raw above and composite below](lanpaint_runs/2026-10-05/track3/top-boundary-raw-composite.png)

![Actual full-width lower source border, raw above and composite below](lanpaint_runs/2026-10-05/track3/bottom-boundary-raw-composite.png)

## Rail diagnostics and uncertainty

The fixed track3 [boundary evaluator](evaluate_boundary.py) follows the upper/left bright metal ridge. It does not score general image quality. The source fit is unchanged: `dx/dy≈−1.971`, angle `−63.10°`, extrapolated `x≈311.76` at `y=320`. Alternate source windows give approximately 1.16° angle / 1.30px endpoint sensitivity from real source curvature and feature selection.

| Composite diagnostic | edge-control | think1 | think2 |
|---|---:|---:|---|
| Detected rows used in far fit, out of 41 | 41 | 16 | 0 |
| Generated/source far angle difference | +8.03° | +3.33°* | invalid |
| Far line extrapolated endpoint offset at y320 | −10.82px | −7.18px* | invalid |
| Near join median offset from source tangent, rows312–319 | −3.60px | −6.24px | invalid |
| Full-width top adjacent-row / nearby-change ratio | 2.57 | 2.43 | diagnostic only |
| Full-width bottom adjacent-row / nearby-change ratio | 1.73 | 1.77 | diagnostic only |

**The think1 far-fit values are partial, not a reliable whole-range improvement measurement.** Its shifted rail leaves the frozen search corridor at rows270–293; only 16 of the 41 planned far rows are detected, and multiple rows hit the corridor edge. The detected segment and true untouched closeup support a steeper direction locally. They do not prove a correct global rail continuation. Its near-join offset worsens in the same diagnostic, and actual inspection still shows the elbow. These quantities must not be collapsed into a winning scalar.

For think2, zero far rows are detected and the rail correspondence fails. The evaluator returns `null` angle/endpoint/near offset, which is **unknown measurement**, not zero error or successful alignment. Independent viewing separately establishes the disconnected rail failure. A distant rail can legitimately curve, so far angle difference alone is not a failure criterion; the immediate join, feature identity and complete composition remain decisive.

Adjacent-row/texture ratios can also flag valid object edges. They are seam alarms and cannot establish image quality. Raw source MAE concerns decoded source reconstruction; the final composite's exact source pixels are guaranteed by hard paste and do not validate the generated-side join.

- [Root composite diagnostic and selected features](boundary_eval/2026-10-05/lanpaint/REPORT.md).
- [Independent raw and composite diagnostic, metrics and untouched crops](boundary_eval/2026-10-05/lanpaint/independent-raw-composite/REPORT.md).

## Timing and validation

Shared model load was 9.99s and shared conditioning 2.49s; edge-context encoding took 0.34s within conditioning. The model/conditioning phases executed once for the suite. Their values appear in each arm record for context and must not be summed as separate model loads. Core time includes all inner denoiser calls, correction noise/updates, target insertion, and outer Euler work. It excludes preview capture, final/preview VAE decode and loading. These are one Mac M4 Max process/seed measurements with warm-up, execution-order, swap and thermal limitations. They show this probe's additional work, not an iPhone latency prediction or a stable cross-run speed ranking.

The complete suite finished in 158.38s according to its metrics; the bounded launch wrapper recorded 159.18s including wrapper overhead, exit0, under a 900s process-group timeout. The runner hash matches its recorded launch hash. CPU preflight passed equation/NFE, source-context separation, absolute layout and token-insertion checks without loading weights.

Independent CPU artifact validation confirms actual forward counts, inner calls and token sums; matching conditions; exact raw/composite control reproduction; correct final dimensions; exact final source; finite saved predicted-clean latent snapshots; recorded composite hashes; and unchanged original JPEG hashes. Runtime records also assert finite final latents and chosen known-source equality at sigma zero. Existing geometry and sampler outputs are retained.

- [Independent artifact validation](lanpaint_runs/2026-10-05/track3/independent_validation.json).
- [Complete suite metrics](lanpaint_runs/2026-10-05/track3/lanpaint_metrics.json).
- [CPU preflight](lanpaint_runs/2026-10-05/track3/preflight.json).
- [Bounded launch, runner hash and exit status](lanpaint_runs/2026-10-05/track3/launch.json).
- [Experimental runner](run_lanpaint_probe.py).

No new checkpoint, training, cloud job, mobile benchmark or Player integration was performed. This outcome rejects these tested correction settings as a complete fix; it does not establish that every official LanPaint configuration or another checkpoint will fail.
