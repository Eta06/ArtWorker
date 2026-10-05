# Completed teacher comparison: 20 steps, 40 steps, and 40 steps with bridge

All three tangent-control teacher outputs still fail complete-image quality. Increasing the faithful teacher from 20 to 40 steps does not fill the white outer regions. Adding the known-latent bridge at 40 steps improves the local guardrail direction further, while the white regions and source/context seam persist. No result is promoted to a seamless outpaint.

The reviewer independently opened completed faithful40 raw/composite finals, the three-way full comparison, raw/composite source-border crops and selected-feature overlay. Every primary path follows the intended upper/left bright metal ridge; all 16 planned near rows are extracted without corridor hits. The overlay was inspected, rather than accepting a numeric match alone.

![Actual completed full composites](full-teacher-comparison.png)

![Actual raw and composite source-join pixels](teacher-rail-raw-composite.png)

![Actual upper source/context border](teacher-upper-border-raw-composite.png)

![Actual lower source border and white failure](teacher-lower-border-raw-composite.png)

## Local and global observations

| Measurement | teacher20 | teacher40 | teacher40 bridge |
|---|---:|---:|---:|
| Known-latent bridge | no | no | yes |
| Near source-tangent angle difference | −1.15° | −1.21° | −0.04° |
| Fitted endpoint dx at y320 | −2.52px | −2.52px | −1.73px |
| Last4 source-tangent median dx | −2.02px | −1.90px | −1.05px |
| Raw source top16 MAE, 0–255 | 5.08 | 5.28 | 9.31 |
| Raw source bottom16 MAE, 0–255 | 3.63 | 3.75 | 7.40 |
| Near-white generated upper pixels | 21.56% | 23.35% | 23.38% |
| Near-white generated lower pixels | 81.86% | 89.04% | 89.06% |
| Actual denoiser evaluations | 20 | 40 | 40 |
| Target-token forwards | 46,080 | 92,160 | 92,160 |

Near-white means every RGB channel is at least 230. It describes the directly viewed blank regions, not a universal image-quality test. Both 40-step outputs have rows y0–73 and y867–1151 with at least 90% near-white pixels, already present in raw decode. The lower generated asphalt extends briefly past source y832 before stopping at the white region. The upper extension invents a shoreline/forest composition; the water and vegetation meet the original source with a visible change at y320.

Faithful40 keeps a smoother raw source-border reconstruction than the bridged run. The bridge brings the generated-side ridge closer to the original tangent but causes a bright short elbow in the raw first source rows; hard compositing restores the original inner rail. Neither local angle improvement nor exact final source pixels removes the surrounding water/context seam or white outer failure. Real source curvature, edge width/color and pixel sampling also limit precision; the local fit is not proof of perfect geometry.

## Matched conditions and independent validation

The faithful40 and bridge40 runs match model/control revisions, quantization, prompt, source, original noise, all sigma nodes, text-prefix length, control context, image size, CFG, no adapter, disabled prefix cache, full-canvas compute and runner hash. The only sampler switch is the known-latent bridge. Their generated outer raw-pixel MAE against each other is 1.87/255: the outputs are close, not byte-identical. The matched comparison establishes that the tested white failure persists both with and without the bridge; it cannot establish a cause beyond these tested conditions.

Teacher20 and faithful40 use the same trained control/source configuration without a bridge, with different step counts/sigma grids. More steps do not solve the white failure in this pair. This is a bounded experiment outcome, not a statement that every ControlNet configuration will fail.

Faithful40 independent CPU validation confirms 40 completed records, 1,280 base-block forwards, 640 control-block forwards, 92,160 target-token forwards, correct latent/image shapes, finite saved latents, recorded latent/composite hashes, exact final resized source and unchanged original JPEG hashes. Both chains process the full 2,304-token target every step; this is not center-out compute growth. Core denoising took 253.79s and the faithful40 suite 273.83s on the Mac, with single-process timing limitations and no iPhone claim.

- [Independent faithful40 validation](../../../controlnet_runs/2026-10-05/track3-40/independent_validation.json).
- [Matched comparison conditions and raw differences](comparison_validation.json).
- [Fresh source-derived diagnostic metrics](metrics.json).
- [Source-derived diagnostic report and selected features](REPORT.md).
- [Faithful40 suite metrics](../../../controlnet_runs/2026-10-05/track3-40/controlnet_metrics.json).

Previous experiment metrics and outputs remain unchanged. The new sparse supported-control experiment is still a separate pending run; this report makes no claim about its guide map, partial latents or future final quality.
