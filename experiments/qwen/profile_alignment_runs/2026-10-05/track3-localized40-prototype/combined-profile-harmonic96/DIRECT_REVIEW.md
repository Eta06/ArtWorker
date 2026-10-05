# Direct review of combined profile and harmonic RGB

Inspected actual native 110×46 joins, enlarged rail crops, the same full-width top crop, and all full outputs. Combining the profile warp with the unchanged original-decoder harmonic residual reduces the rail's positional discontinuity and improves the tone transition around the rail. A clear horizontal water/texture seam remains across the original source boundary. A warmer vegetation correction is visible above the source, and the metal texture still changes. Nearest sampling still shows stair steps; linear sampling compresses/interpolates texture. None of these outputs is accepted as seamless or perfect.

Original generator raw, original JPEG, shared source PNG and all frozen helper/evaluator hashes remain unchanged. Labeled derived-raw intermediates retain the original decoded known square solely for residual computation. Their exteriors match the saved profile variants exactly; final source and far regions remain byte exact. Final correction adds RGB values, including in the nearest-based variant; the no-new-RGB guarantee applies only to its preceding geometry stage.

| Stage | top adjacent-row RGB MAE |
|---|---:|
| original | 16.088 |
| harmonic only | 11.109 |
| linear profile64 alone | 17.127 |
| nearest64 alone | 17.102 |
| linear profile64 + harmonic96 | 12.417 |
| nearest64 + harmonic96 | 12.399 |

The unchanged harmonic stage's high-frequency power ratio relative to the warped input is 1.00174 for the rail patch with linear sampling, and 1.00043 with nearest sampling; no clipping occurs. Geometric sampling itself changes the patch more substantially: high-frequency power ratios relative to original are 1.06569 and 1.80516, respectively. The latter reflects visible aliasing/stair steps rather than improved grain quality. The FFT includes object edges and is a diagnostic, not proof of natural grain.

The fixed source-derived evaluator near endpoint is +0.685 px for the linear combined variant and +0.540 px for nearest combined, against −17.404 px original. Near-angle differences are +3.977° and +4.291°, against +0.300° original. The wider evaluator still reports ~8° far-angle difference. The held-out highlight remains two invalid rows, one absent final row, and only one valid observation. Color correction does not restore that missing identity.

[Detailed metrics and hashes](metrics.json) · [Fixed source-derived evaluator](frozen-source-tangent-eval/REPORT.md) · [Fixed wider evaluator](frozen-boundary-eval/REPORT.md) · [Actual rail comparison](rail-join-comparison.png) · [Actual full-width join](top-fullwidth-join-comparison.png)

Independent audit reproduced both combined images and all spectral ratios exactly. The top residual cap clamps 4.6875% of additive residual channels (unclamped range −27.372 to +27.546), separately from the zero final-output clipping. Enlarged crops retain halo-like shading, an altered inner highlight/short kink, and a bottom grain transition. Quality remains inconclusive.
