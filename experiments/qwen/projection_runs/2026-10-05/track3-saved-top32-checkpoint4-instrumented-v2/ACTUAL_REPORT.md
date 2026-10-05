# Four-update checkpointed VAE projection: completed, quality unaccepted

Parent ran two independent four-update arms from the exact saved-top32 latent input. Internal runtime was34.867s; the external supervisor reported36.049s. Only the frozen F32 VAE was loaded (1,350,961,616 parameter bytes); there were zero transformer calls and no model-weight training. The observed MLX peak was9,905,994,600 bytes (9.226GiB), including measured baseline/load and both arm/final-decode scopes. This is a current-run measurement, not a matched numerical checkpoint speedup or memory-reduction claim.

[Independent state/pixel audit](independent_state_pixel_integrity.json) passes. Both latest atomic states contain iteration4 and finite F32 delta/moment/variance with correct input/source/runner/config provenance. Saved projected tensors equal original base plus saved delta. Outside-collar latent entries remain exact; original source pixels are exact in final composites and exterior pixels equal each new raw. Input NPZ, original model raw and original JPEG remain unchanged. Differentiable/reference decoder error and copied-baseline pixel error are both zero.

Distant decoded pixels are not bit-exact because the VAE couples spatial regions. At least128px away from the source borders, consistency changes1.6815% of channels and derivative1.4038%, each by at most1/255; MAE is0.01682/0.01404. Maximum latent delta is about0.0601, below the0.2 cap. OS thermal-warning text was recorded; no physical Celsius sensor temperature was measured.

| metric | baseline | consistency4 | derivative4 |
|---|---:|---:|---:|
| raw source top16 MAE | 8.857 | 5.041 | 7.571 |
| raw source bottom16 MAE | 6.986 | 4.054 | 5.397 |
| composited one-pixel top derivative MAE | 22.048 | 17.936 | 8.137 |
| composited one-pixel bottom derivative MAE | 7.497 | 6.773 | 3.646 |

Pre-update losses decrease through all four steps: consistency0.004794→0.002130; derivative0.030504→0.009441. These are optimization/reconstruction measurements, not semantic-quality scores. Raw inner-source MAE increases slightly, while the final pasted source remains exact.

## Direct quality evidence

[Frozen source-tangent review](frozen-source-tangent-eval/QUALITY_REVIEW.md) uses unchanged thresholds and directly inspected selected features. Consistency4 is invalid because it loses strong near-join edge observations at y318/y319. Derivative4 remains valid and reduces the final-row offset from+3.930px to−1.993px; near angle changes+0.341°→+0.177°. However its endpoint and last-four median worsen slightly, and a local bend remains. These findings do not establish a repaired rail.

Actual full images and joins were viewed. The one-pixel boundary contrast decreases, but the horizontal water/vegetation/road join, ripple-scale difference and bottom asphalt grain transition remain visible. Derivative high-frequency power versus baseline rises about10.3% in the generated-water patch and13.5% in generated asphalt; raw known asphalt power is13.8% above the original-source patch. FFT patches include object edges, so these are descriptive changes rather than proof of grain preservation.

![Actual full composites](final-comparison.png)

![Actual nearest-enlarged rail joins](rail-join-4x.png)

![Actual raw water join](frozen-source-tangent-eval/raw-water-join-3x.png)

## Failure history and next diagnostic

The [first launch](../track3-saved-top32-checkpoint4-instrumented/launch.json) failed before model loading because supervisor files made the output directory nonempty; it remains failed and preserved. The earlier [40.60GiB projection attempt](../../../PROJECTION_RESULTS.md) remains a completed consistency arm plus an incomplete timed-out derivative arm. Different latent input and four updates prevent a matched speed/memory attribution to checkpointing.

A separate16-update candidate is justified by decreasing losses and unused delta budget, primarily to test derivative continuation. Consistency remains a control that may further weaken rail contrast. It starts from the same original latent input and zero Adam state, preserves an iteration4 snapshot for actual comparison with this run, and then performs12 additional updates. It is latent optimization, not a new model or training result. Sixteen-update quality and runtime are unknown until executed.
