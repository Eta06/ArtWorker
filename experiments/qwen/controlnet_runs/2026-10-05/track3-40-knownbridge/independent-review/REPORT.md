# Independent review: 40-step tangent control with known-latent bridge

The generated upper metal ridge now follows the source's local tangent closely, but the complete image remains **failed for production outpainting**. White outer regions and a visible water/vegetation source-border transition persist. Exact central-source preservation and a small local rail-angle error do not establish a seamless final image.

The reviewer opened actual raw/composite finals, the source-derived feature overlay and raw/composite crops at both source borders. The selected feature is the upper/left bright metal ridge. It has all 16 near rows detected, no corridor hits, and a clearly separated alternative path. The source-derived local diagnostic returns angle difference −0.043°, extrapolated endpoint offset −1.73px at y320, and last-four-row median offset −1.05px. These are local feature measurements; edge width/color, source curvature and pixel sampling add uncertainty. They are not a claim of mathematically perfect geometry.

The old conspicuous generated-side elbow is substantially reduced in the exact-source composite. The raw reconstructed source-border rail still develops a bright short elbow around the first source rows. Pasting the original source restores the inner rail, leaving a smaller position/width/color change at the join. The water and vegetation also change visibly across y320. Broad white regions exist in raw at the top and bottom, before any final source paste.

Pixels with every RGB channel at least 230 occupy 23.38% of the generated upper region and 89.06% of the generated lower region. Rows with at least 90% such pixels span y0–73 and y867–1151. This describes the directly viewed blank regions; brightness alone is not a universal failure detector.

![Actual full composite comparison](full-comparison.png)

![Actual raw/composite rail join compared with the old edge-context output](rail-join-raw-composite.png)

![Actual upper source border, showing remaining water/context seam](upper-source-border-raw-composite.png)

![Actual lower source border and white outer failure](lower-source-border-raw-composite.png)

## Integrity and comparison limits

Independent CPU validation recomputed 40 successful step records, 40 actual base/control evaluations, 1,280 base-block forwards, 640 control-block forwards and 92,160 target-token forwards. All 2,304 target tokens are processed on every step in both chains. This full teacher has no growing-compute demonstration. The resized original 512-square source is exact in the final composite; saved latent shape, finiteness, latent/composite hashes, source file hash and unchanged JPEG hashes passed independently. The runner's optional known-bridge assertion checks known source latents at sigma zero; the CPU artifact validation did not re-encode the VAE to repeat that assertion.

Core denoising was 267.80s; the complete suite took 287.93s on this Mac. These are single-process Mac measurements, not iPhone latency or stable model rankings.

The 20-step tangent teacher used no known-latent bridge. This run changes **both** step count and the bridge, so improvement cannot be attributed solely to either change. Raw source top16/bottom16 MAE increases from 5.08/3.63 in the 20-step tangent arm to 9.31/7.40 here, despite the improved outer tangent. This is another reason not to promote a result using one metric. The comparison with the original edge-context output also changes adapter, conditioning and compute mode; it is visual evidence, not a matched ControlNet-only ablation.

- [Source-derived local rail diagnostic](../../../../boundary_eval/2026-10-05/controlnet40-bridge-source-tangent/REPORT.md).
- [Independent artifact validation](../independent_validation.json).
- [Complete suite metrics](../controlnet_metrics.json).
- [Bounded launch](../launch.json).

The separate faithful 40-step run without a known bridge is not part of this completed review. No quality claim is inferred from its pending state or from a control guide map. No generated pixels were painted, warped or feathered.
