# Independent review of the 20-step trained structural-control teacher

The tangent-control arm substantially improves the local upper guardrail join, but **all three 20-step arms fail complete-image quality**. The tangent-control output leaves large white regions at both ends. The other arms introduce stacked/repeated road scenes, and mask-only adds fake text. A good near-boundary ridge is not a successful final outpaint.

This review opened real raw/composite PNGs, the source-derived selected-feature overlay, and untouched crops. It did not infer quality from the control maps. The control guide is a structural hypothesis, not source truth. No model output was painted, warped or blended.

## Actual source join and complete image

- `tangent-canny`: the selected yellow feature is the intended upper/left bright metal ridge. The old conspicuous elbow is largely removed. A slight source/generated width/color/position change remains visible; the composite is not mathematically seamless. Its upper extension forms a new shoreline/forest scene. There is a white band above approximately y69 and a broad white region from approximately y868 downwards. Both exist in raw decode. Asphalt continues for roughly 36 pixels after the source ends at y832, then stops at the white region.
- `mask-only`: a relatively continuous local rail is retained with a different tangent, but the whole frame contains repeated road/water panels, horizontal composition seams and fake lettering below the source. Absence of white bands is not success.
- `source-canny`: the local ridge resembles mask-only, while the full frame also contains repeated/stacked road scenes and white upper/lower regions.
- Original `edge-context`: the old elbow is clear in the same source-coordinate raw/composite crop. Its overall layout remains more coherent than these failed teacher outputs. It uses Turbo, an image-reference prefix and actual target growth; comparing it to this teacher is qualitative, **not** a ControlNet-only matched ablation.

![Actual full composite comparison](full-comparison.png)

![Actual raw and composite source-join pixels, nearest-neighbor enlargement](rail-join-raw-composite.png)

![Actual upper source border](upper-source-border-raw-composite.png)

![Actual lower source border and the beginning of the white failure](lower-source-border-raw-composite.png)

## Source-derived near-rail diagnostic

The independent [source-tangent diagnostic](../../../../evaluate_source_tangent.py) uses only original source rows320–340 to define its corridor and local tangent. The source angle is −61.94° with endpoint x310.47 at y320. Each teacher arm has all 16 planned near rows detected, no corridor hits, a clearly separated alternative-path score, and small local fit residuals. The selected features were inspected directly; no false post or parallel lower ridge was identified in the tangent-control primary path.

| Arm | Near angle difference | Endpoint dx at y320 | Last4 median dx | Feature extraction |
|---|---:|---:|---:|---|
| mask-only | +7.58° | −1.10px | −2.51px | extracted |
| source-canny | +7.59° | −1.04px | −2.36px | extracted |
| tangent-canny | −1.15° | −2.52px | −2.02px | extracted |

Tangent-canny's direction is closer to the original tangent, while its fitted position remains about2.5 pixels left of the source ridge. These are local measurements of an identified metal edge, not a full-frame quality score. Real source curvature and edge width/color give additional uncertainty; the remaining transition should be judged at actual pixels.

- [Root diagnostic with selected features](../../../../boundary_eval/2026-10-05/controlnet20-source-tangent/REPORT.md).

## Integrity, computation and visible white-region diagnostic

Independent validation recomputed 20 actual step records per arm, 20 base/control denoiser evaluations, 640 base-block forwards, 320 control-block forwards and 46,080 target-token forwards per arm. Both chains use all 2,304 target tokens on every step, with prefix cache disabled. These teacher outputs do **not** demonstrate center-out compute growth. The teacher uses Q4 base, BF16 trained control, CFG 1, text-only prefix, no Turbo adapter and no optional known-latent bridge.

The resized central 512 source is exact in every final composite. Saved final latents have correct shape, are finite, and match their recorded hashes; composite hashes and source file hash match. Original JPEG hashes remain unchanged. Artifact validation passing establishes integrity, not image-quality acceptance.

As an additional descriptive diagnostic, pixels with all RGB channels≥230 cover 21.56% of tangent-canny's generated upper area and 81.86% of its generated lower area. Source-canny values are 21.25% and 10.10%; mask-only is approximately zero. Brightness alone cannot determine failure: the actual broad white blank regions were viewed directly, while mask-only's different failures are visible despite a low near-white count.

The complete 20-step suite took 447.91s in its own metrics on the Mac M4 Max. Core denoising was 128.74/154.80/140.81s for mask-only/source-canny/tangent-canny. These are single-process/order-dependent Mac measurements; no iPhone or stable speed claim follows.

- [Independent CPU artifact validation](../independent_validation.json).
- [Suite metrics](../controlnet_metrics.json).
- [Bounded launch and runner hash](../launch.json).

This report covers completed 20-step finals only. The 40-step known-bridge extension is a separate experiment and was still running when this review was written. No quality claim is made for its partial state.
