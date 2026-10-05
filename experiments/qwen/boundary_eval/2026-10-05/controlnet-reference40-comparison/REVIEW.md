# Independent review of the reference-prefix outpaint combination

Adding the real cover reference and stronger outpaint instructions does not solve full-image composition in this completed run. The upper and lower regions still contain repeated rail/road/water panels with horizontal transitions. The local source-adjacent metal ridge remains well aligned, but the complete outpaint is rejected.

The reviewer opened actual raw/composite finals, the selected-feature overlay and real before/after crops. The primary path is the intended upper/left bright metal ridge, with all 16 near rows detected and no corridor hits. It measures −0.284° local tangent difference, −1.627px extrapolated endpoint offset and −0.813px last-four-row offset. Native sparse40 values are +0.918°, −1.196px and −0.874px. Small local fit differences do not establish a winning complete image, and source curvature/edge sampling limit precision.

Both runs avoid white blanks, but both fail global scene continuity. The reference combination changes the repeated panels' viewpoint and texture without removing them. The source-adjacent raw decoded rail still has a short bright reconstruction elbow inside the known region. Hard compositing restores the original inner rail; the water/vegetation source transition and outer scene panels remain visible.

![Actual complete before/after composites](full-reference-comparison.png)

![Actual raw/composite rail join](rail-raw-composite.png)

![Actual upper source border](upper-source-border-raw-composite.png)

![Actual lower source and outer transition](lower-source-and-outer-transition.png)

## Exact source/reference input and real computation

Independent CPU validation reads persisted real conditioning arrays. The saved reference is the actual 512-square RGB source with constant opaque alpha. Its 1,024 packed reference latents are finite and match the recorded original source-prefix hash `bffd8c1053b8bf45abd7447a565ce6a8caf924508f75315319da0ec3625a7d01`. Saved prompt embeddings and image-slot hashes also match; there are 256 four-token image slots.

Each forward contains 2,304 target image tokens, 1,024 source-reference image tokens and 173 text tokens: 3,328 packed image rows and 3,501 joint attention-query rows. Forty recorded forwards therefore total **92,160 target tokens + 40,960 reference image tokens + 6,920 text rows = 140,040 joint queries**. The base/control block totals are 1,280 / 640. Both streams process the full canvas and reference every step, with no prefix cache; this is not growing-compute streaming.

The saved padded control context has 1,024 literal-zero129 reference image rows followed by the exact saved 2,304 target rows. There are no text rows in that image-context padding. The known mask is exactly the source target rows20–51, and the reference rows have no known-target mask. Saved final target source latents independently match the saved source-context fields at sigma zero. The final source pixels, latent/image shapes, finiteness, hashes and original JPEG preservation also pass.

Literal-zero reference context does **not** guarantee zero completed prefix hints: biased projections and the control chain can still produce them. This experiment separately gates completed hints on the entire text/image prefix after all hints are calculated. The source-reference inputs remain untouched. Existing pinned tiny CPU tests cover exact hint gating and unchanged base-prefix mechanics; actual per-layer real-model hints were not persisted. Source-prefix equality at every real forward is a recorded runtime check, while saved source bytes/hash and exact CPU input concatenation were independently verified.

## Combined changes and limits

Against native sparse40, 21 input/configuration checks match, including model/control revisions, noise, target sigma grid, sparse target-context hash, map hash, source, quantization, CFG, bridge and cache settings. Saved actual source65 fields also match the native sparse run's recorded hash. However, this is an explicitly **combined** experiment:

1. Actual-square prompt vision and an extra 1,024-token image-reference prefix.
2. Stronger explicit outpaint prompt, increasing text-prefix rows from 96 to 173.
3. Completed full-prefix hint gating, including both text and reference positions.

The extra reference also changes the joint layout and attention work. This run cannot isolate which combined change caused a local or semantic difference. It rejects this completed combination as a global fix, not every possible reference/control method.

Core denoising was 403.86s; the suite metrics report 425.00s and the launch wrapper 425.98s including overhead. These are single-process Mac measurements, not stable model-speed or iPhone claims.

- [Independent real reference/target artifact validation](../../../controlnet_runs/2026-10-05/track3-reference40-outpaint/independent_validation.json).
- [Matching inputs and explicitly combined changes](comparison_validation.json).
- [Fresh source-derived metric and selected features](REPORT.md).
- [Completed reference suite metrics](../../../controlnet_runs/2026-10-05/track3-reference40-outpaint/reference_control_metrics.json).
- [Bounded launch](../../../controlnet_runs/2026-10-05/track3-reference40-outpaint/launch.json).

The separate localized-hint experiment is pending. This report makes no quality claim about its partial state, proposed hint mask or a still-downloading native Fill model. Previous experiment metrics and outputs remain unchanged.
