# Independent completed sparse-control review

The supported structural-control gate removes the previous white outer blanks in this matched seed, but the expanded image still fails as a coherent outpaint. The generated top and bottom contain repeated road, rail and water scenes separated into horizontal panels. The central source and local rail are retained; complete-image quality is not accepted.

The reviewer opened saved raw/composite finals, the actual full before/after montage, raw/composite rail and border crops, and the source-derived feature overlay. The guide/support masks were used only for input-integrity checks, not as evidence of generated quality. No output pixels were painted, warped or blended.

![Actual full outputs: full structural control and supported structural control](full-control-comparison.png)

![Actual raw/composite rail join](rail-raw-composite.png)

![Actual upper source border](upper-source-border-raw-composite.png)

![Actual lower source and outer transition](lower-source-and-outer-transition.png)

## Actual geometry and composition

The selected feature in the sparse output is the intended upper/left metal ridge; all 16 near rows are detected without corridor hits. The source-derived fit gives angle difference +0.918°, endpoint offset −1.196px at y320, and last-four-row median offset −0.874px. The full-control bridge comparator gives −0.043°, −1.727px and −1.052px. Both are good local correspondence measurements relative to the early elbow, with real source curvature and edge-width/color uncertainty. Neither establishes a seamless complete image.

The sparse raw decode still has a short bright reconstruction elbow inside the first source rows. Exact-source compositing restores the inner rail, leaving a small width/color transition and a visible water/vegetation boundary. Farther outward, source-adjacent asphalt stops and a separate road/water panel begins. The upper generated area also repeats a road/water/rail view with a horizontal transition. These failures are present in saved generated pixels.

Near-white pixels, defined as every RGB channel at least 230, fall from 23.38% upper / 89.06% lower in the full-control comparator to 0% / 0% in the sparse output. Direct viewing confirms the white blank regions disappeared. It also demonstrates why a blank-region metric cannot promote image quality: those regions were replaced by incompatible repeated scene panels.

## Matched input change and source-channel preservation

The sparse run matches the full40 known-bridge comparator across 23 recorded configuration/input fields, including model/control revisions, quantization, prompt, source, noise, sigma nodes, CFG, no adapter, text-only prefix, disabled cache, full-canvas compute, bridge setting, control-map hash and the complete **pre-gate** 129-channel encoded context hash. The isolated runner is different code; its declared sampling/input change is the supported structural gate, not a new checkpoint or new prompt.

The gate retains the first 64 structural latent channels on the known ROI and within 64 image pixels of the input-derived unknown-region guide. It sets those first 64 channels to literal zero elsewhere. The final 65 mask/source channels remain unchanged. The stored support contains 1,121 retained tokens: all 1,024 source tokens plus 97 guided unknown tokens. Structural fields at the other 1,183 target tokens are zeroed. This is input-conditioning dropout; all 2,304 denoiser target tokens still run in both chains.

Independent CPU checks verified stored support image/packed-token shapes, absolute nearest packing, source-ROI coverage, counts and hashes. Separate float32 and BF16 gate checks confirm byte-exact final65 preservation, exact retained first64 values, literal zero unsupported first64 values and an unchanged input tensor. No weights or GPU were used by those checks.

The actual GPU run asserts source65 equality and supported/unsupported structural invariants against its full-map context before gating. Its pre-gate context hash matches the previous full-control run exactly. Full baseline/gated context tensors were not persisted, so independent CPU validation cannot re-encode the actual VAE fields; it distinguishes the verified gate contract from the recorded actual-runtime assertions.

This matched pair supports the unsupported structural field as a contributor to the tested white-blank behavior. It does not show that sparsity solves global composition or generalizes to another source/seed.

## Artifact validation and compute

Independent validation passed 40 actual step records, 1,280 base-block forwards, 640 control-block forwards, 92,160 target-token forwards, finite saved latents, correct shapes, recorded latent/composite hashes, exact resized512 source in the final composite and unchanged original JPEG hashes. Raw source top16/bottom16 MAE is 9.74/8.20, versus 9.31/7.40 in the full-control bridge comparator. Reconstruction MAE is separate from local geometry and full-scene quality.

Core denoising was 269.67s and the suite completed in 289.16s on the Mac. Zeroing structural inputs does not omit transformer work; this result is not a compute-growth or iPhone-performance demonstration.

- [Matched conditions, support integrity and independent CPU gate checks](comparison_validation.json).
- [Independent sparse artifact validation](../../../controlnet_runs/2026-10-05/track3-sparse40-knownbridge/independent_validation.json).
- [Fresh source-derived metric and selected features](REPORT.md).
- [Completed sparse suite metrics](../../../controlnet_runs/2026-10-05/track3-sparse40-knownbridge/controlnet_metrics.json).
- [Bounded sparse launch](../../../controlnet_runs/2026-10-05/track3-sparse40-knownbridge/launch.json).

The separate reference-prefix/outpaint-prompt combination is pending and changes several factors. This review does not infer its quality from partial files or describe it as a single-factor control.
