# Localized completed-hint experiment: independent review

Localized40 produces one coherent portrait scene and removes the repeated road/lake panels seen in reference40. It still fails the source join: the generated guardrail has an extra upper metal band and a displaced outer edge. The top water/vegetation transition and lower asphalt transition remain visible. This is a useful composition result, not a seamless outpaint or accepted rail geometry.

This review uses completed saved raw/composite images, actual saved conditioning arrays, a fixed source-derived feature diagnostic, and CPU postprocessing artifacts. No model weights were loaded and no GPU inference or Player change was performed.

## Direct image evidence

![Full completed composites](full-comparison.png)

Columns are the earliest edge-context prototype, reference40 with only prefix hints disabled, localized40, and a color-only compositor applied to localized40. The earliest prototype differs in model adapter, conditioning, sampler and growing target compute; it is a qualitative reference and does not isolate a ControlNet effect.

Reference40 repeats a rail/road panel above the source and a new lake/road panel below it. Localized40 instead continues water/vegetation at the top and asphalt at the bottom. No white end margin or extra full road/lake panel is present in localized40. The original person, car and source artwork remain unchanged in the final composite. Upper water texture/color and lower asphalt grain still change at the original square's borders.

![Unmodified true rail crops: raw above, composite below](rail-raw-composite.png)

The generated rail in localized40 is visibly broader above y320. Its upper outer ridge reaches the join to the left of the source rail, leaving a projecting metal band. This is present in the model's raw pixels. Replacing the central square with the exact source removes the decoded inner-source elbow but cannot remove that exterior band.

![Actual upper source border](upper-source-border-raw-composite.png)

![Actual lower source border and nearby exterior](lower-source-border-raw-composite.png)

Crops use nearest-neighbor enlargement where applicable. Yellow border marks sit outside the image and identify y320 or y832; they do not paint the rail. Raw images include VAE reconstruction changes inside the source. Final composites are source-exact at the resized 512 × 512 evaluation source, not a claim of retaining original JPEG-resolution pixels.

## Which feature was measured

The source-only corridor and thresholds are unchanged. The original local outer ridge is fitted using source rows y320–340; generated near rows are y304–319. No guide or output was used to move the corridor.

| Final composite | Feature status | Near tangent difference | Extrapolated horizontal endpoint difference | Last four rows, horizontal offset |
|---|---|---:|---:|---:|
| reference40 | extracted | −0.284° | −1.627 px | −0.813 px |
| localized40 | extracted | +0.300° | −17.404 px | −17.591 px |
| localized40, color only | invalid feature | unavailable | unavailable | unavailable |

![Original pixels used by the diagnostic](source-tangent-untouched-crops.png)

![Feature identity diagnostic](source-tangent-selected-features.png)

Yellow selects the generated outer bright metal ridge; green selects the original outer ridge. In localized40, the generated path is nearly parallel to the source but shifted. The fit extrapolates x293.063 at y320 versus original x310.468. The actual last generated row y319 selects x296.179 versus the source tangent's x312.344. Thus the small angle error does not establish a matching endpoint or rail width. The −17.4 px is a horizontal offset, not a claimed normal-direction rail thickness; the corresponding normal separation of these parallel local lines is approximately 8.2 px.

Purple is a separately traced interior ridge. It is closer to the source-derived corridor, but it is not the source's outer-edge identity and is not detected as a strong edge on y317–319. It cannot justify accepting the extra outer band. The alternative's score is also clearly lower, so the existing evaluator correctly reports the outer ridge as extracted. Extraction remains separate from quality: every output retains `quality_pass: false`.

Color-only correction alters eligibility in the fixed metal detector: the outer ridge's ahead-pixel luminance falls below the 65 threshold at y318, although its horizontal gradient remains strong. Path selection then prefers the interior ridge, which has missing detections on y317–319. It becomes `invalid-feature`, rather than receiving a favorable geometry score. Direct pixels still show the wider rail. This is a detector limitation plus a visible failed join, not evidence that the rail vanished or became correct.

Actual per-pixel eligibility samples and final selected rows are saved in [color_detector_sensitivity.json](color_detector_sensitivity.json), without changing the diagnostic.

The full diagnostic records source-fit sensitivity, all per-row candidates, selected/alternative paths and uncertainty in [metrics.json](metrics.json) and [REPORT.md](REPORT.md). The old generated-corridor evaluator was not retuned.

## Completed-hint localization and actual computation

The comparison retains the same actual-square source prefix, prompt embeddings and image slots, initial target noise, encoded sparse 129-channel target context, zero-padded reference context, structural support and sigma schedule. These shared saved arrays are independently compared, not inferred from a guide image. Source conditioning's final 65 channels remain the same.

The intended forward change is the completed ControlNet hint scalar field. Reference40 already zeroes every text/reference prefix hint while retaining all target hints. Localized40 continues to zero that full prefix, also zeroes known-source target hints and all target hints outside the unknown guide neighborhood, and gives the remaining guide-local unknown hints a 64 px cosine taper. It does not mask base/control hidden states or the control chain before hints are computed. Zero direct known-source hints do not freeze those hidden states against attention effects.

The saved effective field has 97 positive target positions out of 2304. All 1197 text/reference prefix positions and all 1024 known-source target positions are zero. Desired float32 weights, their nearest 16 px packed coordinates, actual BF16-rounded effective values and full joint gate are retained as saved arrays. The gate is based only on input-guide distance. The separate integrity validator checks its geometry, rounding and hashes; this does not establish that the guide yields correct generated geometry.

Each completed run still uses 40 real transformer calls, 1280 base block calls and 640 control block calls. Each call forwards 2304 target image tokens, 1024 reference image tokens and 173 text tokens: 3501 joint queries. Totals are 92,160 target token forwards, 40,960 reference image token forwards and 140,040 joint query forwards. Hint sparsity is not compute sparsity. This is full-canvas teacher inference, not streamed spatial growth or iPhone performance evidence.

Reference40 core denoising took 403.856 s; localized40 took 415.031 s. Completed `launch.json` wrapper elapsed times are 425.984 s and 435.199 s respectively. These are individual current-machine timings, not a speed benchmark.

The separate CPU validator passed 216 integrity checks with zero failures. All seven shared array payloads and their three NPZ containers are exact matches against reference40; desired cosine/nearest-coordinate weights and effective BF16 rounding reproduce exactly. See [independent_validation.json](independent_validation.json) for saved-array, image, source and count checks. Baseline scale 1.0 is verified from the current hash-matched runner call; its top-level metrics schema does not contain that field. Equality of the prefix in each actual GPU model input is a checked recorded runtime assertion; independent CPU inspection verifies saved prefix data and assembly, not a replay of all model calls. Real per-layer completed hint tensors were not saved. Current code hashes and pinned tiny CPU tests support the gate implementation; the saved effective field independently verifies where it is meant to act.

## Generic compositor results

![Actual untouched postprocess rail crops](compositor-rail-untouched.png)

The color-only harmonic correction reduces the visible upper tone mismatch but leaves the extra rail band. DIS flow plus color, tangent-extrapolated DIS plus color, and Farneback plus color also leave a clear width/endpoint discontinuity. None makes this result seamless. DIS variants visibly introduce a small boundary hook; applying a generic small flow cannot remove the large generated outer band while preserving the exact source.

| Compositor variant | Top adjacent-row luminance change, 0–255 | Bottom adjacent-row luminance change, 0–255 |
|---|---:|---:|
| baseline | 15.084 | 5.553 |
| color | 11.048 | 5.318 |
| DIS + color | 14.704 | 5.520 |
| DIS tangent + color | 14.969 | 5.526 |
| Farneback + color | 11.642 | 5.185 |

These describe photometric row changes, not scene quality, rail correspondence or acceptance. Every saved compositor variant independently retains exact source pixels and exact raw exterior pixels farther than its 96 px boundary strip. Input raw/source/JPEG hashes remain unchanged. This verification is in [compositor_independent_validation.json](compositor_independent_validation.json).

![Compositor upper border](compositor-upper-border-untouched.png)

![Compositor lower border](compositor-lower-border-untouched.png)

The next geometry acceptance must inspect the source-adjacent outer ridge, endpoint and complete metal band together. Matching one tangent or a nearer interior ridge is insufficient. Far curvature can remain natural; the decisive failure here is immediately at the source border.
