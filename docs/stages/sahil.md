# Sahil — source preservation independent of placement

2026-10-11. The user requested aligned source-preservation scores and a prompt
recommendation directly in chat, rather than a website gallery. Source position
inside the output is not a selection criterion. No new generation was allowed
or performed.

## Method and coverage

Re-measured all **88 existing outputs** from the three active models only:
GPT Image 2.5 Sunburst, GPT Image 2.5 Flare and Nano Banana 2.1. These active
outputs cover **8 exact prompts**; the ninth historical prompt was tested only
with excluded models. Original receipt, reference and output hashes were
verified. Originals were not edited, composited or regenerated.

The existing detector uses ORB with SIFT fallback and a single global similarity
transform. Free placement has no center-shift penalty. No local deformation or
color correction is applied. Matching still requires at least 12 inliers, a
0.4 inlier fraction, 0.15 source hull coverage and three quadrants. Free-search
bounds remain 0.25–2× expected source scale and ±10° rotation. It cannot certify
an arbitrary transformation outside those bounds.

**77/88** source regions were reliably registered. The other **11** receive no
aligned preservation score, not a fabricated zero. A conservative whole-source
ranking requires ≥99.5% visible source; **61** outputs pass. The other 16
registered outputs retain overlap metrics and visibility fractions, but are
not included in that ranking. This strict coverage gate can exclude slight
edge loss/interpolation as well as actual cropping.

All metrics use the same 512px source grid. SSIM ×100 is structural similarity,
**not a percentage of unchanged pixels**. Within-8 is the percentage of aligned
pixels whose RGB channels each differ by no more than 8/255. Resampling affects
pixel differences; neither measure proves byte-identical source preservation.
Each source's resampling-only control is retained separately.

## Supported recommendation

**Keep the earlier `old-canvas` Sunburst high output/prompt as the baseline.**
That request sends the canvas and original square as two references. It wins
the matched Sunburst high comparison on Karambol and Gripin, and the matched
Sunburst/Flare high comparison on Karambol. This is a recommendation supported
by the available covers, not a statistically established universal winner.
Other prompts have fewer covers and repetitions. Missing cells stay missing;
unequal request counts are not pooled into a misleading global average.

| Prompt | Sunburst high Karambol SSIM ×100 | Gripin SSIM ×100 |
| --- | ---: | ---: |
| old-canvas | 90.79 | 85.18 |
| anchored | 84.10 | 75.62 |
| edge-continuity | 82.21 | 76.34 |
| copy-square | 85.39 | 73.30* |
| canvas-only | 83.17 | 66.97 |
| verbatim | 79.56 | 65.78 |
| short | 64.18 | 72.89 |
| free-layout | 57.83 | 61.93 |

*copy-square Gripin has 96.20% visible source and fails the strict whole-source
gate. Its 73.30 score concerns visible overlap only. It is not included in the
paired whole-source ranking. `old-canvas` averages 87.99 on the matched
Sunburst two-cover comparison, versus 79.86 anchored and 68.54 short.

With `old-canvas`, Sunburst high scores 90.79 for Karambol and 83.47 for Aklın
Hep Bende; Flare high scores 86.09 and 65.44. Nano 1K scores 88.23 and 85.06 on
visible overlap, with 99.38% and 99.36% visibility. These Nano cases are reliable
registrations but are excluded from the conservative whole-source ranking.
Routes differ between old OpenRouter Nano and newer direct Google anchored
attempts, so their comparison is not a controlled prompt-only experiment.

Sunburst high `old-canvas` within-8 pixel fractions are 64.87% Karambol,
69.11% Aklın Hep Bende and 62.46% Gripin. Therefore a 90.79 SSIM value must not
be described as 90.79% of pixels unchanged.

## Artifacts and validation

The offline `aligned_prompt_audit.py` produces orange detected-region outlines,
aligned crops, 50/50 overlays, fixed-scale difference maps, private detailed
JSON, and an 88-row CSV. A static Sunburst eight-prompt/two-cover comparison
sheet was inspected directly and is delivered in chat. There is no new web
report. Private diagnostic images and per-output artifacts remain under the
ignored `results/2026-10-11/sahil/` directory; an anonymous aggregate summary
is versioned as `aligned-prompt-summary.json`.

The serial audit completed in 33.09 seconds with sampled peak RSS **464.69 MiB**
and physical footprint **427.42 MiB**, under a 1 GiB cap and memory-pressure
monitoring. Original receipt and output hashes were rechecked after the run.
All 35 existing and new offline unit tests passed, including asymmetric
placement controls, rejection of unrelated/partial features, equal cover
weight despite repeated requests and missing-condition exclusion. Provider
calls in the test suite are mocks. New provider requests and generation cost
in this stage are both zero.

The rejected wall-cut anchored Karambol example remains rejected on visual
outpaint quality; source preservation measurements do not reverse that choice.
No student training or checkpoint release happened.
