# Yakamoz — Flare and Nano diagnostic sheets

2026-10-11. The user accepted continuing from the earlier old-canvas baseline
and requested the same inline source-preservation visuals for Flare and Nano
Banana that had been delivered for Sunburst. No new inference was performed.

Reused the existing Sahil alignment metrics and diagnostic crops. Verified the
raw image hashes for all existing Flare/Nano outputs before drawing. Generalized
`comparison_image` to accept a model, cover pair and explicit prompt/settings
variants. Each measured example has an orange detected-source outline, aligned
crop and fixed-scale difference heatmap. An unregistered example shows its raw
image and a clear missing-score label, without an unverified orange outline.
An untried model/prompt/cover combination remains labelled untried.

The Flare high sheet contains the eight previously tested prompt variants for
Karambol and Gripin. Karambol is reliably aligned for old-canvas (SSIM×100
86.09), anchored (81.50), copy-square (79.18) and free-layout (56.63). Other
Karambol candidates cannot be scored by the existing reliable detector. Gripin
old-canvas scores 79.68, versus anchored 71.69. Copy-square Gripin was not tried.

The Nano Banana 2.1 sheet shows the two successfully tested exact prompts,
with old-canvas's three resolution settings. It does not imply Nano received
all eight Flare prompt candidates.

| Nano variant | Karambol SSIM ×100 | Aklın Hep Bende SSIM ×100 |
| --- | ---: | ---: |
| old-canvas 1K | 88.23 | 85.06 |
| old-canvas 2K | 88.31 | 84.91 |
| old-canvas 4K | 61.87 | 81.38 |
| anchored 1K | 67.70 | 84.12 |

Nano source visibility percentages are shown explicitly on the sheet when
below Sahil's conservative 99.5% whole-source ranking gate. These are reliable
overlap scores, not certificates of complete source visibility. SSIM is not
the percentage of unchanged pixels. Position is not penalized. Old Nano
requests used OpenRouter, whereas anchored used direct Google, so the route
change is another limitation of this small comparison. No universal conclusion
that a resolution or prompt always wins is supported.

The recommendation remains the liked old-canvas Sunburst high baseline, while
preserving Nano and Flare as active alternatives. There is no new checkpoint,
training-data production or model training in this stage.

Two local static images are delivered directly in chat under the ignored
`results/2026-10-11/yakamoz/` directory: `flare-comparison.png` and
`nano-comparison.png`. Both were visually inspected. No website was changed.
The drawing job took 1.54 seconds, sampled peak RSS 237.91 MiB and physical
footprint 206.77 MiB, under a 1 GiB guard with memory-pressure monitoring.
All 35 offline tests passed; mocked provider tests did not make API calls.
New provider requests and generation cost are both zero.
