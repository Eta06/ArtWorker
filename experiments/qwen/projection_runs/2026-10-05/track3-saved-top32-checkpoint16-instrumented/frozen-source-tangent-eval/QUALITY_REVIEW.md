# Independent 4 versus 16 projection quality comparison

Same frozen source-tangent evaluator and thresholds; source fit is byte-for-byte equal to the 4-update evaluator result. Only the new 16-run frozen-source-tangent-eval subfolder was written. Old outputs and evaluators remain frozen.

| image | extraction | angle difference | endpoint dx | last-four median dx | strong near rows |
|---|---|---:|---:|---:|---:|
| baseline | valid | +0.341° | −2.086 px | −0.975 px | 15/16 |
| consistency4 | invalid | unknown | unknown | unknown | 13/16 |
| consistency16 | invalid | unknown | unknown | unknown | 12/16 |
| derivative4 | valid | +0.177° | −2.366 px | −1.645 px | 16/16 |
| derivative16 | valid | +1.048° | −1.154 px | −1.908 px | 16/16 |

Consistency16 fails both near-edge coverage and local ridge continuity: only 12 of 16 rows are strong, and p90 fit residual is 2.632 px. Its last row remains absent. Actual crops show weakened/flattened outer-edge sections and broken highlight segments. Lower pixel loss therefore does not restore the missing physical rail.

Derivative16 improves the fitted endpoint and y319 offset (−1.201 px versus −1.993 at 4), but worsens angle and last-four median offset. Actual selected x moves 316.611→312.395→311.765→311.143 across y316–319: a 4.217 px step followed by two nearly flat ~−0.63 px/row segments, instead of the source tangent ~−1.876 px/row. Its last two detected gradient strengths rise from 7.360/17.144 at 4 to 30.287/49.466 at 16. The actual rail becomes visibly corrugated, stepped and spiky; valid extraction does not establish coherent metal geometry. No comparable alternative path is found, but the path can follow sharpened fragments of a broken upper band rather than one smooth ridge.

Full scene composition remains coherent, but direct native/enlarged raw and composite comparisons show local quality regression versus 4. Consistency16 washes out fine decoded-known water/asphalt texture. Derivative16 amplifies/reconfigures local metal, wave and asphalt contrast, with an uneven oversharpened border bridge. Water ripple scale/contrast remains different across several rows; vegetation/road tone bands and the bottom asphalt grain transition remain. The final source paste does not make changed raw-known texture into preserved original texture.

Instrumented spectral diagnostics agree with the texture tradeoff. Consistency16 raw-known water and asphalt high-frequency power ratios versus baseline are 0.741 and0.721, down from 1.013 and0.903 at 4; generated bottom asphalt is 0.876. Derivative16 generated rail/water/asphalt ratios are 1.415/1.188/1.288, versus 1.123/1.103/1.135 at 4. These FFT values include object edges; they are not evidence of natural grain restoration. Independent per-row mean color, luminance variation and horizontal texture change are saved across 64 rows of water and asphalt joins to avoid judging only the single boundary pixel.

The derivative boundary metric improves top 8.137→4.406 and bottom 3.646→2.143, while the visible rail deteriorates. This demonstrates objective/visual disagreement, not successful16-step promotion. Both 16 outputs remain unaccepted; the older failed directory and derivative timeout remain failures. Parent audits state/source/far integrity separately. No controlled checkpoint speedup or mobile claim follows from this quality review.

[Per-row edge and texture summary](independent_quality_summary.json) · [Independent raw multi-row texture profiles](independent_raw_row_texture_profiles.json) · [Frozen evaluator report](REPORT.md)

![Actual raw rail4 versus 16](raw-rail-4-vs16-4x.png)

![Actual source-exact rail4 versus 16](composite-rail-4-vs16-4x.png)

![Actual raw multi-row water join](raw-water-4-vs16-3x.png)

![Original source versus raw known water](raw-source-water-4-vs16-3x.png)

![Original source versus raw known asphalt](raw-source-asphalt-4-vs16-4x.png)

![Actual raw asphalt join](raw-bottom-join-4-vs16-3x.png)
