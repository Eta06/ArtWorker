# Source-derived near-boundary rail diagnostic

Coordinates and source tangent use only the original source. The older generated-corridor evaluator is unchanged. Feature extraction is not a quality pass.

| Image | extraction | local tangent difference deg | endpoint dx px | last4 median dx px | invalid reason |
|---|---|---:|---:|---:|---|
| zero-source-hints_scale0.5 | invalid-feature | invalid | invalid | invalid | near ridge is discontinuous or strongly curved/ambiguous in the local fit |
| top32_scale0.5 | feature-extracted | 37.59 | 2.78 | -1.60 |  |
| zero-source-hints_scale1 | invalid-feature | invalid | invalid | invalid | near ridge is discontinuous or strongly curved/ambiguous in the local fit |
| top32_scale1 | feature-extracted | 7.72 | -1.00 | -2.13 |  |

A local edge/post can fool this source-correspondence prior. Yellow selected generated ridge, purple separated alternative, green original ridge, grey source-derived search corridor. Markers outside each crop identify y320. The untouched image remains decisive; metrics do not imply success.

![Actual untouched crops](source-tangent-untouched-crops.png)

![Derived feature-selection diagnostic](source-tangent-selected-features.png)
