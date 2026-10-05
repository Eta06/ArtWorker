# Source-derived near-boundary rail diagnostic

Coordinates and source tangent use only the original source. The older generated-corridor evaluator is unchanged. Feature extraction is not a quality pass.

| Image | extraction | local tangent difference deg | endpoint dx px | last4 median dx px | invalid reason |
|---|---|---:|---:|---:|---|
| baseline | feature-extracted | 0.34 | -2.09 | -0.97 |  |
| consistency16 | invalid-feature | invalid | invalid | invalid | fewer than80 percent of near rows have a strong metal edge; near ridge is discontinuous or strongly curved/ambiguous in the local fit |
| derivative16 | feature-extracted | 1.05 | -1.15 | -1.91 |  |

A local edge/post can fool this source-correspondence prior. Yellow selected generated ridge, purple separated alternative, green original ridge, grey source-derived search corridor. Markers outside each crop identify y320. The untouched image remains decisive; metrics do not imply success.

![Actual untouched crops](source-tangent-untouched-crops.png)

![Derived feature-selection diagnostic](source-tangent-selected-features.png)
