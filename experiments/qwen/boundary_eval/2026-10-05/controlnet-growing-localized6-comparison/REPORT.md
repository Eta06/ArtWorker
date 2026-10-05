# Source-derived near-boundary rail diagnostic

Coordinates and source tangent use only the original source. The older generated-corridor evaluator is unchanged. Feature extraction is not a quality pass.

| Image | extraction | local tangent difference deg | endpoint dx px | last4 median dx px | invalid reason |
|---|---|---:|---:|---:|---|
| old edge-context | invalid-feature | invalid | invalid | invalid | near ridge is discontinuous or strongly curved/ambiguous in the local fit |
| scale0 | invalid-feature | invalid | invalid | invalid | near ridge is discontinuous or strongly curved/ambiguous in the local fit |
| scale0.5 | invalid-feature | invalid | invalid | invalid | near ridge is discontinuous or strongly curved/ambiguous in the local fit |
| scale1 | invalid-feature | invalid | invalid | invalid | near ridge is discontinuous or strongly curved/ambiguous in the local fit |

A local edge/post can fool this source-correspondence prior. Yellow selected generated ridge, purple separated alternative, green original ridge, grey source-derived search corridor. Markers outside each crop identify y320. The untouched image remains decisive; metrics do not imply success.

![Actual untouched crops](source-tangent-untouched-crops.png)

![Derived feature-selection diagnostic](source-tangent-selected-features.png)
