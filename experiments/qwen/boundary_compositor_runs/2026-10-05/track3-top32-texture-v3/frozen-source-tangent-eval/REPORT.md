# Source-derived near-boundary rail diagnostic

Coordinates and source tangent use only the original source. The older generated-corridor evaluator is unchanged. Feature extraction is not a quality pass.

| Image | extraction | local tangent difference deg | endpoint dx px | last4 median dx px | invalid reason |
|---|---|---:|---:|---:|---|
| baseline | feature-extracted | 0.34 | -2.09 | -0.97 |  |
| protected-color | feature-extracted | 0.34 | -2.09 | -0.97 |  |
| texture25-color | feature-extracted | 0.34 | -2.09 | -0.97 |  |
| texture50-color | feature-extracted | 0.34 | -2.09 | -0.97 |  |

A local edge/post can fool this source-correspondence prior. Yellow selected generated ridge, purple separated alternative, green original ridge, grey source-derived search corridor. Markers outside each crop identify y320. The untouched image remains decisive; metrics do not imply success.

![Actual untouched crops](source-tangent-untouched-crops.png)

![Derived feature-selection diagnostic](source-tangent-selected-features.png)
