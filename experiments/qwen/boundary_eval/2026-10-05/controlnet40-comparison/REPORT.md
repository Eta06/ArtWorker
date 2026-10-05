# Source-derived near-boundary rail diagnostic

Coordinates and source tangent use only the original source. The older generated-corridor evaluator is unchanged. Feature extraction is not a quality pass.

| Image | extraction | local tangent difference deg | endpoint dx px | last4 median dx px | invalid reason |
|---|---|---:|---:|---:|---|
| teacher20 | feature-extracted | -1.15 | -2.52 | -2.02 |  |
| teacher40 | feature-extracted | -1.21 | -2.52 | -1.90 |  |
| teacher40 bridge | feature-extracted | -0.04 | -1.73 | -1.05 |  |

A local edge/post can fool this source-correspondence prior. Yellow selected generated ridge, purple separated alternative, green original ridge, grey source-derived search corridor. Markers outside each crop identify y320. The untouched image remains decisive; metrics do not imply success.

![Actual untouched crops](source-tangent-untouched-crops.png)

![Derived feature-selection diagnostic](source-tangent-selected-features.png)
