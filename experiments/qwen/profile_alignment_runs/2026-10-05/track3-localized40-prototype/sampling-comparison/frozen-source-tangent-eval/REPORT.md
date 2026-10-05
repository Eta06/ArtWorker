# Source-derived near-boundary rail diagnostic

Coordinates and source tangent use only the original source. The older generated-corridor evaluator is unchanged. Feature extraction is not a quality pass.

| Image | extraction | local tangent difference deg | endpoint dx px | last4 median dx px | invalid reason |
|---|---|---:|---:|---:|---|
| original | feature-extracted | 0.30 | -17.40 | -17.59 |  |
| profile64 | feature-extracted | 3.93 | 0.66 | -0.38 |  |
| profile96 | feature-extracted | 1.92 | 0.25 | -0.19 |  |
| partial64 | feature-extracted | 2.54 | -5.81 | -6.67 |  |
| nearest64 | feature-extracted | 4.31 | 0.56 | -0.57 |  |

A local edge/post can fool this source-correspondence prior. Yellow selected generated ridge, purple separated alternative, green original ridge, grey source-derived search corridor. Markers outside each crop identify y320. The untouched image remains decisive; metrics do not imply success.

![Actual untouched crops](source-tangent-untouched-crops.png)

![Derived feature-selection diagnostic](source-tangent-selected-features.png)
