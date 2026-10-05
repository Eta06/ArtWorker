# Source profile alignment diagnostic

CPU resampling of the saved localized40 exterior. Source and distant pixels remain exact. This is not a new model run or accepted repair.

Three fitted anchors: outer positive, inner positive, closing negative. The first negative highlight is held out. The strong positive post at x328 is excluded from source identities.

| variant | outer dx | held-out highlight dx | inner dx | closing dx | source exact | distant exterior exact |
|---|---:|---:|---:|---:|---|---|
| original | -17.63 | -18.12 | -8.97 | -3.40 | True | original |
| profile64 | -0.42 | -1.96 | 0.13 | 0.48 | True | True |
| profile96 | -0.23 | -1.34 | 0.18 | 0.53 | True | True |
| partial64 | -6.71 | -6.14 | -0.64 | -0.13 | True | True |
| nearest64 | -0.61 | -2.02 | 0.37 | 0.32 | True | True |

The held-out highlight has only one near-join observation with intact identity; two widened falls are invalid and its last-row identity is absent. Its numeric position therefore cannot establish a successful four-edge repair. Missing/ambiguous edges remain flagged in JSON.

Nearest64 selects exact original RGB tuples; it adds no interpolated RGB values. It can duplicate/remove grain samples and make stair steps. Linear variants interpolate RGB and may soften/alias grain. All variants geometrically stretch/compress texture, bounded by recorded sampling Jacobians.

An extra ridge or post can fool profile matching; anchor residuals are not an acceptance score.

![Native untouched joins](join-untouched-1x.png)

![Enlarged untouched joins](join-untouched-4x.png)

![Separate feature diagnostic](join-feature-diagnostic-4x.png)

![Full comparison](full-comparison.png)
