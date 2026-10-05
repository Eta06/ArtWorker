# Source profile alignment diagnostic

CPU resampling of the saved localized40 exterior. Source and distant pixels remain exact. This is not a new model run or accepted repair.

Three fitted anchors: outer positive, inner positive, closing negative. The first negative highlight is held out. The strong positive post at x328 is excluded from source identities.

| variant | outer dx | held-out highlight dx | inner dx | closing dx | source exact | distant exterior exact |
|---|---:|---:|---:|---:|---|---|
| original | -17.63 | -11.88 | -8.97 | -3.40 | True | original |
| profile64 | -0.42 | 0.62 | 0.13 | 0.48 | True | True |
| profile96 | -0.23 | 0.67 | 0.18 | 0.53 | True | True |
| partial64 | -6.71 | -0.85 | -0.64 | -0.13 | True | True |

Edge polarity and observations are saved per row. Missing/ambiguous edges remain flagged. An extra ridge or post can fool profile matching; anchor residuals are not an acceptance score.

![Native untouched joins](join-untouched-1x.png)

![Enlarged untouched joins](join-untouched-4x.png)

![Separate feature diagnostic](join-feature-diagnostic-4x.png)

![Full comparison](full-comparison.png)
