# Track3 source-boundary diagnostic

This is a local rail feature diagnostic, not an overall quality score. The profile/search corridor is fixed; inspect selected points to reject false edges.

| Image | far angle difference deg | far endpoint dx px | near offset median px | top seam ratio | bottom seam ratio | source exact |
|---|---:|---:|---:|---:|---:|---|
| original | 2.57 | -18.30 | -19.57 | 2.37 | 1.67 | True |
| harmonic-only | 2.57 | -18.30 | -19.61 | 1.74 | 1.60 | True |
| profile64-RGB96 | 8.04 | -1.45 | -2.63 | 1.95 | 1.60 | True |
| nearest64-RGB96 | 7.93 | -1.73 | -2.65 | 1.93 | 1.60 | True |

Far fit: generated rows270–310; source rows320–370. Near offset: detected generated rows312–319 against the source tangent. Positive angle difference means the generated rail is less steep. Signed endpoint dx is generated minus source at y320. Raw source pixels are decoded approximations; composites paste the resized source exactly.

A smaller near offset can be an elbow that connects incompatible far tangents. Report far angle, endpoint, near residual and full images together; do not collapse them into a winning scalar.

Source curvature and rail-edge selection create uncertainty. Per-window sensitivity, rejected rows and competing peaks are in metrics.json. Source fit should remain identical for all later candidates.

![Actual unmodified full outputs](full.png)

![Actual untouched rail crops](rail-crop-untouched.png)

![Selected features, yellow generated and green original, pink source boundary](rail-feature-selection.png)

![Actual untouched lower source-boundary crops](bottom-crop-untouched.png)
