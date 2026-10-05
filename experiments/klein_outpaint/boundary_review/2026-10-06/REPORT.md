# Track3 source-boundary diagnostic

This is a local rail feature diagnostic, not an overall quality score. The profile/search corridor is fixed; inspect selected points to reject false edges.

| Image | far angle difference deg | far endpoint dx px | near offset median px | top seam ratio | bottom seam ratio | source exact |
|---|---:|---:|---:|---:|---:|---|
| klein4-described | 6.05 | -16.60 | -18.59 | 3.82 | 3.17 | True |
| klein28-described | 2.64 | -17.46 | -17.80 | 2.40 | 2.27 | True |
| qwen-v1 | 6.05 | -8.08 | -2.45 | 1.21 | 1.96 | True |
| qwen-v2 | 8.17 | -0.78 | -2.84 | 1.16 | 1.87 | True |

Far fit: generated rows270–310; source rows320–370. Near offset: detected generated rows312–319 against the source tangent. Positive angle difference means the generated rail is less steep. Signed endpoint dx is generated minus source at y320. Raw source pixels are decoded approximations; composites paste the resized source exactly.

A smaller near offset can be an elbow that connects incompatible far tangents. Report far angle, endpoint, near residual and full images together; do not collapse them into a winning scalar.

Source curvature and rail-edge selection create uncertainty. Per-window sensitivity, rejected rows and competing peaks are in metrics.json. Source fit should remain identical for all later candidates.

![Actual unmodified full outputs](full.png)

![Actual untouched rail crops](rail-crop-untouched.png)

![Selected features, yellow generated and green original, pink source boundary](rail-feature-selection.png)

![Actual untouched lower source-boundary crops](bottom-crop-untouched.png)
