# Profile geometry plus original-decoder harmonic RGB

CPU compositor diagnostic only. Saved model raw and all frozen helpers remain unchanged. The original decoded known square is restored inside each labeled derived_raw only for the 96px residual calculation; exact shared source pixels are pasted at the end.

| variant | top RGB adjacent-row MAE | source exact | distant exterior exact | rail-patch high-frequency ratio, combined / warp |
|---|---:|---|---|---:|
| harmonic-only | 11.109 | True | True | control |
| profile64 | 12.417 | True | True | 1.0017 |
| nearest64 | 12.399 | True | True | 1.0004 |

The held-out highlight remains invalid/missing near the join. Harmonic RGB can reduce a photometric seam but cannot restore missing rail geometry. Nearest sampling preserves original RGB tuples before correction, while the additive RGB stage creates new values. Spectral texture includes object edges; it is not proof of retained grain.

![Untouched rail comparisons](rail-join-comparison.png)

![Full-width top join](top-fullwidth-join-comparison.png)

![Full outputs](full-comparison.png)
