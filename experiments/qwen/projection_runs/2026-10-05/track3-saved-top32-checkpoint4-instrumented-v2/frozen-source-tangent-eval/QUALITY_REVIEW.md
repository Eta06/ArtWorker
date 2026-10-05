# Independent review of saved-top32 VAE projection

Same frozen source-tangent evaluator, unchanged source fit and thresholds. This is an independent CPU evaluation of the saved composites, not a new model or projection run.

| image | feature extraction | near angle difference | endpoint dx | last-four median dx | near-row coverage |
|---|---|---:|---:|---:|---:|
| baseline | valid | +0.341° | −2.086 px | −0.975 px | 15/16 |
| consistency4 | invalid | unknown | unknown | unknown | 13/16 |
| derivative4 | valid | +0.177° | −2.366 px | −1.645 px | 16/16 |

Consistency4 has no strong extracted edge at y318/y319 and fails the fixed requirement of three strong last-four rows. It cannot be called a successful rail-geometry result. Baseline and derivative have no comparably likely alternative path; selected points were inspected directly. Derivative reduces the y319 source-tangent offset from +3.930 px to −1.993 px, but the fitted endpoint and last-four median do not improve. It retains a local bend: selected x moves from 316.228 at y317 to 311.711 at y318. A smaller tangent-angle number alone does not establish a seamless rail.

Directly viewed actual full composite/raw images, untouched joins, selected features, raw water joins, original-source/decoded-known water and asphalt patches. The derivative variant reduces the abrupt photometric border contrast at the rail, and consistency makes decoded known top water closer in tone to source. Neither makes the full-width water/vegetation/road join disappear. Water ripple scale/texture still differs across the join, and the bottom asphalt has a persistent grain transition. Raw known texture changes even though the final composites hard-paste the original source. No smoothing filter or manual pixel correction was added in this review.

The instrumented runner reports high-frequency ratios against baseline raw: consistency generated top water 1.02975, known top water 1.01335, known bottom asphalt 0.90324, generated bottom asphalt 1.00190; derivative corresponding values 1.10264, 1.04344, 1.03397 and 1.13509. Against original source, derivative known bottom asphalt is 1.13836. These quantify changed spectral texture, including object edges; they do not prove natural grain restoration. Consistency raw-source MAE is lower at the borders than derivative (top16 5.041 vs7.571; bottom16 4.054 vs5.397), while derivative's one-pixel boundary objective is lower. The objectives trade off and are not semantic quality metrics.

Source/far/state integrity is audited separately by the parent. The earlier failed directory and old derivative timeout remain failed. This different-input four-update run cannot establish a controlled checkpoint memory/speed gain or mobile feasibility. Quality remains unaccepted.

[Fixed evaluator report](REPORT.md) · [Independent summary and per-row points](independent_quality_summary.json)

![Actual raw water join](raw-water-join-3x.png)

![Original source versus raw known water](raw-known-source-water-4x.png)

![Original source versus raw known asphalt](raw-known-source-asphalt-4x.png)

![Actual raw rail join](raw-rail-join-4x.png)

A second independent direct-image review agrees: derivative4 gives a modest immediate attachment improvement, while the selected edge plausibly remains the upper metal edge. Its valid detection can describe a sharpened border bridge with a new bend, not restored physical rail geometry. Water, vegetation, inner-rail and asphalt texture/tone defects remain. No seamless, natural-grain or checkpoint-speedup acceptance follows.
