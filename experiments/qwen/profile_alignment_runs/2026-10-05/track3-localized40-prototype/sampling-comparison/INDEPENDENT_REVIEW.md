# Independent CPU profile diagnostic review

Integrity passed independently: saved float32 fields reproduce every output byte, all source and bottom pixels are exact, far exterior and zero-field support are exact, and nearest64 returns exact original RGB tuples at same-row nearest coordinates for all 589,824 pixels. No model/GPU or frozen optical-flow/evaluator edits occurred.

Directly inspected native, enlarged and full outputs. Strong profile variants substantially reduce the protruding outer band and narrow its fitted width. Linear sampling compresses/interpolates metal/vegetation texture; nearest sampling introduces visible stair steps and duplicated or removed grain samples. The horizontal tone seam remains. The geometry-only top seam luminance jump worsens from 15.084 to about 16.25.

The held-out highlight identity remains invalid at y317/y318 and absent at y319. Only y316 remains a trustworthy near observation; no complete four-edge repair is established. Strong variants also lose closing-edge extraction at some earlier rows. This is an inconclusive compositor, not an accepted seamless result.

Frozen source-derived evaluator keeps its exact source fit and thresholds: near endpoint −17.40 px original becomes +0.66 px linear64 / +0.56 px nearest64. Near angle difference increases from 0.30° to 3.93° / 4.31°. The wider frozen evaluator's far angle difference increases from 2.57° to 8.49° / 8.26°. Better endpoint position therefore comes with a changed local bend.

[Metrics](metrics.json) · [Fixed source-derived evaluator](frozen-source-tangent-eval/REPORT.md) · [Fixed wider evaluator](frozen-boundary-eval/REPORT.md) · [Native untouched join](join-untouched-1x.png)
