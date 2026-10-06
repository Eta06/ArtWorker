# Iris — source-square fidelity audit

6 October 2026. Added an offline audit of existing raw reference outpaints.
The user requested source preservation checks alongside their visual review.
No new generation, provider request, billing, weight change or subagent.
Actual preferences, media and per-output fidelity results stay local in ignored
results; the public stage records method and execution, not the user's choices.

## Measurement

Verify reviewed output identities/SHA256, original cover SHA256, input-reference
checksums, and equality between the request square and its canvas pixels. Read
the intended source rectangle from each original receipt, scale it to actual
returned dimensions, and compare it at its **fixed location** on a common
512×512 analysis grid. A score is unavailable if provenance cannot be verified.

Report luminance SSIM, RGB MAE/PSNR, signed RGB bias, CIELAB Delta E 76 mean/p95,
fraction of pixels with any channel error >8/255, and worst SSIM among sixteen
tiles. None is a calibrated preservation percentage or an aesthetic verdict.
An identity canvas follows the same native-size/resampling pipeline to show
the analysis floor; it is not subtracted from model measurements.

Separately estimate a global similarity transform using mutual ORB matches and
RANSAC. Require broad spatial support, sufficient inliers, bounded scale,
rotation and displacement, and at least 85% visible source. Record failed fits
as unavailable, retaining diagnostic parameters. Aligned metrics cover visible
source only and cannot replace the primary fixed-position score. No local
warping, color correction or center-source pasting is applied before scoring.

The report includes original/output wipe sliders, consistent-scale difference
heatmaps, expected and estimated source rectangles, cover/model filters,
metric/cost ordering and CSV/JSON exports. The existing gallery exposes a report
link when present. Ratings and stable review IDs are unchanged. The report is
an export snapshot, not a live reflection of subsequent browser votes.

## Verification and actual run

Seven known-answer tests passed: identity, known RGB color shift, a localized
redraw, known translation/scale/rotation, featureless registration failure,
receipt-to-output coordinate mapping, and exclusion of invisible source pixels.
The shifted/scaled/rotated control recovers the imposed transform and improves
after global registration; original identity gives SSIM 1 and zero RGB/Lab error.

The serial audit measured 42 existing selected outputs with no provenance
failures. Reliable bounded alignment was available for 39; three remained
unavailable, with no fabricated scores. The guarded run completed in 14.39s:
sampled peak process-group RSS **355.89 MiB**, physical footprint **318.80 MiB**.
The limit was 1 GiB; initial swap was zero, no guard stop occurred. The identity
resampling controls had minimum SSIM 0.99414 and maximum RGB MAE 1.306/255.
Sources and raw outputs were retained unchanged. New provider cost: **$0**.

Live browser checks confirmed filtering, cost ordering, the wipe slider's
100%-output endpoint, its return to 50%, alignment disclosure and displayed
transform/overlap data. The prior local server was no longer running; it was
restarted on the same loopback port, preserving the browser storage origin.
No actual ratings were changed during checks.

## Limits and next work

Common-grid downsampling can miss tiny native-resolution edits. SSIM/Lab/MAE do
not prove unchanged identities, text or logos. Registration is a bounded
feature-based diagnostic and may fail or be biased by concentrated/repetitive
features. Rerun with a fresh export/directory after new choices or outputs.
Defaults audit liked settings; the CLI can also audit every completed setting
of liked models or the entire ledger. Future work can add local face/text/logo
checks and a larger independent cover set. No student training or automatic
teacher selection was performed.

## Publication

Seven changed source/document files, one file per `iris:` commit, grouped in one
guarded push. No local selections, result scores, cover media, generated media,
weights or credentials are included. Method and execution are recorded on the
existing ArtWorker Space Page.
