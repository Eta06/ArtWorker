# Sumak — aligned source-content comparison

6 October 2026. The user clarified that source movement in a teacher output is
permitted. Compare the detected source region with the original after global
alignment, and retain placement as a separate diagnostic. This stage reuses
existing media: no paid generation, weights, student training or subagents.
Individual preferences and model-specific result scores remain local/ignored.

## Method and implementation

Keep the Iris provenance checks and common 512×512 analysis grid. Default
`--placement-mode free` removes the old center-displacement limit, accepts scale
.25–2 and rotation ±10°, and still requires at least twelve mutual ORB/RANSAC
inliers, 40% inlier fraction, 15% source feature-hull coverage, three quadrants
and 85% visible source. The historical bounded mode remains available.

Apply only the recovered global translation, rotation and isotropic scale before
content scoring. No local deformation, color correction or source replacement.
Report SSIM, RGB/Lab errors, worst-tile SSIM and visible coverage. Add explicit
pixel agreement fractions: all RGB channel differences must be ≤8/255 or
≤16/255. These thresholds describe numerical agreement, not semantic identity
or calibrated preservation percentages. Hidden source pixels are excluded.

The report defaults to aligned content; its fixed-location toggle changes
metrics, wipe images, heatmaps and sorting. Each reliable fit adds a 50/50
source/output overlay and visible-pixel mask. Unavailable alignment has no score
and no silent fallback. JSON schema 2 records placement mode; CSV uses separate
fixed/aligned columns. Publish the landing page at the existing URL with a
relative asset/export prefix to the fresh audit directory. Preserve the old
Iris index, measurements and media. Browser ratings are untouched.

## Verification and actual run

Eight known-answer tests passed in 0.681s. The added test recovers an asymmetric
placement that the historical center bounds reject; it checks recovered scale,
position, visibility and improved aligned RGB error. Identity and known color
change also validate the new pixel-agreement threshold.

The sequential offline audit checked 42 selected outputs with no provenance
failures. Reliable aligned comparison was available for 41; one fit remained
unavailable because its feature matches lacked sufficient spatial support. This
is an execution count, not a quality verdict or an automatic teacher selection.
The guarded run completed in **15.85s**, sampled peak process-group RSS
**352.06 MiB**, physical footprint **314.94 MiB**, against a 1 GiB limit. Initial
swap was zero; no guard stop occurred. New generation charge: **$0**.

Live browser checks verified model/cover filtering, aligned/fixed metric and
sorting changes, loaded comparison images, asset/export paths, slider endpoints
and return to 50%, and the unavailable-fit message with empty scores. The stopped
loopback server was restarted on the same port. The previous tab was stuck on
the browser's connection-error page and its control API rejected navigation;
a fresh tab in the same in-app browser successfully loaded the report. No
votes or browser preference storage were changed.

## Training implication and limits

A future mask-conditioned student can receive a source canvas and explicit
known/fill mask for asymmetric expansion. Example: 720×720 source, left 20,
right 100, top 200, bottom 500 produces an 840×1420 canvas with source origin
(20,200). Recovered teacher placement could help build varied training pairs;
the mask identifies generation regions without confusing real black pixels
with missing image. No such student has been trained in this stage.

Content similarity is one admission signal. Boundary continuity and the quality
of generated exterior still need separate visual checks. Downsampling can miss
small text, face or logo edits; feature fits can fail or be biased. At least 85%
visibility does not guarantee that the entire cover is retained. The resampling
control describes fixed-location sampling and is not the full alignment-error
floor. Do not subtract it or infer exact native-pixel preservation.

## Publication

Five changed source/document files, exactly one per `sumak:` commit, grouped in
one memory-guarded push. No preferences, per-model scores, media, credentials or
model payloads are published. Record method, execution, limitations and the
verified publication on the existing ArtWorker Space Page.
