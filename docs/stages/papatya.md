# Papatya — source preservation without a colored placeholder

11 October 2026, Türkiye. The user requested another Nano trial, aligned source
comparison, and investigation of the unavailable Karambol alignment. Existing
Filiz media, source crops and preferences are preserved. No subagents, model
downloads, training or simulator work.

## Attempts and actual charges

Nano Banana 2.1 / 1K was retried once on each of the same three square references.
All three attempts again returned HTTP 402: Google AI Studio prepayment credits
depleted. The OpenRouter key authenticated with remaining limit. This is a billing
failure, not content moderation or a quality result. The only advertised Nano
endpoint was Google AI Studio; no billing/settings changes or credit purchase.
These attempts settled to **$0**, with no automatic POST retries.

Two additional Karambol requests use Sunburst / high and Flare / high with a
new `square-anchored` prompt. Both return 864×1536. Each costs **$0.041082**;
total new charge **$0.082164** agrees with generation `total_cost` records and the
settled account delta. Counters initially lagged; the first report draft was not
reconciled. The final report uses settled records. Zero unknown costs. Historical
Filiz costs are displayed per image but are excluded from the new-charge total.
The [anonymous ledger](../../experiments/openrouter_outpaint/papatya-costs.json)
contains no source identifiers, generation IDs, credentials or local media paths.

## Exact input and prompt

Both square modes send only `source.png`, the reproducible 720×720 source crop,
and request `aspect_ratio: 9:16`. The green diagnostic canvas is **not sent**.
The original short prompt remains available and is used for the Nano retry.
Only the two new Karambol requests use this prompt:

> Outpaint this artwork to a 9:16 portrait. Keep the entire original square unchanged, centered and spanning the full output width. Generate scene continuations only above and below it. Do not zoom out, recompose, redraw or independently resize anything inside the original artwork. Preserve faces, text, logos, clothing, colors, texture, lighting and perspective. Printed graphics must remain printed graphics, not become real objects. Join the extensions seamlessly without blurred or mirrored copies.

This tests stronger placement/content instructions without colored placeholders.
It is one output per model/prompt, not a seed-controlled or repeat-variance study.
The observed OpenAI endpoint exposes reference images, aspect ratio and quality;
it does not advertise a pixel-lock mask parameter. Text instructions alone do not
guarantee untouched source pixels. No source pasting or color correction was used.

## Why registration failed

The original Karambol/Flare has 435/885 ORB inliers, about 49.15%, but the source
features cover only two of four quadrants. An independent SIFT attempt has
215/407 inliers and still only two quadrants. The top half supports an approximate
candidate area; it does not validate one transform for the whole cover. Global
position shift alone does not explain the failure. Recomposition/redrawing is
plausible from inspection, but this is not a measured causal diagnosis.

Gripin/Flare is a detector failure that can be recovered: ORB has 287/738 inliers
(38.89%, below the 40% gate); SIFT has 124/231 (53.68%), 38.59% source feature hull
coverage and all four quadrants. Added ORB → SIFT fallback with the **same** inlier,
coverage, quadrant, visibility and similarity-transform geometry requirements.
Accepted ORB alignments remain unchanged. No local/nonrigid warping or color fit.
The method and failed first attempt are retained in diagnostics.

The final report reuses six Filiz raw images and adds the two new images: **seven
of eight reliably align**, versus four of six in the historical ORB-only Filiz
report. The old Karambol/Flare remains unavailable. Its orange candidate is now
shown explicitly as unverified, with no whole-source preservation percentage.
Green means an assumed centered layout for these square-only inputs, not an image
or placement instruction supplied to the model. Historical reports are untouched.

## Preservation improved, exterior quality did not

Karambol aligned measurements at 512×512:

| Model / prompt | SSIM | RGB channels each within 16/255 |
| --- | ---: | ---: |
| Sunburst / original short | 0.6418 | 69.71% |
| Sunburst / square-anchored | 0.8410 | 80.05% |
| Flare / original short | unavailable | unavailable |
| Flare / square-anchored | 0.8150 | 81.10% |

Direct visual inspection finds obvious horizontal seams at the lower square edge
in the anchored outputs. The lower extension becomes wall background rather than
continuing the subjects/clothes naturally. Stronger source instructions improve
source agreement but regress exterior continuity. **No new output/prompt is
accepted as an overall quality improvement or training target.** Exterior decisions
remain with the user. RGB threshold fractions are numerical tolerance metrics,
not semantic percentages of unchanged artwork; SSIM is not a quality ranking.

The student can eventually accept source plus a separate binary known/fill mask
and independent four-side padding, so mask colors cannot conflict with artwork
colors. That requires a model/pipeline enforcing known pixels and seam validation.
It is next work, not implemented training or a guaranteed fix in this experiment.

## Verification and resources

Nineteen offline tests passed in 1.213 seconds. New controls recover known geometry
through SIFT, reject top-half-only matching and unrelated texture, and ensure the
anchored mode chooses a single square reference. Existing actual-wire checks for
square input, legacy canvas behavior and provenance remain covered.

Serial bounded runs used a 1.5 GiB footprint ceiling and continuous memory/pressure
checks. Nano retry: 3.184 s, sampled peak RSS 132.30 MiB, footprint 82.77 MiB.
Anchored generation: 46.272 s, RSS 93.58 MiB, footprint 63.28 MiB. Three offline
report renders used approximately 290–302 MiB RSS; the final render took 3.122 s,
RSS 289.75 MiB, footprint 252.16 MiB. Initial system swap existed; no pressure
stop occurred. These are sampled Mac process-group measurements, not iPhone RAM.

Live inspection found cross-directory raw links escaping the shallow stage URL.
The renderer now uses private local media symlinks inside the report, preserving
original files without duplicate image copies. The corrected report was rendered
without additional provider calls. Gallery, original/output links, eight-row CSV
and JSON, seven measured alignments, unavailable candidate caption, filtered
Karambol cards and slider clipping were checked. Python compilation and Git
whitespace checks passed. Lazy image loading is retained; no votes were cast.

Local gallery: `http://127.0.0.1:54146/papatya/`; aligned diagnostics:
`http://127.0.0.1:54146/papatya/fidelity/`. Eleven gallery cells: six reused images,
two new images and three fresh Nano errors. Media, source manifests, per-cover
metrics, credentials and preferences stay ignored.

Reproduce a fresh Nano-only run with `square_trial.py --models
google/gemini-nano-banana-2.1 --stage papatya --sources <private-manifest> --results
<fresh-private-directory>`; add `--run --key-file <private-key> --budget-usd 2`
only when paid execution is intended. Anchored mode uses `benchmark.py
--input-mode square-anchored --model <slug> --quality high --track <id>
--source-image <source> --output <fresh-directory> --key-file <private-key>`.
Run generation under `scripts/run_bounded_model.py`. `preservation_comparison.py`
combines `--baseline-ledger`, repeated `--ledger`, `--sources`, `--results` and
`--stage` entirely offline. Existing result directories are not overwritten.

## Publication and next work

Nine changed source/test/document/anonymous-cost files, each in its own `papatya:`
commit, form one publication batch with one memory-guarded push. No AI trailers,
weights, media or credentials. Record verified commit/remote state in the existing
ArtWorker Space Page after publication.

Next: visually compare both prompts, address the external Nano billing failure
before another authorized retry, and investigate mask-conditioned generation for
source integrity plus natural transitions. No student checkpoint, automatic
teacher selection or green-canvas rollback was produced.

Method reference: [OpenCV feature matching](https://docs.opencv.org/4.13.0/d1/de0/tutorial_py_feature_homography.html)
(SIFT/matching reference; our transform is global similarity, not homography).
