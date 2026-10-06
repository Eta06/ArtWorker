# OpenRouter outpainting comparison

Evaluate user-selected hosted models on `track3` (Aklın Hep Bende)
and `track2` (Karambol). The original Kehribar stage covers six models; Zeytin
covers nineteen models and every native resolution/quality setting. These are visual evaluation calls, not student training.

The Ardıç consistency batch adds five new square album covers selected from the
user's YouTube Music library. `cover_consistency.py` compares Sunburst high,
Flare high, GPT Image 2 high and Nano Banana 2.1 1K: twenty serial requests,
without paid POST retries. Teacher generation time is not a selection criterion.
Use a local five-entry source manifest with `id`, `title`, `source_file` (relative
to repository root) and source-file `sha256`. Media and full manifests stay
ignored. Custom sources are accepted by `benchmark.py --source-image`; receipts
pin their original path and hash, and fidelity analysis reconstructs the exact
crop/resize. Historical resource paths remain supported.

```sh
python3 scripts/run_bounded_model.py \
  --output .build/ardic-guard --max-gib 1.5 --timeout 14400 -- \
  python3 experiments/openrouter_outpaint/cover_consistency.py \
  --run --stage ardic --budget-usd 5 \
  --sources /path/inside/repository/selected.json \
  --results experiments/openrouter_outpaint/results/ardic \
  --key-file /path/to/private-key-file
```

Omit `--run` to reconcile existing receipts and rebuild pages without generating
images. Existing definitive receipts are never resubmitted; uncertain requests
pause the batch. A $0.50 preflight reserve bounds dispatch against the batch and
dedicated-key budget; this is not a provider-enforced spending cap. Source/output
hashes, native dimensions, quality/resolution, response/generation charges and
account reconciliation are recorded. Registration uses the same free-placement
method and 512-pixel grid as Sumak. Generated exterior quality remains the user's
decision; registration failures have no aligned content score. Each stage has a
separate browser-vote namespace, with dynamic cover filters and export labels.

A completed generation whose image response is lost remains a billed attempt,
not an accepted output. Reconcile its exact generation ID/cost first; retain the
original transport receipt and any explicitly authorized fresh recovery attempt
under `superseded_attempts`. Never relabel a lost image as a visual success. The
bounded JSON reader now returns once a complete document arrives and retains
response IDs on read errors; this avoids waiting for transport EOF when it never
arrives, without asserting that this caused a particular network failure.

`benchmark.py` uses the dedicated `POST /api/v1/images` endpoint, pinned provider
routing and one output per request. Model capabilities and endpoint prices come
from the live Image Models API. Image-only models can be absent from the ordinary
chat model catalog. Never silently substitute another model slug.

Local resource images are 1280×720 with side bars. Center-crop the true 720×720
artwork and place it on a green 720×1280 target canvas at `[0,280,720,1000]`.
Send the canvas and original square as two references, with the same prompt for
every cover/model. This API route has reference images but no explicit pixel
mask; preserving the source geometry is requested, not enforced. Keep raw output.

All models support 9:16; this common benchmark ratio differs from the local
512×1152 phone experiment. Request the 1K tier where supported and medium quality
for OpenAI. Tiers and quality knobs are not equivalent across providers. Record
actual returned dimensions; do not call this an identical-resolution benchmark.

Run each request serially under `scripts/run_bounded_model.py` with a 0.6 GiB
process-group sampling limit and pressure/swap guards. There are no automatic paid
POST retries. An uncertain request must be reconciled before any retry; a definite
capacity error can be retried in a fresh directory, retaining the original record.

```sh
python3 scripts/run_bounded_model.py \
  --output .build/openrouter-research/guards/example \
  --max-gib 0.6 --timeout 300 -- \
  python3 experiments/openrouter_outpaint/benchmark.py \
  --key-file /path/to/private-key-file \
  --model bytedance-seed/seedream-5-0-flash --track track3 \
  --output experiments/openrouter_outpaint/results/example/seedream/track3
```

Use `summarize.py --key-file ... --results ... --ledger ...` after the batch.
Reconcile `usage.cost` to the generation's `total_cost` using `x-generation-id`.
Do not infer an individual call's charge from an immediate key-usage delta: the
key counter can lag. Credit purchase fees/tax and future training compute are
outside these generation costs. Missing billing remains unknown, not assumed zero.

Credentials, source images, generated images and full local request receipts stay
outside normal Git. The public ledger contains cost, timing, dimensions, hashes,
provider/model identifiers and limitations. Provider output-training permission
and source-artwork redistribution rights are not established by this evaluation.

Sources: [Image API](https://openrouter.ai/docs/guides/overview/multimodal/image-generation),
[generation billing](https://openrouter.ai/docs/api/api-reference/generations/get-request-&-usage-metadata-for-a-generation).

## Zeytin variant batch

See [capabilities](zeytin-capabilities.md) and [pinned plan](zeytin-plan.json).
The batch has 98 comparison cells, reuses eleven identical previous successes,
and withholds five FLUX road-cover cells following the provider's earlier filter.
Recraft is excluded because its endpoint accepts text only. Single-reference
models receive the canvas only, with the second-reference sentence removed.

```sh
python3 scripts/run_bounded_model.py \
  --output .build/openrouter-zeytin-guard --max-gib 1.5 --timeout 7200 -- \
  python3 experiments/openrouter_outpaint/variants.py \
  --plan experiments/openrouter_outpaint/zeytin-plan.json \
  --results experiments/openrouter_outpaint/results/2026-10-06/zeytin \
  --key-file /path/to/private-key-file
```

At most two HTTP requests run concurrently; local image decoding is monitored
under one shared process-group guard. A $45 new-spend ceiling reserves $3 per
in-flight request against both that ceiling and the key's remaining credit. This
is a dispatch guard, not a guarantee against provider billing delays. Unknown
responses pause dispatch; no request with an existing receipt is automatically
resubmitted. The batch can resume definitive completed/failed receipts with a
fresh outer guard directory.

`report_variants.py --key-file ... --results ... --ledger ...` reconciles each
generation, writes JSON/CSV records and a local HTML gallery. It includes all
variants and original raw outputs; overview sheets show the first profile only.
Gallery images link to full-resolution files. Every charge is attributed from
response usage or generation billing, not overlapping per-request key deltas.
Visual selection belongs to the user; no winner or quality score is assigned.

## Local visual preferences

`review_gallery.py` renders the preference gallery from an existing reconciled
ledger without any API calls or new generations. `report_variants.py` uses the
same renderer, so rebuilding reports retains the review interface.

```sh
python3 experiments/openrouter_outpaint/review_gallery.py \
  --ledger experiments/openrouter_outpaint/zeytin-costs.json \
  --results experiments/openrouter_outpaint/results/2026-10-06/zeytin
```

Each available output has mutually exclusive like/dislike buttons. Pressing the
active button clears the choice. Votes persist in browser localStorage, keyed by
model, native profile, cover and raw-image SHA256; a new generated image never
inherits the old output's preference. Model summaries deduplicate models with at
least one liked output and show accepted settings/covers plus rejection counts.
Unavailable images cannot be rated. Filters can show liked, disliked or undecided
outputs. Model-list text and full JSON preference exports are previewed in a
copyable dialog with a download link.

Preferences stay in this browser and origin, and sync between open tabs on that
origin. Clearing browser data or changing the server port loses that storage;
export a backup first. JSON import is not implemented. Native download handling
is browser-dependent; the dialog provides readable/selectable export text.
Storage failures are surfaced instead of claiming persistence. Preferences are
not published to GitHub, Space or OpenRouter, and do not select a teacher
automatically.

## Source-square fidelity (Iris / Sumak)

`source_fidelity.py` evaluates existing raw images against the exact reference
square sent in each request. It makes no network calls. Selection identities,
raw/input hashes, original cover hashes and canvas source pixels are verified
before a score is produced. User preferences, metrics and image diagnostics
stay in ignored local results, not in the public repository or Space.

Run with the existing `experiments/qwen/.venv` (tested: Pillow 12.3.0,
NumPy 2.5.3, OpenCV 4.14.0); no diffusion weights are loaded:

```sh
python3 scripts/run_bounded_model.py \
  --output .build/sumak-fidelity-guard --max-gib 1 --timeout 600 -- \
  experiments/qwen/.venv/bin/python \
  experiments/openrouter_outpaint/source_fidelity.py \
  --ledger experiments/openrouter_outpaint/zeytin-costs.json \
  --selections /path/to/artworker-zeytin-secimler.json \
  --placement-mode free \
  --output experiments/openrouter_outpaint/results/example/fidelity
```

Use a fresh output directory to preserve previous audits. Default scope is
`liked`: accepted individual outputs/settings. `--scope liked-models` includes
all completed settings of models with at least one accepted output;
`--scope all` includes all completed ledger outputs. Raw images are never edited.
The source is sampled on a common 512×512 analysis grid. The expected square
rectangle is scaled from each receipt's layout to the actual output dimensions,
including native tiers which are not exactly 9:16. The full output is resized
with Lanczos; the expected square is sampled with bicubic interpolation.

Sumak makes **aligned source content** the report's default comparison. Source
movement is permitted; translation, rotation and uniform scale are compensated
before measuring the overlap. A separate fixed-location view retains the Iris
layout diagnostic. Switching views changes the wipe, heatmap, metrics and ordering.

Both views report luminance SSIM (11×11 Gaussian,
sigma 1.5, population moments, L=255, K1=.01, K2=.03; exclude five-pixel border),
RGB MAE/PSNR, signed RGB bias, mean/p95 CIELAB Delta E 76, fraction of pixels
with any channel difference >8/255, and the worst SSIM among a 4×4 tile grid.
The report also shows the fraction of evaluated pixels where **all three** RGB
channel differences are at most 8/255, and the more tolerant 16/255 fraction.
These are fidelity diagnostics, not a calibrated preservation percentage,
general image-quality score or proof of unchanged faces/text/logos. Subtle
changes below 512px analysis resolution can be missed. Delta E 76 is a simple
Euclidean Lab distance, not a perceptual neural metric or CIEDE2000.

Registration uses mutual ORB matches with a .75 ratio check and
RANSAC to estimate one translation/rotation/isotropic-scale transform. Require
at least 12 inliers, 40% inlier fraction, 15% source feature hull coverage and
three quadrants. Default `--placement-mode free` accepts scale .25–2, rotation
±10° and at least 85% visible source, without a center-shift limit. It still
requires a reliable global similarity transform; it does not fit arbitrary
deformations. Historical `--placement-mode bounded` retains Iris bounds: scale
.75–1.25, rotation ±5°, center shift ±18% of source width. Unreliable/out-of-bounds
fits are recorded as unavailable, not zero error. Matrices and rejected fit
diagnostics are retained. Aligned metrics cover only visible source pixels;
SSIM uses only windows wholly inside that overlap. Never locally warp, adjust
colors or paste the original source before scoring. Fixed-location scores answer
layout adherence; aligned scores answer content similarity independently of
placement. Neither alone measures outpaint quality outside the source square.

An identity canvas passes through the same native dimensions and analysis
pipeline to report a resampling-only control; its score is not subtracted.
The HTML report includes original/output wipe sliders, fixed-scale difference
maps (mean RGB difference 0–64/255), expected/fitted source rectangles, filters,
metric/cost ordering and CSV/JSON exports. Aligned examples include a 50/50 source
overlay and an explicit visible-pixel mask; unobserved areas are grey in the
overlay and excluded from metrics, not counted as matches. Unavailable fits have
no fabricated score or automatic fixed-score fallback. JSON schema 2 records
placement mode, and CSV columns explicitly separate fixed/aligned measurements.
`render(..., assets=...)` can publish at the existing report URL while retaining
assets and exports in a fresh audit directory. Preserve the old index and audit
files before replacing the landing HTML. The gallery links to `fidelity/`
when the report exists. It is a snapshot of the exported choices; subsequent
browser votes require a fresh export and audit. No votes are altered.

Eight known-answer tests cover identity, color change, local redraw, transform
recovery, featureless rejection, geometry, hidden-pixel exclusion and asymmetric
placement outside the old center bounds. Validation:

```sh
cd experiments/openrouter_outpaint
../qwen/.venv/bin/python -m unittest -v test_source_fidelity.py
```

For a future mask-conditioned student, recovered source placement can define a
known-area mask and its complementary fill region. A 720×720 source with padding
left=20, right=100, top=200, bottom=500 gives an 840×1420 canvas and source origin
(20,200). Explicit masks distinguish missing regions from real black artwork.
Varied training layouts and held-out boundary/content checks are required;
localization alone does not train this capability. No student training is run by
this audit.
