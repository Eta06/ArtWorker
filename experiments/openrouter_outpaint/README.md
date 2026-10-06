# OpenRouter outpainting comparison

Evaluate user-selected hosted models on `track3` (Aklın Hep Bende)
and `track2` (Karambol). The original Kehribar stage covers six models; Zeytin
covers nineteen models and every native resolution/quality setting. These are visual evaluation calls, not student training.

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
