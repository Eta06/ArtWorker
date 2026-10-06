# Zeytin — hosted model and native-tier comparison

6 October 2026. Expanded the Chrome model selection to **19 image-reference
models**: thirteen new models plus the previous six. Excluded the image catalog
page, both Ming models and Meta Muse as requested. Recraft V4.1 Flash was also
excluded because its current Image API endpoint accepts text only.

## Scope and controls

The [live capability table](../../experiments/openrouter_outpaint/zeytin-capabilities.md)
and [endpoint snapshots / test plan](../../experiments/openrouter_outpaint/zeytin-plan.json)
record every native resolution and quality control. This is **49 native profiles
× two covers = 98 comparison cells**. Eleven identical successful Kehribar
outputs are reused without another charge. Five FLUX Aklın Hep Bende cells are
withheld following its earlier provider filter, rather than retrying that input.
Eighty-two new requests were planned. Dua Lipa is not in this benchmark.

All supported native tiers are tested, including auto quality. Grok exposes both
resolution and quality, so its four combinations are included. Krea Medium and
Large are distinct model names, not quality controls. Common 9:16 does not imply
the same actual pixel dimensions, and model-native tiers are not cross-provider
resolution guarantees. No aspect-ratio, seed or background sweep is performed.

## Inputs and output handling

The source is the true 720×720 square, center-cropped from the 1280×720 local
resource. It is centered at [0,280,720,1000] on a 720×1280 green target canvas.
Most endpoints receive this canvas plus the original square. Krea and MAI 2.5
Pro accept at most one reference, so they receive only the canvas, and only the
second-reference sentence is removed from the otherwise shared prompt. These
reference counts and source hashes are recorded per call.

Provider routing is pinned without fallback. Each call requests one image; n=1
is sent only when the endpoint supports n. No enforced pixel mask is available
in this reference-image workflow. Raw returned output is kept without source
pasting or color finishing. Endpoint success does not mean a valid outpaint:
for example, the Krea Turbo Karambol output returned a different room scene
instead of preserving the source cover. The user selects visual quality.

## Execution and billing

Two HTTP requests run concurrently under one continuous macOS process-group
memory, pressure and swap guard; there are no local model-weight loads or new
subagents. The sampled ceiling is 1.5 GiB, and each POST has a 480-second
response timeout. Unknown responses stop dispatch. Existing request receipts
are never automatically resubmitted. A $45 new-spend dispatch ceiling reserves
$3 per in-flight request and checks the dedicated key's remaining credit. This
reservation is not a hard provider billing cap; account counters can lag.

MAI 2.6 and MAI 2.6 Flash returned HTTP400 input-filter errors on Aklın Hep
Bende, while Karambol returned images. Their raw error records are retained.
No attempt was made to bypass provider filters.

**82 actual new requests produced 80 raw images and two HTTP400 input-filter failures.** With eleven reused successful baseline images, the comparison contains **91 images** across 98 cells. The five withheld FLUX cells were not requested.

**New stage charge: $4.7298115. Including the earlier $0.2929000 baseline: $5.0227115.** All 91 successful calls' generation total_cost values exactly matched response usage.cost. The dedicated key's settled account total matched the same combined charge. The two definite HTTP errors had zero additional charge under that reconciliation.

| Model | New stage USD | Including reused baseline USD |
|---|---:|---:|
| openai/gpt-image-2 | 0.5034360 | 0.5034360 |
| google/gemini-3.1-flash-lite-image | 0.0684190 | 0.0684190 |
| krea/krea-2-medium-turbo | 0.0300000 | 0.0300000 |
| krea/krea-2-medium | 0.0600000 | 0.0600000 |
| krea/krea-2-large | 0.1200000 | 0.1200000 |
| microsoft/mai-image-2.5-pro | 0.2376500 | 0.2376500 |
| qwen/qwen-image-3 | 0.1440000 | 0.1440000 |
| qwen/qwen-image-3-pro | 0.2540000 | 0.2540000 |
| x-ai/grok-imagine-image-2.0 | 0.6400000 | 0.6400000 |
| bytedance-seed/seedream-5-0-pro | 0.1920000 | 0.1920000 |
| bytedance-seed/seedream-5-0-lite | 0.1400000 | 0.1400000 |
| microsoft/mai-image-2.6 | 0.0522770 | 0.0522770 |
| microsoft/mai-image-2.6-flash | 0.0237080 | 0.0237080 |
| google/gemini-nano-banana-2.1 | 0.3591615 | 0.4440135 |
| tencent/hy-image-v3.5-preview | 0.1600000 | 0.2080000 |
| bytedance-seed/seedream-5-0-flash | 0.0360000 | 0.0720000 |
| black-forest-labs/flux-3-image | 0.4090000 | 0.4330000 |
| openai/gpt-image-2.5-sunburst | 0.6476800 | 0.6977040 |
| openai/gpt-image-2.5-flare | 0.6524800 | 0.7025040 |

Every native setting, cover, actual resolution, latency and charge is listed in the [per-setting table](../../experiments/openrouter_outpaint/zeytin-costs.md), [CSV](../../experiments/openrouter_outpaint/zeytin-costs.csv) and [generation ledger](../../experiments/openrouter_outpaint/zeytin-costs.json).

Batch elapsed time: **2012.75s**. Peak sampled request-group RSS: **263.56 MiB**; peak sampled physical footprint: **228.88 MiB**. Memory pressure stayed normal and swap growth was zero. No guard stopped the batch. These two overlapping memory metrics must not be added; sampling does not guarantee capture of instantaneous peaks.

Receipt coverage, requested controls, input hashes, original-source hashes, all 91 raw image hashes/dimensions and PNG integrity were verified. This confirms record and file integrity, not visual quality acceptance. Model and cover filters were exercised in the live local gallery.

Ten changed publication files are committed individually and grouped into one guarded push. Images and weights are excluded.

## Records and limitations

The machine-readable ledger records resolution/quality, actual pixels, elapsed
time (POST response plus local decode/write; preflight and account lookups excluded),
model and provider identity, resolved model versions, image hashes,
generation IDs and actual USD charges. Missing charge data is unknown; it is not
assumed free. Krea's endpoint pricing was empty, but real request billing is
recorded. Reused images have their original cost and are excluded from the new
stage spend. Generation total_cost is reconciled with response usage.cost and
the dedicated account delta; concurrent key deltas do not attribute per-call
charges. Credit purchase fees, tax and training costs are excluded.

The local gallery includes every profile, separate model/cover filters, the
original square covers and clickable full-size outputs. Overview sheets show
one first profile per model, not all profiles. JSON/CSV/Markdown tables retain
all profile results. One sample per cover/setting does not establish a general
quality ranking, and no teacher winner is selected.

Outputs, source artwork, private request receipts and credentials stay outside
Git. No student training, distillation or phone integration occurred here.
Output-training and source redistribution eligibility remain unestablished.
Each changed publication file has its own zeytin: commit; one publication push.

Sources: [OpenRouter Image API](https://openrouter.ai/docs/guides/overview/multimodal/image-generation),
[OpenAI image controls](https://developers.openai.com/api/docs/guides/image-generation).
