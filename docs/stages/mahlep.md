# Mahlep — budgeted Sunburst queue and native Google batch fallbacks

2026-10-11. The user authorized generation within separately supplied account
budgets and reaffirmed Sunburst as the main teacher. Nano Banana is reserved
for covers Sunburst refuses or completes without an image. Unused Google credit
does not justify silently changing the primary teacher to Nano.

## Verified batch availability

Native Google `gemini-nano-banana-2.1` exposes `batchGenerateContent` through the
supplied key. Image generation batches receive a 50% discount and target completion
within 24 hours. The route changes from the historical direct Interactions API to
`generateContent` in Batch API; this deserves a small protocol/quality pilot.
No Google generation was submitted during the four-cover Sunburst pilot.

Native OpenAI supports Sunburst image edits in Batch API. However, the user funds
OpenRouter, not a native OpenAI account. OpenRouter's current Batch API rejects
image-output configurations and does not expose the dedicated Image API used
by the selected baseline. No unsupported image batch was submitted there.
Sources: [Google guide](https://ai.google.dev/gemini-api/docs/batch-api),
[Google reference](https://ai.google.dev/api/batch-api),
[Google pricing](https://ai.google.dev/gemini-api/docs/pricing),
[OpenAI batch](https://developers.openai.com/api/docs/guides/batch),
[OpenRouter limitations](https://openrouter.ai/docs/batch-quickstart).

## Execution and billing controls

The new executor freezes the old-canvas prompt hash and two-reference order.
The baseline remains Sunburst `high`, 9:16, green-placeholder canvas then original
square. Benchmark assertions now check prompt hash and reference count **before
the paid POST**, including when endpoint capabilities change.

The private plan pins 1,000 Erguvan source hashes. Production selection interleaves
UZI, Motive, Güneş and Murda first; other Turkish sources precede remaining covers.
No source was downloaded again, and no old experiment was overwritten.

Primary spending uses the greater of known charges and live account-credit
movement, rather than mistaking the API-key limit for account credit. All balance
values and source preferences stay local. A conservative $0.10 per-call reserve
leaves a small remainder; unknown billing, unexpected prices and ambiguous
completion stop subsequent calls. Receipts and intents forbid blind POST retries.
There are no automatic technical retries in this implementation.

Nano fallback batches contain at most two requests each. The production command
can submit only the first protocol pilot; further batch submissions require
inspection of its output. It stores the remote job ID immediately and can retrieve
results without generating again. Source IDs, response count, original/reference
hashes and dimensions are verified before free alignment. Failed batches or
missing outputs are quarantined; missing usage is not labelled free.

Google uses $0.75/M input, $3.75/M text/thinking and $15/M image tokens for Batch.
1K image output alone is $0.0168. Reservations use counted input tokens plus the
8,192 output-token cap at the highest output rate, and a 10% allowance. Usage-based
estimates never become claimed invoice charges. Actual Google balance, taxes and
TRY billing conversion are unavailable through this key; local reservations are
not a provider-side budget cap. The 49.27 TRY/USD conversion is a planning
assumption, not Google's billed exchange rate.

## Paid pilot and source preservation

Four new Sunburst images completed: UZI / Turkish Nightmare, Motive / ŞEHRİME
BOMBALAR, Güneş / NKBİ X YAPAMAM (Remix), Murda / SURF. Each reported and
generation-reconciled charge is **$0.048952**; total **$0.195808**. Each output is
864×1536. Prompt SHA-256 remains
`736716dd0146c1cbda160d21c6615dbe78e494a6e98dc779cb94701bd006efc8`.
All four used the same two canvas/source references. No fallback was needed.

Free source alignment measured SSIM×100 of 81.81, 93.37 and 85.46 for UZI, Motive
and Murda respectively. Their visible-source fractions were 99.91%, 100%, 100%.
The Güneş cover had spatially concentrated matches; it remains **unmeasurable**
with this conservative detector, rather than receiving a fabricated percentage.
These are source-content measurements, not exterior-quality scores, unchanged
pixel percentages or human training approval. Murda had a larger color difference
than Motive. The UZI raw output was visually inspected; the whole pilot has not
received a human acceptance decision.

## Validation and resources

Seven new offline tests passed: credit/receipt movement reconciliation, technical
versus policy failure routing, priority/dedup selection, unknown-submission
protection, frozen two-reference payload, nested REST batch outputs/state, and
missing-usage/paid-thinking estimates. Five existing Google trial tests also
passed. No paid request was made by tests. Whitespace checks passed.

The four-call pilot used the existing 100 ms memory guard, 512 MiB process-group
limit and memory-pressure/swap-growth stops. Duration 124.19 seconds; sampled
peak RSS **136.50 MiB**, physical footprint **86.80 MiB**. The separate four-source
alignment job took 1.70 seconds, RSS **275.81 MiB** / footprint **238.25 MiB**.
RSS and footprint overlap and are not added. Pressure stayed normal. There was
pre-existing system swap; the guard limits growth rather than claiming zero swap.

New agents: 0. Student training/checkpoints: 0. At this publication snapshot the
larger paid queue has not launched. Any later queue status lives in the ignored
private state, and the Space progress entry records its actual launch separately.
Artwork, request bodies, credentials, account balances and job IDs are excluded
from Git. Publication follows one changed file per commit and one guarded push.
