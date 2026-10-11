# Susam — direct Nano batch trial

Date: 2026-10-11.

## User decision and primary stop

The user requested pausing Sunburst and trying approximately 100 covers through
batch generation. The Sunburst pause marker was written immediately. Existing
requests drained, their results were retained, and the Pelit guard exited with
code zero. Its final primary count was 177 completed Sunburst images. Six Nano
images were present at that stop; two previously submitted fallback jobs remained
recoverable. No Sunburst generation is performed by the new trial worker.

## Trial protocol

The private queue saves a fixed selection of 100 distinct previously unattempted
cover hashes, following the existing artist priority. Completed covers are not
regenerated. The teacher is Gemini Nano Banana 2.1 through the native Google
Batch API, at 1K and 9:16. The frozen old-canvas prompt, two reference images,
and 8192 output token cap are preserved. No lower token cap or image-quality
setting was introduced to make the budget appear smaller.

The API's discounted asynchronous route is documented by Google:
[Gemini Batch API](https://ai.google.dev/gemini-api/docs/batch-api).
This trial uses two-image inline jobs, with at most eight jobs outstanding, so
request and response bodies remain small. The 100-cover plan does not mean all
100 requests are submitted immediately. Polls are at least 60 seconds apart in
the production launch. Results and source-registration audits are retained.

## Budget and safety of execution

Existing Google spending and unresolved reservations are carried forward.
Every new job requires room for its worst-case output-token reserve with a 10%
conversion buffer. Reservations are released to the usage-based estimate only
when complete billing token details are returned. Missing usage is not a zero
charge. Exhausted capacity waits for outstanding jobs; exhausted budget with no
outstanding jobs stops the trial. There are no automatic generation POST retries.

Observed previous billed-token estimates suggest roughly 100–120 TL for 100
similar images. This is a planning estimate, not an invoice or guarantee. The
API key does not provide the actual Google billing balance or invoice FX. The
trial preserves the user's existing spending cap; account balances and provider
job identifiers remain private.

## Execution evidence and limitations

The first two previously pending fallback results were retrieved without a new
generation, bringing the older Nano total to eight images. The new 100-cover
trial's first two-image job was accepted; the remaining items are scheduled
under the rolling reservation policy. The worker runs under the existing guard
with a sampled 512 MiB RSS/footprint limit, memory-pressure and swap-growth checks,
and a 24-hour local runtime limit. Provider job IDs and the trial plan survive a
local stop and permit reconciliation without resubmission.

The combined queue and direct-trial suite passed 17 offline tests. New coverage
checks distinct source selection, preservation of completed covers, explicit
submission eligibility, the Sunburst pause prerequisite, stopping at the budget,
and resuming a completed plan without a second generation. No paid requests were
used by these tests. `git diff --check` passed.

A successful response is not an accepted training example. No-image outputs are
quarantined. The source-region alignment audit allows translation and scale;
SSIM is not a percentage of unchanged pixels. Exterior quality, seams and source
preservation still require review. No student training or checkpoint release
occurred. The 100-cover trial is started, not reported as completed.

## Subsequent routing correction

While the first submission wave was running, the user requested running Nano
through OpenRouter batch instead. The Nano pause marker was written immediately;
16 new covers were already submitted in eight native Google jobs and the other
84 were not sent. Submitted jobs and their reservations are preserved. No
OpenRouter generation request was made and no synchronous substitute was used.

Live read-only OpenRouter discovery returned only the ordinary
`google/gemini-nano-banana-2.1` model, served by Google AI Studio with no batch
variant. OpenRouter does provide a Batch API, but its current
[Batch Quickstart](https://openrouter.ai/docs/batch-quickstart) explicitly rejects
chat-completion requests asking for non-text output through `modalities` or
`image_config`. Image inputs are provider-dependent URL references; input image
support does not imply image generation support. The requested OpenRouter image
batch route is therefore unavailable. Both local generation queues remain
paused, pending a decision on whether to resume the remaining Google batches.
