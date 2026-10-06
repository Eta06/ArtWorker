# Ardıç — five-cover teacher consistency

6 October UTC / 7 October 2026 Türkiye. The user requested five new songs and
album covers from their logged-in Chrome YouTube Music account, then controlled
outpainting trials with Sunburst, Flare, GPT Image 2 and Nano Banana 2.1. Teacher
generation time is explicitly not a selection criterion. Source preservation,
exterior quality and actual dataset-example cost are the relevant dimensions.
No subagents, local weight downloads, student training or simulator work.

## Sources and protocol

Selected five distinct album releases from the visible library. Downloaded their
actual square album images, rather than video thumbnails: all are 544×544.
Downloaded five public audio URLs serially using the installed yt-dlp; the
matching source/audio manifests include hashes and ffprobe durations. One
best-audio download returned HTTP 403; explicit audio format 140 succeeded.
Chrome account cookies were not exported. Music, covers, URLs, full receipts,
preferences and per-model source-content scores remain local and ignored.

The new manifest-based runner uses the existing two-reference prompt and
720×1280 green-placeholder layout, with the 720-square source at (0,280).
One high-quality request for each OpenAI model and one 1K Nano request per cover:
twenty comparison cells, pinned providers without fallbacks. Actual returned
OpenAI dimensions are 864×1536; Nano dimensions are 768×1376. Native quality tiers
are different controls, not an identical-resolution experiment. Source images
contain no new detail above their original 544-pixel resolution.

Custom source receipts pin the original relative path, hash, crop and resize.
The fidelity audit verifies that provenance and uses Sumak's free global
translation/scale/rotation alignment on a common 512-pixel analysis grid.
No color correction, local warping or original-source paste into outputs.
All twelve returned images passed provenance checks and produced reliable
aligned comparisons. RGB agreement thresholds, SSIM, color error, visible
coverage and worst-region measurements are diagnostics, not semantic identity,
native pixel equality or automatic training admission. Exterior quality and
boundary continuity still require the user's visual review.

## Actual attempts and billing

Twenty-one paid-request attempts: twelve images, eight definitive HTTP 400
provider filter rejections, and one completed generation whose image response
was lost. Do not count a provider rejection or a missing output as visual quality.

| Requested model/profile | Returned images / five covers | Actual stage charge USD |
|---|---:|---:|
| GPT Image 2.5 Sunburst / high | 3/5 | 0.1468560 |
| GPT Image 2.5 Flare / high | 2/5 | 0.0979040 |
| GPT Image 2 / high | 2/5 | 0.2917440 |
| Nano Banana 2.1 / 1K | 5/5 | 0.2487585 |

Total **$0.7852625**, reconciled exactly against the dedicated key's usage delta.
All thirteen billed generations were confirmed through their exact generation
IDs; there are no unknown charges. The eight definitive rejections reconcile
to zero additional cost. Nano's total includes **$0.0399885** for the lost output
and **$0.0425685** for a subsequent fresh recovery request. See the anonymous
[per-attempt cost ledger](../../experiments/openrouter_outpaint/ardic-costs.json).
Fees/tax, future training compute and visual acceptance costs are excluded.

The original Nano transport timed out after 600.279 seconds and paused dispatch.
Chrome OpenRouter Logs identified the sole matching dedicated-key generation at
the request timestamp. The generation API confirmed completion and its exact
cost; the stored-content API had no recoverable image. Preserved the original
failure receipt, reconciled the attempt as completed-output-unavailable, then
made one fresh recovery request in a separate directory. No blind POST retry or
filter workaround. The recovered image is a new sample, not the missing original.

The JSON reader now consumes bounded chunks, returns once the complete JSON
document is available, and retains response IDs on read errors. Known-answer
tests cover a relay that never closes after valid JSON and a partial-response
timeout. This hardens transport handling; it does not establish the original
timeout's cause. There is a $5 dispatch budget with a $0.50 reserve per request;
this preflight is not a provider-enforced spending cap.

## Validation and resource use

Twelve known-answer tests passed in 0.741 seconds, including independent custom
crop/provenance controls and unsafe-path rejection. Python compilation and Git
whitespace checks passed. The first guarded batch paused after the transport
failure (880.25 seconds); the resumed batch finished in 431.71 seconds. Highest
sampled process-group RSS was **303.64 MiB**, physical footprint **277.42 MiB**,
under a 1.5 GiB ceiling. Pressure stayed normal and swap stayed zero. These are
sampled process-group measurements, not total system or iPhone memory usage.
Final report-only reconciliation/audit used 176.52 MiB RSS and took 6.39 seconds.

The local gallery has five dynamic cover filters, unmodified full-size outputs,
per-attempt costs, explicit missing-output states and its own `ardic` preference
namespace. Historical Zeytin preferences and reports remain intact. Browser
verification checked the four-image UZI filter, all loaded output dimensions,
enabled review buttons, five-cover fidelity filters, aligned/fixed metric
changes, loaded diagnostic images, and CSV/JSON endpoints. No user votes were
changed. Screenshots and five visual comparison sheets are local artifacts.

## Interpretation and next work

The current Image Edit Arena snapshot places Sunburst first, Flare second,
GPT Image 2 **medium** third and Nano 2.1 sixth. Our high-quality GPT Image 2
setting differs from that leaderboard label. The general-edit ranking motivates
these candidates but does not decide album-source preservation. Unequal returned
cover sets also make a single cross-model mean misleading. Compare matching
covers, keep worst-case source changes visible, and include rejection/lost-output
costs separately from image quality.

Keep Nano and Sunburst as candidates for the user's new visual selections;
inspect seams alongside aligned content before choosing teacher examples.
This is a small, mostly single-sample comparison, not a universal model ranking.
No student checkpoint or trained phone model was produced. Training and
redistribution eligibility are not established by these visual evaluation calls.

Sources: [Image Edit Arena](https://arena.ai/leaderboard/image-edit),
[OpenRouter Image API](https://openrouter.ai/docs/guides/overview/multimodal/image-generation),
[generation billing](https://openrouter.ai/docs/api/api-reference/generations/get-generation),
[yt-dlp FAQ](https://github.com/yt-dlp/yt-dlp/wiki/FAQ).

## Publication

Publish eleven changed source/document/anonymous-cost files, exactly one file per
`ardic:` commit, grouped in one memory-guarded push. Media, credentials and local
content scores are excluded. Reconcile this stage and the current priorities on
the existing ArtWorker ChatGPT Space Page after verifying remote publication.
