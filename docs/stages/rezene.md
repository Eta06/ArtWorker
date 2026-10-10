# Rezene — direct Google Nano Banana 2.1 trial

11 October 2026, Türkiye. The user supplied a Google AI Studio credential and
explicitly authorized using Nano Banana directly. Credentials stay outside Git
in a private mode-0600 local file. Existing OpenRouter experiments are preserved;
this stage changes the route, not the historical Kekik routing record.

## Actual requests

Authenticated model listing returned HTTP 200 and advertised
`models/gemini-nano-banana-2.1`. The two successful requests use the direct
`https://generativelanguage.googleapis.com/v1beta/interactions` endpoint, model
`gemini-nano-banana-2.1`, with `store: false`, a single 720×720 source PNG and the
unchanged anchored prompt. Requested output: 9:16, 1K. No green canvas, separate
mask, source paste, color correction, new download, student training or local
model load. Sources: existing Karambol and Aklın Hep Bende covers.

Three POST attempts, no automatic retries:

1. Karambol / PNG output: HTTP 400. The live endpoint said only `image/jpeg` was
   supported for `response_format.mime_type`, despite PNG examples in the current
   official image guide. No image and no usage returned.
2. A fresh Karambol request with JPEG output: HTTP 200, one image.
3. Aklın Hep Bende / JPEG output: HTTP 200, one image.

Both images are natively 768×1376, approximately the requested portrait ratio.
JPEG bytes are decoded and losslessly saved as PNG for analysis; this does not
undo the original JPEG compression. Input remains PNG. The adapter refuses an
existing receipt to avoid repeating an uncertain or charged request.

[Official image guide](https://ai.google.dev/gemini-api/docs/image-generation),
[Interactions API](https://ai.google.dev/api/interactions-api).

## Source preservation

Existing ORB→SIFT registration, unchanged whole-source reliability gates, free
position/scale/rotation, common 512×512 analysis. No local warp or source replacement.
Two prior Sunburst/anchored outputs are reused, with no new OpenAI calls or charges.

| Cover | Direct Nano 2.1 aligned SSIM | Prior Sunburst aligned SSIM | Nano visible source |
| --- | ---: | ---: | ---: |
| Karambol | 0.6770 | 0.8410 | 97.96% |
| Aklın Hep Bende | 0.8412 | 0.8201 | 99.63% |

Both Nano source regions align reliably with features in all four quadrants.
Karambol's source retention is weaker on this sample; the car/photo cover has
better numerical similarity than the reused Sunburst sample. Neither establishes
a general model ranking. There is one sample per cover/model, different native
dimensions and JPEG versus prior PNG output. SSIM is not a percentage of unchanged
content, and registration visibility is not fidelity. Faces, lettering, texture
and smaller details can still change. Exterior quality remains the user's choice;
these outputs are not automatically accepted as training targets.

Local review: `http://127.0.0.1:54146/rezene/` and `/rezene/fidelity/`, including
the previous Sunburst outputs, source overlays, difference maps and orange-region
diagnostics. Media, per-cover measurements and receipts remain ignored by Git.

## Cost and limits

Google's response reports token usage, not actual charged dollars. Current standard
rates: input $1.50/M tokens, image output $30/M, text/thinking output $7.50/M.
Image-only responses can omit internal text from modality breakdowns; the estimator
uses aggregate output minus image tokens, plus separately reported thought tokens.
No cached input, tools or search was requested or reported. These estimates assume
standard paid rates and are not reconciled invoices or account balance measurements.

| Successful request | Estimated USD including input/text/thinking |
| --- | ---: |
| Karambol | $0.043284 |
| Aklın Hep Bende | $0.043059 |
| Total successful requests | **$0.086343** |

The first rejected request has no usage; its actual billing is unknown rather than
asserted $0. Actual billing is unavailable for all three attempts. Published 1K image
output alone is $0.0336, excluding input and text/thinking. Anonymous estimates are
in [rezene-costs.json](../../experiments/openrouter_outpaint/rezene-costs.json).
[Google pricing](https://ai.google.dev/gemini-api/docs/pricing).

Generation process peak sampled RSS: 66.63 MiB. Report peak: 182.77 MiB.
Serial runs use the existing 1.5 GiB process guard, 100 ms pressure/RSS/footprint
sampling and 256 MiB swap-growth stop; sampling is not a kernel memory limit.
No subagents were created. No new model weights or simulator processes were loaded.

Offline validation: 29 tests, including direct routing/source serialization, model
output parsing, internal-text cost estimation and unknown billing display. Python
compilation and whitespace checks pass. Next: user review, additional retained-prompt
stress covers if selected, and explicit training masks; no student checkpoint yet.

Publication follows one changed file per commit and one guarded push. Stage source,
tests, shared unknown-cost UI handling, documentation and anonymous cost record are
published; secrets, source artworks and generated media are excluded.
