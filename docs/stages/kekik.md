# Kekik — prompt search with square references

11 October 2026, Türkiye. The previous anchored prompt is retained verbatim.
The user accepts the existing exterior quality and wants better preservation of
the source artwork. Arbitrary source position, uniform scale and asymmetric
padding are valid; placement alone is not a rejection criterion. This stage tests
prompt wording, not student training, phone deployment or model compression.

## Protocol and prompt search

All new requests go to **OpenRouter**, using the user-provided OpenRouter key.
The image POST endpoint is `https://openrouter.ai/api/v1/images`; there are no
direct Google AI Studio calls or Google credentials. Each request sends exactly
one reproducible 720×720 square PNG and requests 9:16. The green diagnostic canvas
is saved locally but **never transmitted**. Sunburst and Flare use native `high`
quality. Source/prompt checksums and exact wire prompts remain in private receipts.

The [six exact prompt candidates](../../experiments/openrouter_outpaint/prompt-candidates.json)
include the unchanged anchored prompt, concise canvas expansion, free placement,
edge continuity, verbatim retention and a literal copy instruction. The last is
an adaptive experiment, not an advertised pixel-lock feature. The discovered
OpenAI endpoint offers aspect ratio, quality, background, n, input references and
output compression; it does not advertise a source-lock mask parameter.

Screening: 18 new OpenAI images on Karambol and Gripin. The two existing Papatya
anchored Karambol outputs are reused rather than regenerated. Four alternative
prompts cover both images/models; the anchored prompt is newly tested on Gripin.
Expansion: anchored and edge-continuity each run on Aklın Hep Bende and two extra
stress cases, a green pixel-art cover and a typography/hand cover: 12 new images.
Adaptive follow-up: copy-square on Karambol and the two stress cases, both models:
six new images. Two final Sunburst/copy-square checks complete its coverage on
Aklın Hep Bende and Gripin: 38 new images in total. Covers come from the existing
local collection; none are newly
downloaded. Stress cases were selected for their visual properties, not as a
random or independent evaluation set. There is one sample per cell, no controlled
seed, and different prompts have different cover coverage. Numerical differences
therefore do not establish causality or population-level quality rankings.

## Nano routing check

Two new Nano Banana 2.1 / 1K attempts: the normal discovered-provider pin, followed
by one explicit diagnostic omitting the entire provider override. Both are
OpenRouter calls. Both return **HTTP 429**, quota exceeded, with
`metadata.provider_name = Google AI Studio`, no image. The fresh image endpoint
catalog advertises only that provider. Removing our pin did not restore generation.
The error does not identify the exhausted quota, reset time or account-side route;
we do not infer that the user's BYOK change failed or call AI Studio directly.
No settings, billing, credentials or purchases are changed; no automatic POST retry.
[OpenRouter image API](https://openrouter.ai/docs/guides/overview/multimodal/image-generation)
and [BYOK routing](https://openrouter.ai/docs/guides/overview/auth/byok).

## Analysis and review

Sunburst aligned SSIM (higher is more similar, not a semantic identity percentage):

| Cover | Original short | Retained anchored | Edge continuity | Copy square |
| --- | ---: | ---: | ---: | ---: |
| Karambol | 0.642 | 0.841 (reused) | 0.822 | 0.854 |
| Gripin | 0.729 | 0.756 | 0.763 | 0.733 |
| Aklın Hep Bende | 0.625 | 0.820 | 0.701 | 0.640 |
| Typography / hand stress case | not run | 0.701 | 0.469 | 0.864 |
| Green pixel-art stress case | not run | unavailable | unavailable | unavailable |

The anchored prompt stays the general candidate; copy-square is retained as an
additional case-dependent candidate. Its Karambol and typography results improve,
but it regresses on the car/photo cover and Gripin. There is no universally better
prompt. On the four alignable Sunburst covers, anchored's worst-tile minimum is
about 0.495 versus copy-square's 0.363; averages cannot hide a weak local region.
Both comparisons exclude the unavailable pixel-art case and are descriptive only.

Flare/copy-square yields SSIM 0.792 on Karambol and 0.604 on the typography case.
Flare/anchored retains the Gripin logo, which the historical short Flare output
dropped; its SSIM improves from 0.617 to 0.717. Its new Aklın Hep Bende SSIM is
0.661. Several Flare alternatives still cannot align a whole source: Karambol
canvas-only/edge-continuity/verbatim, anchored typography, and every pixel-art
variant. These are not automatically scored as poor or perfect quality.

Direct inspection of source/aligned/overlay/heat-map contact sheets confirms
better retention of lettering, hand and cord geometry in the Sunburst typography
copy result; small texture and detail changes remain. Karambol still has local
differences. Pixel-art outputs can reinterpret the grid/face and are also weak
feature-registration cases. No green placeholder was sent, so placeholder color
collision alone cannot explain these observed source changes. This is an inference
from the controlled input, not a diagnosis of the model's internal mechanism.
No universal pixel preservation, automatic training acceptance or perfect prompt.

The same ORB→SIFT reliability gates and global similarity registration are retained.
Analysis uses a common 512×512 source grid. Translation, global scale and rotation
are compensated; there is no local warp, color correction or source paste.
Unavailable whole-source registration stays unavailable, never a preservation
percentage. Repetitive pixel art is a particularly weak case for feature-based
registration, and visual changes can coexist with detector limitations.

The new local gallery and fidelity report include a prompt/setting filter. The
aligned wipe, 50/50 overlay, heat map and orange-region diagnostics remain available.
CSV exports now also include worst-tile SSIM and exact equality on the analysis
grid. Exact-grid equality and RGB tolerances are not native-pixel or semantic
identity guarantees. Older galleries, images, receipts and preference stores are
preserved. New votes use the `kekik` namespace. Source media, manifests, raw outputs,
per-cover metrics, credentials and generation identifiers remain ignored by Git.

## Actual charges and verification

Forty new image POST attempts: 38 OpenAI images, two Nano HTTP errors. All new
OpenAI outputs are 864×1536. New charges **$1.561256**, settled whole-stage account
delta **$1.561256**, zero unknown costs. Sunburst: 20 images / **$0.821760**;
Flare: 18 / **$0.739496**; Nano: two errors / **$0**. Individual batches initially
had lagging counters, so final reconciliation uses the stage's initial account
snapshot and all generation costs, not a sum of overlapping counter windows.
Eight reused baseline images add no new charge. The
[anonymous cost ledger](../../experiments/openrouter_outpaint/kekik-costs.json)
contains each actual attempt and setting without source identifiers or credentials.

Twenty-four offline tests pass in 1.295 seconds. Tests cover the actual custom
prompt/square wire payload, default pinned versus automatic routing, provenance,
registration gates, stopping before POST on low credits or changed sources,
uncertain-request non-reposting and lagging-account budget control. Changed Python
compiles; prompt catalog validates; Git whitespace checks pass. No extra paid
requests are made during reporting or browser checks.

The report includes 46 images (38 new + 8 reused), **35 reliable whole-source
alignments**, 11 unavailable, and two Nano error gallery cells. Live browser checks
confirm combined model/cover/prompt filtering, wipe clipping and loaded images,
the unavailable-region caption including inlier/coverage gates, and clearing the
prompt filter when selecting favorites. No votes are cast. The 46-row CSV includes
the new fields with unavailable aligned measurements blank; all 46 raw-image URLs
return HTTP 200 through the local report aliases. Gallery and fidelity report:
`http://127.0.0.1:54146/kekik/` and `/kekik/fidelity/`.

All generation/report runs use serial execution, a 1.5 GiB process-group ceiling,
100 ms memory/pressure sampling and a 256 MiB swap-growth stop. Sampled seconds /
RSS MiB / physical footprint MiB:

| Run | Seconds | RSS MiB | Footprint MiB |
| --- | ---: | ---: | ---: |
| probe-run | 1.355 | 47.70 | 32.31 |
| default-route-run | 1.203 | 47.31 | 31.89 |
| round1-run | 440.704 | 265.31 | 221.22 |
| round2-run | 298.094 | 195.94 | 146.67 |
| round3-run | 145.231 | 135.66 | 86.52 |
| round4-run | 55.973 | 291.92 | 242.38 |
| report-run | 16.885 | 319.09 | 281.50 |

No memory/pressure stops. Existing system swap was present; RSS and footprint
share memory and must not be added. These are sampled Mac process measurements,
not iPhone measurements. No local model weights are loaded.

## Reproduction and publication protocol

`benchmark.py --input-mode square --prompt-file <UTF-8-prompt>` supports controlled
prompt variants; `--automatic-routing` omits provider pinning. The original short,
anchored and legacy canvas defaults remain available. Custom prompts are validated
before network access and cannot be used with the canvas mode.

`prompt_search.py --plan <private-plan> --results <fresh-private-directory>
--key-file <private-key> --run --budget-usd <batch-budget>` dispatches serially.
A plan pins model, track, quality/resolution, profile, source/prompt relative paths
and checksums, status and output directory. Without `--run`, it only reconciles
receipts with billing GETs. Completed/error/no-image cells are not reposted;
uncertain outcomes stop. The budget preflight reserves $0.50 per request and uses
the larger of account movement and already reported receipt charges because
account counters lag. This is a soft preflight, not a provider-enforced cost cap.
Run it under `scripts/run_bounded_model.py` with continuous resource monitoring.
`preservation_comparison.py` renders baselines and new ledgers completely offline.

Publish each changed code/test/prompt/document/anonymous-cost file in its own
`kekik:` commit, followed by one guarded push. No model weights, private media or
credentials enter normal Git. Record verified remote state and results in the
existing ArtWorker Space Page. No subagents, model downloads, training, simulator
work or released student checkpoint in this stage.

Next work depends on these observed source failures: retain the strongest tested
candidate, review individual training pairs, and investigate a separate known/fill
mask with enforced source pixels rather than relying on text for an absolute
preservation guarantee. No training eligibility or universal perfect prompt is
claimed here.
