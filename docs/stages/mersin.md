# Mersin — existing prompt/output archive

2026-10-11. The user prohibited further generation and requested every existing
prompt and its outputs. No provider requests were made in this stage; no new
generation cost was incurred. Existing receipt and output files were preserved.

## Quality correction

The user rejected the wall-cut Karambol output from Papatya's `anchored` Sunburst
request, later reused in Kekik and Rezene. Its aligned source SSIM of 0.841 does
not establish good outpainting. Promoting it on source similarity alone failed
to preserve the visual quality the user wanted. This correction concerns that
specific output; it does not imply all anchored outputs are rejected.

The archive highlights the earlier Zeytin Sunburst **high** Karambol output,
marked liked in the user's exported selections. Its actual request used the
long `old-canvas` prompt with two references: a 720×1280 canvas with green
placeholders and the original 720×720 square. The complete request prompt and
actual references are available beside the output, rather than reconstructed
from a later candidate description.

Reference raw SHA-256:
`581659d06d0fbd54bff2252b1b2a4c4eaee8395d61fa058efc1594c58867122e`.

Rejected Papatya raw SHA-256:
`d0f190f3d057acd1d28a89e2db8c9fb0a3a8a8ad8a7620bcef9ddb245beccb22`.

## Actual coverage

The local OpenRouter and direct Google receipt archive contains **9 exact prompt
texts**, **19 normalized models**, **176 requests**, **151 existing images**,
and **25 requests without a returned image** (24 HTTP errors and one completed
response without usable output). The same Nano model accessed through two
providers is counted once; each request retains its route. Symlinked result
aliases are excluded from receipt counting. This is teacher API history, not
an inventory of every local model or training experiment.

| Exact prompt | Requests |
| --- | ---: |
| old-canvas | 108 |
| old-canvas-single | 8 |
| short | 17 |
| anchored | 13 |
| canvas-only | 4 |
| free-layout | 4 |
| edge-continuity | 10 |
| verbatim | 4 |
| copy-square | 8 |

Kekik tested six candidates: anchored, canvas-only, free-layout, edge-continuity,
verbatim and copy-square. Not every prompt was tried with every model. The
matrix explicitly distinguishes untried pairs, failed requests and image
outputs. Prompt grouping uses exact UTF-8 text SHA-256, without whitespace
normalization. An anonymous summary is versioned in
`experiments/openrouter_outpaint/prompt-archive-summary.json`.

## Viewer and validation

Local archive: http://127.0.0.1:54146/mersin/ . Model, cover, prompt and request
status filters show all existing attempts. Each card links the original full
image and actual input references. Historical charges are labelled as reported
or estimated; failure charges are not silently assumed to be zero. Local art,
receipts, previews and the user's votes remain outside normal Git.

The offline builder processed one preview at a time under the memory guard:
4.96 seconds, sampled peak RSS **171.50 MiB**, sampled physical footprint
**159.77 MiB**, a 1 GiB cap and normal memory pressure. Receipt hashes were
checked after building; original output hashes were checked when creating
media aliases. All original bytes were preserved.

All **32** OpenRouter experiment unit tests passed, including three new archive
tests for exact prompt grouping, route normalization/symlink exclusion and
source preservation. Provider calls in those tests are mocks. The browser
showed 176 requests / 151 images in the default view and 13 / 13 for Sunburst
Karambol; the original reference and complete prompt rendered correctly.

Next work is the user's review of existing outputs. No new prompt experiment,
student training, checkpoint release or physical iPhone measurement happened.
