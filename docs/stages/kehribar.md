# Kehribar — OpenRouter outpainting comparison

6 October 2026. The six exact user-selected models were requested on Aklın Hep
Bende (`track3`) and Karambol (`track2`); Dua Lipa was excluded. **13 paid-endpoint
attempts produced 11 images. Total charge: $0.292900.** Eleven successful calls'
Image API `usage.cost` values exactly matched generation `total_cost`; the
dedicated key's usage rose from zero to the same total. Two definite FLUX failures
had no additional charge according to that reconciliation.

## Actual generation costs

| Model | Aklın Hep Bende | Karambol | Total USD |
| --- | ---: | ---: | ---: |
| Seedream 5.0 Flash | 0.018000 | 0.018000 | 0.036000 |
| Nano Banana 2.1 | 0.044526 | 0.040326 | 0.084852 |
| Hy Image 3.5 Preview | 0.024000 | 0.024000 | 0.048000 |
| FLUX 3 Image | 0; no output | 0.024000 | 0.024000 |
| GPT Image 2.5 Sunburst | 0.025012 | 0.025012 | 0.050024 |
| GPT Image 2.5 Flare | 0.025012 | 0.025012 | 0.050024 |

These are observed charges for these requests, not general price guarantees or
1000-image quotes. Credit purchase fees/tax and training compute are excluded.
FLUX's 1K endpoint snapshot listed $0.048; actual charge was $0.024, consistent
with the current promotion. Do not replace real billed amounts with catalog rates.
The [ledger](../../experiments/openrouter_outpaint/kehribar-costs.json) contains
generation IDs, resolved model versions, providers, timings, dimensions and hashes.

## Setup and execution

Original files were 1280×720 with flat side bars; the actual square was cropped
from `[280,0,1000,720]`. Its 720×720 pixels were placed at `[0,280,720,1000]` on a
720×1280 green placeholder canvas. Every model received the same two references
(canvas and original square) and prompt. The shared ratio was 9:16, rather than
the local 512×1152 test. Providers were pinned without automatic fallback.

Request 1K where supported; Sunburst/Flare instead used medium quality. Returned
dimensions differed: Seedream 720×1280; Nano Banana 768×1376; Hy 576×1024; FLUX
768×1360; Sunburst/Flare 864×1536. This is not a comparison at identical pixel
counts. Both reference images are charged where applicable.

| Model | Aklın seconds | Karambol seconds |
| --- | ---: | ---: |
| Seedream | 16.339 | 16.563 |
| Nano Banana | 24.048 | 16.690 |
| Hy | 32.355 | 33.610 |
| FLUX | no image | 69.079 |
| Sunburst | 22.774 | 23.339 |
| Flare | 15.137 | 15.544 |

FLUX track3 first returned HTTP 503 (over capacity). A separate explicit retry
after the batch returned HTTP 400: provider graphic-violence filter. Both receipts
are retained; no attempt was made to bypass the filter or substitute a model.

Calls ran serially with continuous macOS pressure/swap monitoring and a 0.6 GiB
sampled process-group memory ceiling. Peak sampled request-group RSS was
**64.17 MiB**; no guard stopped a call and no new subagent was launched. No local
model weights were loaded. As elsewhere, sampled guards cannot guarantee capture
of every instantaneous peak.

## Direct visual review

All eleven full raw outputs and the two labeled comparison sheets were inspected.
No source restoration, seam blending or color finishing was applied to the output.

| Model | Observations from these samples |
| --- | --- |
| Seedream | Lowest charge. Road/water continue, but the road cover's logo is duplicated at the bottom. Karambol's printed bag becomes a physical hanging bag; source layout moves. |
| Nano Banana | Road cover has coherent water/rail extension without a duplicated logo. Karambol keeps the bag as a print and completes clothing plausibly. Texture/colors and some source details still change. |
| Hy | Road expansion is plausible at the smallest output resolution. Karambol has a severe blue rectangular interruption replacing the lower-right person's continuation; reject that sample. |
| FLUX | Karambol keeps the bag printed and completes bodies, but clothing details/texture change and the upper background is visibly soft. No road output. |
| Sunburst | Road image has an attractive coast/sky continuation; it introduces a sunset and redraws source content. Karambol keeps the print but changes source scale/position, clothing and facial details. |
| Flare | Faster than Sunburst here at the same charge. Karambol keeps the print, with changed clothing/details. The road cover has a discontinuity in the upper rail continuation and changed scene content. |

The reference-image API does not expose an enforced pixel mask in this workflow.
None of the normalized raw centers was pixel-identical to the original. The
ledger's center MAE is a geometry/pixel-change diagnostic, **not a visual score**;
Hy's small MAE does not make its broken Karambol output acceptable. Attractive
raw images are not automatically valid paired outpainting training targets.

## Outcome and next work

Seedream is cheapest in this pilot, but its visible errors make price alone an
insufficient selection criterion. Nano Banana is a promising source-faithful
candidate in these two samples; Flare merits a cost/speed comparison. No universal
teacher was selected from one sample per cover/model.

Next, evaluate a source-preserving conditioning route and accept outputs by seam
continuity, original geometry, clothing/print integrity and billed cost per
accepted image. Resolve provider output-training terms and source-image rights
before producing a training dataset. No student training, distillation, iPhone
integration or model-size change occurred in this stage.

Raw/source images and detailed local receipts remain under the ignored
`experiments/openrouter_outpaint/results/2026-10-06/kehribar/` directory.
Credentials remain in a private local key file and are not in the publication.
The stage publishes five files in five separate `kehribar:` commits, one push.

Sources: [OpenRouter Image API](https://openrouter.ai/docs/guides/overview/multimodal/image-generation),
[generation billing](https://openrouter.ai/docs/api/api-reference/generations/get-request-&-usage-metadata-for-a-generation),
[OpenAI model documentation](https://developers.openai.com/api/docs/models/gpt-image-2.5-sunburst).
