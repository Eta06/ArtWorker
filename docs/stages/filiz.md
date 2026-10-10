# Filiz — single square reference and short prompt

11 October 2026, Türkiye. The user requested three covers with Nano Banana 2.1,
Sunburst and Flare, sending only the square artwork and requesting a portrait
expansion through output parameters. No subagents, downloads, training or iOS work.

## Protocol

Three existing sources: the two earlier comparison covers and one Gripin cover.
The exact reproducible source crop/720×720 resize is retained from previous trials.
Each request sends **one `source.png` reference**, `aspect_ratio: 9:16`, and:

> Extend this square artwork into a seamless 9:16 portrait image by generating the surrounding scene. Keep the original artwork unchanged, fully visible and at the same proportions. Match its colors, lighting, texture and perspective.

Profiles: `google/gemini-nano-banana-2.1` / 1K,
`openai/gpt-image-2.5-sunburst` / high, and
`openai/gpt-image-2.5-flare` / high. Provider routing remains pinned; no fallbacks,
filter workarounds or automatic POST retries. The diagnostic green canvas is saved
locally for the existing fidelity audit but is **not transmitted**. Center placement
is no longer requested. Returned OpenAI images are 864×1536; no exact pixel size is
promised by a native resolution tier.

The default legacy canvas protocol remains intact. Both prompt length/content and
reference count change together, so this is a practical workflow comparison rather
than a controlled causal prompt-only experiment. No seed/repeat sweep was requested.

## Attempts and verified charges

Nine POST attempts: six raw images, three Nano HTTP 402 billing errors. The only
current Nano endpoint, Google AI Studio, reported depleted prepayment credits;
this is an infrastructure failure, not a content rejection or visual quality score.
The OpenRouter key authenticated and had remaining limit. No account/billing
settings were changed and no credits were purchased.

| Model/profile | Images / three covers | Charge USD |
| --- | ---: | ---: |
| Nano Banana 2.1 / 1K | 0/3 | 0 |
| Sunburst / high | 3/3 | 0.122481 |
| Flare / high | 3/3 | 0.122481 |

Each successful output cost **$0.040827**. Total **$0.244962**: all six generation
`total_cost` records agree with response usage costs and the settled key usage
delta; zero unknown charges. Per-request account counters initially lagged.
Three definitive 402 errors reconcile to zero additional charge. See the anonymous
[cost ledger](../../experiments/openrouter_outpaint/filiz-costs.json).

## Preservation and visual limits

All nine receipts confirm a single square reference; six output/source provenance
checks pass. Free global translation/scale/rotation registration is reliable for
four outputs; Karambol/Flare and Gripin/Flare have unavailable alignment scores,
not inferred perfect or zero preservation. No color finishing, local warp or
source pasting was applied. Fixed-location metrics use a diagnostic centered
canvas that was not specified to the model.

Direct visual inspection shows natural portrait expansion is possible without the
green layout. However, Karambol still changes faces/clothing and the source scale;
the short instruction does not guarantee an unchanged cover. Source integrity
requires further evaluation; exterior quality decisions remain with the user.
No teacher was newly accepted and no student dataset or checkpoint was produced.

## Verification, failures and resources

Fifteen offline tests passed in 0.772 seconds. The new wire-payload test independently
decodes the actual request reference and checks square dimensions/pixels, one-image
count, aspect/quality controls and absence of a transmitted placeholder canvas.
Legacy two-reference and one-reference canvas behavior remains covered. Python
compilation and Git whitespace checks passed.

The serial guarded run finished all nine paid calls, then its report rendering
failed because `created_utc` was missing. Original receipts and images were intact.
After repairing the field, a guarded report recovery skipped all existing receipts,
made **no new POST**, and completed successfully. Initial run: 148.72 seconds,
sampled peak RSS **258.56 MiB**, footprint **209.97 MiB**. Report recovery:
2.97 seconds, RSS **159.91 MiB**, footprint **120.11 MiB**. Ceiling: 1.5 GiB;
no memory-pressure/limit stop. Existing system swap was present; it was not a
zero-swap baseline. These are sampled process-group measurements, not iPhone RAM.

Local gallery: `http://127.0.0.1:54146/filiz/`; source-preservation diagnostics at
`/filiz/fidelity/`. Existing Zeytin/Ardıç outputs and votes are preserved; Filiz has
its own preference namespace. The fresh gallery was visibly checked in the in-app
browser, including the three-cover filter, real output links, cost and Nano error
states. Media, source manifests, credentials and per-cover metrics remain ignored.

Reproduction: `square_trial.py --sources <private-three-source-manifest> --results
<private-results-directory>` prepares a gallery offline. Add `--run --key-file
<private-key-path> --budget-usd 2` under `scripts/run_bounded_model.py` to generate.
Existing receipt states are checked before any generation; uncertain outcomes stop
dispatch. A budget reserve is a preflight guard, not a provider-enforced cap.

Next: fix the external Nano billing condition before another explicitly authorized
Nano trial, visually review these six outputs, and compare matched source-preservation
diagnostics with the historical canvas outputs without claiming a single-sample
universal model ranking.

API reference: [OpenRouter Image API](https://openrouter.ai/docs/guides/overview/multimodal/image-generation).

## Publication

The initial five source/document/anonymous-cost files were published as five
single-file `filiz:` commits in one guarded push (`a7faf98`). Subsequent live
fidelity inspection exposed an absent CSV export linked by the shared template.
Added the CSV writer, then regenerated the report without new paid requests and
checked its six-row export. This corrective two-file publication uses two further
single-file commits and one guarded push. No AI trailers, media or credentials.
Record verified publication and the experimental limits in the existing Space Page.
