# Erguvan — frozen teachers and 1,000 new source covers

2026-10-11. The user requested a concrete main/fallback teacher policy and a
new cover corpus from their authorized Chrome YouTube Music account. The
initial 600–700 target was raised to **1,000 new covers**, with the final 300
focused on four requested Turkish artists and adjacent artists in that scene.
No new teacher inference, audio download or student training was performed.

## Teacher policy

Main: OpenRouter `openai/gpt-image-2.5-sunburst`, `high`. Fallback:
`google/gemini-nano-banana-2.1`, `1K`. Flare is comparison-only. Both teachers
retain the exact previously tested **old-canvas** prompt and its two references;
the text matches the published prompt archive, SHA-256
`736716dd0146c1cbda160d21c6615dbe78e494a6e98dc779cb94701bd006efc8`.

[`teacher-policy.json`](../../experiments/openrouter_outpaint/teacher-policy.json)
and [the queue plan](../teacher-pipeline.md) distinguish a definitive refusal,
a confirmed empty response, a transient technical failure and an ambiguous
timeout. They cap retries and require billing reconciliation before a fresh
POST when completion is uncertain. This is a saved policy, not an API executor.
Collection does not trigger paid generations.

Source placement is free: shifted/scaled or asymmetrically expanded covers can
be useful training examples. Detect the actual source footprint, compare its
aligned content without deformation/color correction, and keep full-source
visibility separate from SSIM and RGB differences. No universal acceptance
threshold has been calibrated; SSIM is not an unchanged-pixel percentage.
Unreliable detection, partial sources and visually rejected results require
review. The teacher choice does not finalize or train the local iOS student.

## Collection and provenance

- **1,000 accepted new covers**, in addition to the preserved Defne archive.
- 336 were previously observed discovery candidates not retained in that archive;
  364 came from newly observed liked-artist discographies; 300 came from the
  additional Turkish-scene discovery pass. No direct liked-song cover was newly
  accepted because those observed covers already matched existing artwork.
- The Turkish pass observed 494 release candidates across nine artist catalogs.
  The four requested catalogs were processed first, then adjacent catalogs were
  interleaved. All final 300 accepted covers came from this focused pass.
- Final decoded sizes: **994 at 1024×1024, three at 800×800 and three at 900×900**.
  Files retain downloaded bytes; CDN dimensions do not prove native upload detail.
- Accepted artwork occupies **212,660,864 bytes / 202.81 MiB**.

The authorized Chrome-cookie yt-dlp liked-playlist metadata export contained
1,164 entries. The Music DOM pass captured 882 distinct song URLs, 727 with
loaded artwork; this is not complete artwork coverage of all playlist entries.
The extractor returned video thumbnails, so these were not admitted as covers.
Actual album images came from rendered Music release cards and observed Google
album-image CDN URLs. Cookies were loaded inside yt-dlp, without exporting a
cookie file. All song titles, preferences, artwork, manifests and logs remain
local under the ignored `experiments/cover_collection/results/2026-10-11/erguvan/`.

## Failures and corrections

The first thumbnail-only job supplied a nested playlist to `--load-info-json`.
yt-dlp sanitized its nested entries, resulting in nine failed CLI batches and
409 logical image-unavailable records before any image HTTP download. A
top-level list of individual observed-image info dictionaries fixed this;
an explicit inbound-download retry recovered the eligible records. See the
[pinned yt-dlp implementation](https://raw.githubusercontent.com/yt-dlp/yt-dlp/2026.08.19/yt_dlp/YoutubeDL.py).
These 409 records are retained as failures, not counted as 409 paid requests.
Six additional images failed square/size/detail admission checks and remain
excluded. A browser scrolling call exceeded the tool time limit; discovery was
recovered from persisted DOM exports using a fresh research tab and shorter
scroll batches.

Resume now distinguishes duplicates of pending records from archive duplicates,
so an alias does not prevent the original pending entry from resuming. The
bounded batch cursor advances by the actual batch size, avoiding skipped
candidates near the target when a small batch contains duplicates.

The final attempt ledger contains 1,000 accepted downloads, 1,021 duplicate
release records, 39 duplicate assets, 145 pixel/perceptual duplicates, two
pending-asset aliases and the 415 historical error records above. These are
ledger records, not a billing/request count. Only newly downloaded excluded
copies were discarded; old archives were not edited.

## Validation and resources

The final audit reopened **every accepted file**, checked SHA-256, decoded
normalized pixels and dimensions, and repeated duplicate comparison against
the old archive, original square references and preceding new covers. It
indexed 1,484 baseline files, which include repeated reference copies and do
not mean 1,484 distinct albums. All checks passed. Duplicate detection uses
release/asset identity, normalized RGB hashes, pHash/dHash and RGB difference
limits: it remains a conservative heuristic rather than semantic uniqueness
proof. A representative 24-cover contact sheet was visually inspected; there
was no complete visual review or training-quality acceptance of the corpus.

Two offline duplicate-decision tests passed, covering resize/re-encoding
equivalence versus color-altered/mirrored different artwork. Python compilation
and whitespace checks passed. The frozen prompt matches the existing archive.

All metadata/download/audit jobs used serial batches and the existing sampling
guard with a **512 MiB process-group limit**, memory-pressure and swap-growth
checks every 100 ms. Largest sampled download RSS was **294.70 MiB**, physical
footprint **225.13 MiB**; these overlap and must not be added. The Turkish
extension took **202.79 seconds**. The final audit took **38.54 seconds**,
RSS **150.59 MiB** / footprint **109.45 MiB**. System pressure stayed normal;
each monitored job saw no growth over its initial swap baseline. Existing
system swap was approximately 3.78 GiB, not zero. Chrome is outside the guarded
process group, and sampling is not a kernel-enforced memory ceiling.

**New teacher requests: 0. Generation cost: $0. Audio/video downloads: 0.
New agents: 0. Training runs/checkpoints: 0.**

## Next work and publication scope

Prepare a reviewed small teacher batch from the diverse sources, with an explicit
cover count and spending cap. Record raw outputs, actual charges, alignments,
full-source visibility, masks and human quality decisions for each pair before
training admission. Do not silently replace the frozen baseline or equate a
successful API response with source preservation or exterior quality.

The publication batch contains seven small code/policy/documentation files,
with exactly one changed file per commit and one monitored push. Artwork,
cookies, request bodies, local manifests and model payloads are excluded.
Publication receipts and Git memory telemetry stay under `.build/publication/`;
the verified publication outcome is also recorded in the ArtWorker Space Page.
