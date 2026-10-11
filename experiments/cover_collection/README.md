# Private YouTube Music cover collection

Source discovery uses the user's authorized logged-in browser and rendered album
release links. The original `collect_covers.py` downloader does not access account
cookies, hidden browser state, internal YouTube APIs, or audio streams.
A song with no album-release link
is not admitted as a cover candidate. Public CDN image requests are serial.

Keep all discovery exports, titles, artists, artwork and review files under
`experiments/cover_collection/results/`; this directory is ignored by Git.
Public commits contain code and anonymous stage measurements only.

## Input and collection

Export a JSON array from the authorized browser DOM with `album_url` (an observed
`https://music.youtube.com/browse/MPRE...` release link), `album`, `artists`
(`name`/`url` pairs), `observed_image_url`, `discovery_page`, and optional rendered
`context`/`kind`. Only real `yt3.googleusercontent.com` or
`lh3.googleusercontent.com` width/height image variants are eligible.
Placeholder `data:` images must first be loaded through normal UI scrolling.
Export atomically when a running collector follows discovery updates.

The script changes only the CDN width/height suffix, preserving the observed
asset identity. It saves the returned bytes without locally upscaling them.
Returned dimensions do not establish the original upload's detail resolution.
It bounds responses to 8 MiB, requires square images at least 512 pixels, and
rejects near-solid images for separate review. Repeated release IDs, identical
CDN assets, normalized RGB hashes and close dHash/low-MAE duplicates are skipped.
These filters are conservative heuristics, not a proof of semantic uniqueness.

Example (replace `stage` with the current single-word stage):

```sh
python3 scripts/run_bounded_model.py \
  --output experiments/cover_collection/results/date/stage/guard01 \
  --max-gib 0.5 --timeout 1800 -- \
  experiments/qwen/.venv/bin/python experiments/cover_collection/collect_covers.py \
  --candidates experiments/cover_collection/results/date/stage/candidates.json \
  --output experiments/cover_collection/results/date/stage \
  --stage stage --target 1000 --size 1024 --follow-seconds 1600
```

Resume validates saved file hashes and skips already attempted unchanged assets.
A previously unloaded placeholder can be retried after its real DOM image URL
is observed. Network failures are recorded; they are not automatically retried.
Use a fresh guard directory for each launch. The sampling guard stops on memory
pressure, RSS/physical footprint growth, swap growth or wall-time limits. It is
not a kernel-enforced memory ceiling and its process group excludes Chrome.

## Audit and local review

Write `visits.json` from actual visited page URLs, including observed `artist`
labels where available. Then:

```sh
experiments/qwen/.venv/bin/python experiments/cover_collection/review_collection.py \
  experiments/cover_collection/results/date/stage
```

Audit reopens every saved image, checks hashes, decoded normalized pixels,
release IDs and dimensions. It produces `audit.json`, `thumbs/`, a paginated
`index.html`, `gallery-data.json`, `overview.jpg`, and `pilot-100.json`.
The pilot uses round-robin observed artist labels; it does not measure exterior
quality, assess training eligibility, call teachers, or start training.
Serve the local directory on loopback to review original images, search names,
and filter the first 100 candidates. No media or title-level manifests are
released with the repository.

See [Defne's actual collection and validation](../../docs/stages/defne.md).

## New covers against an existing archive

`collect_unique_covers.py` is a separate thumbnail-only yt-dlp workflow. Use
Chrome cookies only when the user authorizes it. First collect the liked playlist
URLs with yt-dlp's `--cookies-from-browser chrome:Default --flat-playlist
--dump-single-json --skip-download`. That extractor may return video thumbnails,
so these are **not** automatically admitted as album artwork. Match song/release
links to real observed Google album-image URLs in the authorized Music browser.
Unloaded `data:` placeholders are not download candidates.

Pass DOM exports to the new collector with one or more `--baseline-manifest`
paths. It validates baseline hashes, indexes prior covers and original square
references, and excludes old/same-batch release IDs and CDN asset identities.
Downloaded images are compared by normalized RGB hashes and conservative
pHash/dHash plus RGB difference checks. Close-looking artwork editions or crops
can still require review; this is not a semantic uniqueness guarantee.

The collector gives yt-dlp a top-level JSON list of observed-image info records,
with `--load-info-json --skip-download --write-thumbnail`. It never fetches
audio/video streams. Album image identities are preserved when requesting the
1024 CDN size. Optional `--chrome-cookies` loads the authorized Default profile
in yt-dlp without exporting a cookie file or recording values in public files.
Cookie domains remain enforced by yt-dlp's jar. Keep all logs and manifests in
the ignored results directory; Chrome account credentials are never public.

Use the sampling guard with `--max-gib 0.5`; keep batches serial. Resuming checks
accepted file hashes. `--retry-errors` explicitly retries failed **inbound cover
downloads**, not paid image-generation requests. Only newly downloaded duplicate
copies are discarded; old archives stay untouched. Example:

```sh
python3 scripts/run_bounded_model.py --output .build/stage-guard01 \
  --max-gib 0.5 --timeout 1800 -- \
  experiments/qwen/.venv/bin/python experiments/cover_collection/collect_unique_covers.py \
  --candidates experiments/cover_collection/results/date/stage/candidates.json \
  --output experiments/cover_collection/results/date/stage \
  --baseline-manifest experiments/cover_collection/results/2026-10-07/defne/manifest.json \
  --target 1000 --stage stage --yt-dlp /path/to/yt-dlp --chrome-cookies
```

After reaching the target, run `audit_unique_covers.py` with the same baseline.
It reopens every accepted file and repeats cross-archive integrity and duplicate
checks. It saves a private audit and representative 24-cover contact sheet.
It does not create a website, call teachers or approve training examples.
`test_unique_covers.py` exercises encoding/resize equivalence versus different
artwork offline. The frozen teacher/fallback plan is in
[teacher-pipeline.md](../../docs/teacher-pipeline.md).
