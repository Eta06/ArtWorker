# Defne — 1,000 local source covers

7 October 2026 Türkiye. The user changed the collection target from one hundred
to **one thousand** album covers from their YouTube Music account and broader
music discovery. This stage collects original source candidates only. No hosted
image-generation calls, audio downloads, student training, model downloads,
phone work or subagents.

## Actual discovery and downloads

Started from the logged-in Chrome YouTube Music home and liked-music list, whose
UI displayed 1,164 songs. Album-linked song rows, recommendations and fifteen
artists' public pages/discographies provided 1,447 release links; 1,419 had an
observed HTTPS image URL. Thirty-four source-page visit records were retained.
This is not a claim that every liked song or every artist release was covered.
Videos without album-release links were excluded. Account cookies were not
exported, and no internal authenticated music API was used.

Downloaded **1,000** distinct accepted source images serially from Google image
CDN URLs, requesting the 1024-pixel variant of observed asset identities.
Saved the returned image bytes, with album/artist metadata, source/discovery
URLs, dimensions, time and SHA-256. The images total **208,291,707 bytes**
(**198.642 MiB**): **999 at 1024×1024, one at 1000×1000**. These are CDN response
dimensions, not verified original upload detail. No local image upscaling.

There were **1,025 actual HTTP image downloads**, all successful: 1,000 retained
images and 25 decoded-pixel/near-duplicate exclusions. Four additional identical
assets were skipped before another download. Exact normalized RGB and file
hashes are unique across all 1,000 retained covers. Perceptual duplicate
heuristics do not guarantee semantic uniqueness across artwork editions.

## Failures and recoveries

The first DOM export included lazy image placeholders. End/PageDown jumps did
not consistently expose all images; switched to normal wheel scrolling with
post-action DOM reads. Ten early placeholder validation failures and 599 later
placeholder observations were recorded locally, without HTTP image requests.
The original runner classified many of those as duplicate assets; the audit
separately reports all **609 placeholder observations** and the four real asset
duplicates, preserving raw receipts. Updated code labels unloaded placeholders
explicitly and allows fresh observed URLs to resume.

One browser wheel command timed out; inspected the actual scroll position and
continued without replaying unknown page mutations. One ambiguous heading
lookup was corrected to a visible `h1`; three missing album titles were verified
from their real album pages. No retained image has a missing album title; two
lack an artist label. The initial loopback audit request returned HTTP 404 before
the audit file existed; the finished endpoints all returned HTTP 200.

## Integrity, local review and resources

A three-image pilot verified that the CDN size variant returned real square
images. The guarded main run finished in **486.354 seconds**, under a **512 MiB**
process-group ceiling. Peak sampled RSS **197.25 MiB**, physical footprint
**179.17 MiB**; pressure stayed normal and swap stayed zero. Chrome is outside
that process group; these are not total-system or iPhone memory measurements.
The collection source tab was closed after discovery and metadata recovery.

The provisional collection name reused the older Mercan stage. Before any
publication, restored that historical document exactly and renamed this new
collection to Defne. Original guard receipts retain their actual launch paths;
the image files and checksums were preserved across the directory rename.

The full audit reopened and decoded every file, checked SHA-256 against the
manifest, verified unique release IDs and normalized RGB hashes, and validated
actual square dimensions. It passed for all 1,000 covers. Audit/gallery build
**11.496 seconds**, peak RSS **73.55 MiB**. A footer correction for the one
1000-pixel image was rebuilt in **11.238 seconds**, peak RSS **55.56 MiB**.
Python compilation and Git whitespace checks passed.

The private local gallery paginates 48 cards at a time, searches album/artist/ID,
opens the original full-size cover and links to the source album. Actual Chrome
verification checked 1,000 displayed records, loaded visible 192-pixel thumbs,
the 100-candidate filter, a loaded 1024-square modal, a matching search, and
page 21/21 with 40 cards ending at cover1000 and disabled Next. JSON data, audit
and pilot endpoints returned HTTP 200. A browser screenshot is saved locally.

Prepared a **100-cover pilot** using round-robin observed artist groups; all 100
come from different stored group labels. The full collection has 371 such groups
and is not balanced: one artist contributes 246 covers. Labels may include
aliases/case variants, so these are not verified unique-person counts. The pilot
provides variety for the user's subsequent teacher trials; it is not a visual
quality selection or approved training dataset. No individual cover has yet
been automatically admitted for training.

## Cost, limits and publication

**New image-generation API cost: $0.** No teacher generations were requested.
Native source detail, complete visual review, source preservation after future
outpainting, teacher-output eligibility and redistribution rights remain
unestablished. The unexpanded artwork, account-derived preferences, title-level
manifests, source links and full receipts remain private and ignored.

Five small code/document files are the publication batch: one file per `defne:`
commit, one guarded push. Record the verified push and current next step on the
existing ArtWorker Space Page. Future work is to review the 100-cover pilot and
explicitly configure teacher models, profiles and a paid-generation budget
before generating expanded training candidates.
