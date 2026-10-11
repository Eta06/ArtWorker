# Sumbul — live production comparison and orange source-region review

Date: 2026-10-11.

## Requested review

The user requested confirming parallel Nano processing, applying the orange
source-matching test to the generated corpus, and a website showing original
covers, expanded outputs, production inputs and expansion direction.

Nano continues through native asynchronous Google batch jobs, at most eight
two-image jobs outstanding. That is up to sixteen requested covers awaiting
completion, not sixteen local model processes or a guaranteed provider execution
thread count. The existing coordinator audits each returned image in an isolated
serial subprocess. No new generation or paid service call was made by this
gallery stage, and the generation settings/prompt were not changed.

## Local gallery

The gallery reuses the existing loopback preview server at
`http://127.0.0.1:54146/sumbul/`. Source code is tracked; corpus files, individual
reviews, local media aliases and snapshots stay in ignored experiment results.
No model/artwork/credential payload was uploaded for hosting or publication.

Each cover opens a detail view containing the original cover, raw outpaint,
toggleable orange source boundary, aligned before/after wipe, RGB difference
map, SSIM, RGB-within-eight tolerance ratio and measured source visibility.
The actual canvas, square reference, prompt, generation settings, native output
dimensions and reported/estimated generation cost can be inspected. Filters
cover artist/title, teacher model and alignment reliability; sorting and
pagination avoid decoding the entire corpus in the browser. Local preferences
can be exported; no image is automatically accepted for training.

Expansion distances are approximate native-output pixels from the aligned source
polygon's bounding box to each canvas edge. They allow asymmetric placement and
scaling. Rotation is retained; negative distances indicate a clipped source
boundary, not a positive expansion. A reliable registration is required for the
orange polygon and preservation metrics. Failed registration gets no invented
source square, similarity value or expansion amounts.

## Refresh, safeguards and observed verification

A read-only watcher publishes the data snapshot atomically every thirty seconds
when the queue changes. Original assets are read-only symlink targets; thumbnails
are small separate JPEGs. Snapshot subprocesses have the existing pressure check,
a sampled 256 MiB process-group RSS cap and sixty-second per-snapshot limit.
They do not acquire the heavy-model lock used by the teacher worker. When the
producer is terminal, no batch jobs remain, and completed images have audit
results, the watcher publishes the final snapshot and exits.

The first gallery build contained 380 completed images, 361 reliable source
registrations and nineteen unavailable registrations, with zero audits pending.
Later verification found 394 images, 375 reliable registrations, nineteen
unavailable and zero pending; production was still active. These are snapshots,
not final corpus counts. All published media targets were checked to exist.
The first snapshot build sampled 47.39 MiB peak RSS and took 12.243 seconds.

Four offline tests passed: centered/asymmetric geometry, unreliable alignment
without a fake region, negative clipped margins, and a read-only repeated
snapshot that omits unrelated private account fields and retains raw hashes.
`git diff --check` passed. Browser verification covered the live page, a Nano
detail, the wipe slider, the orange-boundary toggle and the unavailable-alignment
filter. The inspected failed example displayed no orange polygon and no
preservation ratio. A screenshot of the Nano before/after detail was saved
privately for the user-facing handoff.

## Remaining work

Nano production is still running within the original budget. The final count
will be published by the watcher. Unavailable registrations require human
inspection; they are not an automatic quality rejection. SSIM is not unchanged
pixel percentage, and the RGB ratio is a tolerance measure on the aligned
512×512 analysis grid. No student training or checkpoint release occurred.
