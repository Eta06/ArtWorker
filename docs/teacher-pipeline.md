# Teacher generation policy

The chosen **main teacher** is OpenRouter `openai/gpt-image-2.5-sunburst`,
quality `high`. The **fallback teacher** is `google/gemini-nano-banana-2.1`,
resolution `1K`. Flare remains a comparison model, outside this production queue.
This selects data-generating teachers, not the final local iOS student.

## Frozen baseline

Both teachers use the exact previously tested `old-canvas` prompt, SHA-256
`736716dd0146c1cbda160d21c6615dbe78e494a6e98dc779cb94701bd006efc8`.
The complete text and settings live in
[`teacher-policy.json`](../experiments/openrouter_outpaint/teacher-policy.json).
The two references are the 720×1280 green-placeholder canvas, with the source
720-square at (0,280), followed by the original 720-square. Prompt changes require
a new version and comparison; they do not silently replace this baseline.

## Per-cover queue

1. Verify source identity and cross-archive deduplication. Prepare references and
   save their hashes, prompt hash, model/profile and intended output ratio.
2. Submit once to Sunburst. Save the request state **before** the POST, then the
   response, generation ID, untouched raw image and billing receipt.
3. A definite safety refusal can receive one Nano attempt. A confirmed completed
   response without an image can receive one fallback after billing reconciliation.
4. For 429/5xx errors, reconcile completion and charges before at most one retry.
   An ambiguous timeout or lost response pauses that item: inspect provider state
   and billing before another POST. Never turn every technical error into another
   paid generation on a different teacher. Limits are two primary attempts
   (only with that reconciled technical retry) and one fallback per cover.
5. Detect the source region using the existing free alignment audit. Translation,
   scale and small rotation are allowed; centering is not a quality criterion.
   Compare aligned source content without local deformation or color correction.
   Keep SSIM, RGB tolerance and source visibility separate. SSIM is not a literal
   unchanged-pixel percentage.
6. Unreliable detection or a partial source goes to review. Never invent a source
   rectangle or admit only a favorable overlapping crop. Review faces, lettering,
   composition, exterior quality and seams as well as numerical measurements.
   No universal SSIM acceptance threshold has been calibrated yet.
7. A visually rejected Sunburst result may receive one Nano fallback within the
   configured budget. A Nano failure goes to quarantine; no recursive retry loop.
   Charge all attempts to that source, including both teachers and lost responses.

## Training geometry

For each approved example, save the original square, expanded target, alignment
transform and known/generated masks. Use the transformed source footprint
(a polygon if rotated), rather than assuming a centered orange square. Actual
left/right/top/bottom margins supply asymmetric expansion examples. Random
placement is useful only when the original content is sufficiently preserved.
At student inference, the original region can be composited back exactly, with
boundary quality evaluated separately; this stage does not implement that model.

Keep artwork, account preferences, cookies, credentials and request bodies local.
The queue remains a **plan**, with no automatic API execution during collection.
Before a paid batch, set its cover count and spending cap explicitly. Existing
teacher trials do not establish general quality across the new cover corpus.
