# Fidan — remove the approved obsolete Qwen and MaskFlow weights

8 October 2026 Türkiye. The user explicitly approved removing the first obsolete
Qwen + MaskFlow weight group. Limited this deletion to the two named model
directories; no other model family, environment, experiment output or media was
removed. No model inference, training, hosted API requests or subagents.

## Verified scope and preservation

- `.build/models/qwen`: 63.918 GiB reported by du before removal.
- `experiments/maskflow/models`: 19.324 GiB before removal.

Checked both targets were actual directories inside this repository, rather
than symlinks, and no running process command referenced either target. Verified
the selected Klein base4B baked INT4 transformer, cached fixed instruction,
standard VAE and Flux2CLI existed with their expected sizes. Checked eight
selected-model directory symlinks; none resolved into the deletion targets.
Archived 150 small JSON/readme/license/notice/cache-metadata files in ignored
`.build/storage-cleanup-2026-10-08/metadata/`. Existing pinned source/download,
license and conversion manifests outside the target directories remain intact.

Removed both directories directly after the explicit deletion instruction;
confirmed both paths absent. Selected component device/inode/size/mtime stamps
are unchanged. Preserved the Qwen and MaskFlow Python environments, MaskFlow
run results, hosted teacher outputs, and the 1,000-cover Defne manifest. Code,
Git history and experimental ControlNet results also remain outside this scope.

## Measured outcome and limits

Combined before-removal du size: **83.242 GiB**. Removal took
**2.427 seconds**. Immediate Data-volume available-space increase:
**83.238 GiB**, leaving **111.010 GiB available** in that sample. APFS shared
extents and concurrent application writes mean du totals and df deltas need not
match exactly. Raw before/after bytes and preserved-component stamps are saved
in the ignored cleanup receipt. No trash-purge operation or unrelated deletion.

This verifies filesystem preservation, not a fresh Klein inference run. Old
Qwen/MaskFlow trials now require their pinned upstream downloads and conversion
steps before rerunning. Their historical successful receipts are retained as
historical records, not claims that the removed weights remain locally present.

Publish this one documentation file in one `fidan:` commit and one memory-guarded
push; reconcile current local availability and the cleanup record on the existing
ArtWorker Space Page. Other model groups remain available for later cleanup.
