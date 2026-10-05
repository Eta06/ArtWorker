# Historical verified download snapshots

These JSON files copy the existing completed-download manifests from the local `.build/models/` cache during the Sedef publication stage (2026-10-06). They preserve source revisions, expected sizes/hashes and the recorded actual verification results. No payload was changed or downloaded again for this publication.

- `flux1-fill-q4.manifest.json`
- `sdxl-inpaint-fp16.manifest.json`
- `sdxl-promax-official.manifest.json`

They describe historical artifacts and include original local paths. They do not claim that a remote file or a different machine's cache was verified today. Model binaries remain local and must be obtained separately under their upstream terms. Qwen and MaskFlow source/verification records also live in their experiment directories.
