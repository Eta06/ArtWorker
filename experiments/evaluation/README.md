# Shared album-cover outpainting evaluation

This directory contains input preparation and output inspection, not a model runtime. It only requires Python and Pillow. Source JPGs are never changed.

`python3 experiments/evaluation/evaluate_outpainting.py prepare` creates three identical-layout evaluation inputs from the player collection:

- `source_square.png`: the decoded original 720×720 center crop, stored as PNG without further resizing. Cropping exactly matches `Track.artwork` in `MusicPlayer.swift` (1280×720 export, x=280, y=0, width=720, height=720). This is lossless with respect to the decoded crop, not a recovery of detail lost when the source JPG was encoded.
- `source_512.png`: shared Lanczos-resized 512×512 reference, used to measure pixel preservation.
- `canvas.png`: 512×1152 RGB canvas; source rectangle `[0, 320, 512, 832]` in XYXY coordinates, with black initial fill above and below.
- `mask.png`: 512×1152 grayscale mask, white=generate and black=preserve. Confirm each runtime's convention before using it; invert only for runtimes requiring the opposite convention.
- `canvas_rgba.png`: same canvas with transparent generation regions, for models accepting RGBA edit input.

The target canvas uses multiples of 64 and is slightly taller than the reference 402×874 phone screen. It is deliberately identical across models. Generated regions are 320 pixels above and below the square, totaling 55.6% of the canvas. `inputs/inputs.json` records paths, geometry and original source SHA-256 checksums.

For a fair first pass, run every available candidate on all three tracks with the same canvas and seed set (for example 42, 123 and 2026). Record the exact model revision, quantization, runtime version, sampler, step count, guidance and prompt. A four-step distilled model and an undistilled model require their own supported inference settings; report these settings instead of forcing incompatible parameters. Compare one full-canvas generation first; progressive-band generation is a separate experiment. Do not mix raw model output with output whose original region has been pasted back in.

Create a JSON manifest with a `runs` array. Paths may be absolute or relative to that manifest:

```json
{
  "runs": [
    {
      "model": "actual model and quantization",
      "track_id": "track1",
      "seed": 42,
      "status": "completed",
      "image": "/absolute/path/to/raw-output.png",
      "elapsed_seconds": 123.4,
      "peak_memory_gb": 18.2,
      "memory_kind": "process RSS / MLX peak / Metal allocation; state which",
      "steps": 4,
      "guidance": 1.0,
      "prompt": "The exact prompt used"
    }
  ]
}
```

Failed attempts should remain in the manifest with `status: "failed"` and an `error`, without an `image`. Memory from RSS, Metal and MLX allocation is not interchangeable. The harness does not independently measure inference time or memory; those fields come from the runner. Mac runtime results are not evidence of iPhone performance.

`python3 experiments/evaluation/evaluate_outpainting.py compare /absolute/path/to/runs.json` creates `comparison/report.json`, `comparison/contact_sheet.png` and `comparison/comparison.html`. The report measures:

- Exact unchanged-pixel fraction and absolute RGB error in the original source region.
- A simple row-difference heuristic across the two source boundaries, compared with nearby row differences. This can flag a seam; it is **not a visual quality metric** and should not establish a winner.
- Output size mismatch or missing output, without silently resizing for numeric metrics. Contact-sheet resizing is only for presentation.

Judge coherent scene continuation, visible seams, color/texture consistency, duplicated objects, unexpected text and overall composition by viewing the full-size output. For a stronger decision, expand from the three local covers to at least ten varied covers and compare anonymously. Three covers and one seed only support a small smoke test, not a broad model ranking.

## Collect actual runtime artifacts

`python3 experiments/evaluation/collect_runs.py` reads currently available Mobile-O, DreamLite, FLUX.2 and Qwen trial metadata and regenerates `aggregate/runs.json`, `aggregate/report.json` and `aggregate/comparison.html`. It never imports a model runtime or runs inference. Rerun it after another model completes; unfinished/failed records remain labeled as their last reported status and missing track/model cells stay empty.

`aggregate/raw_contact_sheet.png` and `aggregate/composite_contact_sheet.png` have covers as rows and model/configuration/seed as columns. `aggregate/contact_sheet.png` stacks both labeled panels. Raw and final source-composited images are never treated as interchangeable outputs. Timing excludes loading only when the source runner reports an appropriate generation or phase time; initialization/loading and complete command wall time remain separate fields.

Memory fields retain their actual measurement definitions: Mobile-O's current MPS allocations are **snapshots**, DreamLite's MPS maxima are **sampled**, Qwen/FLUX MLX allocation peaks are distinct from process RSS. Overlapping unified-memory measurements are not added. These are **Mac runtime trials, not iPhone benchmarks**. Source manifests and unchanged reported metadata are preserved for traceability. Prompts and masking implementations differ between runtimes; the outputs support a small task-oriented smoke comparison, not a strict controlled ranking.

Mobile-O instruction-only and latent-lock trials have separate columns. DreamLite Base and Mobile identities come from the reported model, not a hardcoded label. Manifests carrying `diagnostic_control: true` remain in the JSON report but are excluded from the shared-canvas grid and its numeric metrics; these include recoloring and 1024-square controls. A separately recorded 512×1152 portrait trial using the common track/canvas geometry can appear in the grid even when its image file also served as a diagnostic artifact. Consequently `completed_image_records` and `distinct_completed_image_paths` may differ. The metadata records the actual prompt for each trial.

DreamLite square-context trials explicitly show `square context 1024² → crop`: the model works at 1024² with a 448² source, crops 448×1008, then resizes the output to the common 512×1152 presentation geometry. Their timing and output can be inspected, but they are not equivalent direct-portrait inference settings. Qwen Viggle adapter rank (r128/r256) is part of the column label and metadata. All exact-pixel preservation claims compare `source_512.png`, the resized 512² reference, **not the original 720² decoded crop's pixel map**.

FLUX trial timing is labeled **Chain incl. lazy load** when the source runner reports this scope: encoder and transformer weight loading happen inside the inpainting chain. A zero-second pipeline initialization is only pre-chain setup, not a complete weight-load measurement. Qwen conditioning+generation+decode timings exclude the separately reported load phase; full-attempt elapsed time remains available. Consequently these displayed times should not be compared as identical measurement scopes. FLUX's upstream profile prints `MB` while dividing by 1,048,576; confirmed records are therefore retained as **MiB**. Its macOS process peak memory footprint is recorded separately from RSS and MLX allocations, without summing overlapping values.

When the four selected hard-cover trials have completed, the collector also writes `aggregate/track3-curated.png`: DreamLite Mobile square-context crop, DreamLite Base 28-step coastal portrait, Klein Base 50-step latent-blended portrait, and Qwen Viggle r256 6-step known-latent edge-padding portrait. Labels state their differing methods and timing scopes. `aggregate/validation.json` records artifact existence, shared geometry and source_512 equality without assigning a quality score. Separate Qwen spatial FULL/GROW probes carry `diagnostic_control: true` / `shared_canvas_comparable: false`; their image records are excluded from this aggregate entirely and their manifests are listed as excluded sources for traceability. They are reported separately.
