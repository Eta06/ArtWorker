# Native FLUX.1 Fill teacher

This isolated experiment compares a model trained for masked image completion
against the preceding Qwen experiments. It uses the shared 512 × 1152 canvas,
with the unchanged 512-square source at `[0, 320, 512, 832]`. White mask pixels
are generated; black pixels retain the source conditioning.

The initial quality trial uses **50 steps, guidance 30, seed 42**, following the
[Black Forest Labs Fill example](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev).
The CLI also accepts 30 steps for a later controlled speed comparison. Guidance
is an input to the distilled guidance embedding: each step has one transformer
call and no separate negative branch.

The local checkpoint is the public
[MFlux community Q4 conversion](https://huggingface.co/mflux-community/flux-1-dev-fill-mflux-q4/tree/eebfbaa12c95107169452c7d22622e04771192a3),
pinned at `eebfbaa12c95107169452c7d22622e04771192a3`. Download and verification
are handled separately; this runner never downloads weights. Its default path
is `.build/models/flux1-fill-q4`, with the verified downloader manifest at
`.build/models/flux1-fill-q4.manifest.json`. Before loading weights, it verifies
every file size, each LFS SHA256 and each ordinary Git blob against that manifest.
Both manifest revisions must match the pinned commit. Metrics label this trial
as research use and record the inherited FLUX dev non-commercial model license.

The implementation reproduces the installed MFLUX `Flux1Fill` loop and uses its
`MaskUtil`, prompt encoder, transformer, scheduler and VAE directly. The
inspected runtime revision was `25661d8e853996fec9134998b995edf97369bfbd`.
No installed files or earlier runners are modified. Source RGB is normalized
to `[-1, 1]` and multiplied by `(1 - mask)` **before** VAE encoding, making the
unknown encoder input exactly zero. The model receives 64 dynamic latent
channels, 64 masked-source channels and 256 packed pixel-mask channels: 384
channels per image token.

Every denoising call processes all 2,304 image tokens, through 19 joint and 38
single transformer blocks. A 50-step run therefore performs 50 forward calls,
115,200 image-token forwards, 950 joint-block forwards and 1,900 single-block
forwards. These are full-canvas quality trials; they do not establish growing
computation, streaming generation or iPhone performance.

## CPU gate and launch

```sh
experiments/qwen/.venv/bin/python experiments/flux1_fill/test_cpu_preflight.py
experiments/qwen/.venv/bin/python experiments/flux1_fill/run_fill_trial.py \
  --preflight --track track3 \
  --output experiments/flux1_fill/preflight/track3
```

The CPU gate compares native packing against independent index equations. It
checks mask polarity, normalization, all 8 × 8 pixel-mask offsets, 2 × 2 latent
packing, four/five-dimensional VAE outputs, source placement and 384-channel
input construction. It loads no weights and selects the CPU explicitly. When
the checkpoint is ready, the preflight also loads both local tokenizers without
encoders or network access. Incomplete files produce `preflight_files_incomplete`.

After the downloader marks its manifest `verified`, the parent schedules one
GPU run at a time:

```sh
experiments/qwen/.venv/bin/python experiments/flux1_fill/run_fill_trial.py \
  --track track3 --steps 50 --seed 42 \
  --output experiments/flux1_fill/runs/2026-10-05/track3-50-seed42
```

Use a fresh output directory for each run. A previous `metrics.json` prevents
overwriting the experiment. `--prompt` records an explicit prompt variant;
the default uses the existing shared track prompt unchanged.

## Output and interpretation

- `raw.png`: direct RGB VAE decode, saved **before** source compositing.
- `composite.png`: raw output with only the exact original source rectangle
  pasted back. Generated pixels outside that rectangle remain byte-identical.
- `initial_noise.npz` and `final_latents.npz`: canonical float32 copies, with
  their content hashes recorded in `metrics.json`.
- `metrics.json`: per-step progress, actual call/block/token counts, sigma
  schedule, phase durations, MLX/RSS memory, finiteness checks, input/checkpoint
  provenance and raw source reconstruction error.

No feathering, tonal correction, border retouching, latent known-region clamp
or synthetic line overlay is applied. Inspect both raw and composite images at
the source joins and in the complete outer scene. Execution success and low
source reconstruction error do not imply acceptable outpainting quality.

As of preparation, the CPU conditioning gate and both local tokenizers passed
(CLIP 77 tokens, T5 512 tokens). The checkpoint download was in progress, and no
model inference was launched by this subtask.
