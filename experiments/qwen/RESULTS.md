# Qwen-Image-2.1 album-cover outpainting: actual local trials

Measured on Apple M4 Max, 36 GiB unified memory, using the pinned native MLX
MFLUX runtime in `runtime.json`. These are single-seed experiments, not a
statistical ranking. Earlier runs competed with CPU compilation, and checkpoint
and filesystem caches were warm for later runs. Timing differences across tracks
must not be attributed solely to image content.

The shared output is 512×1152 with the original resized 512×512 cover at
`[0,320,512,832]`. Every successful final composite preserves that source exactly.
The raw decode also remains available so the composite cannot hide reconstruction
or boundary defects. Base and turbo were run at the same 4-bit DiT/encoder
quantization. The VAE checkpoint is F32. The sampler uses flow Euler with per-step
known-latent replacement; it is not a native mask-trained checkpoint or LanPaint.

| Trial | Track | Total seconds | Denoise seconds | Inference MLX peak GiB |
|---|---|---:|---:|---:|
| Base, 40 steps | 1 | 405.64 | 373.06 | 10.83 |
| Base, 40 steps | 2 | 192.29 | 176.40 | 10.89 |
| Base, 40 steps | 3 | 334.92 | 316.58 | 10.89 |
| Viggle v0.2.1 r128, 6 steps | 1 | 69.27 | 49.75 | 10.89 |
| Viggle v0.2.1 r128, 6 steps | 2 | 54.66 | 36.13 | 10.89 |
| Viggle v0.2.1 r128, 6 steps | 3 | 52.73 | 37.49 | 10.89 |
| Viggle v0.2.1 recommended r256, 6 steps | 2 | 45.17 | 31.92 | 10.89 |
| Viggle v0.2.1 recommended r256, 6 steps | 3 | 48.23 | 34.47 | 10.89 |
| r256, edge context for known-region VAE encoding | 3 | 46.14 | 32.67 | 10.89 |

`evaluation_runs.json` links each successful raw output, composite, and complete
metrics record. The table's inference peak is the maximum of conditioning,
denoising and decoding, excluding checkpoint conversion. Loading/quantizing the
original checkpoint reached **24.50 GiB MLX**, recorded separately. Loaded q4
component parameters occupy approximately **9.57 GiB** before adapters and
activations; r128 adds 0.63 GiB and r256 adds 1.27 GiB. Few-step distillation
reduces transformer passes, but does not shrink this base model's weights.

## Inspection of actual results

- Track 1 is mostly black around the moon/person source. Base and r128 continue
  that black composition cleanly without duplicated people, moons or new text.
  This easy background is weak evidence about general scene completion.
- Track 2's r128 result extends the men's clothing only partway, then inserts a
  hard black strip at the bottom. Base40 removes that strip and completes both
  clothing/body continuations more coherently. The top cyan-wall boundary is
  still mildly visible. The recommended r256 adapter also truncates the men
  partway through the generated bottom, then replaces the rest with a hard cyan
  wall strip; the smaller r128 truncation is not the sole cause of this failure.
- Track 3's base, r128 and r256 all continue water, guardrail and asphalt while
  keeping one person and one car. All have visible transitions at the source
  boundaries. Those defects exist in the raw decode as well as the composite.
- The edge-context diagnostic changes only the VAE encoding used for the
  known-region replacement; the vision reference and VAE prefix still receive
  the same gray portrait canvas. It visibly reduces border ringing without
  duplicating the subject/car, but does not fully remove the water-texture seam.
  Raw source-region top16 MAE changes **18.02→8.24 /255**, bottom16 **12.25→4.87**,
  and interior **3.25→3.21**. These are reconstruction errors, not a complete
  perceptual quality score.

The publisher's r256 adapter and the smaller truncated r128 adapter are kept as
distinct trials. Tested previews were finite; the first Base track1 predates the
explicit float-finiteness assertion, although its saved PNG is visually sane.

## Reproducibility and limits

Base: `Qwen/Qwen-Image-2.1@790c92633540aa0cb11d9abf19eb46d861714758`.
Adapter: `Viggle/Qwen-Image-2.1-viggle-turbo@bb26a0f38e5fe6c124aaccc9187a87eed5d9ed13`.
MFLUX: `25661d8e853996fec9134998b995edf97369bfbd`.
Environment versions: `requirements.lock`.
All checkpoints are under ignored `.build/models/qwen`; no Player source changed.

These timings and memory peaks are Mac measurements. They do not establish
iPhone deployment feasibility. The base has a Qwen Research license; production
licensing needs a separate decision. The masked sampler and border-context
changes are local adaptations, so outputs are evidence for this exact pipeline,
not a claim about an official Qwen outpaint benchmark.

## Matched target-growth diagnostic

`run_spatial_trial.py` subsequently ran a matched FULL/GROW pair with one shared
model load and the same actual square-source prefix, prompt, seed, source latent
encoding and sigma schedule. FULL passes 2,304 target tokens on all six steps;
GROW passes 1,280, 1,280, 1,792, 1,792, 2,304, 2,304. Future target tokens are
actually absent from transformer projection and attention before activation.
Absolute final-canvas rotary positions and the fixed prefix cache are preserved.

FULL core denoising took **29.38s**; GROW took **24.33s**: **22.2% fewer target
token forwards**, **17.2% lower measured core time** in one pair. Both use six
transformer calls. Order, thermal state and kernel warm-up limit this timing
evidence. Preview capture and VAE decoding are timed separately. The decoder
runs **after both sampling loops**; captured predictions are genuine intermediate
states, but this is not a live UI streaming benchmark.

The FULL square-prefix result is coherent. GROW's final outer bands contain
grid/ringing artifacts, an extra guardrail, and small invented symbols. The
untrained late-insertion initializer therefore fails quality acceptance. Earlier
active regions keep refining at the same global sigma; no ring is claimed
complete or frozen before the final step. Both outputs are finite and preserve
the original source exactly in the final composite.

The paired trials are marked diagnostic controls and excluded from the common
grid. Details and actual output paths are in `SPATIAL_RESULTS.md`; architecture
and training limitations are in `TOKEN_GROWTH_NOTE.md`. `evaluation_runs.json`
contains nine standard successes and two separate diagnostic successes.

Source/model documentation:
[Qwen checkpoint](https://huggingface.co/Qwen/Qwen-Image-2.1),
[Qwen repository](https://github.com/QwenLM/Qwen-Image-2.1),
[MFLUX reference runtime](https://github.com/mflux-community/mflux/tree/main/src/mflux/models/qwen21/reference),
[Viggle adapter](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo).
