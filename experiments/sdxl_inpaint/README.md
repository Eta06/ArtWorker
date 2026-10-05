# Native SDXL inpainting trial

This isolated experiment uses the trained nine-channel
[SDXL inpainting checkpoint](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1/blob/115134f363124c53c7d878647567d04daf26e41e/README.md),
pinned at `115134f363124c53c7d878647567d04daf26e41e`. The initial settings follow
its published example: **20 requested steps, CFG 8, strength 0.99**.

The untouched Diffusers pipeline reduces those settings to **19 Euler
timesteps**. Each timestep invokes the UNet once with a two-item CFG batch:
19 literal calls and 38 branch-sample evaluations. The runner measures the
actual calls, batch sizes, timesteps and sigma schedule rather than substituting
the requested step count. These are full-canvas trials, with no growing
computation or iPhone performance claim.

Input files are the frozen shared canvas, mask and 512-square source in
`experiments/evaluation/inputs/`. The canvas is 512 × 1152 and the source box
is `[0, 320, 512, 832]`. White mask pixels regenerate; black pixels identify
the source. The normalized masked-source encoder input has exact zero RGB
outside that box. The UNet receives noisy latents (4 channels), the mask (1)
and masked-source latents (4), in that order. No known-region latent replacement
is added to the native nine-channel path.

At strength 0.99, initialization also encodes the complete input canvas and
adds actual sampled noise at the first retained timestep. This differs from
pure-noise initialization. This frozen SDXL canvas has RGB **0** outside the
source. The preceding Fill runner created RGB **128** outside its source, then
erased that exterior to normalized zero for masked-source encoding. SDXL's
complete initial-image encode is an additional, distinct conditioning path:
only the source, mask and geometry match across these model trials, not their
full canvas bytes or initial noise. A gray-canvas SDXL arm would be a separate
experiment. The runner observes the native preparation methods
and saves their real outputs and generator state hashes without replacing them.
The default positive exterior prompt is identical to the controlled FLUX Fill
exterior trial; `--prompt` records an explicit alternative.

## Preparation and launch

Use the existing `.build/dreamlite-venv` environment: Torch 2.14.0,
Diffusers 0.39.0 and Transformers 4.57.3 were verified. The parent owns downloading
the 18 files described in `download_manifest.json`; this runner has no download
path. The checkpoint must be at `.build/models/sdxl-inpaint-fp16`, and the separate
`.build/models/sdxl-inpaint-fp16.manifest.json` must have `status="verified"` and
its exact absolute `local_dir`.
The runner validates pinned hashes again before loading the model.

```sh
.build/dreamlite-venv/bin/python experiments/sdxl_inpaint/test_cpu_preflight.py
.build/dreamlite-venv/bin/python experiments/sdxl_inpaint/run_sdxl_trial.py \
  --preflight --track track3 \
  --output experiments/sdxl_inpaint/preflight/track3
```

After the parent finishes checksum verification and schedules the GPU:

```sh
.build/dreamlite-venv/bin/python experiments/sdxl_inpaint/run_sdxl_trial.py \
  --track track3 --steps 20 --guidance 8 --strength 0.99 --seed 42 \
  --output experiments/sdxl_inpaint/runs/2026-10-05/track3-20-seed42
```

The local loader uses `local_files_only=True`, `variant="fp16"`, safetensors,
FP16 and MPS. Watermarking and automatic mask overlays are disabled. A CPU
generator supplies actual seeded randomness. No attention slicing, VAE tiling,
adapter, refiner, feathering or color correction is enabled. Use a fresh output
directory; an existing `metrics.json` prevents overwriting a previous trial.

## Evidence saved

- `raw.png`: model RGB decode before source pasting, plus the original decoded
  tensor in `raw_decoded_tensor.npz`.
- `composite.png`: only the exact source rectangle is pasted over the raw output;
  all generated exterior pixels remain byte-identical.
- `actual_noise.npz`, `initial_latents.npz`, `masked_source_latents.npz`,
  `mask_latents.npz` and `final_latents.npz`: observed native tensors with shape,
  device, dtype, finiteness and canonical float32 hashes.
- `metrics.json`: pipeline/runtime/checkpoint/source provenance, generator state
  hashes, actual UNet/CFG counts, per-step finiteness, phase durations, raw source
  reconstruction error, process peak RSS and sampled MPS memory values.

MPS memory figures are peaks of recorded samples, not a claim of continuously
measured peak device memory. Observer synchronizations, finite checks and saved
arrays add overhead, so the timing describes this instrumented trial.

Inspect both source joins and the complete generated scene directly. Runtime
success, exact source pasting and low reconstruction error do not establish
acceptable image quality. Sources and model metadata are recorded separately
in `SOURCES.md` and `download_manifest.json`.

The CPU conditioning gate passed, including a 207,636-parameter random tiny
SDXL-style UNet. Read-only audits found no runner or observer blocker. Preparation
did not load the pretrained model or run MPS; the checkpoint download was still
in progress when this runner was prepared.

## Separate strength-one initialization probe

`run_sdxl_strength1_trial.py` is a frozen copy of the preceding runner, with
native `strength=1` initialization. It preserves the original runner and test.
The source, mask, geometry, positive prompt, CFG 8 and seed 42 remain the same.
The probe processes all **20 timesteps / 40 CFG branch-samples** and starts with
sampled noise multiplied by the native scheduler's initial noise sigma. It
skips encoding the complete black-exterior initial canvas. The masked-source
VAE encode still supplies the trained nine-channel conditioning.

This is a controlled investigation of the dark exterior in the preceding
strength-0.99 result. The published model example uses strength 0.99; the probe
does not assume strength 1 is a quality improvement. Same seed also does not
give matching noise: strength 0.99 consumes the complete initial-image VAE
posterior draw before the noise draw, while strength 1 draws noise first and
then the masked-source posterior. The metrics record this difference and the
actual generator state transitions.

The additional CPU gate invokes native `prepare_latents`, checks FP16 and
FP32 draws against independent `torch.randn`, verifies exact generator
advancement, rejects any full-image encode or image-plus-noise call, and checks
the complete frozen Euler schedule. The inherited CPU gate is retained and
reported separately, including its strength-0.99 schedule checks.

```sh
.build/dreamlite-venv/bin/python experiments/sdxl_inpaint/test_strength1_cpu_preflight.py
.build/dreamlite-venv/bin/python experiments/sdxl_inpaint/run_sdxl_strength1_trial.py \
  --preflight --track track3 \
  --output experiments/sdxl_inpaint/preflight/track3-strength1
```

Parent-only GPU launch after the CPU gate:

```sh
.build/dreamlite-venv/bin/python experiments/sdxl_inpaint/run_sdxl_strength1_trial.py \
  --track track3 --steps 20 --guidance 8 --strength 1 --seed 42 \
  --output experiments/sdxl_inpaint/runs/2026-10-05/track3-20-cfg8-strength1-seed42
```

It retains the same raw/composite outputs and observers, with additional checks
that initial latents equal actual noise times sigma, full-image encoding is
skipped, only one masked-source conditioning encode occurs, and all requested
Euler timesteps are executed. Both CPU gates passed without pretrained model
loading or GPU execution during preparation.
