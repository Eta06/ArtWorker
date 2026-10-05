# Official-component ProMax repaint/outpaint trial

This isolated runner follows the author's
[pinned outpainting recipe](https://github.com/xinsir6/ControlNetPlus/blob/b48420576eac63c04388cb65fb74513cbd17405a/promax/controlnet_union_test_outpainting.py)
using native Diffusers 0.39.0 on the existing Torch 2.14.0 environment. It uses
the ordinary **four-channel SDXL base**, **eight-task ProMax Union** and
**madebyollin FP16-fix VAE**. The prior nine-channel inpaint checkpoint is not
loaded. Existing runners and installed runtime files are preserved.

The verified local assembly is `.build/models/sdxl-promax-official/`, with
ordinary base files in `base/`, renamed native ProMax config/weights in
`controlnet/`, and the separately loaded replacement VAE in `vae_fix/`.
The parent-owned `.build/models/sdxl-promax-official.manifest.json` pins all
three repository revisions and 20 destination files. The runner requires its
verified status, rechecks pinned hashes, and checks component loading reports
for missing, unexpected or mismatched keys. The VAE's published container is
F32; the official recipe loads/casts it to FP16.

The first trial uses **30 requested steps, CFG 5, strength 0.9999, scale 1**,
Euler ancestral scheduling, guess mode false and control guidance throughout
the run. Native strength trimming gives **29 actual iterations**: 29 ControlNet
calls and 29 UNet calls, each with a two-item CFG batch, or 58 branch-sample
evaluations per model. The actual counters and timesteps are checked and saved.

The benchmark keeps the frozen 512 × 1152 canvas, 512-square cover at
`[0, 320, 512, 832]`, white-unknown mask, seed 42 and the positive exterior
prompt used in the previous Fill/SDXL trial. This 589,824-pixel canvas differs
from the author's preprocessing near 1024² pixels; it is a controlled benchmark
resolution, not an exact reproduction of the demo.

Native Union receives `control_image=[control_rgb]`, `control_mode=[7]`. The
RGB control image retains the known source and has black RGB zero outside it;
the model's control preprocessing range is `[0, 1]`. Mode 7 becomes an eight-slot
one-hot repaint tensor for both CFG branches. The native one-image list is used,
not the legacy eight-image-slot interface. MHA axes are unchanged.

## Source preservation and RNG evidence

The native four-channel pipeline replaces known-region latents after every
solver step. Until the last step, it uses the original source latents noised
with the original initialization noise at the **next** timestep. The last
replacement uses clean source latents. The observer captures the actual solver
output, actual `add_noise` output, source latents and mask, then verifies the
callback tensor exactly equals the native equation. It does not add another
clamp or consume extra noise for this verification.

Euler ancestral scheduling samples new noise on every iteration. The runner
records CPU-generator state transitions around actual scheduler calls and
preserves the wrapped method's signature so native Diffusers still forwards
that generator. This RNG path is different from the preceding Euler and Fill
trials; same seed does not mean matching noise across models.

No watermark or automatic mask overlay is enabled. `raw.png` and the original
decoded tensor are saved before the separate source-only composite. That
composite pastes the original source rectangle exactly and preserves all
generated exterior pixels unchanged. No feathering, color correction, adapter,
refiner, tiling or artificial structural overlay is added.

## CPU gate and parent-only launch

```sh
.build/dreamlite-venv/bin/python experiments/sdxl_promax/test_cpu_preflight.py
.build/dreamlite-venv/bin/python experiments/sdxl_promax/run_promax_trial.py \
  --preflight --track track3 \
  --output experiments/sdxl_promax/preflight/track3
```

After the parent queues the single GPU trial:

```sh
.build/dreamlite-venv/bin/python experiments/sdxl_promax/run_promax_trial.py \
  --track track3 --steps 30 --guidance 5 --strength 0.9999 --seed 42 \
  --output experiments/sdxl_promax/runs/2026-10-05/track3-30-repaint7-seed42
```

Use a fresh output folder. An existing `metrics.json` prevents overwriting a
previous run. The loader is offline and selects exact local safetensors. The
parent handles downloads and bounded GPU scheduling; preparation uses no
pretrained model loading or MPS execution.

Metrics retain input/checkpoint/runtime hashes, model classes and dtypes,
actual prepared RGB control, mask, source posterior latents, initial noise,
final latents, raw source reconstruction error, call counts, source-clamp
checks, finite predictions/residuals, timing, peak process RSS and sampled MPS
memory. Synchronization and observation add overhead; these are instrumented
trial timings. Sampled memory peaks are not continuously measured device peaks.

Directly inspect the rail through the source boundary and the complete outer
scene. Runtime success, finite tensors, a verified latent clamp or exact pixel
compositing do not establish acceptable outpainting quality. This teacher
experiment makes no phone-runtime, spatial-growth or model-superiority claim.
Detailed pinned-source and assembly evidence is in
`../sdxl_inpaint/PROMAX_FEASIBILITY.md` and `promax_download_manifest.json`.

## Separate strength-one initialization probe

`run_promax_strength1_trial.py` is an isolated copy; the frozen official-recipe
runner and its CPU gate are preserved. It keeps the same source, mask, prompt,
seed 42, task 7, CFG 5, scale 1 and 30 requested steps, while fixing strength
to 1. The native schedule retains all **30 ControlNet and 30 UNet calls**,
or **60 CFG branch samples per model**, with 29 next-timestep source replacements
and a final clean-source replacement.

Unlike the nine-channel SDXL strength-one probe, native four-channel ProMax
still encodes the complete canvas to obtain source-anchor latents before it
samples initialization noise. The actual initial full-canvas tensor must equal
the actual sampled noise times `scheduler.init_noise_sigma`; initialization
must not call `scheduler.add_noise`. The source-anchor encode, masked-source
posterior and subsequent per-step source replacements remain active.

The initial RNG operation order therefore matches the preceding ProMax 0.9999
path: full source posterior, initialization noise, then masked-source posterior.
The initial `add_noise` formula itself consumes no RNG. Strength one changes
the initialization formula and retains an additional ancestral solver step.
Equality of real-run noise is determined from the saved actual tensors, rather
than inferred from the seed.

```sh
.build/dreamlite-venv/bin/python experiments/sdxl_promax/test_strength1_cpu_preflight.py
.build/dreamlite-venv/bin/python experiments/sdxl_promax/run_promax_strength1_trial.py \
  --preflight --track track3 \
  --output experiments/sdxl_promax/preflight/track3-strength1
```

The parent can queue this separate probe after numerical-parity triage:

```sh
.build/dreamlite-venv/bin/python experiments/sdxl_promax/run_promax_strength1_trial.py \
  --track track3 --steps 30 --guidance 5 --strength 1 --seed 42 \
  --output experiments/sdxl_promax/runs/2026-10-05/track3-30-repaint7-strength1-seed42
```

Raw output, exact-source compositing, strict offline component loading and
observational metrics follow the original runner. CPU checks establish native
initialization and conditioning behavior; image quality requires the parent’s
actual model trial and direct inspection.
