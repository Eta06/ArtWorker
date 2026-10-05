# DreamLite Mobile local trial

The official 4-step distilled DreamLite Mobile weights were downloaded and run on
the local Apple Silicon Mac with PyTorch MPS on 2026-09-30. This experiment does not
modify the music player or establish performance on an actual iPhone.

Initial generic-prompt Mobile portrait results were poor. Subsequent controls
confirmed that ordinary editing works and that **Base 28-step track3 portrait
outpainting produces a coherent coastal continuation**. This is promising for
the task, with seams remaining; it is not a three-cover quality win or phone
performance claim. The Base checkpoint was also downloaded and tested.

## What was downloaded

- Model: `carlofkl/DreamLite-mobile`, `diffusers` branch commit
  `6695c3f4be230f0493fa5dbf78be3bc4d3bb2ab4`.
- Weight repository size: **5,071,894,300 bytes**. The Qwen3-VL encoder is about
  4.255 GB, the 0.39B UNet is about 780 MB, and the tiny VAE is about 4.9 MB.
- Location: `.build/models/dreamlite/mobile` (ignored model weights).
- Official implementation commit: `a6e20c8cc94027f37dd7c5a81b0b3b472aa18409`.
- All LFS weights and tokenizers were SHA-256 verified against Hugging Face
  metadata; see `model_manifest.json`.
- Code license: Apache 2.0. Weights license: CC BY-NC 4.0; this is a local research
  trial, not a commercially redistributable model package.

The live Hugging Face API reported `gated=false` and allowed unauthenticated
downloads. The GitHub README still says that access must be requested.

## Reproduction

```sh
uv venv --python 3.12 .build/dreamlite-venv
uv pip install --python .build/dreamlite-venv/bin/python \
  torch 'transformers==4.57.3' 'diffusers==0.39.0' \
  'huggingface-hub==0.36.2' accelerate pillow psutil safetensors torchvision einops
.build/dreamlite-venv/bin/python experiments/dreamlite/run_mobile.py
```

The tested versions are recorded in each `metrics.json`. The newest Diffusers
0.40/main dependency requires Hugging Face Hub 1.x, which conflicts with the
official Transformers 4.57.3 pin. Diffusers 0.39.0 includes DreamLite and works with
Hub 0.36.2. This dependency combination was imported and actually executed.

## Shared album-cover comparison

All three native instruction-edit runs use the shared 512 × 1152 input canvas,
source rectangle `[0, 320, 512, 832]`, seed 42, 4 steps, and bfloat16 MPS. The
instruction requests seamless expansion of the black bands above and below.

DreamLite Mobile exposes `image` instruction editing, **no native `mask_image`
parameter**. Each raw result therefore edits the whole canvas. A second saved
`*_composite.png` restores the original resized source rectangle exactly. It
does not claim the model itself protected the original pixels.

| Cover | Generation including conditioning and decode | Raw center mean absolute RGB error (0–255) | Composite center byte exact |
|---|---:|---:|---|
| Un Día | 4.665 s | 50.547 | yes |
| karambol | 1.307 s | 85.123 | yes |
| Aklın Hep Bende | 1.233 s | 22.034 | yes |

Cold pipeline loading took 4.200 seconds. The first generation includes initial
MPS setup costs. Sampled peaks across loading and the three runs were 5.300 GB
MPS allocated memory, 6.367 GB MPS driver memory, and 0.743 GB process RSS. These
measurements overlap in unified memory and must not be added together. This run
uses an unquantized bfloat16 encoder, unlike the official iOS 4-bit encoder.

**Visual result: unsuitable for this outpainting task in this configuration.**
The outer areas contain tiled collages, duplicated people/cars, unwanted text,
and black horizontal gaps. Restoring the source square protects the cover but
does not make the generated extensions coherent. Inspect the saved raw images
as well as composites; the source preservation in composites is postprocessing.

Paths:

- `results/mobile-4step-seed42/`: three raw images, three composites, metrics.
- `results-mobile.log`: native instruction trial execution.

## Controlled source-latent restoration

One additional trial on track3 keeps the same model, input, prompt, seed, and 4
steps. After each FlowMatch update, the protected source region is restored at
the next sigma using `(1 - sigma) * source_latent + sigma * initial_noise`.
This is an experimental sampler adaptation, not a trained masked model.

```sh
.build/dreamlite-venv/bin/python experiments/dreamlite/run_mobile.py \
  --tracks track3 --latent-preservation \
  --output-dir experiments/dreamlite/results/mobile-4step-latent-blend-seed42
```

Generation took 1.825 seconds; raw center error improved from 22.034 to 6.472, but
the outer regions retained the same collage and black-gap failures. Source
latent preservation alone did not solve the task. The final original composite
again preserves the source rectangle exactly.

The 512 × 1152 portrait format may be outside the model's dominant square training
distribution, but later controls below also show prompt and model-variant effects.
These initial results do not prove that portrait dimensions alone cause failure.

## Ordinary-edit, square and Base controls

The base model revision is `751cb8dbb9072a8c8ffd8684e0f254b50f20531b`, with the
same encoder/VAE byte hashes as Mobile. The two identical large components are
hard-linked locally; only the different 780 MB UNet and small configuration/token
files required additional downloads. All Base LFS hashes were also verified;
see `base_remote_manifest.json`.

```sh
.build/dreamlite-venv/bin/python experiments/dreamlite/run_controls.py
.build/dreamlite-venv/bin/python experiments/dreamlite/run_controls.py \
  --variant base --tests recolor square_outpaint portrait_outpaint
```

These controls use track3 and scene-specific instructions. Square outpainting
places a 512-square source in a 1024-square canvas at `[256,256,768,768]` and
expands all four sides. This is a diagnostic configuration, distinct from the
shared phone canvas. The normal recolor edit uses the full image at 1024-square.

| Variant | Test | Generation seconds | Observed output |
|---|---|---:|---|
| Mobile / 4 steps | 1024-square car recolor | 2.981 | Red car becomes blue; recognizably same man/coast, some geometry changes |
| Mobile / 4 steps | 1024-square outpaint | 2.186 | Plausible sea/road/vegetation; leftover black frame and mismatched source edges |
| Base / 28 steps | 1024-square car recolor | 37.801 | Blue car with open door retained better; details still change |
| Base / 28 steps | 1024-square outpaint | 41.717 | Coherent coastal scene; car/source boundary and camera geometry differ |
| Base / 28 steps | 512 × 1152 portrait outpaint | 32.356 | Coherent sea and road without repeated people/cars; color/road seams remain |
| Mobile / 4 steps | Same detailed portrait prompt | 7.777 | Improved over generic collage, but disconnected channel above and grass replacing road below |

The successful recolor controls confirm the loaded pipeline can perform useful
instruction edits. The same detailed portrait prompt works better with Base
than Mobile in this one-cover trial. This suggests preservation/continuation
quality was lost in the fast variant; it is not proof of a universal gap across
seeds or covers. The earlier generic-prompt failure should not be attributed
solely to portrait dimensions or used to reject the complete DreamLite family.

Base's process sampled up to 9.459 GB MPS driver memory and 5.877 GB MPS allocated
memory. Mobile square controls used up to 7.448 GB driver memory. Measurements
remain Mac/bfloat16 results, with an unquantized encoder.

Control inputs, outputs and metrics are in `results/*controls*`. The directly
comparable track3 portrait records are in:

- `results/base-28step-portrait-coastal-prompt-seed42/metrics.json`
- `results/mobile-portrait-coastal-prompt-seed42/metrics.json`

## Tested square-context phone workaround

`run_square_context.py` executed the following configuration with Mobile on all
three covers, 4 steps and seed 42:

1. Generate in 1024-square model space with a 448-square source at
   `[288,288,736,736]`.
2. Crop `[288,8,736,1016]`, which gives 448 × 1008.
3. LANCZOS resize to 512 × 1152, placing the resized source exactly at
   `[0,320,512,832]`.
4. Restore the original shared 512 source pixels in the final composite.

This keeps the phone geometry and uses the official square working resolution.
It changes model context, source size in model space, and work resolution, so it
must be reported as a separate configuration.

| Cover | Generation seconds | Visual observation |
|---|---:|---|
| Un Día | 5.125 | Repeated glowing moon-like dots; black borders/gaps |
| karambol | 3.720 | Clean cyan background without duplicate people/text; black frame lines and a color seam remain |
| Aklın Hep Bende | 4.201 | Coherent sea/vegetation outside source, but black gap and lower road replaced with vegetation |

All final source rectangles are byte exact. Peak MPS driver memory was 7.444 GB,
and allocated MPS memory was 5.301 GB. This configuration improves over the initial
generic portrait collage, particularly on the plain cyan studio background. It
still does not reliably produce seamless outpainting. The native square raw,
cropped phone raw, input square and final source composites are retained in
`results/mobile-square-context-phone-seed42/`.

```sh
.build/dreamlite-venv/bin/python experiments/dreamlite/run_square_context.py
```

Base square-context inference remains unrun. Base's already tested direct
portrait track3 result is the stronger coherent-scene quality reference.

The square-context model sees 1,048,576 generated-canvas pixels versus 589,824 in
the direct portrait trial. About 80.9% of the model-space square is outside the
448-square source; only the central 448 × 1008 crop survives in the phone image.
This is not an equal-work or equal-context benchmark, even though final dimensions
and restored source pixels are identical. Prompts are also scene-specific.

Source inspection found another relevant difference: Diffusers 0.39 resizes edit
images to 512-square before VLM conditioning without preserving aspect ratio;
the cloned official custom pipeline uses a 256-square resize instead. Direct
portrait inputs are therefore squeezed in the VLM reference, while square inputs
are not. This may contribute to output differences; it has not been isolated in
a controlled experiment. The VAE condition still uses the requested output
dimensions. No inference runtime was silently patched for the comparisons.

## iOS implementation implications

The official deployment reference uses fp16 CoreML UNet/VAE with a 4-bit MLX
Qwen3-VL encoder. The included export scripts and Swift pipeline fix 1024-square
dimensions: UNet input `[1, 4, 128, 256]`, VAE latent `[1, 4, 128, 128]`.
For the tested portrait canvas, static CoreML exports would instead need UNet
input `[1, 4, 144, 128]` and VAE latent `[1, 4, 144, 64]`, together with Swift shape
updates. The published square iOS example is not ready-made portrait outpainting.

Primary references:

- [Official repository](https://github.com/ByteVisionLab/DreamLite)
- [Mobile weights](https://huggingface.co/carlofkl/DreamLite-mobile/tree/diffusers)
- [Official iOS deployment guide](https://github.com/ByteVisionLab/DreamLite/blob/main/deploy/README.md)
- [Weight license](https://github.com/ByteVisionLab/DreamLite/blob/main/WEIGHTS_LICENSE)
