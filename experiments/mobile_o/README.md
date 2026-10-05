# Mobile-O local album-cover trial

Actual experiment performed 2026-09-30 on the local Apple Silicon Mac, not a
physical iPhone. Player sources and original JPG/MP3 resources were unchanged.

## Downloaded checkpoints and source

- [Mobile-O source](https://github.com/Amshaker/Mobile-O), commit
  `91c255080a0130846fe5b5cd54ff5af3c05ab2b9`, under `.build/model-research/Mobile-O`.
- [Unified Python checkpoint](https://huggingface.co/Amshaker/Mobile-O-0.5B),
  revision `92b5ef6a90a770c631153c3c32be0bbe049e2a8d`, under
  `.build/models/mobile-o/python`. Safetensors file: 4,775,658,842 bytes.
- [iOS/CoreML/MLX checkpoint](https://huggingface.co/Amshaker/Mobile-O-0.5B-iOS),
  revision `56828ca7076d594dc38239627f0c4edb41a508b1`, under
  `.build/models/mobile-o/ios`. Downloaded the app's FP32 transformer/VAE decoder,
  connector, vision encoder and 4-bit MLX LLM, not all alternate ANE variants.
  Selected iOS package total: **3,632,691,267 bytes** across 24 files.
- SANA transformer/VAE/scheduler JSON configurations under
  `.build/models/mobile-o/sana-config`. No duplicate SANA base weights downloaded;
  trained tensors for these components are already in Mobile-O's unified weights.

The 0.5B label refers to the language model. The loaded whole model has
**1,664,722,343 parameters**. Its checkpoint also contains 1,207 redundant
`base_model.model.*` tensors, all of which have corresponding top-level keys.
The adapter loads the complete top-level model and rejects missing weights.

Mobile-O code, models and mobile app are **CC BY-NC-SA 4.0** (research and
noncommercial), per the repository's license. This is a benchmark candidate,
not a permissive commercial product dependency.

## Reproduce

An isolated Python 3.12 environment is at `.build/mobile-o-venv`. Relevant tested
versions: PyTorch 2.14.0, torchvision 0.29.0, transformers 4.51.3,
diffusers 0.35.2, timm 1.0.30. The upstream timm 0.6.13 pin lacks the
`timm.layers` import required by the published source, so a current timm was used.
CUDA-only dependencies (xformers, bitsandbytes, flash-attn, DeepSpeed) were not
required for this inference adapter.

```sh
.build/mobile-o-venv/bin/python -u experiments/mobile_o/run_trial.py
```

The adapter follows the published image-conditioned editing path on MPS:
source canvas → MobileCLIP visual tokens + Qwen hidden states → conditioning
projector → SANA DiT → AE decode. It initializes modules from config before
loading all trained weights. It overrides the upstream sampler's hardcoded
16×16 latent size with **16×36**, yielding the shared **512×1152** portrait
canvas without rescaling a generated square.

This is **semantic instruction editing, with no native mask input or source
latent lock**. The whole blank-padded canvas is the image reference. The original
512×512 square is pasted back at `[0,320,512,832]` only after decoding. Raw and
composited PNGs are both retained so preservation is not confused with the model's
behavior. This portrait PyTorch path was actually run; it is not the fixed-square
published CoreML path.

## Actual result

One seed (42), 20 diffusion steps, guidance 1.5, shared evaluation inputs.

| Cover | Full inference, seconds | Source max error after compositing | Visual observation |
|---|---:|---:|---|
| track1 | 4.91 | 0 | Repeated moon/person, invented lettering, mismatched seams |
| track2 | 4.74 | 0 | Repeated faces/title, invented lettering, mismatched seams |
| track3 | 4.99 | 0 | Repeated cars/open roof, geometry does not continue source |

Cold model loading was 2.86 seconds; complete three-cover process was 20.20
seconds. The first separate smoke test took 10.29 seconds for track1, before
subsequent cached execution; cold-start kernel costs matter.

End-of-track MPS allocations were **3.37 GB current / 6.64 GB driver**.
Process peak RSS was 0.734 GB. MPS end allocations are **not a measured true
transient peak**, and RSS alone excludes much of the GPU allocation. These Mac
numbers are not iPhone latency or memory claims.

Artifacts: `results/trial.json`, `trial.log`, and each
`results/track*-seed42-{raw,composited}.png`.

**Conclusion:** Runtime and portrait generation succeeded, but all three covers
failed this untrained instruction-only outpainting method. The model produces
alternate artwork rather than spatially continuing the original. Small size and
speed alone do not establish outpainting suitability. Training or a conditioned
masked sampler would be a separate experiment.

## Second actual trial: experimental latent restoration

The prepared `--latent-lock` adapter was subsequently run on all three covers,
same seed 42, 20 steps and 512×1152 canvas. Baseline artifacts are retained intact.

```sh
.build/mobile-o-venv/bin/python -u experiments/mobile_o/run_trial.py \
  --latent-lock --output experiments/mobile_o/results_latent_lock
```

This is a **training-free RePaint-style known-latent restoration adapter**, not
a native mask-trained Mobile-O model and not the full RePaint jump/resampling
algorithm. The existing AE encodes the source canvas. At every denoising step,
preserved-region latents are replaced by the encoded original at the next flow
sigma: `(1 - sigma) * known_latent + sigma * original_noise`. The source is still
composited exactly after decoding. The model receives the same image-conditioned
instruction as before; no fine-tuning or model-weight changes were made.

| Cover | Inference, seconds | Baseline raw source MAE | Locked raw source MAE | Quality after exact composite |
|---|---:|---:|---:|---|
| track1 | 5.80 | 63.15 | 7.38 | Duplicate person/moon and invented title remain |
| track2 | 8.10 | 108.80 | 10.85 | Duplicate faces/title and invented clothing text remain |
| track3 | 8.57 | 42.20 | 6.67 | Duplicate cars and incorrect continuation remain |

MAE uses RGB channel units 0–255, within the raw decoded source rectangle before
compositing. Its improvement shows that latent restoration protects known content
much better. It is **not an outpainting quality score**. Nonzero raw error is
expected from the lossy AE and boundary receptive fields; the exact composited
source maximum error remains zero for all three.

The locked variant makes parts of the boundary transition smoother, but does not
solve content collisions or semantic duplication. None of these three results is
suitable for ArtWorker production outpainting. A mask-trained editor or adapted
student remains necessary to test.

Cold model load: 3.20 seconds. Full process: 28.82 seconds. Sampled MPS driver
peaks were approximately **6.64–6.65 GB**, with current allocation about 3.37 GB.
Samples were taken after AE encoding, every synchronized diffusion step and
decoding; they can miss higher transient allocations inside an operator. These
are Mac PyTorch measurements, not physical-iPhone or CoreML results.

Artifacts: `results_latent_lock/trial.json`,
`results_latent_lock/comparison.json`, `latent_lock.log`, and six new raw/composited
PNGs. Trial metadata uses absolute artifact paths.

## iOS source verification

The README advertises `edit <prompt> + image`, but current
`Mobile-O-App/app/MobileO/ViewModels/ChatViewModel.swift` routes only `generate`
to image generation and attached images to understanding. No edit branch is
implemented. `MobileOGenerator.swift` extracts text-only conditioning.

The released export script fixes DiT and VAE latents to 16×16 (512×512 image);
only conditioning sequence length is flexible. The downloaded native package
therefore does not establish portrait generation support. Source `ModelVariant`
currently exposes FP32 CoreML, with `.cpuAndGPU`; the alternative ANE split
packages on the Hub are not wired through that enum.

The authors explicitly require a physical iPhone 15 or later and state Simulator
unsupported. Their 3–4s/512² and <2GB claims are author-reported and were not
verified on a phone here. Native editing and portrait support need further export
and Swift work before any ArtWorker integration.
