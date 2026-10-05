# Çınar — mobile and distillation direction, 2026-10-06

This is a research plan with measured Mac feasibility, not an iPhone performance claim. The current Qwen full edit export is 10.277 GB of weights, including the image/text encoder and VAE; task adapters are additional. The sequential native loader reduces working memory but does not make this a small mobile model.

## Separate the optimizations

1. **Quantization:** fewer bits per parameter. The current tensor-wise native Q4 export demonstrates a bounded conversion from the cached complete Qwen checkpoint. This does not reduce parameter count or denoising steps. Full checkpoint semantics and geometry still require evaluation.
2. **Step distillation:** fewer denoiser evaluations. [Viggle v0.3](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo/tree/009a44a895ef85f7e643c80fdca9543795248867) supplies a six-step adapter, including rank128/256 and merged formats. It keeps the large base network. Combining a general speed adapter and the outpaint adapter is experimental; visual equivalence cannot be inferred from successful loading. We downloaded and verified the 679,604,800-byte rank128 adapter, avoiding a second entire merged checkpoint.
3. **Architecture distillation:** train a genuinely smaller denoiser/conditioner. This is the route toward broad device support, but there is no trained ArtWorker student or accepted teacher yet. A tiny denoiser paired with a multi-gigabyte VLM is not a tiny full system.

## Compact baseline actually executed

[MI-GAN](https://github.com/Picsart-AI-Research/MI-GAN) is an older ICCV 2023 baseline, selected for its complete system size, not recency. The author-linked ONNX pipeline is 28,079,181 bytes; both code and separate weight licenses are MIT. Three CPU trials completed in 0.51–0.66 seconds with 0.36–0.40 GiB sampled footprint. They fail this large portrait extension: scenery/texture hallucinations and visible joins. Ten sequential 64-row strips cost 2.07 seconds and accumulated severe drift. This establishes a small deployable architecture direction, not acceptable ArtWorker quality or an iPhone result.

Klein base's tensor-wise INT4 export is 2,180,050,128 bytes. A separate version with all 88 outpaint adapter pairs applied before its one quantization is 2,180,050,216 bytes. Qwen3 text weights add 2,263,022,417 bytes and standard VAE adds 168,120,878 bytes: the present full Klein system is about 4.61 GB, excluding tokenizer/config overhead. The 2.18 GB transformer alone is not the full mobile footprint. Export peak footprint was 0.71 GiB unbaked / 1.06 GiB baked. Runtime key/shape/dtype validation passed; decoding/visual acceptance remain separate. Caching the fixed fill instruction's embeddings could eliminate the large text encoder from a narrowly scoped inference product, but that optimization has not been implemented or benchmarked here.

## Feasible student investigation

Once a teacher passes the boundary/scene evaluation, prepare paired original/extended pictures from licensed sources, with random crop placement, aspect ratio, extension size and scene type. Include water, road markings, guardrails, architecture, artwork, text and people. Keep image families separated between train/validation/test. Album covers currently serve as local evaluation examples; they are not automatically licensed training or public dataset material.

First fine-tune task conditioning on a compact diffusion backbone using the source image, unknown-area mask and target geometry. A provisional design target is a 0.4–1B denoiser with a compact image conditioner and cached or fixed instruction embeddings. Those are hypotheses and size targets, not an implemented architecture or promised image quality. Train source registration and boundary geometry explicitly; preserve the original artwork in the final compositor and evaluate the raw model separately.

Only then distill the trajectory to 4–8 steps, evaluate full-system memory, and test mixed-bit compression. Step-distillation alone cannot shrink a 4B/7B backbone. Teacher/adapter/student redistribution terms must be reviewed before releasing weights; Qwen uses its Research license, while Klein's inspected 4B base/adapter licenses are Apache-2.0. Our code license does not relicense third-party weights.

## iOS implementation path

[Apple's Core ML diffusion package](https://github.com/apple-aiml-research/ml-stable-diffusion) demonstrates a Swift pipeline, just-in-time component loading through `reduceMemory`, device-specific compute-unit selection and compressed weights. [Core ML Tools' optimization guide](https://apple.github.io/coremltools/docs-guides/source/opt-stable-diffusion.html) covers palettization and activation quantization. These provide deployment techniques; they do not supply a ready Qwen/Klein converter or prove that every operation will execute on the Neural Engine. Apple's published Stable Diffusion phone benchmarks concern different architectures and resolutions and must not be substituted for ours.

Compile and profile the actual student with fixed resolution buckets. Test on physical devices spanning older/lower-memory phones and recent Pro phones, with playback active. Record cold/warm generation time, peak memory, model storage, thermal behavior, battery and termination. Lower resolution/fewer steps and cached artwork are potential device-specific modes; acceptable behavior across all phones remains a goal to validate.

## Progressive output

Keep true progressive spatial generation separate from denoising previews or a reveal animation. A progressive student would need joint training with expanding active regions and boundary context. Re-running a complete phone canvas at every reveal is not the desired speed optimization. Current trials generate the complete target once; they do not claim center-outward streaming.

## Acceptance before training

Use matched seeds/geometry and review raw plus exact-source composite. Reject duplicate subjects, re-framing, seams, mismatched texture scale and disconnected/wrongly angled structure. Numeric source integrity is a compositor check, not an outpaint score. Extend successful candidates beyond three covers to held-out licensed images and multiple seeds before accepting a teacher or launching costly training.

## Additional current mask-conditioned candidate

[Alibaba-PAI Z-Image Turbo Fun Union 2.1](https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1/tree/5155fc56d17821007d6f62ac192c09e0f0e72016) was verified on 2026-10-06 at revision `5155fc56d17821007d6f62ac192c09e0f0e72016`. The current 2602 eight-step full/lite control files are 6,712,485,600 / 2,016,627,488 bytes, in addition to the Z-Image backbone, text encoder and VAE. The author's card describes inpaint conditioning and earlier mask-leakage fixes. The local pinned MFLUX ControlSpec exposes pose/canny/hed/depth/mlsd, but no mask-conditioned inpaint mode; its zero-padded generic control encoding is not a verified inpaint recipe. Neither payload was downloaded or run. This remains a genuine additional candidate requiring its correct reference pipeline and a bounded export, rather than silently treating a generic image control as outpaint.
