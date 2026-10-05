# Control conditioning utility

`conditioning.py` follows the [pinned VideoX Qwen 2.1 pipeline](https://github.com/aigc-apps/VideoX-Fun/blob/4b7b6402a1e0f0406bd6801fb66c0a00bd922621/videox_fun/pipeline/pipeline_qwenimage21_control.py#L636-L668). Its control layout is `[control latents 64, known mask 1, masked-source latents 64]`. Pixel masks use white for regeneration; the packed mask uses the inverse, 1 for known source. Source RGB is normalized to `[-1,1]`, regeneration RGB becomes zero, and opaque alpha `+1` is appended afterward. Missing control produces literal zero latent channels, not an encoded black image.

The [denoising loop](https://github.com/aigc-apps/VideoX-Fun/blob/4b7b6402a1e0f0406bd6801fb66c0a00bd922621/videox_fun/pipeline/pipeline_qwenimage21_control.py#L677-L750) disables prefix KV caching with active control. The [official conditioning path](https://github.com/aigc-apps/VideoX-Fun/blob/4b7b6402a1e0f0406bd6801fb66c0a00bd922621/videox_fun/pipeline/pipeline_qwenimage21_control.py#L580-L603) carries no reference-image prefix. An optional zero-padding helper for a separate image-prefix experiment is explicitly labelled experimental; it does not validate that model variant.

## API

```python
ctx, meta = encode_control_context(
    mx, np, Image, vae, latent_creator,
    source_rgb, source_rect, width, height, control_rgb=None,
)
```

For the shared 512×1152 canvas and `[0,320,512,832]` source rectangle, `ctx.shape == (1,2304,129)`; channel 64 contains 1 at latent rows 20 through 51. The existing MLX reference VAE normalizes its mean latents internally. A raw, unnormalized VAE wrapper must normalize before returning from `encode`; this helper does not double-normalize.

The helper returns the masked VAE output dtype. Before a transformer forward, cast the whole context to the prompt/DiT dtype, as the official pipeline does: `ctx = ctx.astype(prompt_embeds.dtype)`.

- `prepare_control_pixels` exposes the source canvas, white-generation pixel mask, known latent mask and normalized RGBA inputs before encoding.
- `fullcanvas_control_rgba` accepts a full-size caller-provided RGB Canny/MLSD/control rendering and appends opaque alpha.
- `place_source_control` places a source-size RGB detector rendering on a black canvas. It draws no continuation beyond the source.
- `gather_control_context` selects final-canvas absolute raster IDs when target computation grows. Encode the control canvas once; do not resize it at each growth step.
- `control_prefix_padding` adds zero context for reference image latent tokens only. Text tokens must not be included in that count.

The helpers never resize or change source bytes. They do not infer barrier geometry, perform Canny/MLSD detection, enforce final pixel preservation, or load ControlNet weights. Source-exact final compositing remains the caller's responsibility. Conditioning preparation alone is not model parity or visual-quality evidence.

## CPU preflight

```sh
experiments/qwen/.venv/bin/python experiments/qwen/controlnet_port/conditioning.py
```

Passed on 2026-10-05 with MLX's CPU device and a distinguishable-channel mock VAE/packer. Verified normalization and alpha, known rows 20:52, known mask polarity, channel packing, unchanged source/control bytes, literal zero control without a VAE call, absolute growth gather, experimental reference padding, and an offset source rectangle. No model weights or GPU computation were used.

A separate read-only review found no blocking normalization, mask, alpha or channel-layout error. This review also confirmed the caller-side dtype cast above.
