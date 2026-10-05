"""Qwen-Image 2.1 ControlNet conditioning, independent of the model loader.

The VideoX control pipeline at 4b7b6402a1e0f0406bd6801fb66c0a00bd922621
uses 64 control latents, one *known* mask channel, and 64 masked-source
latents. Images supplied here are already at their intended pixel size.
No geometry, detector, weights, or device setup happens during import.
"""

from __future__ import annotations

import hashlib


PIPELINE_COMMIT = "4b7b6402a1e0f0406bd6801fb66c0a00bd922621"
PIPELINE_URL = (
    "https://github.com/aigc-apps/VideoX-Fun/blob/" + PIPELINE_COMMIT
    + "/videox_fun/pipeline/pipeline_qwenimage21_control.py"
)


def _dimensions(width, height, source_rect):
    if not isinstance(width, int) or not isinstance(height, int):
        raise TypeError("Canvas width/height must be integer pixel counts")
    if width < 32 or height < 32 or width % 32 or height % 32:
        raise ValueError("Qwen 2.1 canvas dimensions must be multiples of 32")
    if len(source_rect) != 4 or any(not isinstance(v, int) for v in source_rect):
        raise TypeError("source_rect must contain four integer XYXY coordinates")
    x1, y1, x2, y2 = source_rect
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("source_rect must be a nonempty rectangle inside the canvas")
    return x1, y1, x2, y2


def _rgb_array(np, Image, value, width, height, label):
    """Accept bytes only; do not silently rescale, resize, or composite alpha."""
    if isinstance(value, Image.Image):
        if value.mode not in ("RGB", "RGBA"):
            raise ValueError(f"{label} must be RGB or opaque RGBA")
        value = np.asarray(value)
    array = np.asarray(value)
    if array.dtype != np.uint8 or array.shape not in ((height, width, 3), (height, width, 4)):
        raise ValueError(f"{label} must be uint8 RGB/RGBA at {width}x{height}; resizing is explicit")
    if array.shape[-1] == 4 and not np.all(array[..., 3] == 255):
        raise ValueError(f"{label} must have opaque alpha")
    return np.ascontiguousarray(array[..., :3]).copy()


def fullcanvas_control_rgba(np, Image, control_rgb, width, height):
    """Prepare a caller-supplied Canny/MLSD/etc. canvas without drawing new lines.

    The returned HWC RGBA bytes have constant alpha 255. A detector output
    or manually specified geometry must already span this exact canvas.
    """
    rgb = _rgb_array(np, Image, control_rgb, width, height, "control_rgb")
    alpha = np.full((height, width, 1), 255, dtype=np.uint8)
    return np.concatenate([rgb, alpha], axis=-1)


def place_source_control(np, Image, source_control_rgb, source_rect, width, height):
    """Place an existing source detector output on black; do not extrapolate it."""
    x1, y1, x2, y2 = _dimensions(width, height, source_rect)
    source = _rgb_array(np, Image, source_control_rgb, x2 - x1, y2 - y1, "source_control_rgb")
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[y1:y2, x1:x2] = source
    return fullcanvas_control_rgba(np, Image, canvas, width, height)


def prepare_control_pixels(np, Image, source_rgb, source_rect, width, height, control_rgb=None):
    """Build official normalized pixels and the nearest-resized known mask.

    Input mask semantics are white=regenerate, black=known. The packed latent
    mask is their inverse: known=1, regenerate=0. Regenerated source RGB is
    zero in normalized coordinates *before* constant alpha 1 is appended.
    """
    x1, y1, x2, y2 = _dimensions(width, height, source_rect)
    source = _rgb_array(np, Image, source_rgb, x2 - x1, y2 - y1, "source_rgb")
    source_canvas = np.zeros((height, width, 3), dtype=np.uint8)
    source_canvas[y1:y2, x1:x2] = source
    known = np.zeros((height, width), dtype=np.float32)
    known[y1:y2, x1:x2] = 1.0
    generate_mask = ((1.0 - known) * 255).astype(np.uint8)
    normalized = source_canvas.astype(np.float32) / 127.5 - 1.0
    masked_rgb = normalized * known[..., None]
    alpha = np.ones((height, width, 1), dtype=np.float32)
    masked_rgba = np.concatenate([masked_rgb, alpha], axis=-1)
    masked_pixels = masked_rgba.transpose(2, 0, 1)[None, :, None]
    lat_h, lat_w = height // 16, width // 16
    # torch.interpolate(mode="nearest") uses floor(output_index * input/output).
    ys = np.arange(lat_h, dtype=np.int64) * height // lat_h
    xs = np.arange(lat_w, dtype=np.int64) * width // lat_w
    known_latent = known[ys[:, None], xs[None, :]][None, None, None]
    control_pixels = None
    if control_rgb is not None:
        rgba = fullcanvas_control_rgba(np, Image, control_rgb, width, height)
        # Normalize RGB, then append alpha +1 (opaque), as VideoX _rgba does.
        rgb = rgba[..., :3].astype(np.float32) / 127.5 - 1.0
        control_pixels = np.concatenate([rgb, alpha], axis=-1).transpose(2, 0, 1)[None, :, None]
    metadata = {
        "pipeline_commit": PIPELINE_COMMIT,
        "canvas_size": [width, height],
        "source_rect_xyxy": list(source_rect),
        "source_uint8_sha256": hashlib.sha256(source.tobytes()).hexdigest(),
        "pixel_mask_polarity": "white=regenerate, black=known",
        "latent_mask_polarity": "1=known, 0=regenerate",
        "known_pixel_count": int(known.sum()),
        "known_latent_tokens": int(known_latent.sum()),
        "latent_grid": [lat_h, lat_w],
        "control_supplied": control_rgb is not None,
        "unknown_normalized_rgb": 0.0,
        "rgba_alpha": 1.0,
        "automatic_geometry_inference": False,
    }
    return {
        "source_canvas_rgb": source_canvas,
        "generate_mask_uint8": generate_mask,
        "known_latent": known_latent,
        "masked_source_pixels": masked_pixels,
        "control_pixels": control_pixels,
        "metadata": metadata,
    }


def encode_control_context(mx, np, Image, vae, latent_creator, source_rgb, source_rect,
                           width, height, control_rgb=None):
    """Return (packed_context [1,H/16*W/16,129], metadata).

    ``vae.encode`` must return mean/mode latents already normalized by the
    Qwen 2.1 channel mean/std, as the existing MLX reference VAE does. The
    caller controls device and dtype. No weights are loaded by this utility.
    """
    prepared = prepare_control_pixels(np, Image, source_rgb, source_rect, width, height, control_rgb)
    expected_shape = (1, height // 16 * (width // 16), 64)
    masked = latent_creator.pack_latents(vae.encode(mx.array(prepared["masked_source_pixels"])))
    if masked.shape != expected_shape:
        raise ValueError(f"Masked VAE output packed as {masked.shape}; expected {expected_shape}")
    if prepared["control_pixels"] is None:
        control = mx.zeros_like(masked)
    else:
        control = latent_creator.pack_latents(vae.encode(mx.array(prepared["control_pixels"])))
        if control.shape != expected_shape:
            raise ValueError(f"Control VAE output packed as {control.shape}; expected {expected_shape}")
        control = control.astype(masked.dtype)
    known = latent_creator.pack_latents(mx.array(prepared["known_latent"])).astype(masked.dtype)
    context = mx.concatenate([control, known, masked], axis=-1)
    if context.shape != (expected_shape[0], expected_shape[1], 129):
        raise AssertionError("Unexpected ControlNet conditioning channel layout")
    metadata = dict(prepared["metadata"])
    metadata.update({
        "packed_shape": list(context.shape),
        "channel_ranges": {"control": [0, 64], "known_mask": [64, 65], "masked_source": [65, 129]},
        "control_encode_count": 0 if control_rgb is None else 1,
        "source_encode_count": 1,
        "reference_image_prefix": "none in official control pipeline",
    })
    return context, metadata


def gather_control_context(mx, control_context, absolute_target_ids):
    """Gather from the final-canvas raster order; never re-encode on growth."""
    if control_context.ndim != 3 or control_context.shape[0] != 1 or control_context.shape[-1] != 129:
        raise ValueError("Expected one packed final-canvas ControlNet context")
    return control_context[:, mx.array(absolute_target_ids, dtype=mx.int32)]


def control_prefix_padding(mx, control_context, reference_tokens):
    """Experimental image-prefix adapter, not the official text-only recipe.

    Adds literal zeros for reference IMAGE latent tokens, not text tokens.
    A caller must separately verify its ControlNet/DiT layout and use no
    prefix KV cache when control is active.
    """
    if not isinstance(reference_tokens, int) or reference_tokens < 0:
        raise ValueError("reference_tokens must be a nonnegative integer")
    if control_context.ndim != 3 or control_context.shape[0] != 1 or control_context.shape[-1] != 129:
        raise ValueError("Expected one packed ControlNet context")
    zeros = mx.zeros((1, reference_tokens, 129), dtype=control_context.dtype)
    return mx.concatenate([zeros, control_context], axis=1)


def cpu_preflight():
    """Check normalization, polarity, packing and growth gathers without weights."""
    import mlx.core as mx
    import numpy as np
    from PIL import Image

    mx.set_default_device(mx.cpu)
    width, height, rect = 512, 1152, [0, 320, 512, 832]
    source = np.zeros((512, 512, 3), dtype=np.uint8)
    source[..., 0], source[..., 1], source[..., 2] = 23, 141, 217
    source[0, :, 0] = np.arange(512, dtype=np.uint16) % 256
    before = source.copy()
    captured = []

    class MockVAE:
        def encode(self, pixels):
            values = np.asarray(pixels)
            captured.append(values.copy())
            assert values.shape == (1, 4, 1, height, width)
            token_pattern = np.arange((height // 16) * (width // 16), dtype=np.float32).reshape(1, 1, 1, height // 16, width // 16)
            channels = np.arange(64, dtype=np.float32)[None, :, None, None, None]
            return mx.array(token_pattern + channels * 10000 + len(captured) * 1000000)

    class MockPacker:
        @staticmethod
        def pack_latents(latents):
            if latents.ndim == 5:
                latents = latents[:, :, 0]
            b, c, h, w = latents.shape
            return latents.transpose(0, 2, 3, 1).reshape(b, h * w, c)

    prepared = prepare_control_pixels(np, Image, source, rect, width, height)
    keep = prepared["known_latent"][0, 0, 0]
    expected_keep = np.zeros((72, 32), dtype=np.float32)
    expected_keep[20:52] = 1
    assert np.array_equal(keep, expected_keep)
    assert np.all(prepared["generate_mask_uint8"][:320] == 255)
    assert np.all(prepared["generate_mask_uint8"][320:832] == 0)
    assert np.all(prepared["generate_mask_uint8"][832:] == 255)
    ctx, meta = encode_control_context(mx, np, Image, MockVAE(), MockPacker,
                                       Image.fromarray(source), rect, width, height)
    actual = np.asarray(ctx)
    assert len(captured) == 1 and ctx.shape == (1, 2304, 129)
    assert np.all(actual[..., :64] == 0), "Missing control was VAE-encoded"
    assert np.array_equal(actual[0, :, 64].reshape(72, 32), expected_keep)
    pixels = captured[0]
    assert np.all(pixels[:, :3, :, :320] == 0)
    assert np.all(pixels[:, :3, :, 832:] == 0)
    assert np.all(pixels[:, 3] == 1)
    expected_rgb = (source.astype(np.float32) / 127.5 - 1).transpose(2, 0, 1)
    assert np.array_equal(pixels[0, :3, 0, 320:832], expected_rgb)
    expected_pattern = np.arange(2304, dtype=np.float32) + 1000000
    assert np.array_equal(actual[0, :, 65], expected_pattern)
    assert np.array_equal(actual[0, :, 128], expected_pattern + 63 * 10000)

    control = np.zeros((height, width, 3), dtype=np.uint8)
    control[90:95, 8:502] = 255
    control_before = control.copy()
    captured.clear()
    controlled, controlled_meta = encode_control_context(mx, np, Image, MockVAE(), MockPacker,
                                                        source, rect, width, height, control)
    controlled_np = np.asarray(controlled)
    assert len(captured) == 2
    assert np.array_equal(controlled_np[0, :, 0], np.arange(2304, dtype=np.float32) + 2000000)
    assert np.array_equal(controlled_np[..., 65:], actual[..., 65:])
    assert np.all(captured[1][:, 3] == 1)
    assert np.array_equal(captured[1][0, :3, 0], (control.astype(np.float32) / 127.5 - 1).transpose(2, 0, 1))
    assert np.array_equal(source, before) and np.array_equal(control, control_before)

    ids = np.arange(16 * 32, 56 * 32, dtype=np.int32)
    gathered = np.asarray(gather_control_context(mx, ctx, ids))
    assert np.array_equal(gathered, actual[:, ids])
    assert np.array_equal(gathered[0, :, 64].reshape(40, 32)[4:36], np.ones((32, 32)))
    padded = np.asarray(control_prefix_padding(mx, ctx, 1024))
    assert padded.shape == (1, 3328, 129)
    assert np.all(padded[:, :1024] == 0) and np.array_equal(padded[:, 1024:], actual)
    # A horizontal and vertical offset exercises nearest coordinates beyond the shared crop.
    offset = prepare_control_pixels(np, Image, np.zeros((64, 64, 3), dtype=np.uint8),
                                    [32, 48, 96, 112], 128, 160)
    offset_expected = np.zeros((10, 8), dtype=np.float32)
    offset_expected[3:7, 2:6] = 1
    assert np.array_equal(offset["known_latent"][0, 0, 0], offset_expected)
    return {
        "status": "passed", "device": "CPU", "model_weights_loaded": False,
        "source_unchanged": True, "control_unchanged": True,
        "packed_shape": list(ctx.shape), "known_source_latent_rows": [20, 52],
        "known_source_latent_tokens": meta["known_latent_tokens"],
        "known_mask_channel": 64, "unknown_source_rgb_normalized_zero": True,
        "rgba_alpha_constant_one": True, "no_control_is_literal_zero_latents": True,
        "absolute_gather_exact": True, "experimental_prefix_padding_exact": True,
        "offset_rectangle_mask_exact": True,
        "vae_calls_without_control": meta["source_encode_count"],
        "vae_calls_with_control": controlled_meta["source_encode_count"] + controlled_meta["control_encode_count"],
    }


if __name__ == "__main__":
    import json
    print(json.dumps(cpu_preflight(), indent=2))
