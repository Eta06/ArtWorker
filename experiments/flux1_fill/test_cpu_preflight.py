#!/usr/bin/env python3
"""Weight-free, CPU-only parity checks for MFLUX's native FLUX.1 Fill inputs.

Run with the existing MLX environment:
    experiments/qwen/.venv/bin/python experiments/flux1_fill/test_cpu_preflight.py

The NumPy references use pixel/channel indices, rather than copying the native
reshape/transpose implementation. Synthetic images are temporary; no model,
weight loader, network access, or GPU computation is needed.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]


class MockVAE:
    """Record the normalized encoder input and emit deterministic 16-channel data."""

    def __init__(self, mx, height: int, width: int, temporal_axis: bool = True):
        self.mx = mx
        self.height = height
        self.width = width
        self.temporal_axis = temporal_axis
        self.inputs: list[np.ndarray] = []
        channels = np.arange(16, dtype=np.float32)[:, None, None]
        rows = np.arange(height // 8, dtype=np.float32)[None, :, None]
        columns = np.arange(width // 8, dtype=np.float32)[None, None, :]
        self.latents = ((channels * 17 + rows * 3 + columns * 5) % 257) / 128 - 1
        self.latents = self.latents[None, :, None] if temporal_axis else self.latents[None]

    def encode(self, image):
        self.mx.eval(image)
        self.inputs.append(np.array(image, dtype=np.float32, copy=True))
        return self.mx.array(self.latents)


def _normalized_rgb(image: Image.Image) -> np.ndarray:
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / np.float32(255)
    return (pixels * np.float32(2) - np.float32(1)).transpose(2, 0, 1)[None]


def _binary_mask(image: Image.Image) -> np.ndarray:
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / np.float32(255)
    return (pixels >= np.float32(0.5)).astype(np.float32).transpose(2, 0, 1)[None]


def _reference_pack(latents: np.ndarray, height: int, width: int) -> np.ndarray:
    """Token (y,x), feature (channel,dy,dx) selects latent (2y+dy,2x+dx)."""
    if latents.ndim == 5:
        if latents.shape[2] != 1:
            raise ValueError("The VAE temporal axis must be a singleton")
        latents = latents[:, :, 0]
    channels = latents.shape[1]
    token_ids = np.arange((height // 16) * (width // 16))[:, None]
    feature_ids = np.arange(channels * 4)[None, :]
    token_y, token_x = token_ids // (width // 16), token_ids % (width // 16)
    channel = feature_ids // 4
    dy, dx = (feature_ids % 4) // 2, feature_ids % 2
    return latents[0, channel, 2 * token_y + dy, 2 * token_x + dx][None]


def _reference_unpack(packed: np.ndarray, height: int, width: int) -> np.ndarray:
    """Invert the token/feature index equation without reshape or transpose."""
    channels = packed.shape[-1] // 4
    channel = np.arange(channels)[:, None, None]
    y = np.arange(height // 8)[None, :, None]
    x = np.arange(width // 8)[None, None, :]
    token = (y // 2) * (width // 16) + x // 2
    feature = channel * 4 + (y % 2) * 2 + x % 2
    return packed[0, token, feature][None]


def _reference_reshape_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """Channel 8*dy+dx retains one pixel offset from every 8x8 block."""
    channel = np.arange(64)[:, None, None]
    y = np.arange(height // 8)[None, :, None]
    x = np.arange(width // 8)[None, None, :]
    return mask[0, 0, 8 * y + channel // 8, 8 * x + channel % 8][None]


def _reference_packed_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """Compute all 256 features directly from the original full-resolution mask.

    Feature = 4*(8*pixel_dy+pixel_dx) + 2*latent_dy + latent_dx.
    This differs from simply flattening a 16x16 image tile in row order.
    """
    token_ids = np.arange((height // 16) * (width // 16))[:, None]
    feature_ids = np.arange(256)[None, :]
    token_y, token_x = token_ids // (width // 16), token_ids % (width // 16)
    pixel_dy, pixel_dx = (feature_ids // 4) // 8, (feature_ids // 4) % 8
    latent_dy, latent_dx = (feature_ids % 4) // 2, feature_ids % 2
    return mask[0, 0, 16 * token_y + 8 * latent_dy + pixel_dy,
                16 * token_x + 8 * latent_dx + pixel_dx][None]


def _synthetic_rgb(height: int, width: int) -> Image.Image:
    y, x = np.indices((height, width), dtype=np.int32)
    pixels = np.stack(((19 * x + 7 * y + 31) % 256,
                       (3 * x + 29 * y + 97) % 256,
                       (23 * x + 11 * y + 173) % 256), axis=-1).astype(np.uint8)
    return Image.fromarray(pixels)


def _check_conditioning(mx, MaskUtil, FluxLatentCreator, image, mask, directory,
                        height, width, temporal_axis):
    image_path = directory / "image.png"
    mask_path = directory / "mask.png"
    image.save(image_path)
    mask.save(mask_path)
    # PIL is the independent resizing reference; no native ImageUtil helper here.
    scaled_image = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    scaled_mask = mask.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    expected_mask = _binary_mask(scaled_mask)
    expected_input = _normalized_rgb(scaled_image) * (np.float32(1) - expected_mask)
    vae = MockVAE(mx, height, width, temporal_axis=temporal_axis)
    static = MaskUtil.create_masked_latents(vae, height, width, image_path, mask_path)
    mx.eval(static)
    actual = np.asarray(static)
    np.testing.assert_array_equal(vae.inputs[0], expected_input)
    expected_source = _reference_pack(vae.latents, height, width)
    expected_packed_mask = _reference_packed_mask(expected_mask, height, width)
    expected_static = np.concatenate((expected_source, expected_packed_mask), axis=-1)
    np.testing.assert_array_equal(actual, expected_static)
    assert len(vae.inputs) == 1, "The masked source must be encoded exactly once"
    assert actual.shape == (1, (height // 16) * (width // 16), 320)
    # The real transformer receives 64 dynamic channels followed by 320 static ones.
    dynamic = FluxLatentCreator.create_noise(seed=9043, height=height, width=width)
    hidden = mx.concatenate((dynamic, static), axis=-1)
    mx.eval(hidden)
    assert hidden.shape == (1, (height // 16) * (width // 16), 384)
    np.testing.assert_array_equal(np.asarray(hidden)[..., 64:], expected_static)
    return vae, expected_mask, actual, list(hidden.shape)


def run_preflight() -> dict:
    """Assert installed native Fill conditioning matches independent CPU references."""
    import mlx.core as mx

    # Set the device before importing the native utilities or creating any arrays.
    mx.set_default_device(mx.cpu)
    from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
    from mflux.models.flux.variants.fill.mask_util import MaskUtil
    from mflux.utils.image_util import ImageUtil

    assert mx.default_device() == mx.cpu
    checks = []

    # Black/white polarity and the exact threshold, including unequal RGB channels.
    levels = np.array([0, 1, 63, 127, 128, 192, 254, 255], dtype=np.uint8)
    threshold_rgb = np.stack((levels, levels[::-1], np.roll(levels, 3)), axis=-1)[None]
    threshold_image = Image.fromarray(threshold_rgb)
    native_mask = ImageUtil.to_array(threshold_image, is_mask=True)
    native_rgb = ImageUtil.to_array(threshold_image)
    mx.eval(native_mask, native_rgb)
    np.testing.assert_array_equal(np.asarray(native_mask), _binary_mask(threshold_image))
    np.testing.assert_array_equal(np.asarray(native_rgb), _normalized_rgb(threshold_image))
    assert float(np.asarray(native_mask)[0, 0, 0, 0]) == 0  # black is known
    assert float(np.asarray(native_mask)[0, 0, 0, -1]) == 1  # white is unknown
    assert float(np.asarray(native_rgb)[0, 0, 0, 0]) == -1  # normalized black is -1
    checks.append("white_unknown_black_known_and_threshold_128")

    # Unique values expose every offset, channel, and rectangular token ordering.
    height, width = 32, 48
    latent_np = np.arange(16 * (height // 8) * (width // 8), dtype=np.float32)
    latent_np = latent_np.reshape(1, 16, height // 8, width // 8)
    for temporal_axis in (False, True):
        latent = latent_np[:, :, None] if temporal_axis else latent_np
        packed = FluxLatentCreator.pack_latents(mx.array(latent), height, width)
        mx.eval(packed)
        expected = _reference_pack(latent, height, width)
        np.testing.assert_array_equal(np.asarray(packed), expected)
        unpacked = FluxLatentCreator.unpack_latents(packed, height, width)
        mx.eval(unpacked)
        np.testing.assert_array_equal(np.asarray(unpacked), _reference_unpack(expected, height, width))
        np.testing.assert_array_equal(np.asarray(unpacked), latent_np)
    checks.append("latent_pack_unpack_4d_and_singleton_5d")

    y, x = np.indices((height, width), dtype=np.int32)
    nonuniform = (((x * 11 + y * 7 + (x // 3) * (y // 5)) % 17) < 8).astype(np.float32)
    coordinate_ramp = (y * width + x).astype(np.float32)
    for first_channel in (nonuniform, coordinate_ramp):
        # Different extra RGB channels confirm that reshape_mask selects channel zero.
        # The ramp makes every pixel distinct, so no channel permutation can hide
        # behind repeated zero/one values in the realistic binary-mask case.
        mask_np = np.stack((first_channel, 1 - nonuniform, (x % 2).astype(np.float32)), axis=0)[None]
        reshaped = MaskUtil.reshape_mask(mx.array(mask_np), height, width)
        mask_packed = FluxLatentCreator.pack_latents(reshaped, height, width, num_channels_latents=64)
        mx.eval(reshaped, mask_packed)
        np.testing.assert_array_equal(np.asarray(reshaped), _reference_reshape_mask(mask_np, height, width))
        np.testing.assert_array_equal(np.asarray(mask_packed), _reference_packed_mask(mask_np, height, width))
        np.testing.assert_array_equal(_reference_unpack(np.asarray(mask_packed), height, width),
                                      _reference_reshape_mask(mask_np, height, width))
    checks.append("all_8x8_pixel_channels_then_2x2_latent_patches")

    with tempfile.TemporaryDirectory(prefix="flux1-fill-cpu-") as temporary:
        directory = Path(temporary)
        # A resizing case checks native loading, Lanczos resampling, normalization,
        # binarization, and final concatenation together, with both VAE output ranks.
        small_image = _synthetic_rgb(21, 31)
        sy, sx = np.indices((19, 29), dtype=np.int32)
        small_mask_pixels = ((sx * 53 + sy * 71) % 256).astype(np.uint8)
        small_mask = Image.fromarray(small_mask_pixels).convert("RGB")
        for temporal_axis in (False, True):
            _check_conditioning(mx, MaskUtil, FluxLatentCreator, small_image, small_mask,
                                directory, height, width, temporal_axis)
        checks.append("native_loading_resizing_masked_rgb_and_static_concat")

        # Exact half-open source box: 512x512 at y=320..831 on 512x1152.
        height, width = 1152, 512
        source_box = (0, 320, 512, 832)
        resized_source = _synthetic_rgb(47, 43).resize((512, 512), Image.Resampling.LANCZOS)
        canvas = _synthetic_rgb(height, width)
        canvas.paste(resized_source, (source_box[0], source_box[1]))
        mask_pixels = np.full((height, width, 3), 255, dtype=np.uint8)
        mask_pixels[320:832, :, :] = 0
        mask_image = Image.fromarray(mask_pixels)
        vae, geometry_mask, static, transformer_shape = _check_conditioning(
            mx, MaskUtil, FluxLatentCreator, canvas, mask_image, directory,
            height, width, temporal_axis=True,
        )
        known_rgb = _normalized_rgb(resized_source)
        np.testing.assert_array_equal(vae.inputs[0][:, :, 320:832, :], known_rgb)
        np.testing.assert_array_equal(np.asarray(canvas)[320:832], np.asarray(resized_source))
        unknown = np.broadcast_to(geometry_mask.astype(bool), vae.inputs[0].shape)
        assert np.count_nonzero(vae.inputs[0][unknown]) == 0
        assert not np.any(vae.inputs[0][unknown] == -1), "Unknown input must be normalized zero, not black"
        assert np.all(geometry_mask[:, :, 319] == 1)
        assert np.all(geometry_mask[:, :, 320] == 0)
        assert np.all(geometry_mask[:, :, 831] == 0)
        assert np.all(geometry_mask[:, :, 832] == 1)
        # Source boundaries align with token rows 20..51; exterior tokens retain 1.
        token_rows = np.arange((height // 16) * (width // 16)) // (width // 16)
        expected_unknown_tokens = (token_rows < 20) | (token_rows >= 52)
        np.testing.assert_array_equal(static[0, :, 64:],
                                      np.broadcast_to(expected_unknown_tokens[:, None], (2304, 256)))
        checks.append("512x1152_source_box_exact_resized_rgb_and_unknown_normalized_zero")

    # Missing inputs return an empty tensor without invoking the encoder.
    empty = MaskUtil.create_masked_latents(vae, height, width, None, None)
    mx.eval(empty)
    assert empty.shape == (1, 0, 0)
    assert len(vae.inputs) == 1
    checks.append("missing_input_empty_conditioning_without_encode")
    assert mx.default_device() == mx.cpu
    return {
        "status": "passed",
        "device": "cpu",
        "weights_loaded": False,
        "downloads_required": False,
        "checks": checks,
        "geometry": {
            "width": width,
            "height": height,
            "source_box": list(source_box),
            "source_box_convention": "half-open [left, top, right, bottom]",
            "source_pixels_exact": True,
            "known_pixels": int(np.count_nonzero(geometry_mask[0, 0] == 0)),
            "unknown_pixels": int(np.count_nonzero(geometry_mask[0, 0] == 1)),
            "known_token_rows": [20, 52],
            "known_normalized_rgb_max_abs_error": 0.0,
            "unknown_normalized_rgb": 0.0,
            "normalized_black_rgb": -1.0,
            "mock_vae_shape": list(vae.latents.shape),
            "static_shape": list(static.shape),
            "masked_source_channels": 64,
            "mask_channels": 256,
            "transformer_input_shape": transformer_shape,
        },
        "scope": "conditioning parity only; no inference or generated-image quality test",
    }


def test_cpu_preflight():
    assert run_preflight()["status"] == "passed"


def main() -> int:
    try:
        result = run_preflight()
    except Exception as error:
        print(json.dumps({"status": "failed", "error": f"{type(error).__name__}: {error}"}, indent=2))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
