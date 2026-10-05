#!/usr/bin/env python3
"""CPU-only native Diffusers SDXL inpaint conditioning checks without checkpoints.

Run: .build/dreamlite-venv/bin/python experiments/sdxl_inpaint/test_cpu_preflight.py

Native processors and pipeline utility methods are used without constructing a
pipeline. A mock encoder supplies deterministic four-channel latents. One tiny
random UNet checks the batched CFG forward contract; no pretrained weights,
downloads, GPU execution, image generation, or VAE fidelity test is involved.
"""

from __future__ import annotations

import json
import os
import sys
from importlib.metadata import version
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class DummyScheduler:
    """Order-one scheduler with a visible input scale and recorded begin index."""

    order = 1

    def __init__(self, torch, steps=20, scale=0.25):
        self.timesteps = torch.arange(steps - 1, -1, -1, device="cpu", dtype=torch.float32)
        self.begin_indices = []
        self.scale_calls = []
        self.scale = scale

    def set_begin_index(self, index):
        self.begin_indices.append(index)

    def scale_model_input(self, latents, timestep):
        self.scale_calls.append({"shape": list(latents.shape), "device": latents.device.type})
        return latents * self.scale


class MockPipeline:
    """Only the attributes needed by native preparation and timestep methods."""

    vae_scale_factor = 8

    def __init__(self, torch, encoded_latents, scheduler):
        self.torch = torch
        self.encoded_latents = encoded_latents
        self.scheduler = scheduler
        self.encoder_inputs = []

    def _encode_vae_image(self, image, generator):
        assert image.device.type == "cpu"
        self.encoder_inputs.append(image.detach().cpu().numpy().copy())
        return self.torch.from_numpy(self.encoded_latents.copy()).to(device="cpu", dtype=image.dtype)


def _synthetic_rgb(np, Image, height, width):
    y, x = np.indices((height, width), dtype=np.int32)
    pixels = np.stack(((17 * x + 7 * y + 29) % 256,
                       (3 * x + 23 * y + 101) % 256,
                       (29 * x + 11 * y + 173) % 256), axis=-1).astype(np.uint8)
    return Image.fromarray(pixels)


def _reference_rgb(np, image):
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / np.float32(255)
    return (pixels * np.float32(2) - np.float32(1)).transpose(2, 0, 1)[None]


def _reference_mask(np, image):
    pixels = np.asarray(image.convert("L"), dtype=np.float32) / np.float32(255)
    return (pixels >= np.float32(0.5)).astype(np.float32)[None, None]


def _reference_nearest(np, mask, height, width):
    # Nearest interpolation samples floor(output_index * input_size / output_size).
    target_height, target_width = height // 8, width // 8
    ys = np.floor(np.arange(target_height) * mask.shape[-2] / target_height).astype(np.int64)
    xs = np.floor(np.arange(target_width) * mask.shape[-1] / target_width).astype(np.int64)
    return mask[:, :, ys[:, None], xs[None, :]]


def _tiny_unet_cfg(torch, np, UNet2DConditionModel, latent_mask, source_latents):
    """One SDXL-style random UNet call handles two CFG branches in one CPU batch."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        # Keep this test deterministic without seeding any CUDA/MPS generator.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(1947)
            unet = UNet2DConditionModel(
                sample_size=8,
                in_channels=9,
                out_channels=4,
                down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
                up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
                block_out_channels=(16, 32),
                layers_per_block=1,
                norm_num_groups=8,
                cross_attention_dim=16,
                attention_head_dim=4,
                addition_embed_type="text_time",
                addition_time_embed_dim=4,
                projection_class_embeddings_input_dim=8 + 6 * 4,
            ).to(device="cpu").eval()
        assert unet.config.in_channels == 9
        assert unet.config.time_cond_proj_dim is None
        assert all(parameter.device.type == "cpu" for parameter in unet.parameters())
        assert all(parameter.dtype == torch.float32 for parameter in unet.parameters())
        calls = []

        def record_forward(module, arguments):
            calls.append({"shape": list(arguments[0].shape), "device": arguments[0].device.type})

        hook = unet.register_forward_pre_hook(record_forward)
        noisy_np = (np.arange(4 * 8 * 8, dtype=np.float32) / np.float32(127) - 1).reshape(1, 4, 8, 8)
        noisy = torch.from_numpy(noisy_np.copy()).to("cpu")
        # This crop crosses the source boundary, so both mask values are exercised.
        mask = latent_mask[:, :, 36:44, :8]
        conditioning = source_latents[:, :, 36:44, :8]
        scheduler = DummyScheduler(torch)
        doubled_noisy = torch.cat((noisy, noisy), dim=0)
        scaled_noisy = scheduler.scale_model_input(doubled_noisy, scheduler.timesteps[0])
        hidden = torch.cat((scaled_noisy, mask, conditioning), dim=1)
        expected_hidden = np.concatenate((np.repeat(noisy_np, 2, axis=0) * np.float32(0.25),
                                          mask.numpy(), conditioning.numpy()), axis=1)
        np.testing.assert_array_equal(hidden.numpy(), expected_hidden)
        assert hidden.shape == (2, 9, 8, 8)
        assert scheduler.scale_calls == [{"shape": [2, 4, 8, 8], "device": "cpu"}]
        assert bool(torch.any(mask == 0)) and bool(torch.any(mask == 1))
        prompt = torch.from_numpy(np.arange(2 * 4 * 16, dtype=np.float32).reshape(2, 4, 16) / 128)
        pooled = torch.from_numpy(np.arange(2 * 8, dtype=np.float32).reshape(2, 8) / 16)
        time_ids = torch.tensor([[1152, 512, 0, 0, 1152, 512]] * 2, device="cpu", dtype=torch.float32)
        with torch.inference_mode():
            prediction = unet(hidden, scheduler.timesteps[0], encoder_hidden_states=prompt,
                              added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids},
                              return_dict=False)[0]
            unconditional, conditional = prediction.chunk(2)
            guided = unconditional + 5.0 * (conditional - unconditional)
        hook.remove()
        assert calls == [{"shape": [2, 9, 8, 8], "device": "cpu"}]
        assert prediction.shape == (2, 4, 8, 8)
        assert guided.shape == (1, 4, 8, 8)
        assert bool(torch.isfinite(prediction).all()) and bool(torch.isfinite(guided).all())
        expected_guided = prediction.numpy()[:1] + np.float32(5) * (prediction.numpy()[1:] - prediction.numpy()[:1])
        np.testing.assert_array_equal(guided.numpy(), expected_guided)
        return {
            "random_weights_only": True,
            "parameter_count": sum(parameter.numel() for parameter in unet.parameters()),
            "sdxl_addition_embed_type": unet.config.addition_embed_type,
            "input_shape": list(hidden.shape),
            "output_shape": list(prediction.shape),
            "guided_shape": list(guided.shape),
            "unet_calls": len(calls),
            "branch_samples": int(prediction.shape[0]),
            "guidance_scale": 5.0,
            "only_noisy_channels_scaled": True,
            "device": "cpu",
        }
    finally:
        torch.set_num_threads(previous_threads)


def run_preflight() -> dict:
    """Validate native image/mask preparation, timestep trimming, and CFG shape."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch

    # CPU is selected before Diffusers imports, processors, arrays, or parameters.
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float32)
    import numpy as np
    from PIL import Image
    from diffusers import UNet2DConditionModel
    from diffusers.image_processor import VaeImageProcessor
    from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_inpaint import (
        StableDiffusionXLInpaintPipeline,
    )

    assert torch.get_default_device().type == "cpu"
    image_processor = VaeImageProcessor(vae_scale_factor=8)
    mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False,
                                      do_binarize=True, do_convert_grayscale=True)
    checks = []

    # Threshold testing remains separate from geometry, including 127/128.
    levels = np.array([0, 1, 63, 127, 128, 192, 254, 255], dtype=np.uint8)
    threshold_image = Image.fromarray(np.tile(levels, (8, 1)))
    threshold_mask = mask_processor.preprocess(threshold_image, height=8, width=8)
    np.testing.assert_array_equal(threshold_mask.numpy(), _reference_mask(np, threshold_image))
    assert threshold_mask.device.type == "cpu"
    assert float(threshold_mask[0, 0, 0, 0]) == 0
    assert float(threshold_mask[0, 0, 0, -1]) == 1
    checks.append("white_unknown_black_known_threshold_128")

    width, height = 512, 1152
    source_box = [0, 320, 512, 832]
    source = _synthetic_rgb(np, Image, 47, 43).resize((512, 512), Image.Resampling.LANCZOS)
    canvas = _synthetic_rgb(np, Image, height, width)
    canvas.paste(source, (0, 320))
    mask_pixels = np.full((height, width), 255, dtype=np.uint8)
    mask_pixels[320:832, :] = 0
    mask_image = Image.fromarray(mask_pixels)
    normalized = image_processor.preprocess(canvas, height=height, width=width).to(dtype=torch.float32)
    mask = mask_processor.preprocess(mask_image, height=height, width=width)
    reference_rgb = _reference_rgb(np, canvas)
    reference_mask = _reference_mask(np, mask_image)
    np.testing.assert_array_equal(normalized.numpy(), reference_rgb)
    np.testing.assert_array_equal(mask.numpy(), reference_mask)
    masked_image = normalized * (mask < 0.5)
    expected_masked_rgb = reference_rgb * (reference_mask < np.float32(0.5))
    np.testing.assert_array_equal(masked_image.numpy(), expected_masked_rgb)
    np.testing.assert_array_equal(masked_image.numpy()[:, :, 320:832], _reference_rgb(np, source))
    np.testing.assert_array_equal(np.asarray(canvas)[320:832], np.asarray(source))
    unknown = np.broadcast_to(reference_mask.astype(bool), expected_masked_rgb.shape)
    assert np.count_nonzero(masked_image.numpy()[unknown]) == 0
    assert not np.any(masked_image.numpy()[unknown] == -1)
    assert np.all(reference_mask[:, :, 319] == 1) and np.all(reference_mask[:, :, 320] == 0)
    assert np.all(reference_mask[:, :, 831] == 0) and np.all(reference_mask[:, :, 832] == 1)
    checks.append("512x1152_exact_center_source_and_unknown_normalized_zero")

    latent_height, latent_width = height // 8, width // 8
    encoded = (np.arange(4 * latent_height * latent_width, dtype=np.float32) / np.float32(32768) - 1)
    encoded = encoded.reshape(1, 4, latent_height, latent_width)
    dummy = MockPipeline(torch, encoded, DummyScheduler(torch))
    expected_latent_mask = _reference_nearest(np, reference_mask, height, width)
    for cfg in (False, True):
        latent_mask, source_latents = StableDiffusionXLInpaintPipeline.prepare_mask_latents(
            dummy, mask, masked_image, batch_size=1, height=height, width=width,
            dtype=torch.float32, device=torch.device("cpu"), generator=None,
            do_classifier_free_guidance=cfg,
        )
        batch = 2 if cfg else 1
        np.testing.assert_array_equal(dummy.encoder_inputs[-1], expected_masked_rgb)
        np.testing.assert_array_equal(latent_mask.numpy(), np.repeat(expected_latent_mask, batch, axis=0))
        np.testing.assert_array_equal(source_latents.numpy(), np.repeat(encoded, batch, axis=0))
        noisy_np = np.arange(4 * latent_height * latent_width, dtype=np.float32).reshape(
            1, 4, latent_height, latent_width) / np.float32(8192)
        noisy = torch.from_numpy(np.repeat(noisy_np, batch, axis=0).copy())
        concatenated = torch.cat((noisy, latent_mask, source_latents), dim=1)
        expected_concat = np.concatenate((np.repeat(noisy_np, batch, axis=0),
                                          np.repeat(expected_latent_mask, batch, axis=0),
                                          np.repeat(encoded, batch, axis=0)), axis=1)
        np.testing.assert_array_equal(concatenated.numpy(), expected_concat)
        assert concatenated.shape == (batch, 9, 144, 64)
        assert all(tensor.device.type == "cpu" for tensor in (latent_mask, source_latents, concatenated))
    assert len(dummy.encoder_inputs) == 2
    checks.append("native_nearest_mask_resize_and_noisy4_mask1_masked_source4")

    # Nonuniform small masks expose nearest-neighbor coordinates that a rectangular
    # source mask aligned to eight pixels would not distinguish from other modes.
    small_height, small_width = 32, 48
    y, x = np.indices((small_height, small_width), dtype=np.int32)
    irregular_pixels = ((((13 * x + 7 * y + (x // 3) * (y // 5)) % 17) < 8) * 255).astype(np.uint8)
    irregular_image = Image.fromarray(irregular_pixels)
    irregular = mask_processor.preprocess(irregular_image, height=small_height, width=small_width)
    supplied_latents = torch.zeros((1, 4, small_height // 8, small_width // 8), device="cpu")
    calls_before = len(dummy.encoder_inputs)
    irregular_latent, bypass_latents = StableDiffusionXLInpaintPipeline.prepare_mask_latents(
        dummy, irregular, supplied_latents, 1, small_height, small_width, torch.float32,
        torch.device("cpu"), None, False,
    )
    np.testing.assert_array_equal(irregular_latent.numpy(), _reference_nearest(
        np, _reference_mask(np, irregular_image), small_height, small_width))
    assert len(dummy.encoder_inputs) == calls_before
    assert torch.equal(bypass_latents, supplied_latents)
    checks.append("nonuniform_nearest_sampling_and_four_channel_latent_bypass")

    timesteps, actual_steps = StableDiffusionXLInpaintPipeline.get_timesteps(
        dummy, num_inference_steps=20, strength=0.99, device=torch.device("cpu"),
    )
    np.testing.assert_array_equal(timesteps.numpy(), np.arange(18, -1, -1, dtype=np.float32))
    assert actual_steps == 19 and len(timesteps) == 19
    assert dummy.scheduler.begin_indices == [1]
    checks.append("native_twenty_requested_strength_099_nineteen_actual_order_one_steps")

    tiny = _tiny_unet_cfg(torch, np, UNet2DConditionModel, latent_mask, source_latents)
    checks.append("tiny_random_sdxl_unet_cfg_two_branches_one_batched_cpu_call")
    assert torch.get_default_device().type == "cpu"
    return {
        "status": "passed",
        "device": "cpu",
        "runtime": {"torch": version("torch"), "diffusers": version("diffusers")},
        "pretrained_weights_loaded": False,
        "full_pipeline_loaded": False,
        "downloads_required": False,
        "checks": checks,
        "geometry": {
            "width": width, "height": height, "source_box": source_box,
            "source_box_convention": "half-open [left, top, right, bottom]",
            "source_pixels_exact": True, "known_normalized_rgb_max_abs_error": 0.0,
            "unknown_normalized_rgb": 0.0, "normalized_black_rgb": -1.0,
            "known_pixels": int(np.count_nonzero(reference_mask == 0)),
            "unknown_pixels": int(np.count_nonzero(reference_mask == 1)),
            "latent_mask_shape": [1, 1, latent_height, latent_width],
            "latent_source_shape": [1, 4, latent_height, latent_width],
            "input_channels": {"noisy": 4, "mask": 1, "masked_source": 4, "total": 9},
            "input_shape_without_cfg": [1, 9, latent_height, latent_width],
            "input_shape_with_cfg": [2, 9, latent_height, latent_width],
        },
        "timestep_trim": {
            "requested_steps": 20, "strength": 0.99, "scheduler_order": 1,
            "begin_index": 1, "actual_steps": actual_steps, "actual_unet_calls": len(timesteps),
            "cfg_branch_samples_for_one_image": 2 * len(timesteps),
            "count_rule": "Count actual loop calls from len(timesteps), not requested steps.",
            "strength_099_initialization": "Native pipeline adds noise to encoded full init_image; not pure-noise init.",
        },
        "tiny_random_unet": tiny,
        "scope": "conditioning and forward-contract checks only; no pretrained inference or output-quality claim",
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
