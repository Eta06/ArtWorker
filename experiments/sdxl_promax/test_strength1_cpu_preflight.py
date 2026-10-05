#!/usr/bin/env python3
"""Weight-free CPU gate for native four-channel Union strength-one inpainting.

The native pipeline must still sample full-source VAE latents for known-region
replacement before drawing pure initialization noise. A mock VAE posterior
consumes a deterministic draw; no pretrained VAE or full pipeline is loaded.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path
from types import MethodType, SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / ".build/models/sdxl-promax-official"
SCHEDULER_CONFIG = MODEL / "base/scheduler/scheduler_config.json"
OLD_GATE = Path(__file__).with_name("test_cpu_preflight.py")


def _state_hash(generator):
    return hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()


class MockPosterior:
    """Known mean/std and one real CPU random draw per posterior sample."""

    def __init__(self, torch, shape, dtype, events, label):
        self.torch, self.shape, self.dtype = torch, shape, dtype
        self.events, self.label = events, label

    def sample(self, generator):
        before = _state_hash(generator)
        draw = self.torch.randn(self.shape, generator=generator, device="cpu", dtype=self.dtype)
        sample = self.torch.full(self.shape, 0.25, device="cpu", dtype=self.dtype)
        sample += self.torch.full(self.shape, 0.125, device="cpu", dtype=self.dtype) * draw
        self.events.append({"phase": self.label, "generator_before": before,
                            "generator_after": _state_hash(generator)})
        return sample


class MockVAE:
    def __init__(self, torch):
        self.torch = torch
        self.config = SimpleNamespace(force_upcast=False, scaling_factor=0.13025, latent_channels=4)
        self.inputs, self.events = [], []

    def encode(self, image):
        assert image.device.type == "cpu" and image.shape == (1, 3, 1152, 512)
        self.inputs.append(image.detach().clone())
        label = "full_source_vae_posterior" if len(self.inputs) == 1 else "masked_source_vae_posterior"
        return SimpleNamespace(latent_dist=MockPosterior(
            self.torch, (1, 4, 144, 64), image.dtype, self.events, label))


class MockPipeline:
    vae_scale_factor = 8

    def __init__(self, torch, scheduler, control_processor, native_pipeline):
        self.scheduler, self.control_image_processor = scheduler, control_processor
        self.vae = MockVAE(torch)
        # Exercise the installed encoder utility as well as native prepare_latents.
        self._encode_vae_image = MethodType(native_pipeline._encode_vae_image, self)


def _reference_posterior(torch, generator, dtype):
    shape = (1, 4, 144, 64)
    draw = torch.randn(shape, generator=generator, device="cpu", dtype=dtype)
    return (torch.full(shape, 0.25, device="cpu", dtype=dtype)
            + torch.full(shape, 0.125, device="cpu", dtype=dtype) * draw) * 0.13025


def _clamp_steps(torch, np, scheduler, timesteps, initial_latents, noise, image_latents,
                 latent_mask, generator, reference_generator, add_noise_calls):
    """Real solver/add_noise calls versus independent NumPy pixel selection."""
    latents = initial_latents.clone()
    source_np, noise_np = image_latents.numpy().copy(), noise.numpy().copy()
    unknown = np.broadcast_to(latent_mask.numpy().astype(bool), source_np.shape)
    draws = []
    for index, timestep in enumerate(timesteps):
        scheduler.scale_model_input(latents, timestep)
        prediction = latents * 0.01 + index * 0.001
        before = _state_hash(generator)
        solver = scheduler.step(prediction, timestep, latents, generator=generator, return_dict=False)[0]
        # Every native ancestral step draws in model_output.dtype, even the last
        # step whose sigma_up is zero. Advance only the independent generator.
        torch.randn(prediction.shape, generator=reference_generator, device="cpu", dtype=prediction.dtype)
        assert torch.equal(generator.get_state(), reference_generator.get_state())
        assert before != _state_hash(generator)
        draws.append({"step": index + 1, "generator_after": _state_hash(generator)})
        if index < len(timesteps) - 1:
            next_timestep = timesteps[index + 1]
            known = scheduler.add_noise(image_latents, noise, next_timestep[None])
            assert scheduler.step_index == index + 1
            assert torch.equal(scheduler.timesteps[scheduler.step_index], next_timestep)
            sigma = np.asarray(scheduler.sigmas[scheduler.step_index].item(), dtype=source_np.dtype)
            expected_known = source_np + noise_np * sigma
            np.testing.assert_array_equal(known.numpy(), expected_known)
        else:
            known, expected_known = image_latents, source_np
        latents = (1 - latent_mask) * known + latent_mask * solver
        expected = np.where(unknown, solver.numpy(), expected_known)
        np.testing.assert_array_equal(latents.numpy(), expected)
        np.testing.assert_array_equal(latents.numpy()[~unknown], expected_known[~unknown])
        np.testing.assert_array_equal(latents.numpy()[unknown], solver.numpy()[unknown])
        np.testing.assert_array_equal(noise.numpy(), noise_np)
        assert bool(torch.isfinite(latents).all())
    assert len(draws) == 30 and len(add_noise_calls) == 29
    np.testing.assert_array_equal(latents.numpy()[~unknown], source_np[~unknown])
    return {"steps_checked": 30, "initial_add_noise_calls": 0,
            "post_step_known_add_noise_calls": 29, "final_step_add_noise_calls": 0,
            "original_noise_unchanged": True, "final_known_latents_clean_and_exact": True,
            "independent_numpy_max_abs_error": 0.0,
            "ancestral_rng_draws": 30, "final_step_rng_draw_retained": True,
            "ancestral_generator_advancement_exact": True}


def _run_cpu(torch):
    import numpy as np
    from PIL import Image
    from diffusers import EulerAncestralDiscreteScheduler
    from diffusers.image_processor import VaeImageProcessor
    from diffusers.pipelines.controlnet.pipeline_controlnet_union_inpaint_sd_xl import (
        StableDiffusionXLControlNetUnionInpaintPipeline as Pipeline,
    )

    # The preserved .9999 gate independently covers native task-7 one-hot values,
    # CFG model plumbing, and the tiny nonzero Union/UNet residual forward.
    spec = importlib.util.spec_from_file_location("promax_preserved_cpu_gate", OLD_GATE)
    inherited = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inherited)
    old_result = inherited.run_preflight()
    assert old_result["status"] == "passed"
    shared = json.loads((ROOT / "experiments/evaluation/inputs/inputs.json").read_text())
    track = next(item for item in shared["tracks"] if item["id"] == "track3")
    source = Image.open(track["source_512"]).convert("RGB")
    canvas = Image.open(track["canvas"]).convert("RGB")
    mask_image = Image.open(track["mask"]).convert("L")
    width, height = shared["canvas_size"]
    assert (width, height) == (512, 1152) and shared["source_rect_xyxy"] == [0, 320, 512, 832]
    assert source.size == (512, 512) and canvas.size == mask_image.size == (width, height)
    source_np, canvas_np = np.asarray(source), np.asarray(canvas)
    np.testing.assert_array_equal(canvas_np[320:832], source_np)
    mask_np = np.ones((1, 1, height, width), dtype=np.float32)
    mask_np[:, :, 320:832] = 0
    np.testing.assert_array_equal(np.asarray(mask_image), mask_np[0, 0] * 255)
    control_image = Image.new("RGB", (width, height), (0, 0, 0))
    control_image.paste(source, (0, 320))
    image_processor = VaeImageProcessor(vae_scale_factor=8)
    mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False,
                                      do_binarize=True, do_convert_grayscale=True)
    control_processor = VaeImageProcessor(vae_scale_factor=8, do_convert_rgb=True, do_normalize=False)
    image = image_processor.preprocess(canvas, height=height, width=width)
    mask = mask_processor.preprocess(mask_image, height=height, width=width)
    reference_rgb = (canvas_np.astype(np.float32) / np.float32(255) * 2 - 1).transpose(2, 0, 1)[None]
    np.testing.assert_array_equal(image.numpy(), reference_rgb)
    np.testing.assert_array_equal(mask.numpy(), mask_np)
    masked_image = image * (mask < 0.5)
    np.testing.assert_array_equal(masked_image.numpy(), reference_rgb * (mask_np < 0.5))
    unknown = np.broadcast_to(mask_np.astype(bool), masked_image.shape)
    assert np.count_nonzero(masked_image.numpy()[unknown]) == 0
    config_bytes = SCHEDULER_CONFIG.read_bytes()
    checks, clamp_results = [], []
    schedule = None
    for dtype in (torch.float32, torch.float16):
        scheduler = EulerAncestralDiscreteScheduler.from_config(json.loads(config_bytes))
        scheduler.set_timesteps(30, device="cpu")
        dummy = MockPipeline(torch, scheduler, control_processor, Pipeline)
        timesteps_before = scheduler.timesteps.clone()
        timesteps, actual_steps = Pipeline.get_timesteps(dummy, 30, 1.0, torch.device("cpu"))
        assert actual_steps == len(timesteps) == 30 and scheduler.begin_index == 0 and scheduler.order == 1
        assert torch.equal(timesteps, timesteps_before)
        schedule = {"class": type(scheduler).__name__, "order": scheduler.order,
                    "requested_steps": 30, "strength": 1.0, "actual_steps": 30, "begin_index": 0,
                    "expected_controlnet_calls": 30, "expected_unet_calls": 30,
                    "expected_cfg_branch_samples_per_model": 60, "guess_mode": False,
                    "timesteps": timesteps.tolist(), "sigmas": scheduler.sigmas.tolist(),
                    "init_noise_sigma": float(scheduler.init_noise_sigma),
                    "config_path": str(SCHEDULER_CONFIG),
                    "config_sha256": hashlib.sha256(config_bytes).hexdigest()}
        generator = torch.Generator(device="cpu").manual_seed(42)
        reference_generator = torch.Generator(device="cpu").manual_seed(42)
        before = _state_hash(generator)
        original_add_noise = scheduler.add_noise
        add_noise_calls = []

        def forbid_initial_add_noise(*args, **kwargs):
            raise AssertionError("Native strength-one initialization must not call scheduler.add_noise")

        scheduler.add_noise = forbid_initial_add_noise
        input_image = image.to(dtype=dtype)
        input_before = input_image.clone()
        latents, noise, image_latents = Pipeline.prepare_latents(
            dummy, batch_size=1, num_channels_latents=4, height=height, width=width,
            dtype=dtype, device=torch.device("cpu"), generator=generator, latents=None,
            image=input_image, timestep=timesteps[:1], is_strength_max=True,
            add_noise=True, return_noise=True, return_image_latents=True,
        )
        expected_source = _reference_posterior(torch, reference_generator, dtype)
        expected_noise = torch.randn((1, 4, 144, 64), generator=reference_generator, device="cpu", dtype=dtype)
        assert len(dummy.vae.inputs) == len(dummy.vae.events) == 1
        assert torch.equal(dummy.vae.inputs[0], input_image) and torch.equal(input_image, input_before)
        assert torch.equal(image_latents, expected_source)
        assert torch.equal(noise, expected_noise)
        assert torch.equal(latents, noise * scheduler.init_noise_sigma)
        assert torch.equal(latents, expected_noise * scheduler.init_noise_sigma)
        assert torch.equal(generator.get_state(), reference_generator.get_state())
        assert all(value.device.type == "cpu" and value.dtype == dtype for value in (latents, noise, image_latents))
        pure_noise_without_posterior = torch.randn(noise.shape, generator=torch.Generator(device="cpu").manual_seed(42),
                                                   device="cpu", dtype=dtype)
        assert not torch.equal(noise, pure_noise_without_posterior)
        control = Pipeline.prepare_control_image(dummy, control_image, width, height, 1, 1,
                    torch.device("cpu"), dtype, None, "default", True, False)
        expected_control = torch.from_numpy((np.asarray(control_image).astype(np.float32) / np.float32(255))
                                            .transpose(2, 0, 1)[None]).to(dtype=dtype)
        assert torch.equal(control, expected_control.repeat(2, 1, 1, 1))
        assert control.shape == (2, 3, height, width) and float(control.min()) == 0 and float(control.max()) <= 1
        state_before_mask = _state_hash(generator)
        mask_cfg, masked_source_cfg = Pipeline.prepare_mask_latents(
            dummy, mask, masked_image, 1, height, width, dtype, torch.device("cpu"), generator, True)
        expected_masked_source = _reference_posterior(torch, reference_generator, dtype)
        assert len(dummy.vae.inputs) == len(dummy.vae.events) == 2
        assert torch.equal(dummy.vae.inputs[1], masked_image.to(dtype=dtype))
        assert torch.equal(masked_source_cfg, expected_masked_source.repeat(2, 1, 1, 1))
        assert torch.equal(generator.get_state(), reference_generator.get_state())
        np.testing.assert_array_equal(mask_cfg.float().numpy(), np.repeat(mask_np[:, :, ::8, ::8], 2, axis=0))
        assert state_before_mask != _state_hash(generator)

        def observed_add_noise(*args, **kwargs):
            state = _state_hash(generator)
            result = original_add_noise(*args, **kwargs)
            assert state == _state_hash(generator)
            add_noise_calls.append(True)
            return result

        scheduler.add_noise = observed_add_noise
        init_state_after = dummy.vae.events[0]["generator_after"]
        checks.append({"dtype": str(dtype), "seed": 42, "shape": list(noise.shape),
                       "full_source_encode_calls_during_prepare_latents": 1,
                       "masked_source_encode_calls_after_prepare_latents": 1,
                       "full_source_vae_latents_exact": True,
                       "native_noise_exact_after_source_posterior_draw": True,
                       "initial_latents_exact_noise_times_init_sigma": True,
                       "initialization_add_noise_calls": 0, "generator_advancement_exact": True,
                       "noise_differs_from_seed_only_first_draw": True,
                       "generator_before": before, "generator_after_source_posterior": init_state_after,
                       "posterior_events": dummy.vae.events})
        clamp_results.append({"dtype": str(dtype), **_clamp_steps(
            torch, np, scheduler, timesteps, latents, noise, image_latents, mask_cfg[:1],
            generator, reference_generator, add_noise_calls)})
    # The final source paste is a distinct output operation, checked with a
    # nonuniform synthetic raw image; the latent clamp does not promise RGB identity.
    raw_np = (np.arange(height * width * 3, dtype=np.uint32).reshape(height, width, 3) % 251).astype(np.uint8)
    composite = Image.fromarray(raw_np).copy()
    composite.paste(source, (0, 320))
    composite_np = np.asarray(composite)
    outside = mask_np[0, 0].astype(bool)
    np.testing.assert_array_equal(composite_np[320:832], source_np)
    np.testing.assert_array_equal(composite_np[outside], raw_np[outside])
    np.testing.assert_array_equal(np.asarray(Image.fromarray(raw_np)), raw_np)
    return {"status": "passed", "device": "cpu", "strength": 1.0,
            "runtime": {"torch": version("torch"), "diffusers": version("diffusers")},
            "pretrained_weights_loaded": False, "full_pipeline_loaded": False, "downloads_required": False,
            "initialization": "native pure noise times init sigma; sampled source latents retained for replacement",
            "rng_order": ["full_source_vae_posterior", "initial_noise", "masked_source_vae_posterior",
                          "30 ancestral solver draws including final zero-sigma-up step"],
            "initial_noise_rng_order_matches_promax_strength_09999": True,
            "initialization_checks": checks, "schedule": schedule, "known_latent_clamp": clamp_results,
            "geometry": {"width": width, "height": height, "source_box": [0, 320, 512, 832],
                         "source_input_pixels_exact": True, "control_rgb_black_exterior": True,
                         "control_mode": [7], "cfg_control_shape": [2, 3, height, width],
                         "normalized_masked_source_unknown_zero": True,
                         "source_composite_pixels_exact": True, "composite_outside_raw_pixels_exact": True},
            "inherited_strength_09999_gate": {"path": str(OLD_GATE),
                "sha256": hashlib.sha256(OLD_GATE.read_bytes()).hexdigest(), "status": old_result["status"],
                "strength": old_result["schedule"]["strength"],
                "tiny_random_plumbing": old_result["tiny_random_plumbing"]},
            "scope": "CPU native initialization/RNG/schedule/clamp and inherited random plumbing; no pretrained quality acceptance"}


def run_preflight() -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch

    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float32)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = _run_cpu(torch)
        assert torch.get_default_device().type == "cpu"
        return result
    finally:
        torch.set_num_threads(previous_threads)


def test_strength1_cpu_preflight():
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
