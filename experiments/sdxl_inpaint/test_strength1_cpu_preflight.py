#!/usr/bin/env python3
"""CPU-only proof of native SDXL strength-one pure-noise initialization.

Run with .build/dreamlite-venv/bin/python. No pipeline, VAE, UNet, pretrained
weights, GPU, or download is used. The frozen local Euler configuration is
loaded as JSON and its scheduler is constructed on CPU. Native utility methods
run on a mock self whose forbidden image-encoding paths raise immediately.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCHEDULER_CONFIG = ROOT / ".build/models/sdxl-inpaint-fp16/scheduler/scheduler_config.json"


class MockPipeline:
    vae_scale_factor = 8

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.encode_calls = 0

    def _encode_vae_image(self, *args, **kwargs):
        self.encode_calls += 1
        raise AssertionError("Strength one must not encode the full RGB init image")


def _state_hash(generator) -> str:
    return hashlib.sha256(generator.get_state().cpu().numpy().tobytes()).hexdigest()


def run_preflight() -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch

    # Select CPU before importing Diffusers or constructing arrays/schedulers.
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float32)
    from diffusers import EulerDiscreteScheduler
    from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_inpaint import (
        StableDiffusionXLInpaintPipeline,
    )

    assert torch.get_default_device().type == "cpu"
    config_bytes = SCHEDULER_CONFIG.read_bytes()
    scheduler = EulerDiscreteScheduler.from_config(json.loads(config_bytes))
    scheduler.set_timesteps(20, device=torch.device("cpu"))
    assert scheduler.order == 1
    assert scheduler.timesteps.device.type == "cpu"
    assert scheduler.sigmas.device.type == "cpu"
    add_noise_calls = []

    def forbidden_add_noise(*args, **kwargs):
        add_noise_calls.append(True)
        raise AssertionError("Strength one must not add noise to encoded init-image latents")

    scheduler.add_noise = forbidden_add_noise
    dummy = MockPipeline(scheduler)
    timesteps_before = scheduler.timesteps.clone()
    timesteps, actual_steps = StableDiffusionXLInpaintPipeline.get_timesteps(
        dummy, num_inference_steps=20, strength=1.0, device=torch.device("cpu"),
    )
    assert actual_steps == 20 and len(timesteps) == 20
    assert scheduler.begin_index == 0
    assert torch.equal(timesteps, timesteps_before)
    sigma = scheduler.init_noise_sigma
    assert float(sigma) > 0

    height, width = 1152, 512
    shape = (1, 4, height // 8, width // 8)
    checks = []
    # FP16 matches the real runner's noise draw. FP32 checks the same native
    # contract separately; sampling FP32 then casting does not match an FP16 draw.
    for dtype in (torch.float32, torch.float16):
        generator = torch.Generator(device="cpu").manual_seed(42)
        reference_generator = torch.Generator(device="cpu").manual_seed(42)
        before = generator.get_state().clone()
        assert torch.equal(before, reference_generator.get_state())
        # Native prepare_latents accesses image.shape even when no image encoding
        # is required. Supply three RGB channels so the latent-input branch is off.
        image = torch.full((1, 3, height, width), 0.25, device="cpu", dtype=dtype)
        image_before = image.clone()
        result = StableDiffusionXLInpaintPipeline.prepare_latents(
            dummy, batch_size=1, num_channels_latents=4, height=height, width=width,
            dtype=dtype, device=torch.device("cpu"), generator=generator,
            latents=None, image=image, timestep=timesteps[:1],
            is_strength_max=True, add_noise=True, return_noise=True,
            return_image_latents=False,
        )
        assert isinstance(result, tuple) and len(result) == 2
        latents, noise = result
        reference_noise = torch.randn(shape, generator=reference_generator,
                                      device="cpu", dtype=dtype)
        reference_latents = reference_noise * sigma
        assert noise.shape == shape and latents.shape == shape
        assert noise.device.type == "cpu" and latents.device.type == "cpu"
        assert noise.dtype == dtype and latents.dtype == dtype
        assert torch.equal(noise, reference_noise), "Native noise differs from an independent CPU torch.randn draw"
        assert torch.equal(latents, noise * sigma), "Native latents must be exactly noise times init_noise_sigma"
        assert torch.equal(latents, reference_latents)
        assert torch.equal(generator.get_state(), reference_generator.get_state())
        assert not torch.equal(before, generator.get_state()), "Noise sampling must advance the CPU generator"
        assert torch.equal(image, image_before), "RGB input must remain unchanged"
        assert bool(torch.isfinite(noise).all()) and bool(torch.isfinite(latents).all())
        assert dummy.encode_calls == 0 and not add_noise_calls
        checks.append({
            "dtype": str(dtype), "shape": list(shape), "seed": 42,
            "native_noise_exact_independent_torch_randn": True,
            "initial_latents_exact_noise_times_init_sigma": True,
            "generator_advancement_exact": True,
            "generator_state_before_sha256": hashlib.sha256(before.numpy().tobytes()).hexdigest(),
            "generator_state_after_sha256": _state_hash(generator),
            "noise_sha256_float32": hashlib.sha256(noise.float().numpy().tobytes()).hexdigest(),
            "initial_latents_sha256_float32": hashlib.sha256(latents.float().numpy().tobytes()).hexdigest(),
            "full_init_image_vae_encode_calls": dummy.encode_calls,
            "scheduler_add_noise_calls": len(add_noise_calls),
        })

    assert torch.get_default_device().type == "cpu"
    return {
        "status": "passed", "device": "cpu",
        "runtime": {"torch": version("torch"), "diffusers": version("diffusers")},
        "pretrained_weights_loaded": False, "full_pipeline_loaded": False,
        "downloads_required": False, "strength": 1.0,
        "initialization": "native pure noise multiplied by init_noise_sigma",
        "scheduler": {
            "class": type(scheduler).__name__, "order": scheduler.order,
            "config_path": str(SCHEDULER_CONFIG),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "init_noise_sigma": float(sigma),
            "requested_steps": 20, "actual_steps": actual_steps,
            "timesteps": timesteps.tolist(), "sigmas": scheduler.sigmas.tolist(),
            "begin_index": scheduler.begin_index,
            "expected_unet_calls": len(timesteps),
            "expected_cfg_branch_samples_for_one_image": 2 * len(timesteps),
        },
        "checks": checks,
        "full_init_image_vae_encode_calls": dummy.encode_calls,
        "scheduler_add_noise_calls": len(add_noise_calls),
        "scope": "pure-noise initialization and schedule only; masked-source encoding remains part of real inpainting",
    }


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
