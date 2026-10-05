#!/usr/bin/env python3
"""Weight-free CPU checks for native ProMax Union task-7 SDXL inpainting.

Uses frozen track-3 pixels, native processors/utility methods, the pinned Euler
ancestral config, and tiny random eight-task Union/four-channel UNet models.
The known-latent replacement equation is checked against independent NumPy
indexing with real scheduler steps/add_noise; no full pipeline is constructed.
No pretrained weights, GPU, download, VAE fidelity, or quality test is involved.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / ".build/models/sdxl-promax-official"
MANIFEST = ROOT / "experiments/sdxl_inpaint/promax_download_manifest.json"


class MockPipeline:
    vae_scale_factor = 8

    def __init__(self, torch, scheduler, control_processor, encoded):
        self.torch = torch
        self.scheduler = scheduler
        self.control_image_processor = control_processor
        self.encoded = encoded
        self.encoder_inputs = []

    def _encode_vae_image(self, image, generator):
        assert image.device.type == "cpu"
        self.encoder_inputs.append(image.detach().numpy().copy())
        return self.torch.from_numpy(self.encoded.copy()).to(device="cpu", dtype=image.dtype)


def _known_latent_clamp(torch, np, scheduler, timesteps, latent_mask):
    """Native inline replacement equation versus independently selected NumPy pixels."""
    shape = (1, 4, 144, 64)
    image_np = np.random.default_rng(9037).standard_normal(shape).astype(np.float32)
    image_latents = torch.from_numpy(image_np.copy())
    original_noise = torch.randn(shape, generator=torch.Generator(device="cpu").manual_seed(42), device="cpu")
    noise_np = original_noise.numpy().copy()
    mask_np = latent_mask.numpy()
    unknown = np.broadcast_to(mask_np.astype(bool), shape)
    latents = scheduler.add_noise(image_latents, original_noise, timesteps[:1])
    init_sigma = np.float32(scheduler.sigmas[scheduler.begin_index].item())
    np.testing.assert_array_equal(latents.numpy(), image_np + noise_np * init_sigma)
    ancestral_generator = torch.Generator(device="cpu").manual_seed(719)
    clamp_noise_calls = 0
    checks = []
    for index, timestep in enumerate(timesteps):
        scheduler.scale_model_input(latents, timestep)
        prediction = latents * 0.01 + float(index) * 0.001
        denoised = scheduler.step(prediction, timestep, latents,
                                  generator=ancestral_generator, return_dict=False)[0]
        denoised_np = denoised.numpy().copy()
        if index < len(timesteps) - 1:
            next_timestep = timesteps[index + 1]
            known = scheduler.add_noise(image_latents, original_noise, next_timestep[None])
            clamp_noise_calls += 1
            # Native scheduler.step has advanced step_index to the NEXT sigma.
            next_index = scheduler.begin_index + index + 1
            assert scheduler.step_index == next_index
            assert torch.equal(scheduler.timesteps[next_index], next_timestep)
            sigma = np.float32(scheduler.sigmas[next_index].item())
            expected_known = image_np + noise_np * sigma
            np.testing.assert_array_equal(known.numpy(), expected_known)
        else:
            known = image_latents
            expected_known = image_np
        # This is the installed four-channel pipeline's post-scheduler equation.
        latents = (1 - latent_mask) * known + latent_mask * denoised
        independent = np.where(unknown, denoised_np, expected_known)
        np.testing.assert_array_equal(latents.numpy(), independent)
        np.testing.assert_array_equal(latents.numpy()[~unknown], expected_known[~unknown])
        np.testing.assert_array_equal(latents.numpy()[unknown], denoised_np[unknown])
        np.testing.assert_array_equal(original_noise.numpy(), noise_np)
        assert bool(torch.isfinite(latents).all())
        checks.append({"step": index + 1, "uses_next_timestep_noise": index < len(timesteps) - 1,
                       "known_latents_exact": True, "unknown_denoised_latents_unchanged": True})
    assert clamp_noise_calls == 28
    np.testing.assert_array_equal(latents.numpy()[~unknown], image_np[~unknown])
    return {"steps_checked": len(checks), "initial_add_noise_calls": 1,
            "post_step_known_add_noise_calls": clamp_noise_calls,
            "original_noise_unchanged": True, "final_known_latents_clean_and_exact": True,
            "final_step_add_noise_calls": 0, "independent_numpy_max_abs_error": 0.0,
            "ancestral_steps_use_separate_stochastic_draws": True}


def _tiny_union_forward(torch, np, ControlNetUnionModel, UNet2DConditionModel, control):
    """One CPU batch-two task-7 Union call supplies residuals to a four-channel UNet."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(8321)
        common = dict(in_channels=4, block_out_channels=(16, 32), layers_per_block=1,
                      norm_num_groups=8, cross_attention_dim=16, attention_head_dim=4,
                      addition_embed_type="text_time", addition_time_embed_dim=4,
                      projection_class_embeddings_input_dim=8 + 6 * 4)
        union = ControlNetUnionModel(
            **common, down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            conditioning_channels=3, conditioning_embedding_out_channels=(4, 8, 12, 16),
            num_control_type=8, num_trans_channel=16, num_trans_head=4,
            num_trans_layer=1, num_proj_channel=16,
        ).to("cpu").eval()
        unet = UNet2DConditionModel(
            **common, sample_size=8, out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        ).to("cpu").eval()
    assert union.task_embedding.shape == (8, 16)
    assert not union.transformer_layes[0].attn.batch_first
    assert all(parameter.device.type == "cpu" for model in (union, unet) for parameter in model.parameters())
    # Native residual heads initialize to zero. Nonzero synthetic projections make
    # this a plumbing check that exercises actual residual tensors in the UNet.
    with torch.no_grad():
        union.controlnet_cond_embedding.conv_out.weight.fill_(0.01)
        for projection in (*union.controlnet_down_blocks, union.controlnet_mid_block):
            projection.weight.fill_(0.01)
    noisy_np = np.arange(4 * 8 * 8, dtype=np.float32).reshape(1, 4, 8, 8) / 127 - 1
    sample = torch.cat((torch.from_numpy(noisy_np), torch.from_numpy(noisy_np)), dim=0)
    control_patch = control[:, :, 288:352, :64]
    assert control_patch.shape == (2, 3, 64, 64)
    prompt = torch.from_numpy(np.arange(2 * 4 * 16, dtype=np.float32).reshape(2, 4, 16) / 128)
    pooled = torch.from_numpy(np.arange(2 * 8, dtype=np.float32).reshape(2, 8) / 16)
    time_ids = torch.tensor([[1152, 512, 0, 0, 1152, 512]] * 2, device="cpu", dtype=torch.float32)
    control_type = torch.zeros((2, 8), device="cpu")
    control_type[:, 7] = 1
    assert torch.equal(control_type.sum(dim=1), torch.ones(2, device="cpu"))
    added = {"text_embeds": pooled, "time_ids": time_ids}
    union_calls, unet_calls = [], []

    def record_union(module, args, kwargs):
        union_calls.append({"sample": list(args[0].shape), "controls": len(kwargs["controlnet_cond"]),
                            "control_type": list(kwargs["control_type"].shape),
                            "control_type_idx": kwargs["control_type_idx"], "guess_mode": kwargs["guess_mode"]})

    def record_unet(module, args, kwargs):
        unet_calls.append({"sample": list(args[0].shape),
                           "residuals": len(kwargs["down_block_additional_residuals"])})

    handles = [union.register_forward_pre_hook(record_union, with_kwargs=True),
               unet.register_forward_pre_hook(record_unet, with_kwargs=True)]
    try:
        with torch.inference_mode():
            down, mid = union(sample, torch.tensor(500.0, device="cpu"), encoder_hidden_states=prompt,
                              controlnet_cond=[control_patch], control_type=control_type,
                              control_type_idx=[7], conditioning_scale=1.0, guess_mode=False,
                              added_cond_kwargs=added, return_dict=False)
            assert any(bool(torch.count_nonzero(residual)) for residual in (*down, mid))
            prediction = unet(sample, torch.tensor(500.0, device="cpu"), encoder_hidden_states=prompt,
                              down_block_additional_residuals=down, mid_block_additional_residual=mid,
                              added_cond_kwargs=added, return_dict=False)[0]
            unconditional, conditional = prediction.chunk(2)
            guided = unconditional + 5.0 * (conditional - unconditional)
        assert len(union_calls) == len(unet_calls) == 1
        assert union_calls[0] == {"sample": [2, 4, 8, 8], "controls": 1, "control_type": [2, 8],
                                  "control_type_idx": [7], "guess_mode": False}
        assert unet_calls[0]["sample"] == [2, 4, 8, 8]
        assert prediction.shape == (2, 4, 8, 8) and guided.shape == (1, 4, 8, 8)
        assert all(bool(torch.isfinite(value).all()) for value in (*down, mid, prediction, guided))
        expected_guided = prediction.numpy()[:1] + np.float32(5) * (prediction.numpy()[1:] - prediction.numpy()[:1])
        np.testing.assert_array_equal(guided.numpy(), expected_guided)
        return {"random_weights_only": True, "controlnet_calls": 1, "unet_calls": 1,
                "cfg_branch_samples_per_model": 2, "latent_input_shape": [2, 4, 8, 8],
                "control_input_shape": [2, 3, 64, 64], "control_type_shape": [2, 8],
                "control_type_idx": [7], "guess_mode": False,
                "residual_shapes": [list(value.shape) for value in down], "mid_shape": list(mid.shape),
                "nonzero_residuals_exercised": True, "synthetic_nonzero_residual_heads": True,
                "legacy_attention_batch_first_false_preserved": True,
                "union_parameters": sum(value.numel() for value in union.parameters()),
                "unet_parameters": sum(value.numel() for value in unet.parameters())}
    finally:
        for handle in handles:
            handle.remove()


def _run_cpu(torch):
    import numpy as np
    from PIL import Image
    from diffusers import ControlNetUnionModel, EulerAncestralDiscreteScheduler, UNet2DConditionModel
    from diffusers.image_processor import VaeImageProcessor
    from diffusers.pipelines.controlnet.pipeline_controlnet_union_inpaint_sd_xl import (
        StableDiffusionXLControlNetUnionInpaintPipeline as Pipeline,
    )

    manifest = json.loads(MANIFEST.read_text())
    assert manifest["pipeline"]["num_control_type"] == 8
    assert manifest["pipeline"]["control_mode"] == [7]
    assert manifest["pipeline"]["unet_in_channels"] == 4
    config_path = MODEL / "base/scheduler/scheduler_config.json"
    scheduler = EulerAncestralDiscreteScheduler.from_config(json.loads(config_path.read_text()))
    scheduler.set_timesteps(30, device="cpu")
    assert scheduler.order == 1
    shared = json.loads((ROOT / "experiments/evaluation/inputs/inputs.json").read_text())
    track = next(item for item in shared["tracks"] if item["id"] == "track3")
    source = Image.open(track["source_512"]).convert("RGB")
    canvas = Image.open(track["canvas"]).convert("RGB")
    mask_image = Image.open(track["mask"]).convert("L")
    width, height = shared["canvas_size"]
    assert (width, height) == (512, 1152) and shared["source_rect_xyxy"] == [0, 320, 512, 832]
    assert source.size == (512, 512) and canvas.size == mask_image.size == (512, 1152)
    source_np, canvas_np = np.asarray(source), np.asarray(canvas)
    expected_mask = np.ones((1, 1, height, width), dtype=np.float32)
    expected_mask[:, :, 320:832] = 0
    np.testing.assert_array_equal(np.asarray(mask_image), expected_mask[0, 0] * 255)
    np.testing.assert_array_equal(canvas_np[320:832], source_np)
    control_image = Image.new("RGB", (width, height), (0, 0, 0))
    control_image.paste(source, (0, 320))
    control_processor = VaeImageProcessor(vae_scale_factor=8, do_convert_rgb=True, do_normalize=False)
    image_processor = VaeImageProcessor(vae_scale_factor=8)
    mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False,
                                      do_binarize=True, do_convert_grayscale=True)
    init_image = image_processor.preprocess(canvas, height=height, width=width)
    mask = mask_processor.preprocess(mask_image, height=height, width=width)
    reference_rgb = (canvas_np.astype(np.float32) / np.float32(255) * 2 - 1).transpose(2, 0, 1)[None]
    np.testing.assert_array_equal(init_image.numpy(), reference_rgb)
    np.testing.assert_array_equal(mask.numpy(), expected_mask)
    masked_image = init_image * (mask < 0.5)
    np.testing.assert_array_equal(masked_image.numpy(), reference_rgb * (expected_mask < 0.5))
    unknown_rgb = np.broadcast_to(expected_mask.astype(bool), masked_image.shape)
    assert np.count_nonzero(masked_image.numpy()[unknown_rgb]) == 0
    expected_encoded = np.arange(4 * 144 * 64, dtype=np.float32).reshape(1, 4, 144, 64) / 32768 - 1
    dummy = MockPipeline(torch, scheduler, control_processor, expected_encoded)
    control = Pipeline.prepare_control_image(
        dummy, control_image, width, height, 1, 1, torch.device("cpu"), torch.float32,
        None, "default", do_classifier_free_guidance=True, guess_mode=False,
    )
    expected_control = (np.asarray(control_image).astype(np.float32) / np.float32(255)).transpose(2, 0, 1)[None]
    np.testing.assert_array_equal(control.numpy(), np.repeat(expected_control, 2, axis=0))
    assert control.shape == (2, 3, 1152, 512) and control.device.type == "cpu"
    assert np.count_nonzero(control.numpy()[np.repeat(unknown_rgb, 2, axis=0)]) == 0
    assert float(control.min()) == 0 and float(control.max()) <= 1
    np.testing.assert_array_equal(control.numpy()[0, :, 320:832].transpose(1, 2, 0),
                                  source_np.astype(np.float32) / np.float32(255))
    mask_cfg, encoded_cfg = Pipeline.prepare_mask_latents(
        dummy, mask, masked_image, 1, height, width, torch.float32, torch.device("cpu"), None, True,
    )
    np.testing.assert_array_equal(mask_cfg.numpy(), np.repeat(expected_mask[:, :, ::8, ::8], 2, axis=0))
    np.testing.assert_array_equal(encoded_cfg.numpy(), np.repeat(expected_encoded, 2, axis=0))
    np.testing.assert_array_equal(dummy.encoder_inputs[0], reference_rgb * (expected_mask < 0.5))
    assert len(dummy.encoder_inputs) == 1
    timesteps, actual_steps = Pipeline.get_timesteps(dummy, 30, 0.9999, torch.device("cpu"))
    assert actual_steps == len(timesteps) == 29 and scheduler.begin_index == 1
    assert torch.equal(timesteps, scheduler.timesteps[1:])
    initial_schedule = {"timesteps": timesteps.tolist(), "sigmas": scheduler.sigmas.tolist()}
    clamp = _known_latent_clamp(torch, np, scheduler, timesteps, mask_cfg.chunk(2)[0])
    tiny = _tiny_union_forward(torch, np, ControlNetUnionModel, UNet2DConditionModel, control)
    return {
        "status": "passed", "device": "cpu",
        "runtime": {"torch": version("torch"), "diffusers": version("diffusers")},
        "pretrained_weights_loaded": False, "full_pipeline_loaded": False, "downloads_required": False,
        "checks": ["frozen_source_and_mask_pixels", "control_rgb_zero_black_exterior_range_zero_one",
                   "native_cfg_control_batch_two_task_seven", "native_nearest_mask_and_masked_rgb_zero",
                   "native_ancestral_thirty_requested_twenty_nine_retained",
                   "next_timestep_original_noise_known_latent_clamp_and_final_clean_source",
                   "tiny_eight_task_union_four_channel_unet_residual_forward"],
        "geometry": {"width": width, "height": height, "source_box": [0, 320, 512, 832],
                     "source_pixels_exact": True, "control_unknown_rgb": 0.0,
                     "masked_source_unknown_normalized_rgb": 0.0,
                     "latent_mask_shape_without_cfg": [1, 1, 144, 64],
                     "cfg_control_image_shape": list(control.shape), "unet_in_channels": 4},
        "schedule": {"class": type(scheduler).__name__, "order": scheduler.order,
                     "requested_steps": 30, "strength": 0.9999, "actual_steps": actual_steps,
                     "begin_index": scheduler.begin_index, "expected_controlnet_calls": 29,
                     "expected_unet_calls": 29, "expected_cfg_branch_samples_per_model": 58,
                     "guess_mode": False, **initial_schedule},
        "known_latent_clamp": clamp, "tiny_random_plumbing": tiny,
        "resolution_scope": {"benchmark": [512, 1152], "author_recipe_approximate_area": 1024 ** 2,
                             "exact_author_demo_resolution_reproduction": False},
        "scope": "CPU conditioning/clamp/plumbing parity; no pretrained MPS inference or image-quality acceptance",
    }


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
