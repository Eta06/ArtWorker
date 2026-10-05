#!/usr/bin/env python3
"""CPU-only orientation, native Fill static320, noise64, assembly fixtures."""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from local_context_inputs import prepare_patches, assemble_composite, assemble_raw_context_diagnostic
from test_cpu_preflight import _check_conditioning


def run_local_context_cpu():
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    from mflux.models.flux.variants.fill.mask_util import MaskUtil
    from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
    from mflux.models.common.config.config import Config
    from mflux.models.common.config.model_config import ModelConfig
    yy, xx = np.indices((512, 512), dtype=np.uint16)
    pixels = np.stack((yy // 2, yy % 256, (xx * 37 + yy) % 256), axis=-1).astype(np.uint8)
    source = Image.fromarray(pixels, mode="RGB")
    before = pixels.copy()
    patches = prepare_patches(np, Image, source)
    checks = {}
    noises = []
    with tempfile.TemporaryDirectory(prefix="fill-local-context-cpu-") as temporary:
        directory = Path(temporary)
        for name, patch in patches.items():
            rect = patch["metadata"]["known_patch_rect_xyxy"]
            crop_rect = patch["metadata"]["source_crop_xyxy"]
            expected_strip = before[crop_rect[1]:crop_rect[3]]
            assert np.array_equal(np.asarray(patch["source_strip"]), expected_strip)
            assert np.array_equal(np.asarray(patch["canvas"])[rect[1]:rect[3]], expected_strip)
            known_token_rows = (20, 28) if name == "upper" else (0, 8)
            for temporal_axis in (False, True):
                vae, mask, static, hidden_shape = _check_conditioning(mx, MaskUtil, FluxLatentCreator,
                    patch["canvas"], patch["mask"], directory, 448, 512, temporal_axis)
                normalized_strip = (expected_strip.astype(np.float32) / np.float32(255) * np.float32(2) - np.float32(1)).transpose(2, 0, 1)[None]
                assert np.array_equal(vae.inputs[0][:, :, rect[1]:rect[3]], normalized_strip)
                unknown = np.broadcast_to(mask.astype(bool), vae.inputs[0].shape)
                assert np.all(vae.inputs[0][unknown] == 0)
                token_rows = np.arange(896) // 32
                unknown_tokens = (token_rows < known_token_rows[0]) | (token_rows >= known_token_rows[1])
                assert np.array_equal(static[0, :, 64:], np.broadcast_to(unknown_tokens[:, None], (896, 256)))
                assert hidden_shape == [1, 896, 384] and static.shape == (1, 896, 320)
            noise = FluxLatentCreator.create_noise(seed=42, width=512, height=448)
            mx.eval(noise)
            noises.append(np.asarray(noise).copy())
            config = Config(model_config=ModelConfig.dev_fill(), num_inference_steps=50, width=512, height=448,
                guidance=30.0, scheduler="linear")
            assert len(config.scheduler.sigmas) == 51
            checks[name] = {"source_crop": crop_rect, "known_patch_rect": rect,
                "known_token_rows": list(known_token_rows), "target_tokens": 896,
                "known_tokens": 256, "unknown_tokens": 640, "native_static_shape": [1, 896, 320],
                "native_hidden_shape": [1, 896, 384], "masked_known_rgb_exact": True,
                "unknown_normalized_rgb_zero": True, "mask256_polarity_and_geometry_exact": True,
                "steps": 50, "guidance": 30.0, "sigma_count": 51,
                "noise_sha256_float32": hashlib.sha256(noises[-1].astype(np.float32).tobytes()).hexdigest()}
    assert np.array_equal(noises[0], noises[1]), "Independent same-seed same-shape calls must share initial noise values"
    # Different ramps expose reversed strips, wrong offset and raw overwrite.
    upper_pixels = np.stack(np.broadcast_arrays(np.arange(448)[:, None] % 256,
        np.arange(512)[None, :] % 256, np.full((448, 512), 71)), axis=-1).astype(np.uint8)
    lower_pixels = np.stack(np.broadcast_arrays((np.arange(448)[:, None] + 43) % 256,
        np.arange(512)[None, :] % 256, np.full((448, 512), 193)), axis=-1).astype(np.uint8)
    upper, lower = Image.fromarray(upper_pixels), Image.fromarray(lower_pixels)
    assembled = assemble_composite(np, Image, source, upper, lower)
    assert np.array_equal(np.asarray(assembled)[:320], upper_pixels[:320])
    assert np.array_equal(np.asarray(assembled)[320:832], before)
    assert np.array_equal(np.asarray(assembled)[832:], lower_pixels[128:])
    diagnostic = assemble_raw_context_diagnostic(Image, source, upper, lower)
    assert np.array_equal(np.asarray(diagnostic)[:448], upper_pixels)
    assert np.array_equal(np.asarray(diagnostic)[448:704], before[128:384])
    assert np.array_equal(np.asarray(diagnostic)[704:], lower_pixels)
    assert np.array_equal(np.asarray(source), before)
    assert np.array_equal(np.asarray(upper), upper_pixels) and np.array_equal(np.asarray(lower), lower_pixels)
    return {"status": "passed", "device": "CPU only", "model_weights_loaded": False,
            "downloads": False, "source_orientation_exact": True, "source_unmodified": True,
            "assembled_source_exact": True, "generated_regions_raw_exact": True, "raw_crops_unmodified": True,
            "independent_calls_same_seed_noise_exact": True, "patches": checks,
            "total_calls": 2, "planned_total_nfe": 100, "planned_target_token_forwards": 89600,
            "true_growth_or_streaming": False,
            "runtime_source_sha256": {name: hashlib.sha256(Path(inspect.getsourcefile(module)).read_bytes()).hexdigest()
                                      for name, module in (("mask_util", MaskUtil), ("latent_creator", FluxLatentCreator), ("config", Config))}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("local_context_cpu_validation.json"))
    args = parser.parse_args()
    result = run_local_context_cpu()
    result["helper_sha256"] = hashlib.sha256(Path(__file__).with_name("local_context_inputs.py").read_bytes()).hexdigest()
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "device": result["device"], "target_tokens_per_call": 896,
                      "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
