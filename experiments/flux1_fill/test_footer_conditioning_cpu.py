#!/usr/bin/env python3
"""CPU-only meaningful invariants for the manual footer conditioning probe."""
from __future__ import annotations

import ast
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from footer_conditioning_inputs import remove_footer_for_condition, assemble_lower_only, PATCH_ROI
from run_fill_trial import ROOT, file_sha256

BASELINE = ROOT / "experiments/flux1_fill/runs/2026-10-05/track3-local-context50-seed42"


def _ast_operations(path, name=None):
    tree = ast.parse(path.read_text())
    if name:
        tree = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    found = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        target = node.targets[0].id
        if target not in ("latents", "hidden", "prediction") or not isinstance(node.value, ast.Call):
            continue
        function = ast.unparse(node.value.func)
        if function in ("config.scheduler.scale_model_input", "mx.concatenate", "model.transformer", "config.scheduler.step"):
            found[function] = ast.dump(node, include_attributes=False)
    if len(found) != 4:
        raise AssertionError("Expected all four native denoising operations")
    return found


def run_cpu_audit():
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    from mflux.models.flux.variants.fill.mask_util import MaskUtil
    from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
    from mflux.models.common.config.config import Config
    from mflux.models.common.config.model_config import ModelConfig
    from test_cpu_preflight import _check_conditioning
    from run_footer_conditioning_trial import native_step

    baseline_suite = json.loads((BASELINE / "metrics.json").read_text())
    old_runner = Path(__file__).with_name("run_local_context_fill_trial.py")
    runner = Path(__file__).with_name("run_footer_conditioning_trial.py")
    if file_sha256(old_runner) != baseline_suite["runner_sha256"]:
        raise AssertionError("Frozen old local runner changed")
    old_ops, new_ops = _ast_operations(old_runner), _ast_operations(runner, "native_step")
    if old_ops != new_ops:
        raise AssertionError("Native denoising operations no longer AST-match")
    shared = json.loads((ROOT / "experiments/evaluation/inputs/inputs.json").read_text())
    track = next(item for item in shared["tracks"] if item["id"] == "track3")
    source = Image.open(track["source_512"]).convert("RGB")
    lower = BASELINE / "lower"
    before = Image.open(lower / "input_canvas.png").convert("RGB")
    generation_mask = Image.open(lower / "input_mask.png").convert("L")
    original = np.asarray(source).copy()
    baseline_pixels = np.asarray(before).copy()
    after, roi_mask, info = remove_footer_for_condition(np, Image, source, before)
    source_copy = np.asarray(source).copy()
    px0, py0, px1, py1 = PATCH_ROI
    mask_np = np.asarray(generation_mask)
    if not np.all(mask_np[:128] == 0) or not np.all(mask_np[128:] == 255):
        raise AssertionError("Native generation mask polarity changed")
    if np.count_nonzero(np.asarray(roi_mask)) != 37 * 39:
        raise AssertionError("Manual ROI geometry changed")
    if not info["actual_changed_pixels"] or info["actual_changed_pixels"] > 37 * 39:
        raise AssertionError("Actual edit missing or expanded beyond ROI")
    if not np.array_equal(np.asarray(before), baseline_pixels) or not np.array_equal(source_copy, original):
        raise AssertionError("Helper mutated source inputs")
    if not np.array_equal(np.asarray(after)[125:128], baseline_pixels[125:128]):
        raise AssertionError("Last three known rows at generation boundary changed")

    static_saved = np.load(lower / "static_conditioning.npz")["static"]
    captured = []
    with tempfile.TemporaryDirectory(prefix="fill-footer-cpu-") as temporary:
        for image in (before, after):
            vae, _, static, hidden_shape = _check_conditioning(mx, MaskUtil, FluxLatentCreator,
                image, generation_mask, Path(temporary), 448, 512, False)
            captured.append((vae.inputs[0].copy(), static.copy()))
            if hidden_shape != [1, 896, 384]:
                raise AssertionError("Native conditioning shape changed")
    before_normalized, before_static = captured[0]
    after_normalized, after_static = captured[1]
    normalized_delta = np.any(before_normalized != after_normalized, axis=1)[0]
    if np.any(normalized_delta & (np.asarray(roi_mask) == 0)):
        raise AssertionError("Actual normalized VAE input changed beyond ROI")
    if not np.array_equal(before_static[..., 64:], after_static[..., 64:]):
        raise AssertionError("Native mask256 changed between condition arms")
    if not np.array_equal(after_static[..., 64:], static_saved[..., 64:]):
        raise AssertionError("Native packed mask256 not exact saved baseline")
    if not np.all(after_normalized[:, :, 128:] == 0):
        raise AssertionError("Unknown masked normalized pixels nonzero")

    noise = np.load(lower / "initial_noise.npz")["latents"].copy()
    baseline_metrics = json.loads((lower / "metrics.json").read_text())
    noise_hash = hashlib.sha256(noise.tobytes()).hexdigest()
    if noise.dtype != np.float32 or noise.shape != (1, 896, 64) or noise_hash != baseline_metrics["initial_noise_sha256_float32"]:
        raise AssertionError("Saved baseline noise invalid")
    restored = mx.array(noise); mx.eval(restored)
    if not np.array_equal(np.asarray(restored), noise):
        raise AssertionError("CPU saved noise restoration not exact")
    saved_sigmas = np.asarray(baseline_metrics["scheduler_sigmas"], dtype=np.float32)
    config = Config(model_config=ModelConfig.dev_fill(), num_inference_steps=50, width=512, height=448, guidance=30.0, scheduler="linear")
    current_sigmas = np.asarray(config.scheduler.sigmas)
    schedule_delta = float(np.abs(saved_sigmas - current_sigmas).max())
    if schedule_delta > 1e-7:
        raise AssertionError("CPU native schedule unexpectedly far from GPU baseline")
    # A nonzero deterministic mock prediction verifies native step inputs and
    # Euler state update, without loading or evaluating any actual model weights.
    calls = []
    class Transformer:
        def __call__(self, *, t, config, hidden_states, prompt_embeds, pooled_prompt_embeds):
            calls.append((t, tuple(hidden_states.shape), config.guidance))
            return hidden_states[..., :64] * mx.array(0.125, dtype=mx.float32)
    class TinyModel:
        transformer = Transformer()
    fake_static = mx.array(after_static)
    pred, updated, hidden = native_step(mx, TinyModel(), config, restored, fake_static, mx.zeros((1, 2, 4)), mx.zeros((1, 4)), 7)
    mx.eval(pred, updated, hidden)
    expected = noise + (noise * np.float32(0.125)) * np.float32(current_sigmas[8] - current_sigmas[7])
    if not np.array_equal(np.asarray(hidden[..., :64]), noise) or not np.array_equal(np.asarray(hidden[..., 64:]), after_static):
        raise AssertionError("Native concatenation not exact")
    if not np.array_equal(np.asarray(updated), expected) or calls != [(7, (1, 896, 384), 30.0)]:
        raise AssertionError("Native Euler mock fixture failed")

    # Different row ramps reveal mistaken top/bottom/offset source assembly.
    yy, xx = np.indices((448, 512))
    upper_arr = np.stack((yy % 256, xx % 256, (yy + xx) % 256), axis=-1).astype(np.uint8)
    lower_arr = np.stack(((yy + 57) % 256, (xx + 29) % 256, (yy * 3 + xx) % 256), axis=-1).astype(np.uint8)
    upper, lower_raw = Image.fromarray(upper_arr), Image.fromarray(lower_arr)
    assembled = assemble_lower_only(np, Image, source, upper, lower_raw)
    assembled_np = np.asarray(assembled)
    if not np.array_equal(assembled_np[:320], upper_arr[:320]) or not np.array_equal(assembled_np[320:832], original) or not np.array_equal(assembled_np[832:], lower_arr[128:]):
        raise AssertionError("Original source/reused upper/actual lower assembly failed")
    if not np.array_equal(np.asarray(lower_raw), lower_arr):
        raise AssertionError("Raw model-like fixture was overwritten")
    # Reject a source/canvas pair with wrong context rather than silently edit it.
    wrong = np.asarray(before).copy(); wrong[0, 0, 0] ^= 1
    try:
        remove_footer_for_condition(np, Image, source, Image.fromarray(wrong))
    except AssertionError:
        mismatch_rejected = True
    else:
        raise AssertionError("Mismatched source context was silently accepted")
    return {"status": "passed", "device": "CPU only", "model_weights_loaded": False,
        "native_denoise_assignments_ast_exact": True, "manual_roi_pixels": 1443,
        "actual_changed_condition_pixels": info["actual_changed_pixels"],
        "original_source_and_baseline_canvas_unmodified": True,
        "input_canvas_pixels_outside_roi_exact": True, "last_three_known_rows_exact": True,
        "native_normalized_vae_pixels_changed_only_roi": True,
        "native_mask256_saved_baseline_exact": True,
        "unknown_normalized_rgb_zero": True, "saved_noise_restoration_exact": True,
        "saved_noise_sha256_float32": noise_hash,
        "saved_gpu_schedule_sha256_float32": hashlib.sha256(saved_sigmas.tobytes()).hexdigest(),
        "cpu_native_schedule_max_delta": schedule_delta,
        "gpu_schedule_and_prompt_exact_checks_pending": True,
        "native_mock_euler_step_exact": True, "exact_source_and_generated_assembly_fixture": True,
        "mismatched_context_rejected": mismatch_rejected,
        "old_runner_sha256": file_sha256(old_runner), "runner_sha256": file_sha256(runner),
        "helper_sha256": file_sha256(Path(__file__).with_name("footer_conditioning_inputs.py")),
        "cpu_test_sha256": file_sha256(Path(__file__))}


if __name__ == "__main__":
    result = run_cpu_audit()
    destination = Path(__file__).with_name("footer_probe") / "cpu_validation.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
