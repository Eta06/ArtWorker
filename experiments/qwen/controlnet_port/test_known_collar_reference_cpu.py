#!/usr/bin/env python3
"""CPU-only random-weight tests for the isolated known-top-collar extension.

Tests real canvas indexing and small32-block/16-hint models. No checkpoint,
download, Metal execution, or image-quality claim is involved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
mx.set_default_device(mx.cpu)  # Must precede every fixture/model array.
import numpy as np
from PIL import Image
from mflux.models.common.config import ModelConfig
from mflux.models.qwen21.model.qwen21_transformer.qwen21_layout import QwenImage21Layout
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer

from conditioning import control_prefix_padding
from control import ControlBranch
from reference_control import assemble_reference_input, target_known_bridge, validate_reference_layout
from localized_reference_control import (prepare_local_hint_weights as old_field,
                                         localized_reference_controlled_forward as old_forward)
from known_collar_reference_control import (prepare_local_hint_weights as copied_field,
    prepare_known_collar_hint_weights, known_collar_reference_controlled_forward,
    runtime_dependency_hashes)
from test_reference_control_cpu import base_trace, digest, randomize


def array(value):
    """FP32 cast is lossless for BF16 values; NumPy cannot expose MLX BF16 buffers."""
    mx.eval(value)
    if value.dtype == mx.bfloat16:
        value = value.astype(mx.float32)
        mx.eval(value)
    return np.asarray(value).copy()


def tensor_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def expect_rejected(callback):
    try:
        callback()
    except (ValueError, TypeError):
        return True
    raise AssertionError("Invalid input was accepted")


def field_case():
    """Pixel oracle and production512x1152 packed-row addressing."""
    width, height, rect = 512, 1152, [0, 320, 512, 832]
    guide = np.zeros((height, width, 3), dtype=np.uint8)
    guide[256:320, 112] = 255
    guide[320:384, 112] = 255  # Known guide seeds must not affect unknown field.
    guide[864:928, 304] = 255
    before = guide.copy()
    known = np.zeros((height, width), dtype=bool)
    known[320:832] = True
    old_pixels, old_packed, old_info = old_field(np, Image, width, height, rect, guide, 64, "cosine")
    copied_pixels, copied_packed, copied_info = copied_field(np, Image, width, height, rect, guide, 64, "cosine")
    assert np.array_equal(old_pixels, copied_pixels) and np.array_equal(old_packed, copied_packed)
    assert old_info == copied_info
    records = []
    for collar_pixels, weight in ((32, 1.0), (32, 0.375), (64, 0.75), (0, 1.0), (32, 0.0)):
        pixels, packed, info = prepare_known_collar_hint_weights(
            np, Image, width, height, rect, guide, 64, "cosine", collar_pixels, weight)
        expected = old_pixels.copy()
        expected[320:320 + collar_pixels] = np.float32(weight)
        assert np.array_equal(pixels, expected)
        assert np.array_equal(pixels[~known], old_pixels[~known])
        assert np.array_equal(packed.reshape(72, 32), expected[::16, ::16])
        assert np.all(pixels[320 + collar_pixels:832] == 0)
        assert np.all(packed.reshape(72, 32)[(320 + collar_pixels) // 16:52] == 0)
        changed = np.flatnonzero(np.any(packed.reshape(72, 32) != old_packed.reshape(72, 32), axis=1))
        expected_rows = list(range(20, 20 + collar_pixels // 16)) if weight else []
        assert changed.tolist() == expected_rows
        assert info["known_collar_positive_tokens"] == (collar_pixels // 16 * 32 if weight else 0)
        assert info["known_outside_collar_positive_tokens"] == 0
        assert info["old_target_weights_float32_sha256"] == tensor_digest(old_packed)
        assert info["target_weights_float32_sha256"] == tensor_digest(packed)
        assert info["pixel_weights_float32_sha256"] == tensor_digest(pixels)
        if not collar_pixels or not weight:
            assert np.array_equal(packed, old_packed) and np.array_equal(pixels, old_pixels)
        records.append({"collar_pixels": collar_pixels, "weight": weight,
                        "changed_packed_rows": changed.tolist(), "metadata": info})
    assert np.array_equal(guide, before)
    empty = np.zeros_like(guide)
    empty_pixels, empty_packed, _ = prepare_known_collar_hint_weights(
        np, Image, width, height, rect, empty, known_collar_weight=0.375)
    assert np.all(empty_pixels[:320] == 0) and np.all(empty_pixels[352:] == 0)
    assert np.all(empty_pixels[320:352] == np.float32(0.375))
    assert np.count_nonzero(empty_packed) == 64
    return {"status": "passed", "canvas": [width, height], "source_rect": rect,
            "old_copied_local_field_byte_exact": True, "unknown_cosine64_field_byte_exact": True,
            "top32_only_rows20_and21": True, "top64_only_rows20_through23": True,
            "known_interior_and_bottom_zero": True, "zero_collar_or_scalar_exact_old_field": True,
            "empty_unknown_guide_does_not_disable_declared_known_collar": True,
            "input_guide_unmodified": True, "cases": records}


def invalid_field_cases():
    guide = np.zeros((128, 128, 3), dtype=np.uint8)
    common = dict(np=np, Image=Image, width=128, height=128, source_rect=[32, 32, 96, 96], guide_rgb=guide)
    cases = []
    for name, overrides in [
        ("negative_collar", {"known_collar_pixels": -32}),
        ("unaligned_collar16", {"known_collar_pixels": 16}),
        ("undeclared_collar48", {"known_collar_pixels": 48}),
        ("too_large_collar96", {"known_collar_pixels": 96}),
        ("float_collar", {"known_collar_pixels": 32.0}),
        ("bool_collar", {"known_collar_pixels": False}),
        ("string_collar", {"known_collar_pixels": "32"}),
        ("negative_weight", {"known_collar_weight": -0.01}),
        ("weight_above_one", {"known_collar_weight": 1.01}),
        ("nan_weight", {"known_collar_weight": float("nan")}),
        ("infinite_weight", {"known_collar_weight": float("inf")}),
        ("string_weight", {"known_collar_weight": "0.5"}),
        ("negative_radius", {"radius_pixels": -1}),
        ("unbounded_radius", {"radius_pixels": 129}),
        ("invalid_shape", {"shape": "triangle"}),
        ("unaligned_source_top", {"source_rect": [32, 33, 96, 97]}),
        ("collar_exceeds_source_height", {"source_rect": [32, 32, 96, 48]}),
        ("source_outside_canvas", {"source_rect": [0, 32, 129, 96]}),
        ("empty_source", {"source_rect": [32, 32, 32, 96]}),
        ("float_source_coordinates", {"source_rect": [32, 32.0, 96, 96]}),
        ("nonbyte_guide", {"guide_rgb": guide.astype(np.float32)}),
        ("guide_shape", {"guide_rgb": guide[:64]}),
    ]:
        expect_rejected(lambda overrides=overrides: prepare_known_collar_hint_weights(**{**common, **overrides}))
        cases.append(name)
    return {"status": "passed", "rejected_cases": cases}


def full_canvas_geometry():
    """Known-target bridge oracle is independent of the new hint field."""
    slots = mx.array([False] * 7 + [True] * 256 + [False] * 10)
    layout = QwenImage21Layout.create(slots, [(1, 32, 32), (1, 72, 32)], (16, 56, 56))
    layout_info = validate_reference_layout(np, layout, 1024, 2304)
    rng = np.random.default_rng(1771)
    reference_np = rng.standard_normal((1, 1024, 64), dtype=np.float32)
    target_np = rng.standard_normal((1, 2304, 64), dtype=np.float32)
    noise_np = rng.standard_normal((1, 2304, 64), dtype=np.float32)
    context_np = rng.standard_normal((1, 2304, 129), dtype=np.float32)
    context_np[..., 64] = 0
    context_np[:, 20 * 32:52 * 32, 64] = 1
    known = context_np[..., 64] > 0.5
    records = []
    for dtype_name, dtype in (("float32", mx.float32), ("bfloat16", mx.bfloat16)):
        reference, target, noise, context = [mx.array(value).astype(dtype) for value in
                                             (reference_np, target_np, noise_np, context_np)]
        snapshots = [array(value) for value in (reference, target, noise, context)]
        padded = control_prefix_padding(mx, context, 1024)
        assert np.all(array(padded[:, :1024]) == 0)
        assert np.array_equal(array(padded[:, 1024:]), snapshots[3])
        for sigma in (1.0, 0.5, 0.0):
            bridged = target_known_bridge(target, noise, context, sigma)
            expected = ((1 - sigma) * context[:, :, 65:].astype(mx.float32)
                        + sigma * noise.astype(mx.float32)).astype(dtype)
            actual = array(bridged)
            assert np.array_equal(actual[known], array(expected)[known])
            assert np.array_equal(actual[~known], snapshots[1][~known])
            model_input = assemble_reference_input(reference, bridged)
            assert np.array_equal(array(model_input[:, :1024]), snapshots[0])
            # Simulate an arbitrary denoiser update, then restore the known bridge.
            updated = bridged + mx.ones_like(bridged) * 0.125
            restored = target_known_bridge(updated, noise, context, sigma)
            assert np.array_equal(array(restored)[known], array(expected)[known])
            assert np.array_equal(array(restored)[~known], array(updated)[~known])
            assert np.array_equal(array(assemble_reference_input(reference, restored)[:, :1024]), snapshots[0])
            records.append({"dtype": dtype_name, "sigma": sigma, "known_target_exact": True,
                            "unknown_target_untouched": True, "reference_input_exact_before_and_after_update": True})
        for value, before in zip((reference, target, noise, context), snapshots):
            assert np.array_equal(array(value), before)
    expect_rejected(lambda: target_known_bridge(assemble_reference_input(reference, target), noise, padded, 0.0))
    return {**layout_info, "status": "passed", "context_channels": 129,
            "source_hidden_noise_context_unmodified": True, "literal_zero129_reference_padding": True,
            "target_context_bitexact": True, "reference_input_bridge_rejected": True, "cases": records}


def tiny_case(padded_prompt=False, dtype_name="float32"):
    dtype = {"float32": mx.float32, "bfloat16": mx.bfloat16}[dtype_name]
    previous_precision = ModelConfig.precision
    ModelConfig.precision = dtype
    try:
        return _tiny_case(padded_prompt, dtype_name, dtype)
    finally:
        ModelConfig.precision = previous_precision


def _tiny_case(padded_prompt, dtype_name, dtype):
    base = Qwen21Transformer(in_channels=8, out_channels=8, num_layers=32,
        attention_head_dim=8, num_attention_heads=4, context_in_dim=12,
        axes_dims_rope=(2, 2, 4), mlp_ratio=3)
    branch = ControlBranch(dim=32, num_heads=4, head_dim=8, mlp_ratio=3, num_layers=32, in_channels=129)
    randomize(base, 1831 + int(padded_prompt))
    randomize(branch, 1832 + int(padded_prompt))
    base.set_dtype(dtype)
    branch.set_dtype(dtype)
    rng = np.random.default_rng(1833)
    layout = QwenImage21Layout.create(mx.array([False, True, False, False, False]),
                                     [(1, 2, 2), (1, 8, 8)], (2, 2, 4))
    reference = mx.array(rng.standard_normal((1, 4, 8), dtype=np.float32)).astype(dtype)
    target = mx.array(rng.standard_normal((1, 64, 8), dtype=np.float32)).astype(dtype)
    hidden = assemble_reference_input(reference, target)
    prompt = mx.array(rng.standard_normal((1, 5, 12), dtype=np.float32)).astype(dtype)
    context = control_prefix_padding(mx, mx.array(rng.standard_normal((1, 64, 129), dtype=np.float32)).astype(dtype), 4)
    snapshots = [array(value) for value in (hidden, reference, target, context)]
    timestep = mx.array([0.5], dtype=dtype)
    prompt_mask = mx.array([[True, True, True, True, False]]) if padded_prompt else None
    guide = np.zeros((128, 128, 3), dtype=np.uint8)
    guide[16, 16] = guide[40, 40] = 255
    _, old_packed, _ = old_field(np, Image, 128, 128, [32, 32, 96, 96], guide, 64, "cosine")
    _, packed, info = prepare_known_collar_hint_weights(np, Image, 128, 128, [32, 32, 96, 96],
                                                      guide, 64, "cosine", 32, 0.375)
    old_alpha, alpha = mx.array(old_packed), mx.array(packed)
    known = np.zeros((8, 8), dtype=bool); known[2:6, 2:6] = True
    collar = np.zeros((8, 8), dtype=bool); collar[2:4, 2:6] = True
    known, collar = known.reshape(-1), collar.reshape(-1)
    outside = ~known & (old_packed.reshape(-1) == 0)
    assert outside.any() and collar.any() and (known & ~collar).any()
    common = dict(control_scale=0.7, cache=None, encoder_hidden_states_mask=prompt_mask)
    if dtype == mx.bfloat16:
        # Existing trace hooks use NumPy's unsupported BF16 buffer format.
        # Exercise real BF16 forwards without enabling those FP32-only hooks.
        base_output = array(base.forward_reference(hidden, prompt, timestep, layout, cache=None,
                                                   encoder_hidden_states_mask=prompt_mask))
        old_output = array(old_forward(base, branch, hidden, prompt, timestep, layout, context, old_alpha, **common))
        new_output = array(known_collar_reference_controlled_forward(
            base, branch, hidden, prompt, timestep, layout, context, alpha, **common))
        for forward in (old_forward, known_collar_reference_controlled_forward):
            for collar_pixels, weight in ((0, 1.0), (32, 0.0)):
                _, disabled, _ = prepare_known_collar_hint_weights(np, Image, 128, 128, [32, 32, 96, 96],
                                                                  guide, 64, "cosine", collar_pixels, weight)
                disabled_output = forward(base, branch, hidden, prompt, timestep,
                    layout, context, mx.array(disabled), **common)
                assert np.array_equal(array(disabled_output), old_output)
            zero = forward(base, branch, hidden, prompt, timestep, layout, context, mx.zeros_like(alpha), **common)
            zero_scale = forward(base, branch, hidden, prompt, timestep, layout, context, alpha,
                                 control_scale=0.0, encoder_hidden_states_mask=prompt_mask)
            assert np.array_equal(array(zero), base_output) and np.array_equal(array(zero_scale), base_output)
        for value, before in zip((hidden, reference, target, context), snapshots):
            assert np.array_equal(array(value), before)
        assert np.isfinite(new_output).all() and np.max(np.abs(new_output - old_output)) > 0.001
        return {"status": "passed", "dtype": dtype_name, "padded_prompt": padded_prompt,
                "base_blocks": 32, "control_hints": 16, "trace_enabled": False,
                "trace_limitation": "Existing NumPy trace hooks cannot expose MLX BF16 buffers; full hint/prefix traces verified separately in FP32.",
                "lossless_bfloat16_to_float32_comparison": True,
                "collar_zero_or_scalar_zero_exact_old_forward": True,
                "old_and_new_allzero_hint_exact_base": True, "old_and_new_control_scale_zero_exact_base": True,
                "hidden_source_target_and_context129_unmodified": True,
                "new_vs_old_output_max_abs": float(np.max(np.abs(new_output - old_output))),
                "trained_quality_claim": False, "field_metadata": info}
    old_trace, trace = {}, {}
    old_output = old_forward(base, branch, hidden, prompt, timestep, layout, context, old_alpha,
                             trace=old_trace, **common)
    prediction = known_collar_reference_controlled_forward(base, branch, hidden, prompt, timestep,
                    layout, context, alpha, trace=trace, **common)
    base_initial, base_prefixes, base_output = base_trace(base, hidden, prompt, timestep, layout, prompt_mask)
    assert len(trace["target_hints_after"]) == len(trace["target_hints_before"]) == 16
    assert len(trace["base_prefix_after_block"]) == 32
    assert np.all(trace["joint_gate"][:, :layout.prefix_length] == 0)
    raw_hint_hashes, collar_maxima = [], []
    for i in range(16):
        raw, after = trace["target_hints_before"][i], trace["target_hints_after"][i]
        assert np.array_equal(raw, old_trace["target_hints_before"][i])
        assert np.array_equal(trace["prefix_hints_before"][i], old_trace["prefix_hints_before"][i])
        expected = array(mx.array(raw) * alpha.astype(dtype))
        assert np.array_equal(after, expected)
        assert np.all(after[:, outside] == 0) and np.all(after[:, known & ~collar] == 0)
        assert np.all(trace["prefix_hints_after"][i] == 0)
        assert np.array_equal(after[:, ~known], old_trace["target_hints_after"][i][:, ~known])
        collar_max = float(np.max(np.abs(after[:, collar]).astype(np.float32)))
        assert collar_max > 0.005
        collar_maxima.append(collar_max)
        raw_hint_hashes.append(tensor_digest(raw))
    assert np.array_equal(base_initial, trace["base_prefix_initial"])
    for expected, actual, old_actual in zip(base_prefixes, trace["base_prefix_after_block"], old_trace["base_prefix_after_block"]):
        assert np.array_equal(expected, actual) and np.array_equal(actual, old_actual)
    disabled_outputs = []
    for collar_pixels, weight in ((0, 1.0), (32, 0.0)):
        _, disabled, _ = prepare_known_collar_hint_weights(np, Image, 128, 128, [32, 32, 96, 96],
                                                          guide, 64, "cosine", collar_pixels, weight)
        disabled_output = known_collar_reference_controlled_forward(base, branch, hidden, prompt, timestep,
                                      layout, context, mx.array(disabled), **common)
        assert np.array_equal(array(disabled_output), array(old_output))
        disabled_outputs.append({"collar_pixels": collar_pixels, "weight": weight, "old_output_exact": True})
    for forward in (old_forward, known_collar_reference_controlled_forward):
        zero = forward(base, branch, hidden, prompt, timestep, layout, context, mx.zeros_like(alpha), **common)
        zero_scale = forward(base, branch, hidden, prompt, timestep, layout, context, alpha,
                             control_scale=0.0, encoder_hidden_states_mask=prompt_mask)
        assert np.array_equal(array(zero), base_output) and np.array_equal(array(zero_scale), base_output)
    for value, before in zip((hidden, reference, target, context), snapshots):
        assert np.array_equal(array(value), before)
    assert np.isfinite(array(prediction)).all()
    difference = float(np.max(np.abs(array(prediction).astype(np.float32) - array(old_output).astype(np.float32))))
    assert difference > 0.001
    invalid = [mx.array(np.full((1, 64, 1), value, dtype=np.float32)) for value in (-0.1, 1.1, np.nan, np.inf)]
    invalid.append(mx.ones((1, 63, 1)))
    for field in invalid:
        expect_rejected(lambda field=field: known_collar_reference_controlled_forward(
            base, branch, hidden, prompt, timestep, layout, context, field, **common))
    expect_rejected(lambda: known_collar_reference_controlled_forward(
        base, branch, hidden, prompt, timestep, layout, context, alpha, cache=[]))
    expect_rejected(lambda: known_collar_reference_controlled_forward(
        base, branch, hidden, prompt, timestep, layout, context[:, :60], alpha))
    return {"status": "passed", "dtype": dtype_name, "padded_prompt": padded_prompt,
            "base_blocks": 32, "control_hints": 16, "all16_raw_hints_exact_against_old": True,
            "raw_target_hint_sha256": raw_hint_hashes, "known_collar_hint_max_abs_by_layer": collar_maxima,
            "prefix_gate_and_all16_completed_prefix_hints_zero": True,
            "base_prefix_trace_exact_against_old_and_base_all32_blocks": True,
            "collar_after_hints_nonzero_and_scaling_exact": True,
            "known_interior_bottom_and_outside_unknown_after_hints_zero": True,
            "unknown_after_hints_exact_against_old": True,
            "collar_zero_or_scalar_zero_exact_old_forward": disabled_outputs,
            "old_and_new_allzero_hint_exact_base": True, "old_and_new_control_scale_zero_exact_base": True,
            "hidden_source_target_and_context129_unmodified": True,
            "new_vs_old_output_max_abs": difference, "invalid_fields_context_cache_rejected": True,
            "known_target_hidden_frozen_claim": False, "trained_quality_claim": False, "field_metadata": info}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("known_collar_cpu_validation.json"))
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    names = ("control.py", "conditioning.py", "reference_control.py", "localized_reference_control.py",
             "known_collar_reference_control.py", "sparse_conditioning.py", "test_reference_control_cpu.py",
             "test_localized_reference_cpu.py")
    source_hashes_before = {name: digest(directory / name) for name in names}
    runtime_before = runtime_dependency_hashes()
    report = {"status": "passed", "device": "CPU only", "dtypes": ["float32", "bfloat16"],
              "real_models_loaded": False, "checkpoint_weights_used": False, "downloads": False,
              "scope": "Known-top-collar post-chain hint mechanics; full32/16 joint computation. No image quality, checkpoint parity, or iPhone performance claim.",
              "fields": field_case(), "invalid_fields": invalid_field_cases(),
              "full_canvas_geometry": full_canvas_geometry(),
              "tiny_cases": [tiny_case(False, "float32"), tiny_case(True, "float32"), tiny_case(True, "bfloat16")],
              "source_code_sha256": source_hashes_before,
              "runtime_dependency_hashes": runtime_before, "script_sha256": digest(__file__)}
    assert source_hashes_before == {name: digest(directory / name) for name in names}, "Dependency changed during CPU validation"
    assert runtime_before == runtime_dependency_hashes(), "Imported runtime changed during CPU validation"
    report["dependency_files_unchanged_during_validation"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"status": report["status"], "device": report["device"], "tiny_cases": len(report["tiny_cases"]),
                      "dtypes": report["dtypes"], "prefix_exact_all32_blocks": True,
                      "output": str(args.output.resolve())}), flush=True)


if __name__ == "__main__":
    main()
