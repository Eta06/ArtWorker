#!/usr/bin/env python3
"""CPU random-weight gate tests; no downloads or real model/checkpoint load."""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
from pathlib import Path

import mlx.core as mx
mx.set_default_device(mx.cpu)
import numpy as np
from PIL import Image
from mflux.models.qwen21.model.qwen21_transformer.qwen21_layout import QwenImage21Layout
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer
from conditioning import control_prefix_padding
from control import ControlBranch
from reference_control import assemble_reference_input, reference_controlled_forward
from localized_reference_control import prepare_local_hint_weights, localized_reference_controlled_forward
from test_reference_control_cpu import array, base_trace, digest, randomize


def runtime_dependency_hashes():
    names = ("qwen21_attention", "qwen21_transformer_block", "qwen21_layout",
             "qwen21_transformer", "qwen21_norm_out", "qwen21_text_projection",
             "qwen21_time_text_embed", "qwen21_rope")
    return {name: {"path": str(Path(importlib.import_module(
            "mflux.models.qwen21.model.qwen21_transformer." + name).__file__).resolve()),
            "sha256": digest(importlib.import_module(
            "mflux.models.qwen21.model.qwen21_transformer." + name).__file__)} for name in names}


def field_case():
    width, height, rect = 160, 160, [96, 96, 128, 128]
    guide = np.zeros((height, width, 3), dtype=np.uint8)
    guide[16, 16] = 255
    guide[110, 110] = 255  # excluded known seed must not enlarge support.
    before = guide.copy()
    yy, xx = np.ogrid[:height, :width]
    oracle_distance = np.sqrt((xx - 16) ** 2 + (yy - 16) ** 2)
    known = np.zeros((height, width), dtype=bool)
    known[96:128, 96:128] = True
    records = []
    for shape in ("hard", "cosine"):
        pixels, packed, info = prepare_local_hint_weights(np, Image, width, height, rect, guide, 64, shape)
        if shape == "hard":
            expected = (oracle_distance <= 64).astype(np.float32)
            assert pixels[16, 80] == 1 and pixels[16, 81] == 0
        else:
            expected = np.where(oracle_distance < 64,
                0.5 * (1 + np.cos(np.pi * oracle_distance / 64)), 0).astype(np.float32)
            assert pixels[16, 16] == 1 and abs(float(pixels[16, 48]) - 0.5) < 1e-7
            assert pixels[16, 80] == 0 and pixels[16, 81] == 0
        expected[known] = 0
        assert np.max(np.abs(pixels - expected)) < 2e-7
        assert np.array_equal(packed.reshape(10, 10), pixels[::16, ::16])
        assert np.all(pixels[known] == 0)
        assert pixels[61, 61] > 0 and pixels[62, 62] == 0
        assert info["unknown_guide_pixels"] == 1
        assert np.isfinite(pixels).all() and pixels.min() >= 0 and pixels.max() <= 1
        records.append({"shape": shape, "oracle_max_abs": float(np.max(np.abs(pixels - expected))),
                        "metadata": info})
    assert np.array_equal(guide, before)
    empty = np.zeros_like(guide)
    assert not prepare_local_hint_weights(np, Image, width, height, rect, empty)[0].any()
    known_only = empty.copy()
    known_only[110, 110] = 255
    assert not prepare_local_hint_weights(np, Image, width, height, rect, known_only)[0].any()
    zero_radius, _, _ = prepare_local_hint_weights(np, Image, width, height, rect, guide, 0)
    assert zero_radius.sum() == 1 and zero_radius[16, 16] == 1
    multiple = empty.copy()
    multiple[16, 16] = multiple[16, 140] = 255
    distances = np.minimum(oracle_distance, np.sqrt((xx - 140) ** 2 + (yy - 16) ** 2))
    expected = np.where(distances < 64, 0.5 * (1 + np.cos(np.pi * distances / 64)), 0).astype(np.float32)
    expected[known] = 0
    actual = prepare_local_hint_weights(np, Image, width, height, rect, multiple)[0]
    assert np.max(np.abs(actual - expected)) < 2e-7
    return {"status": "passed", "cases": records, "known_seed_excluded": True,
            "empty_guide_all_zero": True, "radius_zero_exact": True, "multiple_seed_min_distance_exact": True,
            "guide_unmodified": True, "nearest_samples_exact": True}


def tiny_case(padded_prompt=False, shape="cosine"):
    base = Qwen21Transformer(in_channels=8, out_channels=8, num_layers=32,
        attention_head_dim=8, num_attention_heads=4, context_in_dim=12,
        axes_dims_rope=(2, 2, 4), mlp_ratio=3)
    branch = ControlBranch(dim=32, num_heads=4, head_dim=8, mlp_ratio=3, num_layers=32, in_channels=129)
    randomize(base, 812 + int(padded_prompt))
    randomize(branch, 813 + int(padded_prompt))
    rng = np.random.default_rng(814)
    slots = mx.array([False, True, False, False, False])
    layout = QwenImage21Layout.create(slots, [(1, 2, 2), (1, 8, 8)], (2, 2, 4))
    reference = mx.array(rng.standard_normal((1, 4, 8), dtype=np.float32))
    hidden = assemble_reference_input(reference, mx.array(rng.standard_normal((1, 64, 8), dtype=np.float32)))
    hidden_before, reference_before = array(hidden), array(reference)
    prompt = mx.array(rng.standard_normal((1, 5, 12), dtype=np.float32))
    context = control_prefix_padding(mx, mx.array(rng.standard_normal((1, 64, 129), dtype=np.float32)), 4)
    context_before = array(context)
    timestep = mx.array([0.5], dtype=mx.float32)
    prompt_mask = mx.array([[True, True, True, True, False]]) if padded_prompt else None
    guide = np.zeros((128, 128, 3), dtype=np.uint8)
    guide[16, 16] = guide[40, 40] = 255
    _, packed, info = prepare_local_hint_weights(np, Image, 128, 128, [32, 32, 96, 96], guide, 64, shape)
    alpha = mx.array(packed)
    field = packed.reshape(-1)
    known = np.zeros((8, 8), dtype=bool)
    known[2:6, 2:6] = True
    known = known.reshape(-1)
    outside = ~known & (field == 0)
    supported = field > 0
    assert known.any() and outside.any() and supported.any()
    common = dict(control_scale=0.7, encoder_hidden_states_mask=prompt_mask, cache=None)
    reference_trace, local_trace = {}, {}
    reference_prediction = reference_controlled_forward(base, branch, hidden, prompt, timestep,
        layout, context, prefix_hints_disabled=True, trace=reference_trace, **common)
    prediction = localized_reference_controlled_forward(base, branch, hidden, prompt, timestep,
        layout, context, alpha, trace=local_trace, **common)
    base_initial, base_prefixes, base_output = base_trace(base, hidden, prompt, timestep, layout, prompt_mask)
    assert len(local_trace["target_hints_after"]) == 16
    for before_ref, before_local in zip(reference_trace["prefix_hints_before"], local_trace["prefix_hints_before"]):
        assert np.array_equal(before_ref, before_local)
    for before_ref, before_local, after in zip(reference_trace["target_hints_before"], local_trace["target_hints_before"], local_trace["target_hints_after"]):
        assert np.array_equal(before_ref, before_local)
        assert np.array_equal(after, before_local * packed)
        assert np.all(after[:, known] == 0) and np.all(after[:, outside] == 0)
        assert np.abs(after[:, supported]).max() > 0.01
        if shape == "hard":
            assert np.array_equal(after[:, supported], before_local[:, supported])
    assert all(np.all(hint == 0) for hint in local_trace["prefix_hints_after"])
    assert np.array_equal(base_initial, local_trace["base_prefix_initial"])
    assert all(np.array_equal(a, b) for a, b in zip(base_prefixes, local_trace["base_prefix_after_block"]))
    ones = localized_reference_controlled_forward(base, branch, hidden, prompt, timestep,
        layout, context, mx.ones_like(alpha), **common)
    zeros = localized_reference_controlled_forward(base, branch, hidden, prompt, timestep,
        layout, context, mx.zeros_like(alpha), **common)
    assert np.array_equal(array(ones), array(reference_prediction))
    assert np.array_equal(array(zeros), base_output)
    zero_scale = localized_reference_controlled_forward(base, branch, hidden, prompt, timestep,
        layout, context, alpha, control_scale=0.0, encoder_hidden_states_mask=prompt_mask)
    assert np.array_equal(array(zero_scale), base_output)
    target_effect = float(np.abs(array(prediction) - base_output).max())
    assert target_effect > 0.01 and np.isfinite(array(prediction)).all()
    assert np.array_equal(array(hidden), hidden_before) and np.array_equal(array(reference), reference_before)
    assert np.array_equal(array(context), context_before)
    try:
        localized_reference_controlled_forward(base, branch, hidden, prompt, timestep,
            layout, context, alpha, cache=[])
    except ValueError:
        cache_rejected = True
    else:
        cache_rejected = False
    assert cache_rejected
    return {"status": "passed", "padded_prompt": padded_prompt, "shape": shape,
            "base_blocks": 32, "control_hints": 16, "raw_hints_full_joint_unchanged_exactly": True,
            "prefix_hint_zero_exact": True, "outside_unknown_hint_zero_exact": True,
            "known_source_target_hint_zero_exact": True, "supported_hint_nonzero": True,
            "soft_scaling_exact": True, "base_prefix_trace_exact_all32_blocks": True,
            "alpha_one_exact_validated_reference_gated_forward": True, "alpha_zero_exact_base": True,
            "scale_zero_exact_base": True, "source_hidden_context_unmodified": True,
            "target_output_effect_max_abs": target_effect, "cache_rejected": True,
            "known_target_hidden_trace_preservation_claim": False, "field_metadata": info}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("localized_cpu_validation.json"))
    args = parser.parse_args()
    report = {"status": "passed", "device": "CPU only", "dtype": "float32", "real_models_loaded": False,
              "checkpoint_weights_used": False, "downloads": False,
              "scope": "Post-chain hint localization mechanics; full32/16 computation; no trained quality or quantized checkpoint parity.",
              "fields": field_case(), "tiny_cases": [tiny_case(False, "cosine"), tiny_case(True, "cosine"), tiny_case(False, "hard")],
              "source_code_sha256": {name: digest(Path(__file__).with_name(name)) for name in
                                     ("control.py", "conditioning.py", "reference_control.py", "localized_reference_control.py", "sparse_conditioning.py", "test_reference_control_cpu.py")},
              "runtime_dependency_hashes": runtime_dependency_hashes(), "script_sha256": digest(__file__)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "device": report["device"], "cases": len(report["tiny_cases"]),
                      "prefix_trace_exact_all32_blocks": True, "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
