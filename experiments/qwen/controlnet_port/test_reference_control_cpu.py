#!/usr/bin/env python3
"""CPU-only random-weight tests for the experimental reference extension.

No upstream download, checkpoint, CUDA/MPS, Metal computation, or real model
construction is used. This validates extension mechanics, not trained quality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
mx.set_default_device(mx.cpu)
import numpy as np
from mlx import nn
from mlx.utils import tree_flatten
from mflux.models.qwen21.model.qwen21_transformer.qwen21_layout import QwenImage21Layout
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer_block import Qwen21TransformerBlock

from conditioning import control_prefix_padding, cpu_preflight
from control import ControlBranch, controlled_forward
from reference_control import (assemble_reference_input, reference_controlled_forward,
                               target_known_bridge, validate_reference_layout)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array(value):
    mx.eval(value)
    return np.asarray(value).copy()


def randomize(module, seed):
    rng = np.random.default_rng(seed)
    weights = []
    for name, param in tree_flatten(module.parameters()):
        values = rng.standard_normal(param.shape, dtype=np.float32)
        if ".norm_q.weight" in name or ".norm_k.weight" in name:
            values = 1 + values * 0.1
        elif name == "txt_in.text_norm.weight":
            values *= 0.1
        else:
            values *= 0.10 if param.ndim == 1 else 0.12
        weights.append((name, mx.array(values)))
    module.load_weights(weights, strict=True)
    mx.eval(module.parameters())


def base_trace(base, hidden, prompt, timestep, layout, prompt_mask):
    """Independent uncached base-only joint trace; assert official output parity."""
    images, text = base.img_in(hidden), base.txt_in(prompt)
    joint = mx.concatenate([text, mx.zeros((1, layout.target_tokens // 4, text.shape[-1]), dtype=text.dtype)], axis=1)
    joint = joint[:, layout.repeat_indices]
    joint[:, layout.image_indices] = images
    initial = array(joint[:, :layout.prefix_length])
    key_valid = None
    if prompt_mask is not None:
        mask = mx.concatenate([prompt_mask.astype(mx.bool_), mx.ones((1, layout.target_tokens // 4), dtype=mx.bool_)], axis=1)
        key_valid = mask[:, layout.repeat_indices]
        key_valid[:, layout.image_indices] = True
    time_rows = mx.concatenate([timestep.reshape(-1), mx.zeros((1,), dtype=joint.dtype)])
    time = base.time_text_embed(time_rows, joint.dtype)
    modulation = base.modulation(time)
    traces = []
    for block in base.transformer_blocks:
        joint, stored = block.forward_reference(joint, modulation, layout.target_mask,
            layout, layout.rope, cached=None, extract=False, key_valid=key_valid)
        assert stored is None
        traces.append(array(joint[:, :layout.prefix_length]))
    scale = Qwen21TransformerBlock.select_rows(base.norm_out.linear(nn.silu(time)), layout.target_mask)
    prediction = base.proj_out(base.norm_out(joint, scale))[:, -layout.target_tokens:]
    expected = base.forward_reference(hidden, prompt, timestep, layout, cache=None,
                                      encoder_hidden_states_mask=prompt_mask)
    assert np.array_equal(array(prediction), array(expected))
    return initial, traces, array(expected)


def geometry_case():
    slots = mx.array([False] * 7 + [True] * 256 + [False] * 10)
    layout = QwenImage21Layout.create(slots, [(1, 32, 32), (1, 72, 32)], (16, 56, 56))
    metadata = validate_reference_layout(np, layout, 1024, 2304)
    reference = mx.array(np.arange(1024 * 64, dtype=np.float32).reshape(1, 1024, 64))
    reference_before = array(reference)
    target = mx.array(-np.arange(2304 * 64, dtype=np.float32).reshape(1, 2304, 64))
    target_before = array(target)
    context = np.zeros((1, 2304, 129), dtype=np.float32)
    context[0, 20 * 32:52 * 32, 64] = 1
    context[..., 65:] = np.arange(2304, dtype=np.float32)[None, :, None] + 200000
    context = mx.array(context)
    padded = control_prefix_padding(mx, context, 1024)
    padded_np = array(padded)
    assert padded_np.shape == (1, 3328, 129)
    assert np.all(padded_np[:, :1024] == 0)
    assert np.array_equal(padded_np[:, 1024:], array(context))
    assert padded_np[:, :1024, 64].sum() == 0
    assert padded_np[..., 64].sum() == 1024
    for sigma in (1.0, 0.5, 0.0):
        bridged = target_known_bridge(target, target, context, sigma)
        model_input = assemble_reference_input(reference, bridged)
        assert np.array_equal(array(model_input[:, :1024]), reference_before)
        assert np.array_equal(array(reference), reference_before)
    bridge_zero = array(target_known_bridge(target, target, context, 0.0))
    assert np.array_equal(bridge_zero[:, 20 * 32:52 * 32], array(context[:, 20 * 32:52 * 32, 65:]))
    assert np.array_equal(bridge_zero[:, :20 * 32], target_before[:, :20 * 32])
    assert np.array_equal(bridge_zero[:, 52 * 32:], target_before[:, 52 * 32:])
    try:
        target_known_bridge(assemble_reference_input(reference, target), target, padded, 0.0)
    except ValueError:
        wrong_bridge_rejected = True
    else:
        wrong_bridge_rejected = False
    assert wrong_bridge_rejected
    return {**metadata, "status": "passed", "shape": list(padded.shape),
            "reference_control_padding_literal_zero129": True, "target_context_exact": True,
            "target_known_mask_excludes_reference": True, "reference_input_exact_after_target_bridge": True,
            "source_latents_unmodified": True, "bridge_sigma_zero_known_target_exact": True,
            "bridge_unknown_target_unmodified": True, "bridge_reference_input_rejected": True}


def tiny_case(padded_prompt):
    # All 16 hints are exercised with 32 very small base blocks.
    base = Qwen21Transformer(in_channels=8, out_channels=8, num_layers=32,
        attention_head_dim=8, num_attention_heads=4, context_in_dim=12,
        axes_dims_rope=(2, 2, 4), mlp_ratio=3)
    branch = ControlBranch(dim=32, num_heads=4, head_dim=8, mlp_ratio=3,
        num_layers=32, in_channels=129)
    randomize(base, 640 + int(padded_prompt))
    randomize(branch, 641 + int(padded_prompt))
    rng = np.random.default_rng(642)
    slots = mx.array([False, True, False, False, False])
    layout = QwenImage21Layout.create(slots, [(1, 2, 2), (1, 2, 4)], (2, 2, 4))
    layout_info = validate_reference_layout(np, layout, 4, 8)
    reference = mx.array(rng.standard_normal((1, 4, 8), dtype=np.float32))
    reference_before = array(reference)
    target = mx.array(rng.standard_normal((1, 8, 8), dtype=np.float32))
    hidden = assemble_reference_input(reference, target)
    hidden_before = array(hidden)
    prompt = mx.array(rng.standard_normal((1, 5, 12), dtype=np.float32))
    target_context = mx.array(rng.standard_normal((1, 8, 129), dtype=np.float32))
    context = control_prefix_padding(mx, target_context, 4)
    timestep = mx.array([0.5], dtype=mx.float32)
    prompt_mask = mx.array([[True, True, True, True, False]]) if padded_prompt else None
    common = dict(control_scale=1.0, cache=None, encoder_hidden_states_mask=prompt_mask)
    full_trace, gated_trace = {}, {}
    full = reference_controlled_forward(base, branch, hidden, prompt, timestep,
        layout, context, prefix_hints_disabled=False, trace=full_trace, **common)
    official = controlled_forward(base, branch, hidden, prompt, timestep,
        layout, context, **common)
    assert np.array_equal(array(full), array(official)), "Ungated extension changed existing port math"
    gated = reference_controlled_forward(base, branch, hidden, prompt, timestep,
        layout, context, prefix_hints_disabled=True, trace=gated_trace, **common)
    base_initial, base_prefixes, base_output = base_trace(base, hidden, prompt, timestep, layout, prompt_mask)
    assert len(full_trace["prefix_hints_before"]) == len(gated_trace["prefix_hints_after"]) == 16
    assert all(np.isfinite(value).all() for value in full_trace["prefix_hints_before"])
    prefix_hint_max = [float(np.abs(value).max()) for value in full_trace["prefix_hints_before"]]
    assert all(value > 0.01 for value in prefix_hint_max), "Zero129 reference rows unexpectedly disabled hints"
    assert all(np.all(value == 0) for value in gated_trace["prefix_hints_after"])
    assert all(np.array_equal(before, after) for before, after in zip(
        gated_trace["target_hints_before"], gated_trace["target_hints_after"]))
    assert all(np.array_equal(a, b) for a, b in zip(
        full_trace["prefix_hints_before"], gated_trace["prefix_hints_before"]))
    assert np.array_equal(base_initial, gated_trace["base_prefix_initial"])
    assert all(np.array_equal(expected, actual) for expected, actual in zip(
        base_prefixes, gated_trace["base_prefix_after_block"]))
    full_effect = float(np.abs(array(full) - base_output).max())
    gated_effect = float(np.abs(array(gated) - base_output).max())
    arm_difference = float(np.abs(array(full) - array(gated)).max())
    assert full_effect > 0.01 and gated_effect > 0.01 and arm_difference > 0.01
    assert np.array_equal(array(reference), reference_before)
    assert np.array_equal(array(hidden), hidden_before)
    for disabled in (False, True):
        zero = reference_controlled_forward(base, branch, hidden, prompt, timestep,
            layout, context, control_scale=0.0, prefix_hints_disabled=disabled,
            encoder_hidden_states_mask=prompt_mask)
        assert np.array_equal(array(zero), base_output)
    try:
        reference_controlled_forward(base, branch, hidden, prompt, timestep,
            layout, context, cache=[])
    except ValueError:
        cache_rejected = True
    else:
        cache_rejected = False
    assert cache_rejected
    return {**layout_info, "status": "passed", "padded_prompt": padded_prompt,
            "base_blocks": 32, "control_hints": 16, "ungated_existing_port_output_exact": True,
            "ungated_prefix_hint_max_abs_by_layer": prefix_hint_max,
            "zero129_reference_rows_produce_nonzero_prefix_hints": True,
            "completed_prefix_hints_zeroed_exactly": True, "target_hints_unchanged_exactly": True,
            "control_chain_unchanged_exactly": True, "base_prefix_initial_exact": True,
            "base_prefix_trace_exact_all32_blocks": True, "source_input_unmodified": True,
            "full_control_target_effect_max_abs": full_effect,
            "gated_control_target_effect_max_abs": gated_effect,
            "full_vs_gated_target_difference_max_abs": arm_difference,
            "scale_zero_vs_base_exact_both_arms": True, "cache_rejected": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("reference_cpu_validation.json"))
    args = parser.parse_args()
    report = {"status": "passed", "device": "CPU only", "dtype": "float32",
              "checkpoint_weights_used": False, "real_models_loaded": False, "downloads": False,
              "scope": "Experimental adapter mechanics and uncached same-backend prefix parity; no trained image quality or quantized checkpoint parity.",
              "conditioning": cpu_preflight(), "full_canvas_geometry": geometry_case(),
              "tiny_random_weight_cases": [tiny_case(False), tiny_case(True)],
              "control_port_sha256": digest(Path(__file__).with_name("control.py")),
              "reference_helper_sha256": digest(Path(__file__).with_name("reference_control.py")),
              "conditioning_sha256": digest(Path(__file__).with_name("conditioning.py")),
              "script_sha256": digest(__file__)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "device": report["device"],
                      "tiny_random_weight_cases": len(report["tiny_random_weight_cases"]),
                      "prefix_trace_exact_all32_blocks": True, "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
