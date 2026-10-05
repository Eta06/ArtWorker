#!/usr/bin/env python3
"""CPU-only absolute gathers and paired six-step growth mechanics tests."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
mx.set_default_device(mx.cpu)
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_geometry_trial import (active_ids, active_layout, insertion_state, six_sigmas,
                                cpu_layout_smoke, cpu_insertion_smoke, cpu_geometry_smoke)
from mflux.models.qwen21.model.qwen21_transformer.qwen21_layout import QwenImage21Layout
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer
from control import ControlBranch
from localized_reference_control import prepare_local_hint_weights, localized_reference_controlled_forward, runtime_dependency_hashes
from growing_localized_control import gather_active_conditioning, edge_known_flow_bridge, predicted_clean_with_known
from reference_control import assemble_reference_input
from test_reference_control_cpu import array, digest, randomize

HEIGHTS = (640, 896, 1152, 1152, 1152, 1152)


def gather_case():
    context_np = np.arange(2304 * 129, dtype=np.float32).reshape(1, 2304, 129)
    hint_np = np.arange(2304, dtype=np.float32).reshape(1, 2304, 1) / 2304
    context, hints = mx.array(context_np), mx.array(hint_np)
    slots = mx.array([False] * 7 + [True] * 256 + [False] * 10)
    full_layout = QwenImage21Layout.create(slots, [(1, 32, 32), (1, 72, 32)], (16, 56, 56))
    rows = []
    for height in HEIGHTS:
        ids, _ = active_ids(np, 512, 1152, height)
        layout = active_layout(mx, np, QwenImage21Layout, slots, (1, 32, 32), full_layout, ids, height, 512, (16, 56, 56))
        active_context, padded, active_hints = gather_active_conditioning(np, context, hints, ids)
        assert np.array_equal(array(active_context), context_np[:, ids])
        assert np.array_equal(array(active_hints), hint_np[:, ids])
        assert np.all(array(padded[:, :1024]) == 0)
        assert np.array_equal(array(padded[:, 1024:]), context_np[:, ids])
        assert padded.shape[1] == 1024 + len(ids) and layout.target_tokens == len(ids)
        for actual, original in zip(layout.rope, full_layout.rope):
            assert np.array_equal(array(actual[:layout.prefix_length]), array(original[:full_layout.prefix_length]))
            assert np.array_equal(array(actual[layout.prefix_length:]), array(original)[full_layout.prefix_length + ids])
        rows.append({"active_height": height, "absolute_rows": [int(ids[0] // 32), int(ids[-1] // 32 + 1)],
                     "target_tokens": len(ids), "future_tokens_absent": 2304 - len(ids)})
    assert sum(row["target_tokens"] for row in rows) == 12288
    return {"status": "passed", "steps": rows, "target_token_forward_sum": 12288,
            "reference_image_token_forward_sum": 6144, "literal_zero129_reference_padding": True,
            "context_and_hint_same_absolute_ids_exact": True, "full_canvas_rope_gather_exact": True,
            "inactive_tokens_absent_from_control_and_base_inputs": True}


def trajectory_case(padded_prompt):
    base = Qwen21Transformer(in_channels=8, out_channels=8, num_layers=32,
        attention_head_dim=8, num_attention_heads=2, context_in_dim=12,
        axes_dims_rope=(2, 2, 4), mlp_ratio=1)
    branch = ControlBranch(dim=16, num_heads=2, head_dim=8, mlp_ratio=1, num_layers=32, in_channels=129)
    randomize(base, 1121 + int(padded_prompt)); randomize(branch, 1122 + int(padded_prompt))
    rng = np.random.default_rng(1123)
    width, height, heights = 512, 192, (128, 160, 192, 192, 192, 192)
    slots = mx.array([False, True, False, False, False])
    source_shape = (1, 2, 2)
    full_layout = QwenImage21Layout.create(slots, [source_shape, (1, 12, 32)], (2, 2, 4))
    source = mx.array(rng.standard_normal((1, 4, 8), dtype=np.float32))
    source_before = array(source)
    prompt = mx.array(rng.standard_normal((1, 5, 12), dtype=np.float32))
    prompt_mask = mx.array([[True, True, True, True, False]]) if padded_prompt else None
    noise = mx.array(rng.standard_normal((1, 384, 8), dtype=np.float32))
    known_clean = mx.array(rng.standard_normal((1, 128, 8), dtype=np.float32))
    edge_ids = ((np.clip(np.arange(12), 4, 7) - 4)[:, None] * 32 + np.arange(32)[None, :]).reshape(-1).astype(np.int32)
    edge_guess = known_clean[:, mx.array(edge_ids)]
    context_np = rng.standard_normal((1, 384, 129), dtype=np.float32)
    context_np[:, :, 64] = 0; context_np[:, 4 * 32:8 * 32, 64] = 1
    context = mx.array(context_np)
    guide = np.zeros((height, width, 3), dtype=np.uint8)
    guide[48, 80] = guide[80, 80] = 255
    _, hint_np, _ = prepare_local_hint_weights(np, Image, width, height, [0, 64, 512, 128], guide, 64, "cosine")
    hints = mx.array(hint_np)
    sigmas = mx.array(six_sigmas(np, 2304), dtype=mx.float32)

    def run(scale, controlled):
        latents = previous_ids = previous_clean = None
        rows = []
        for step, active_height in enumerate(heights):
            ids, _ = active_ids(np, width, height, active_height)
            index = mx.array(ids, dtype=mx.int32)
            if previous_ids is None or len(ids) != len(previous_ids):
                latents = insertion_state(mx, np, ids, previous_ids, latents, previous_clean, edge_guess, noise, sigmas[step], True)
            inserted = array(latents)
            layout = active_layout(mx, np, QwenImage21Layout, slots, source_shape, full_layout, ids, active_height, width, (2, 2, 4))
            active_context, padded_context, alpha = gather_active_conditioning(np, context, hints, ids, reference_tokens=4)
            known_mask_np = (ids // 32 >= 4) & (ids // 32 < 8)
            known_mask = mx.array(known_mask_np[None, :, None])
            active_clean, active_noise = edge_guess[:, index], noise[:, index]
            latents = edge_known_flow_bridge(latents, known_mask, active_clean, active_noise, sigmas[step])
            before = latents
            model_input = assemble_reference_input(source, latents)
            assert model_input.shape[1] == 4 + len(ids) == padded_context.shape[1]
            assert np.array_equal(array(model_input[:, :4]), source_before)
            assert np.all(array(alpha)[0, known_mask_np] == 0)
            timestep = (sigmas[step:step + 1] * 1000).astype(latents.dtype) / 1000
            if controlled:
                prediction = localized_reference_controlled_forward(base, branch, model_input, prompt, timestep,
                    layout, padded_context, alpha, control_scale=scale, cache=None,
                    encoder_hidden_states_mask=prompt_mask)
                assert len(branch.control_blocks) == 16
            else:
                prediction = base.forward_reference(model_input, prompt, timestep, layout, cache=None,
                    encoder_hidden_states_mask=prompt_mask)
            advanced = (latents.astype(mx.float32) + (sigmas[step + 1] - sigmas[step]) * prediction.astype(mx.float32)).astype(latents.dtype)
            latents = edge_known_flow_bridge(advanced, known_mask, active_clean, active_noise, sigmas[step + 1])
            clean = predicted_clean_with_known(before, prediction, known_mask, active_clean, sigmas[step])
            assert np.array_equal(array(clean)[0, known_mask_np], array(active_clean)[0, known_mask_np])
            expected_known = ((1 - sigmas[step + 1]) * active_clean.astype(mx.float32) + sigmas[step + 1] * active_noise.astype(mx.float32)).astype(latents.dtype)
            assert np.array_equal(array(latents)[0, known_mask_np], array(expected_known)[0, known_mask_np])
            rows.append({"ids": ids.copy(), "inserted": inserted, "before": array(before),
                         "velocity": array(prediction), "after": array(latents), "clean": array(clean),
                         "target_tokens": len(ids), "branch_blocks": 16 if controlled else 0})
            previous_ids, previous_clean = ids, clean
        return array(latents), rows

    zero, zero_steps = run(0.0, True)
    baseline, base_steps = run(0.0, False)
    for controlled, plain in zip(zero_steps, base_steps):
        for key in ("ids", "inserted", "before", "velocity", "after", "clean"):
            assert np.array_equal(controlled[key], plain[key]), key
    assert np.array_equal(zero, baseline)
    nonzero, nonzero_steps = run(0.5, True)
    effect = float(np.abs(nonzero - baseline).max())
    assert effect > 0.001 and np.isfinite(nonzero).all()
    assert np.array_equal(nonzero[:, 4 * 32:8 * 32], array(known_clean))
    assert np.array_equal(zero[:, 4 * 32:8 * 32], array(known_clean))
    assert np.array_equal(array(source), source_before) and np.array_equal(array(context), context_np)
    assert all(row["branch_blocks"] == 16 for row in zero_steps + nonzero_steps)
    return {"status": "passed", "padded_prompt": padded_prompt, "base_blocks": 32, "control_blocks": 16,
            "steps": 6, "active_heights": list(heights), "scale_zero_paired_trajectory_exact": True,
            "insertions_velocities_bridges_clean_exact_vs_uncached_base": True,
            "sigma_zero_edge_known_target_exact": True, "source_prefix_and_control_context_unmodified": True,
            "future_target_tokens_absent_both_chains": True, "scale_zero_extra_branch_blocks": 96,
            "target_token_forward_sum": sum(row["target_tokens"] for row in zero_steps),
            "nonzero_control_final_effect_max_abs": effect,
            "cached_trained_checkpoint_comparison": "unverified; random CPU mechanics compare uncached baseline"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("growing_localized_cpu_validation.json"))
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    report = {"status": "passed", "device": "CPU only", "dtype": "float32", "real_models_loaded": False,
              "checkpoint_weights_used": False, "adapter_weights_used": False, "downloads": False,
              "scope": "Six-step absolute growth/control/edge bridge mechanics with32 tiny base blocks and16 control blocks; no Turbo/control checkpoint parity or image-quality proof.",
              "gathers": gather_case(), "trajectories": [trajectory_case(False), trajectory_case(True)],
              "old_geometry_smokes": {"layout": cpu_layout_smoke(mx, np, QwenImage21Layout),
                                      "insertion": cpu_insertion_smoke(mx, np), "geometry": cpu_geometry_smoke(mx, np)},
              "source_code_sha256": {name: digest(directory / name) for name in
                                     ("control.py", "conditioning.py", "reference_control.py", "localized_reference_control.py", "growing_localized_control.py", "sparse_conditioning.py", "test_reference_control_cpu.py")},
              "old_geometry_runner_sha256": digest(directory.parent / "run_geometry_trial.py"),
              "runtime_dependency_hashes": runtime_dependency_hashes(), "script_sha256": digest(__file__)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "device": report["device"], "paired_trajectories": 2,
                      "target_tokens_planned": 12288, "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
