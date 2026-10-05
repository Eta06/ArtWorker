#!/usr/bin/env python3
"""CPU saved-input and tiny six-step top32 mechanics; no trained weights."""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
mx.set_default_device(mx.cpu)
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mflux.models.qwen21.model.qwen21_transformer.qwen21_layout import QwenImage21Layout
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer
from control import ControlBranch
from localized_reference_control import (prepare_local_hint_weights,
    localized_reference_controlled_forward, runtime_dependency_hashes)
from known_collar_reference_control import prepare_known_collar_hint_weights
from growing_localized_control import (gather_active_conditioning,
    edge_known_flow_bridge, predicted_clean_with_known)
from growing_known_collar_saved_inputs import (HEIGHTS, RECIPE_KEYS,
    validate_saved_growing_baseline, array_sha, assert_disjoint_output,
    assert_saved_files_unchanged, bf16_values, file_sha)
from reference_control import assemble_reference_input
from run_geometry_trial import active_ids, active_layout, insertion_state, six_sigmas
from test_reference_control_cpu import base_trace, randomize

ROOT = Path(__file__).resolve().parents[3]
BASELINE = ROOT / "experiments/qwen/controlnet_runs/2026-10-05/track3-growing-localized6"


def array(value):
    value = value.astype(mx.float32)
    mx.eval(value)
    return np.asarray(value).copy()


def saved_geometry_case(baseline=BASELINE):
    arrays, report, old = validate_saved_growing_baseline(np, Image, baseline)
    restored = {}
    for name, value in arrays.items():
        if name in ("image_slots", "support", "old_requested_target_hint", "requested_top32_hint", "sigmas"):
            continue
        tensor = mx.array(value).astype(mx.bfloat16)
        assert np.array_equal(array(tensor), value), f"Actual MLX BF16 restore differs: {name}"
        restored[name] = tensor
    requested = mx.array(arrays["requested_top32_hint"]).astype(mx.bfloat16)
    assert np.array_equal(array(requested), arrays["expected_effective_top32_hint"])
    slots = mx.array(arrays["image_slots"], dtype=mx.bool_)
    full = QwenImage21Layout.create(slots, [(1, 32, 32), (1, 72, 32)], (16, 56, 56))
    assert full.prefix_length == 1197 and full.target_tokens == 2304
    source_before = array(restored["source_latents"])
    context_before = array(restored["target_context"])
    input_snapshots = {name: array(restored[name]) for name in
        ("source_latents", "prompt_embeds", "target_context", "noise", "edge_known_target", "edge_guess")}
    steps = []
    for step, height in enumerate(HEIGHTS):
        ids, window = active_ids(np, 512, 1152, height)
        active = active_layout(mx, np, QwenImage21Layout, slots, (1, 32, 32), full,
                              ids, height, 512, (16, 56, 56))
        context, padded, hints = gather_active_conditioning(np,
            restored["target_context"], requested, ids, 1024)
        assert np.array_equal(array(context), arrays["target_context"][:, ids])
        assert np.array_equal(array(hints), arrays["expected_effective_top32_hint"][:, ids])
        assert padded.shape == (1, 1024+len(ids), 129) and np.all(array(padded[:, :1024]) == 0)
        assert np.array_equal(array(padded[:, 1024:]), context_before[:, ids])
        for current, original in zip(active.rope, full.rope):
            assert np.array_equal(array(current[:1197]), array(original[:1197]))
            assert np.array_equal(array(current[1197:]), array(original)[1197+ids])
        local_collar = np.flatnonzero((ids >= 640) & (ids < 704))
        start = {640: 128, 896: 384, 1152: 640}[height]
        assert np.array_equal(local_collar, np.arange(start, start+64))
        known_np = (ids >= 640) & (ids < 1664)
        other_known = known_np & ~((ids >= 640) & (ids < 704))
        assert np.all(array(hints)[0, other_known] == 0)
        mask = mx.array(known_np[None, :, None])
        index = mx.array(ids)
        clean, noise = restored["edge_guess"][:, index], restored["noise"][:, index]
        target = mx.full((1, len(ids), 64), -2, dtype=mx.bfloat16)
        for sigma in (float(arrays["sigmas"][step]), 0.0):
            result = edge_known_flow_bridge(target, mask, clean, noise, mx.array(sigma, dtype=mx.float32))
            expected_known = bf16_values(np, np.float32(1-sigma)*arrays["edge_guess"][:, ids]
                                           + np.float32(sigma)*arrays["noise"][:, ids])
            expected = np.where(known_np[None, :, None], expected_known, np.float32(-2))
            assert np.array_equal(array(result), expected)
            hidden = assemble_reference_input(restored["source_latents"], result)
            assert hidden.shape == (1, 1024+len(ids), 64)
            assert np.array_equal(array(hidden[:, :1024]), source_before)
        steps.append({"step": step+1, "height": height, "window_xyxy": list(window),
            "target_tokens": len(ids), "future_targets_absent": 2304-len(ids),
            "local_collar_positions_half_open": [start, start+64],
            "absolute_target_ids_sha256": array_sha(ids), "context129_gather_sha256": array_sha(array(context)),
            "hint_gather_sha256": array_sha(array(hints)), "prefix_exact": True,
            "absolute_full_canvas_rope_exact": True, "known_flow_bridge_and_sigma_zero_exact": True})
    assert np.array_equal(array(restored["source_latents"]), source_before)
    assert np.array_equal(array(restored["target_context"]), context_before)
    for name, snapshot in input_snapshots.items():
        assert np.array_equal(array(restored[name]), snapshot), f"Restored input changed: {name}"
    assert_saved_files_unchanged(report)
    return {"status": "passed", "actual_mlx_bf16_restores_exact": list(restored),
        "actual_mlx_top32_rounding_exact": True, "layout_prefix_tokens": 1197,
        "target_token_forward_sum": sum(row["target_tokens"] for row in steps),
        "source_prefix_context_noise_and_edges_immutable": True, "steps": steps}, arrays, report, old


def trajectory_case(padded_prompt=False):
    base = Qwen21Transformer(in_channels=8, out_channels=8, num_layers=32,
        attention_head_dim=8, num_attention_heads=2, context_in_dim=12,
        axes_dims_rope=(2, 2, 4), mlp_ratio=1)
    branch = ControlBranch(dim=16, num_heads=2, head_dim=8, mlp_ratio=1, num_layers=32, in_channels=129)
    randomize(base, 2171+int(padded_prompt)); randomize(branch, 2172+int(padded_prompt))
    rng = np.random.default_rng(2173)
    width, height, heights = 512, 192, (128, 160, 192, 192, 192, 192)
    slots = mx.array([False, True, False, False, False]); source_shape = (1, 2, 2)
    full = QwenImage21Layout.create(slots, [source_shape, (1, 12, 32)], (2, 2, 4))
    source = mx.array(rng.standard_normal((1, 4, 8), dtype=np.float32))
    prompt = mx.array(rng.standard_normal((1, 5, 12), dtype=np.float32))
    noise = mx.array(rng.standard_normal((1, 384, 8), dtype=np.float32))
    known_clean = mx.array(rng.standard_normal((1, 128, 8), dtype=np.float32))
    edge_ids = ((np.clip(np.arange(12), 4, 7)-4)[:, None]*32+np.arange(32)[None, :]).reshape(-1).astype(np.int32)
    edge_guess = known_clean[:, mx.array(edge_ids)]
    context_np = rng.standard_normal((1, 384, 129), dtype=np.float32)
    context_np[:, :, 64] = 0; context_np[:, 128:256, 64] = 1
    context = mx.array(context_np)
    guide = np.zeros((height, width, 3), dtype=np.uint8); guide[48, 80] = guide[144, 80] = 255
    _, old_np, _ = prepare_local_hint_weights(np, Image, width, height, [0, 64, 512, 128], guide, 64, "cosine")
    _, disabled_np, _ = prepare_known_collar_hint_weights(np, Image, width, height, [0, 64, 512, 128], guide, 64, "cosine", 0, 1)
    _, new_np, _ = prepare_known_collar_hint_weights(np, Image, width, height, [0, 64, 512, 128], guide, 64, "cosine", 32, 1)
    assert np.array_equal(disabled_np, old_np)
    changed = np.flatnonzero((new_np != old_np).reshape(-1)); assert np.array_equal(changed, np.arange(128, 192))
    sigmas = mx.array(six_sigmas(np, 2304), dtype=mx.float32)
    prompt_mask = mx.array([[True, True, True, True, False]]) if padded_prompt else None
    snapshots = [array(v) for v in (source, prompt, noise, known_clean, edge_guess, context)]

    def run(field, scale=.5, controlled=True, traces=False):
        hints = mx.array(field)
        latents = previous_ids = previous_clean = None
        records = []
        for step, active_height in enumerate(heights):
            ids, _ = active_ids(np, width, height, active_height); index = mx.array(ids)
            if previous_ids is None or len(ids) != len(previous_ids):
                latents = insertion_state(mx, np, ids, previous_ids, latents, previous_clean, edge_guess, noise, sigmas[step], True)
                if previous_ids is not None:
                    assert np.array_equal(array(latents)[:, np.searchsorted(ids, previous_ids)], records[-1]["after"])
            inserted = array(latents)
            layout = active_layout(mx, np, QwenImage21Layout, slots, source_shape, full, ids, active_height, width, (2, 2, 4))
            active_context, padded, alpha = gather_active_conditioning(np, context, hints, ids, 4)
            known_np = (ids >= 128) & (ids < 256)
            known_mask = mx.array(known_np[None, :, None]); clean, active_noise = edge_guess[:, index], noise[:, index]
            latents = edge_known_flow_bridge(latents, known_mask, clean, active_noise, sigmas[step])
            before = latents; hidden = assemble_reference_input(source, latents)
            assert hidden.shape[1] == padded.shape[1] == 4+len(ids)
            assert np.array_equal(array(hidden[:, :4]), snapshots[0])
            timestep = (sigmas[step:step+1]*1000).astype(latents.dtype)/1000
            trace = {} if traces else None
            if controlled:
                prediction = localized_reference_controlled_forward(base, branch, hidden, prompt, timestep, layout,
                    padded, alpha, control_scale=scale, cache=None, encoder_hidden_states_mask=prompt_mask, trace=trace)
            else:
                prediction = base.forward_reference(hidden, prompt, timestep, layout, cache=None, encoder_hidden_states_mask=prompt_mask)
            assert prediction.shape == latents.shape
            if traces:
                old_trace = {}; old_alpha = mx.array(old_np[:, ids])
                localized_reference_controlled_forward(base, branch, hidden, prompt, timestep, layout, padded,
                    old_alpha, control_scale=scale, cache=None, encoder_hidden_states_mask=prompt_mask, trace=old_trace)
                initial, prefixes, _ = base_trace(base, hidden, prompt, timestep, layout, prompt_mask)
                assert len(trace["base_prefix_after_block"]) == 32 and len(trace["target_hints_after"]) == 16
                assert np.array_equal(initial, trace["base_prefix_initial"]) and np.all(trace["joint_gate"][:, :layout.prefix_length] == 0)
                for expected, new_prefix, old_prefix in zip(prefixes, trace["base_prefix_after_block"], old_trace["base_prefix_after_block"]):
                    assert np.array_equal(expected, new_prefix) and np.array_equal(new_prefix, old_prefix)
                collar = (ids >= 128) & (ids < 192); other_known = known_np & ~collar
                assert np.all(array(alpha)[0, other_known] == 0)
                for i in range(16):
                    assert np.array_equal(trace["target_hints_before"][i], old_trace["target_hints_before"][i])
                    assert np.array_equal(trace["target_hints_after"][i][:, ~known_np], old_trace["target_hints_after"][i][:, ~known_np])
                    assert np.all(trace["prefix_hints_after"][i] == 0)
                    assert np.all(trace["target_hints_after"][i][:, other_known] == 0)
                    assert np.max(np.abs(trace["target_hints_after"][i][:, collar])) > .001
            advanced = (latents.astype(mx.float32)+(sigmas[step+1]-sigmas[step])*prediction.astype(mx.float32)).astype(latents.dtype)
            latents = edge_known_flow_bridge(advanced, known_mask, clean, active_noise, sigmas[step+1])
            previous_clean = predicted_clean_with_known(before, prediction, known_mask, clean, sigmas[step])
            expected = ((1-sigmas[step+1])*clean+sigmas[step+1]*active_noise).astype(latents.dtype)
            assert np.array_equal(array(latents)[0, known_np], array(expected)[0, known_np])
            assert np.array_equal(array(previous_clean)[0, known_np], array(clean)[0, known_np])
            records.append({"ids": ids, "inserted": inserted, "before": array(before),
                "velocity": array(prediction), "after": array(latents), "clean": array(previous_clean)})
            previous_ids = ids
        assert np.array_equal(array(latents)[:, 128:256], snapshots[3])
        return array(latents), records

    baseline, base_rows = run(np.zeros_like(old_np), controlled=False)
    for field, scale in ((np.zeros_like(old_np), .5), (old_np, 0), (new_np, 0)):
        final, rows = run(field, scale)
        assert np.array_equal(final, baseline)
        for controlled, plain in zip(rows, base_rows):
            for key in plain:
                assert np.array_equal(controlled[key], plain[key]), key
    old_final, old_rows = run(old_np)
    disabled_final, disabled_rows = run(disabled_np)
    assert np.array_equal(old_final, disabled_final)
    for old_row, disabled_row in zip(old_rows, disabled_rows):
        for key in old_row:
            assert np.array_equal(old_row[key], disabled_row[key])
    enabled, _ = run(new_np, traces=True)
    effect = float(np.max(np.abs(enabled-old_final))); assert effect > .001 and np.isfinite(enabled).all()
    for value, original in zip((source, prompt, noise, known_clean, edge_guess, context), snapshots):
        assert np.array_equal(array(value), original)
    return {"status": "passed", "padded_prompt": padded_prompt, "base_blocks": 32, "control_blocks": 16,
        "steps": 6, "tiny_active_heights": list(heights), "zero_collar_trajectory_exact_old_localized": True,
        "allzero_hint_and_controlscale_zero_trajectories_exact_uncached_base": True,
        "all32_prefix_trace_exact_vs_old_and_base_every_step": True,
        "all16_raw_hints_same_before_gate_and_unknown_hints_unchanged": True,
        "all16_enabled_collar_hints_nonzero": True, "other_known_and_prefix_hints_zero": True,
        "sigma_zero_edge_known_bridge_exact": True, "source_context_noise_and_edge_inputs_immutable": True,
        "newly_inserted_target_old_states_retained_exact": True, "future_targets_absent_both_chains": True,
        "top32_vs_old_final_max_abs": effect, "trained_image_quality_claim": False}


def rejection_cases(baseline, arrays, old):
    rejected = []
    suite = {key: copy.deepcopy(old[key]) for key in RECIPE_KEYS}
    suite.update(known_collar_arms=["top32"], known_collar_scalar_weight=1,
                 runtime_dependency_hashes=copy.deepcopy(old["runtime_dependency_hashes"]))
    validate_saved_growing_baseline(np, Image, baseline, suite)
    def reject(name, callback):
        try:
            callback()
        except (ValueError, KeyError) as exc:
            rejected.append({"case": name, "reason": str(exc)})
        else:
            raise AssertionError(f"Invalid saved inputs accepted: {name}")
    for name, value in (("seed", 43), ("prompt", "changed"), ("steps", 40), ("active_heights", [1152]*6),
        ("control_scales", [1]), ("known_bridge_extension", False), ("known_collar_arms", ["none", "top32"]),
        ("known_collar_scalar_weight", .5), ("base_prefix_cache_enabled", True)):
        altered = copy.deepcopy(suite); altered[name] = value
        reject("current_recipe_"+name, lambda altered=altered: validate_saved_growing_baseline(np, Image, baseline, altered))
    for name, output in (("same", baseline), ("child", baseline/"new"), ("parent", baseline.parent)):
        reject("output_overlap_"+name, lambda output=output: validate_saved_growing_baseline(np, Image, baseline, output=output))
    altered = copy.deepcopy(suite); next(iter(altered["runtime_dependency_hashes"].values()))["sha256"] = "changed"
    reject("actual_runtime_binding", lambda: validate_saved_growing_baseline(np, Image, baseline, altered))
    with tempfile.TemporaryDirectory(prefix="artworker-growing-top32-cpu-") as temporary:
        isolated = Path(temporary)/"baseline"; shutil.copytree(baseline, isolated)
        relocated = copy.deepcopy(old)
        for arm in relocated["runs"].values():
            for key in ("raw_path", "composite_path"):
                arm[key] = str(isolated/Path(arm[key]).relative_to(baseline))
            for preview in arm["previews"]:
                for key in ("latent_path", "raw_path", "composite_path"):
                    preview[key] = str(isolated/Path(preview[key]).relative_to(baseline))
        for key, arm in relocated["runs"].items():
            (isolated/key/"metrics.json").write_text(json.dumps(arm))
        (isolated/"growing_localized_metrics.json").write_text(json.dumps(relocated))
        validate_saved_growing_baseline(np, Image, isolated)
        noise_path = isolated/"shared/target_noise.npz"
        corrupted = arrays["noise"].copy(); corrupted[0, 0, 0] += 1
        np.savez_compressed(noise_path, noise=corrupted)
        reject("saved_noise_hash", lambda: validate_saved_growing_baseline(np, Image, isolated))
        shutil.copy2(baseline/"shared/target_noise.npz", noise_path)
        edge_path = isolated/"shared/edge_known_target.npz"
        guess = arrays["edge_guess"].copy(); guess[0, 0, 0] += 1
        np.savez_compressed(edge_path, known=arrays["edge_known_target"], edge_guess=guess)
        reject("saved_edge_guess_clipped_raster_mapping", lambda: validate_saved_growing_baseline(np, Image, isolated))
        shutil.copy2(baseline/"shared/edge_known_target.npz", edge_path)
        bad = copy.deepcopy(relocated); bad["runs"]["tangent-canny/scale1"]["step_records"][0]["future_target_tokens_absent"] = 0
        (isolated/"growing_localized_metrics.json").write_text(json.dumps(bad))
        (isolated/"tangent-canny/scale1/metrics.json").write_text(json.dumps(bad["runs"]["tangent-canny/scale1"]))
        reject("future_targets_count", lambda: validate_saved_growing_baseline(np, Image, isolated))
        retained = Path(temporary)/"retained.txt"; retained.write_text("unchanged")
        retained_report = {"persisted_file_sha256": {str(retained): file_sha(retained)}}
        assert_saved_files_unchanged(retained_report); retained.write_text("changed")
        reject("retained_file_mutation", lambda: assert_saved_files_unchanged(retained_report))
        assert_disjoint_output(Path(temporary)/"new", baseline)
    return {"status": "passed", "rejected_cases": rejected}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("growing_known_collar_cpu_validation.json"))
    args = parser.parse_args(); baseline = args.baseline.resolve()
    here = Path(__file__).resolve().parent
    helper = here/"growing_known_collar_saved_inputs.py"
    before = {name: file_sha(here/name) for name in ("growing_known_collar_saved_inputs.py", "growing_localized_control.py",
        "localized_reference_control.py", "known_collar_reference_control.py", "known_collar_saved_baseline.py")}
    runtime = runtime_dependency_hashes()
    geometry, arrays, retained, old = saved_geometry_case(baseline)
    report = {"status": "passed", "device": "CPU only", "model_weights_loaded": False,
        "checkpoint_weights_used": False, "downloads": False, "actual_gpu_baseline_rerun_performed": False,
        "saved_geometry": geometry, "six_step_tiny_trajectories": [trajectory_case(False), trajectory_case(True)],
        "rejections": rejection_cases(baseline, arrays, old), "actual_saved_baseline_report": retained,
        "helper_sha256": file_sha(helper), "script_sha256": file_sha(__file__),
        "frozen_growing_helper_sha256": file_sha(here/"growing_localized_control.py"),
        "frozen_localized_helper_sha256": file_sha(here/"localized_reference_control.py"),
        "runtime_dependency_hashes": runtime, "source_code_sha256": before,
        "scope": "Saved tensor integrity, exact BF16 restoration, true absolute growth and tiny32/16 hint mechanics; no trained quality or old cached equality claim."}
    assert before == {name: file_sha(here/name) for name in before}
    assert runtime == runtime_dependency_hashes()
    assert_saved_files_unchanged(retained)
    report["dependency_and_saved_files_unchanged_during_validation"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix+".tmp")
    temporary.write_text(json.dumps(report, indent=2)+"\n"); temporary.replace(args.output)
    print(json.dumps({"status": report["status"], "device": report["device"], "paired_trajectory_cases": 2,
        "target_tokens_per_trained_arm": 12288, "retained_files": len(retained["persisted_file_sha256"]),
        "output": str(args.output.resolve())}), flush=True)


if __name__ == "__main__":
    main()
