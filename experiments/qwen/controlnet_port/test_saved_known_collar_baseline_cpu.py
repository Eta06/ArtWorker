#!/usr/bin/env python3
"""Persisted-input and rejection checks, CPU only, without model weights."""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import tempfile
from pathlib import Path

import mlx.core as mx
mx.set_default_device(mx.cpu)
import numpy as np
from PIL import Image

from known_collar_saved_baseline import (array_sha, assert_disjoint_output, assert_saved_files_unchanged,
                                         bf16_values, file_sha, validate_saved_baseline)
from reference_control import target_known_bridge


def array(value):
    value = value.astype(mx.float32)
    mx.eval(value)
    return np.asarray(value).copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("saved_known_collar_cpu_validation.json"))
    args = parser.parse_args()
    preflight = args.preflight.resolve()
    suite = json.loads((preflight / "preflight.json").read_text())
    baseline = Path(suite["saved_baseline_validation"]["baseline_directory"])
    hint = np.load(preflight / "shared/hint_fields/top32/hint_target_weights_float32.npy", allow_pickle=False)
    support = np.load(preflight / "shared/support_tokens.npy", allow_pickle=False)
    persisted, report, old = validate_saved_baseline(np, Image, baseline, suite, {"top32": hint}, support)
    restored = {}
    for key, value in persisted.items():
        if key in ("image_slots", "old_requested_target_hint", "sigmas"):
            continue
        tensor = mx.array(value).astype(mx.bfloat16)
        check = array(tensor)
        if not np.array_equal(check, value):
            raise AssertionError(f"Actual MLX CPU BF16 restore differs: {key}")
        restored[key] = tensor
    restored_hint = array(mx.array(hint).astype(mx.bfloat16))
    if not np.array_equal(restored_hint, persisted["expected_effective_top32_hint"]):
        raise AssertionError("Independent NumPy BF16 rounding differs from MLX top32 field")
    bridge_tests = []
    context = restored["target_context"]
    noise = restored["noise"]
    latent = mx.full((1, 2304, 64), -2, dtype=mx.bfloat16)
    known = persisted["target_context"][:, :, 64:65] > .5
    for sigma in (1.0, float(persisted["sigmas"][20]), 0.0):
        result = array(target_known_bridge(latent, noise, context, mx.array(sigma, dtype=mx.float32)))
        expected_clean = persisted["target_context"][:, :, 65:]
        expected_bridge = bf16_values(np, np.float32(1-sigma) * expected_clean + np.float32(sigma) * persisted["noise"])
        expected = np.where(known, expected_bridge, np.float32(-2))
        if not np.array_equal(result, expected):
            raise AssertionError("Persisted bridge differs from independent BF16 flow mixture")
        bridge_tests.append({"sigma": sigma, "known_exact": True, "unknown_latents_exact": True, "output_sha256": array_sha(result)})
    rejected = []
    def reject(name, changed_suite=None, changed_hint=None, changed_support=None, path=None):
        try:
            validate_saved_baseline(np, Image, path or baseline, changed_suite or suite,
                                    {"top32": hint if changed_hint is None else changed_hint},
                                    support if changed_support is None else changed_support)
        except (ValueError, KeyError) as exc:
            rejected.append({"case": name, "exception": type(exc).__name__, "reason": str(exc)})
        else:
            raise AssertionError(f"Invalid case accepted: {name}")
    for name, value in (("seed", 43), ("prompt", "different prompt"), ("steps", 39),
                        ("control_scale", .5), ("known_bridge_extension", False),
                        ("known_collar_arms", ["none", "top32"]), ("known_collar_scalar_weight", .5),
                        ("structural_modes", ["source-canny"]), ("source_rect_xyxy", [0, 336, 512, 848]),
                        ("base_manifest_sha256", "changed")):
        changed = copy.deepcopy(suite); changed[name] = value
        reject("recipe_" + name, changed_suite=changed)
    changed = copy.deepcopy(suite)
    first = next(iter(changed["runtime_dependency_hashes"]))
    changed["runtime_dependency_hashes"][first]["sha256"] = "changed"
    reject("runtime_dependency", changed_suite=changed)
    bad_hint = hint.copy(); bad_hint[0, 22*32, 0] = 1
    reject("hint_extra_source_row", changed_hint=bad_hint)
    bad_hint = hint.copy(); bad_hint[0, 0, 0] = .5
    reject("hint_unknown_outside_support", changed_hint=bad_hint)
    bad_hint = hint.copy(); bad_hint[0, 20*32, 0] = 0
    reject("hint_missing_top32_token", changed_hint=bad_hint)
    bad_support = support.copy(); bad_support[0, 0, 0] = ~bad_support[0, 0, 0]
    reject("structural_support_mutation", changed_support=bad_support)
    with tempfile.TemporaryDirectory(prefix="artworker-saved-input-audit-") as temporary:
        isolated = Path(temporary) / "baseline"
        shutil.copytree(baseline, isolated)
        noise_path = isolated / "shared/target_noise.npz"
        corrupted = persisted["noise"].copy(); corrupted[0, 0, 0] += np.float32(1)
        np.savez_compressed(noise_path, noise=corrupted)
        reject("persisted_noise_hash", path=isolated)
        shutil.copy2(baseline / "shared/target_noise.npz", noise_path)
        context_path = isolated / "shared/tangent-canny_contexts.npz"
        with np.load(context_path, allow_pickle=False) as data:
            contents = {name: data[name].copy() for name in data.files}
        contents["target_context"][0, 640, 65] += np.float32(1)
        np.savez_compressed(context_path, **contents)
        reject("persisted_masked_source_hash", path=isolated)
        shutil.copy2(baseline / "shared/tangent-canny_contexts.npz", context_path)
        arm_path = isolated / "tangent-canny/localized-hints/metrics.json"
        contents = json.loads(arm_path.read_text()); contents["sigmas"][1] = .99
        arm_path.write_text(json.dumps(contents))
        reject("suite_arm_schedule_disagreement", path=isolated)
        for name, output in (("same", baseline), ("child", baseline / "new-output"), ("parent", baseline.parent)):
            try:
                assert_disjoint_output(output, baseline)
            except ValueError as exc:
                rejected.append({"case": "output_overlap_" + name, "reason": str(exc)})
            else:
                raise AssertionError("Overlapping output accepted")
        assert_disjoint_output(Path(temporary) / "separate-output", baseline)
        retained = isolated / "retained.txt"
        retained.write_text("unchanged")
        retention_report = {"persisted_file_sha256": {str(retained): file_sha(retained)}}
        assert_saved_files_unchanged(retention_report)
        retained.write_text("mutated")
        try:
            assert_saved_files_unchanged(retention_report)
        except ValueError as exc:
            rejected.append({"case": "baseline_mutation_after_initial_validation", "reason": str(exc)})
        else:
            raise AssertionError("Baseline mutation accepted")
    here = Path(__file__).parent
    result = {
        "status": "passed", "device": "CPU only", "model_weights_loaded": False,
        "actual_gpu_baseline_rerun_performed": False, "persisted_validation": report,
        "actual_mlx_bf16_restores_exact": list(restored), "actual_mlx_top32_rounding_exact": True,
        "persisted_bridge_cases": bridge_tests, "invalid_input_cases": rejected,
        "preflight_sha256": file_sha(preflight / "preflight.json"),
        "source_code_sha256": {name: file_sha(here / name) for name in
                               ("known_collar_saved_baseline.py", "run_known_collar_saved_baseline_trial.py",
                                "known_collar_reference_control.py", "localized_reference_control.py", "reference_control.py")},
        "script_sha256": file_sha(__file__),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "device": result["device"], "invalid_cases": len(rejected), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
