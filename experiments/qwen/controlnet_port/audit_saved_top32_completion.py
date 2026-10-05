#!/usr/bin/env python3
"""Independent persisted-array and exact-pixel audit, NumPy/Pillow only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from known_collar_saved_baseline import (array_sha, assert_saved_files_unchanged,
                                         bf16_values, file_sha, validate_saved_baseline)


def npz(path):
    with np.load(path, allow_pickle=False) as value:
        return {k: value[k].copy() for k in value.files}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    directory = args.run.resolve()
    suite = json.loads((directory / "saved_known_collar_reference_metrics.json").read_text())
    if suite["status"] != "success":
        raise ValueError(f"Final audit requires completed success, currently {suite['status']}/{suite.get('phase')}")
    if list(suite["runs"]) != ["tangent-canny/top32"] or suite.get("actual_gpu_baseline_rerun_performed") is not False:
        raise ValueError("Expected only the new top32 arm and no repeated GPU baseline")
    support_dir = directory / "shared"
    hint = np.load(support_dir / "hint_fields/top32/hint_target_weights_float32.npy", allow_pickle=False)
    support = np.load(support_dir / "support_tokens.npy", allow_pickle=False)
    saved, baseline_report, old = validate_saved_baseline(np, Image, suite["saved_baseline_validation"]["baseline_directory"], suite, {"top32": hint}, support)
    assert_saved_files_unchanged(suite["saved_baseline_validation"])
    for data in suite["runtime_dependency_hashes"].values():
        assert file_sha(data["path"]) == data["sha256"], "Runtime file changed"
    for name, expected in suite["source_code_sha256"].items():
        assert file_sha(Path(__file__).with_name(name)) == expected, f"Recorded code changed: {name}"
    assert file_sha(Path(__file__).with_name("run_known_collar_saved_baseline_trial.py")) == suite["runner_sha256"]
    reference = npz(support_dir / "reference_conditioning.npz")
    context = npz(support_dir / "tangent-canny_contexts.npz")
    noise = npz(support_dir / "target_noise.npz")
    for name in ("source_latents", "prompt_embeds", "image_slots"):
        assert np.array_equal(reference[name], saved[name]), f"Actual restored reference differs: {name}"
    for name in ("target_context", "padded_image_context"):
        assert np.array_equal(context[name], saved[name]), f"Actual restored context differs: {name}"
    assert np.array_equal(context["structural_support"], support)
    assert np.array_equal(noise["noise"], saved["noise"])
    assert np.array_equal(np.asarray(suite["sigmas"], dtype=np.float32), saved["sigmas"])
    actual_hints = npz(support_dir / "hint_fields/top32/effective_hint_weights.npz")
    assert np.array_equal(actual_hints["requested_target_weights"], hint)
    assert np.array_equal(actual_hints["target_weights"], bf16_values(np, hint))
    assert np.all(actual_hints["joint_weights"][:, :1197] == 0)
    assert np.array_equal(actual_hints["joint_weights"][:, 1197:], actual_hints["target_weights"])
    changed = np.flatnonzero((actual_hints["target_weights"] != saved["old_effective_target_hint"]).reshape(-1))
    assert np.array_equal(changed, np.arange(640, 704))
    arm = suite["runs"]["tangent-canny/top32"]
    arm_dir = directory / "tangent-canny/top32"
    assert json.loads((arm_dir / "metrics.json").read_text()) == arm
    assert arm["status"] == "success" and arm["transformer_calls"] == 40
    assert arm["base_block_calls"] == 1280 and arm["control_block_calls"] == 640
    assert arm["source_prefix_exact_each_forward"] and len(arm["step_records"]) == 40
    assert all(r["source_prefix_input_exact"] for r in arm["step_records"])
    assert suite["reference_encoding"]["source_vae_encodes"] == 0 and suite["rng_draws_for_noise"] == 0
    assert suite["control_conditioning"]["tangent-canny"]["new_control_encode_count"] == 0
    final = npz(arm_dir / "final_latents.npz")
    assert final["latents"].shape == (1, 2304, 64) and np.isfinite(final["latents"]).all()
    assert array_sha(final["latents"]) == arm["final_latents_sha256"]
    assert np.array_equal(final["latents"][:, 640:1664], saved["target_context"][:, 640:1664, 65:])
    assert np.array_equal(final["source_prefix_latents"], saved["source_latents"])
    assert np.array_equal(final["sigmas"], saved["sigmas"])
    assert np.array_equal(final["final_canvas_ids"], np.arange(2304))
    assert final["source_rect_xyxy"].tolist() == [0, 320, 512, 832]
    source = np.asarray(Image.open(suite["source_512_path"]).convert("RGB"))
    raw = np.asarray(Image.open(arm_dir / "raw.png").convert("RGB"))
    composite = np.asarray(Image.open(arm_dir / "composite.png").convert("RGB"))
    assert raw.shape == composite.shape == (1152, 512, 3)
    assert np.array_equal(composite[320:832], source)
    assert np.array_equal(composite[:320], raw[:320]) and np.array_equal(composite[832:], raw[832:])
    for name in ("raw", "composite"):
        assert file_sha(arm_dir / f"{name}.png") == arm[f"{name}_sha256"]
    assert suite["saved_baseline_files_unchanged_after_trial"] is True
    assert_saved_files_unchanged(suite["saved_baseline_validation"])
    result = {
        "status": "passed", "device": "CPU NumPy/Pillow only; no MLX or model imports",
        "actual_gpu_baseline_rerun_performed": False, "actual_new_transformer_calls": 40,
        "actual_common_persisted_tensors_exact": ["source_latents", "prompt_embeds", "image_slots", "target_context", "padded_image_context", "noise", "sigmas"],
        "actual_only_changed_hint_ids": changed.tolist(), "actual_hint_prefix_zero": True,
        "source_reference_exact_each_forward": True, "final_known_bridge_exact": True,
        "final_source_pixels_exact": True, "final_exterior_equals_raw_pixels": True,
        "baseline_files_unchanged_independently": True, "code_and_runtime_files_current": True,
        "new_final_latents_sha256": arm["final_latents_sha256"], "new_raw_sha256": arm["raw_sha256"],
        "old_final_latents_sha256": baseline_report["frozen_baseline_final_latents_sha256"],
        "old_raw_sha256": baseline_report["frozen_baseline_raw_sha256"],
        "quality_verdict": "not assessed by this integrity audit", "script_sha256": file_sha(__file__),
    }
    output = directory / "independent_saved_top32_integrity.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "passed", "source_exact": True, "known_bridge_exact": True, "new_calls": 40, "hint_changed_tokens": len(changed), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
