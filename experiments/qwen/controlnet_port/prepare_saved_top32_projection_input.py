#!/usr/bin/env python3
"""CPU-only exact NPZ/raw copy; add a separate provenance manifest only."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from known_collar_saved_baseline import assert_disjoint_output, array_sha, file_sha

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "experiments/qwen/controlnet_runs/2026-10-05/track3-saved-top32-outpaint/tangent-canny/top32/final_latents.npz"
DEFAULT_OUTPUT = ROOT / "experiments/qwen/projection_inputs/2026-10-05/track3-saved-top32"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    original = args.input.resolve()
    output = args.output.resolve()
    assert_disjoint_output(output, original.parent)
    if output.exists():
        raise FileExistsError("Refusing to overwrite a retained projection input")
    arm = json.loads((original.parent / "metrics.json").read_text())
    suite_dir = original.parent.parent.parent
    suite = json.loads((suite_dir / "saved_known_collar_reference_metrics.json").read_text())
    integrity_path = suite_dir / "independent_saved_top32_integrity.json"
    integrity = json.loads(integrity_path.read_text())
    if suite.get("status") != "success" or arm != suite["runs"]["tangent-canny/top32"] or arm.get("status") != "success":
        raise ValueError("Input must be the completed saved-top32 result")
    if integrity.get("status") != "passed" or not integrity.get("final_source_pixels_exact") or not integrity.get("final_known_bridge_exact"):
        raise ValueError("Input requires independent exact source/bridge integrity")
    shared = json.loads((ROOT / "experiments/evaluation/inputs/inputs.json").read_text())
    track = next(t for t in shared["tracks"] if t["id"] == "track3")
    rect = shared["source_rect_xyxy"]
    if shared["canvas_size"] != [512, 1152] or rect != [0, 320, 512, 832]:
        raise ValueError("Unexpected shared canvas/source geometry")
    with np.load(original, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    required = {"latents", "final_canvas_ids", "source_rect_xyxy", "sigmas", "source_prefix_latents"}
    if set(arrays) != required:
        raise ValueError("Unexpected final NPZ fields; refuse an incomplete or ambiguous copy")
    latent = arrays["latents"]
    if latent.shape != (1, 2304, 64) or latent.dtype != np.float32 or not np.isfinite(latent).all():
        raise ValueError("Expected finite full-canvas F32 packed latents")
    if array_sha(latent) != arm["final_latents_sha256"] or array_sha(latent) != integrity["new_final_latents_sha256"]:
        raise ValueError("Final latent tensor differs from completed result")
    if not np.array_equal(arrays["final_canvas_ids"], np.arange(2304)) or not np.array_equal(arrays["source_rect_xyxy"], rect):
        raise ValueError("Final IDs or source rectangle changed")
    if not np.array_equal(arrays["sigmas"], np.asarray(suite["sigmas"], dtype=np.float32)):
        raise ValueError("Final sigma metadata differs from actual completed steps")
    prefix = arrays["source_prefix_latents"]
    if prefix.shape != (1, 1024, 64) or prefix.dtype != np.float32 or not np.isfinite(prefix).all() or array_sha(prefix) != arm["source_prefix_sha256"]:
        raise ValueError("Original square reference latent metadata changed")
    raw = original.parent / "raw.png"
    composite = original.parent / "composite.png"
    if file_sha(raw) != arm["raw_sha256"] or file_sha(composite) != arm["composite_sha256"]:
        raise ValueError("Retained input images changed")
    source = np.asarray(Image.open(track["source_512"]).convert("RGB"))
    raw_pixels = np.asarray(Image.open(raw).convert("RGB"))
    composite_pixels = np.asarray(Image.open(composite).convert("RGB"))
    if raw_pixels.shape != (1152, 512, 3) or not np.array_equal(composite_pixels[320:832], source):
        raise ValueError("Invalid raw or non-exact original source pixels")
    output.mkdir(parents=True)
    destination = output / "final_latents.npz"
    shutil.copy2(original, destination)
    shutil.copy2(raw, output / "raw.png")
    with np.load(destination, allow_pickle=False) as data:
        for key, value in arrays.items():
            if data[key].dtype != value.dtype or not np.array_equal(data[key], value):
                raise AssertionError(f"Exact metadata copy changed array: {key}")
    if file_sha(destination) != file_sha(original) or file_sha(output / "raw.png") != file_sha(raw):
        raise AssertionError("Copied NPZ/raw files differ from originals")
    report = {
        "status": "passed", "device": "CPU NumPy/Pillow only", "gpu_work": False, "models_loaded": False,
        "original_final_npz": str(original), "copied_final_npz": str(destination),
        "original_npz_file_sha256": file_sha(original), "copied_npz_file_sha256": file_sha(destination),
        "all_npz_arrays_and_dtypes_bitwise_unchanged": True,
        "array_metadata": {k: {"shape": list(v.shape), "dtype": str(v.dtype), "sha256": array_sha(v)} for k, v in arrays.items()},
        "original_raw_png": str(raw), "original_raw_file_sha256": file_sha(raw),
        "copied_raw_file_sha256": file_sha(output / "raw.png"), "copied_raw_pixels_sha256": array_sha(raw_pixels),
        "original_source_pixels_exact": True, "source_rect_xyxy": rect,
        "source_512_file_sha256": file_sha(track["source_512"]), "source_pixels_sha256": array_sha(source),
        "original_jpg_file_sha256": file_sha(track["original_jpg"]),
        "completed_run_metrics_sha256": file_sha(suite_dir / "saved_known_collar_reference_metrics.json"),
        "independent_completed_run_integrity_sha256": file_sha(integrity_path),
        "weights_trained": False, "projection_run_performed": False,
        "purpose": "Exact saved-top32 projection input; separate source/provenance metadata only, no altered latent or image values",
        "script_sha256": file_sha(__file__),
    }
    (output / "input_metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "passed", "all_arrays_and_files_exact": True, "gpu_work": False, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
