#!/usr/bin/env python3
"""Independently validate the saved localized40/reference40 artifacts on CPU.

No sampler, model weights or MLX is imported. This checks persisted arrays,
input-derived geometry and recorded counts; it does not replay model forwards
or establish visual quality. The strict reference validator remains unchanged.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
ARM = "tangent-canny/localized-hints"
REFERENCE_ARM = "tangent-canny/prefix-hints-disabled"
MODE = "tangent-canny"


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def image_array(path, mode="RGB"):
    with Image.open(path) as im:
        return np.asarray(im.convert(mode))


def npz_arrays(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def bf16_round_to_nearest_even(value):
    """Convert finite float32 values to BF16 values without an inference runtime."""
    bits = np.ascontiguousarray(value, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def reconstruct_geometry(guide_rgb):
    """Fixed target geometry independently encoded from the supplied input image."""
    known = np.zeros((1152, 512), dtype=np.bool_)
    known[320:832, :] = True
    unknown_guide = np.any(guide_rgb != 0, axis=2) & ~known
    distances = cv2.distanceTransform(
        (~unknown_guide).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    pixels = np.zeros((1152, 512), dtype=np.float32)
    within = distances < 64
    pixels[within] = np.float32(0.5) * (
        np.float32(1) + np.cos(np.pi * distances[within] / np.float32(64))
    )
    pixels[unknown_guide] = 1
    pixels[known] = 0
    # Floor/nearest input coordinates for 72 x 32 packing, in y-major order.
    ys = np.floor(np.arange(72) * 1152 / 72).astype(np.int64)
    xs = np.floor(np.arange(32) * 512 / 32).astype(np.int64)
    packed = np.ascontiguousarray(pixels[np.ix_(ys, xs)].reshape(1, 2304, 1))
    support_pixels = known | ((distances <= 64) & ~known)
    support = np.ascontiguousarray(support_pixels[np.ix_(ys, xs)].reshape(1, 2304, 1))
    known_target = known[np.ix_(ys, xs)].reshape(-1)
    return pixels, packed, support_pixels, support, known_target, unknown_guide


def validate(suite, reference, output):
    metrics_path = suite / "localized_reference_metrics.json"
    baseline_metrics_path = reference / "reference_control_metrics.json"
    s, b = read_json(metrics_path), read_json(baseline_metrics_path)
    report = {
        "status": "running",
        "scope": "CPU-only saved artifact integrity; no visual quality conclusion",
        "suite": str(suite), "reference_suite": str(reference),
        "checks": {}, "file_hashes": {}, "array_hashes": {},
        "shared_array_comparison": {}, "provenance": {},
    }
    checks = report["checks"]

    def check(name, result):
        checks[name] = bool(result)

    def audit_file(name, path, expected=None):
        actual = file_sha(path)
        item = {"path": str(Path(path).resolve()), "sha256": actual}
        if expected is not None:
            item.update({"recorded_sha256": expected, "matches_recorded": actual == expected})
            check("file_hash:" + name, actual == expected)
        report["file_hashes"][name] = item
        return actual

    def audit_array(name, array, expected=None):
        actual = array_sha(array)
        item = {"shape": list(array.shape), "dtype": str(array.dtype), "sha256": actual}
        if expected is not None:
            item.update({"recorded_sha256": expected, "matches_recorded": actual == expected})
            check("array_hash:" + name, actual == expected)
        report["array_hashes"][name] = item
        return actual

    for label, metadata in (("localized", s), ("reference", b)):
        check(label + ":complete", metadata.get("status") == "success" and metadata.get("phase") == "complete")
        check(label + ":steps40", metadata["steps"] == metadata["nfe"] == 40)
        check(label + ":layout", metadata["layout"] == {
            "reference_image_tokens": 1024, "target_tokens": 2304,
            "joint_prefix_tokens": 1197, "prefix_text_tokens": 173,
            "context_rows": 3328, "control_text_padding_rows": 0,
            "full_prefix_hint_gate_uses_target_mask": True,
        })
    check("localized:single_expected_arm", list(s["runs"]) == [ARM])
    check("reference:single_expected_arm", list(b["runs"]) == [REFERENCE_ARM])
    matched_settings = ["track", "seed", "steps", "cfg", "adapter", "prompt", "prompt_style",
                        "width", "height", "source_rect_xyxy", "known_bridge_extension",
                        "structural_support", "model_revision", "base_quantization", "control_revision",
                        "upstream_revision", "sigmas", "prefix_cache_enabled", "growing_compute"]
    for key in matched_settings:
        check("shared_setting:" + key, s[key] == b[key])
    check("target_geometry", (s["width"], s["height"], s["source_rect_xyxy"]) == (512, 1152, [0, 320, 512, 832]))
    check("hint_recipe", s["hint_shape"] == "cosine" and s["hint_radius_pixels"] == 64 and s["control_scale"] == 1)
    check("full_compute_recorded", not s["growing_compute"] and not s["prefix_cache_enabled"] and not s["hint_localization_changes_compute"])
    audit_file("localized_metrics", metrics_path)
    audit_file("reference_metrics", baseline_metrics_path)

    shared_names = ["reference_conditioning.npz", "target_noise.npz", f"{MODE}_contexts.npz"]
    local_arrays, baseline_arrays = {}, {}
    for name in shared_names:
        local_arrays.update(npz_arrays(suite / "shared" / name))
        baseline_arrays.update(npz_arrays(reference / "shared" / name))
        local_file_hash = audit_file("localized_shared:" + name, suite / "shared" / name)
        baseline_file_hash = audit_file("reference_shared:" + name, reference / "shared" / name)
        # Archive byte equality is reported separately from actual array equality.
        report.setdefault("shared_container_byte_equal", {})[name] = local_file_hash == baseline_file_hash
    expected_shapes = {
        "source_latents": (1, 1024, 64), "prompt_embeds": (1, 429, 4096),
        "image_slots": (429,), "noise": (1, 2304, 64),
        "target_context": (1, 2304, 129), "padded_image_context": (1, 3328, 129),
        "structural_support": (1, 2304, 1),
    }
    expected_hashes = {
        "source_latents": lambda m: m["reference_encoding"]["source_prefix_sha256"],
        "prompt_embeds": lambda m: m["prompt_embeds_sha256"],
        "image_slots": lambda m: m["image_slots_bool_sha256"],
        "noise": lambda m: m["noise_sha256"],
        "target_context": lambda m: m["control_conditioning"][MODE]["target_context_sha256_float32"],
        "padded_image_context": lambda m: m["control_conditioning"][MODE]["padded_image_context_sha256_float32"],
        "structural_support": lambda m: m["structural_support_metadata"]["support_tokens_bool_sha256"],
    }
    for name, shape in expected_shapes.items():
        a, z = local_arrays[name], baseline_arrays[name]
        exact = a.shape == z.shape and a.dtype == z.dtype and np.array_equal(a, z)
        report["shared_array_comparison"][name] = {"shape_dtype_and_values_exact": bool(exact)}
        check("shared_array:" + name, exact)
        for label, value, metadata in (("localized", a, s), ("reference", z, b)):
            check(label + ":shape:" + name, value.shape == shape)
            check(label + ":dtype:" + name, value.dtype == (np.bool_ if name in ("image_slots", "structural_support") else np.float32))
            check(label + ":finite:" + name, np.isfinite(value).all())
            audit_array(label + ":" + name, value, expected_hashes[name](metadata))
    source, prefix = image_array(s["source_512_path"]), local_arrays["source_latents"]
    slots, target, padded = local_arrays["image_slots"], local_arrays["target_context"], local_arrays["padded_image_context"]
    check("image_slots_256_and_173_text", slots.sum() == 256 and (~slots).sum() == 173)
    check("literal_zero_reference129", np.all(padded[:, :1024] == 0))
    check("target_exact_after_padding", np.array_equal(padded[:, 1024:], target))
    known = np.zeros((72, 32), dtype=np.float32)
    known[20:52] = 1
    check("target_known_mask_exact", np.array_equal(target[0, :, 64].reshape(72, 32), known))
    known_flat = known.reshape(-1).astype(bool)
    keep = local_arrays["structural_support"].reshape(-1)
    check("unsupported_structural64_literal_zero", np.all(target[:, ~keep, :64] == 0))
    check("known_structural_support_all_true", keep[known_flat].all())
    check("source65_exact_against_reference", np.array_equal(target[:, :, 64:], baseline_arrays["target_context"][:, :, 64:]))

    for label, p, metadata in (("localized", suite, s), ("reference", reference, b)):
        rgba = image_array(p / "source_rgba.png", "RGBA")
        check(label + ":actual_square_rgba", rgba.shape == (512, 512, 4))
        check(label + ":rgba_source_pixels_exact", np.array_equal(rgba[:, :, :3], source))
        check(label + ":opaque_source_alpha", np.all(rgba[:, :, 3] == 255))
        audit_file(label + ":source512", metadata["source_512_path"], metadata["source_512_sha256"])
        audit_file(label + ":source_rgba", p / "source_rgba.png")
        audit_array(label + ":source_rgb", source, metadata["source_uint8_sha256"])
        audit_array(label + ":source_rgba", rgba, metadata["source_rgba_uint8_sha256"])
        audit_file(label + ":control_map", metadata["map_paths"][MODE], metadata["map_sha256"][MODE])
        audit_file(label + ":input_guide", metadata["guide_path"], metadata["structural_support_metadata"]["guide_file_sha256"])

    hint_info = s["localized_hint_weights"]
    guide_rgb = image_array(s["guide_path"])
    check("guide_shape", guide_rgb.shape == (1152, 512, 3))
    pixels, packed, support_pixels, support, known_reconstructed, unknown_guide = reconstruct_geometry(guide_rgb)
    hint_pixels = np.load(suite / "shared" / "hint_pixel_weights_float32.npy", allow_pickle=False)
    hint_target = np.load(suite / "shared" / "hint_target_weights_float32.npy", allow_pickle=False)
    support_saved = np.load(suite / "shared" / "support_tokens.npy", allow_pickle=False)
    effective = npz_arrays(suite / "shared" / "effective_hint_weights.npz")
    weights, joint, requested = (effective[key] for key in ("target_weights", "joint_weights", "requested_target_weights"))
    check("known_geometry_independent_exact", np.array_equal(known_reconstructed, known_flat))
    check("cosine_pixels_independent_exact", np.array_equal(pixels, hint_pixels))
    check("nearest16_desired_weights_independent_exact", np.array_equal(packed, hint_target))
    check("requested_weights_exact", np.array_equal(requested, packed))
    check("support_independent_exact", np.array_equal(support, local_arrays["structural_support"]) and np.array_equal(support, support_saved))
    support_png = image_array(suite / "shared" / "support.png", "L")
    check("support_display_exact", np.array_equal(support_png, support_pixels.astype(np.uint8) * 255))
    hint_png = image_array(suite / "shared" / "hint_weights.png", "L")
    check("hint_display_exact", np.array_equal(hint_png, np.clip(pixels * 255, 0, 255).round().astype(np.uint8)))
    check("effective_target_shape", weights.shape == (1, 2304, 1))
    check("effective_joint_shape", joint.shape == (1, 3501, 1))
    check("all1197_prefix_weights_literal_zero", np.all(joint[:, :1197] == 0))
    check("effective_joint_tail_exact", np.array_equal(joint[:, 1197:], weights))
    check("all1024_known_target_weights_literal_zero", np.all(weights[:, known_flat] == 0) and np.all(requested[:, known_flat] == 0))
    check("effective_weights_finite_and_bounded", np.isfinite(weights).all() and np.all((weights >= 0) & (weights <= 1)))
    check("positive97_unknown_cells_only", np.count_nonzero(weights) == 97 and np.count_nonzero(weights[:, ~known_flat]) == 97)
    check("positive_cells_equal_requested", np.array_equal(weights > 0, requested > 0))
    check("positive_hint_cells_within_structural_support", np.all(keep[(weights > 0).reshape(-1)]))
    check("effective_exact_bf16_rne", np.array_equal(weights, bf16_round_to_nearest_even(requested)))
    check("effective_stored_float32", all(value.dtype == np.float32 for value in (weights, joint, requested)))
    check("bf16_precision_recorded", hint_info["effective_dtype"] == "mlx.core.bfloat16")
    check("guide_unknown_count", int(unknown_guide.sum()) == hint_info["unknown_guide_pixels"] == 2031)
    check("positive_pixel_count", int(np.count_nonzero(pixels)) == hint_info["positive_hint_pixels"] == 27864)
    check("support_token_count", int(support.sum()) == s["structural_support_metadata"]["supported_target_tokens"] == 1121)
    check("support_pixel_count", int(support_pixels.sum()) == s["structural_support_metadata"]["supported_pixels"] == 290010)
    check("desired_weight_sum", float(packed.sum()) == hint_info["target_hint_weight_sum"])
    audit_array("guide_rgb", guide_rgb, hint_info["guide_rgb_uint8_sha256"])
    audit_array("hint_pixels_desired", hint_pixels, hint_info["pixel_weights_float32_sha256"])
    audit_array("hint_target_desired", hint_target, hint_info["target_weights_float32_sha256"])
    audit_array("hint_requested_target", requested, hint_info["target_weights_float32_sha256"])
    audit_array("hint_effective_target", weights, hint_info["effective_target_weights_float32_sha256"])
    audit_array("hint_effective_joint", joint, hint_info["effective_joint_weights_float32_sha256"])
    audit_array("support_pixels_reconstructed", support_pixels, s["structural_support_metadata"]["support_pixels_bool_sha256"])
    audit_array("support_tokens_saved", support_saved, s["structural_support_metadata"]["support_tokens_bool_sha256"])
    for name in ("hint_pixel_weights_float32.npy", "hint_target_weights_float32.npy", "support_tokens.npy", "effective_hint_weights.npz", "hint_weights.json", "support.json"):
        audit_file("localized_shared:" + name, suite / "shared" / name)
    audit_file("hint_display", suite / "shared" / "hint_weights.png", hint_info["display_png_sha256"])
    audit_file("support_display", suite / "shared" / "support.png", s["structural_support_metadata"]["support_png_sha256"])
    report["hint_geometry"] = {
        "shape": "cosine", "radius_pixels": 64, "unknown_guide_pixels": int(unknown_guide.sum()),
        "positive_hint_pixels": int(np.count_nonzero(pixels)), "positive_target_tokens": int(np.count_nonzero(weights)),
        "known_target_tokens_zero": int(known_flat.sum()), "joint_prefix_tokens_zero": 1197,
        "desired_sum": float(requested.sum()), "effective_sum": float(weights.sum()),
        "bf16_rounded_cells": int(np.count_nonzero(weights != requested)),
        "maximum_bf16_rounding_error": float(np.max(np.abs(weights - requested))),
        "support_target_tokens": int(support.sum()), "support_pixels": int(support_pixels.sum()),
        "numpy_version": np.__version__, "opencv_version": cv2.__version__,
    }

    report["recorded_runtime"] = {}
    for label, p, metadata, arm in (("localized", suite, s, ARM), ("reference", reference, b, REFERENCE_ARM)):
        run = metadata["runs"][arm]
        records = run["step_records"]
        check(label + ":run_complete", run["status"] == "success")
        check(label + ":40_recorded_nfe", len(records) == run["transformer_calls"] == run["steps"] == 40)
        check(label + ":step_sequence", [row["step"] for row in records] == list(range(1, 41)))
        for field, aggregate, count in (("base_blocks", "base_block_calls", 32), ("control_blocks", "control_block_calls", 16),
                                        ("target_tokens_forwarded", "target_token_forward_sum", 2304),
                                        ("reference_image_tokens_forwarded", "reference_image_token_forward_sum", 1024),
                                        ("joint_query_tokens", "joint_query_token_forward_sum", 3501)):
            check(label + ":per_call_and_total:" + field, all(row[field] == count for row in records) and sum(row[field] for row in records) == run[aggregate] == count * 40)
        check(label + ":step_sigmas_match", all(row["sigma"] == metadata["sigmas"][i] and row["next_sigma"] == metadata["sigmas"][i + 1] for i, row in enumerate(records)))
        check(label + ":recorded_prefix_assertions", run["source_prefix_exact_each_forward"] and all(row["source_prefix_input_exact"] for row in records))
        directory = p / arm
        latents = npz_arrays(directory / "final_latents.npz")["latents"]
        raw, composite = image_array(directory / "raw.png"), image_array(directory / "composite.png")
        check(label + ":final_latent_shape_finite", latents.shape == (1, 2304, 64) and np.isfinite(latents).all())
        check(label + ":final_known_source_latents_exact", np.array_equal(latents[:, known_flat], local_arrays["target_context"][:, known_flat, 65:]))
        check(label + ":saved_prefix_cpu_assembly_exact", np.array_equal(np.concatenate([prefix, latents], axis=1)[:, :1024], prefix))
        check(label + ":raw_and_composite_shapes", raw.shape == composite.shape == (1152, 512, 3))
        check(label + ":composite_source_pixels_exact", np.array_equal(composite[320:832], source))
        check(label + ":run_source_prefix_hash", run["source_prefix_sha256"] == array_sha(prefix))
        check(label + ":run_noise_hash", run["noise_sha256"] == array_sha(local_arrays["noise"]))
        audit_array(label + ":final_latents", latents, run["final_latents_sha256"])
        audit_file(label + ":final_latents_npz", directory / "final_latents.npz")
        audit_file(label + ":raw_image", directory / "raw.png", run["raw_sha256"])
        audit_file(label + ":composite_image", directory / "composite.png", run["composite_sha256"])
        report["recorded_runtime"][label] = {
            "nfe": len(records), "base_blocks": run["base_block_calls"], "control_blocks": run["control_block_calls"],
            "target_token_forward_sum": run["target_token_forward_sum"],
            "reference_token_forward_sum": run["reference_image_token_forward_sum"],
            "joint_query_token_forward_sum": run["joint_query_token_forward_sum"],
            "denoising_seconds": run["phases"]["denoising_seconds"],
            "full_suite_elapsed_seconds": metadata["elapsed_seconds"],
        }

    for label, metadata, runner in (("localized", s, "run_localized_reference_control_trial.py"), ("reference", b, "run_reference_control_trial.py")):
        audit_file(label + ":runner", ROOT / "experiments/qwen/controlnet_port" / runner, metadata["runner_sha256"])
        for name, expected in metadata["source_code_sha256"].items():
            audit_file(label + ":code:" + name, ROOT / "experiments/qwen/controlnet_port" / name, expected)
        for name, info in metadata.get("runtime_dependency_hashes", {}).items():
            audit_file(label + ":runtime_module:" + name, info["path"], info["sha256"])
    # Inspect the pinned source as data; never import either inference runner.
    reference_runner = ROOT / "experiments/qwen/controlnet_port/run_reference_control_trial.py"
    syntax = ast.parse(reference_runner.read_text())
    calls = [node for node in ast.walk(syntax) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "reference_controlled_forward"]
    scale_keywords = [keyword.value for call in calls for keyword in call.keywords
                      if keyword.arg == "control_scale"]
    baseline_scale_one = len(calls) == len(scale_keywords) == 1 and isinstance(scale_keywords[0], ast.Constant) and scale_keywords[0].value == 1.0
    check("reference_scale_one_from_pinned_runner_source", baseline_scale_one)
    report["provenance"]["baseline_control_scale"] = {
        "value": 1.0 if baseline_scale_one else None,
        "evidence_kind": "inferred from hash-matched runner source; not a stored metrics field or runtime trace",
        "runner_call_line": calls[0].lineno if calls else None,
        "runner_path": str(reference_runner),
        "explicit_control_scale_keyword": baseline_scale_one,
    }
    manifest_path = ROOT / "experiments/evaluation/inputs/inputs.json"
    audit_file("input_manifest", manifest_path)
    for track in read_json(manifest_path)["tracks"]:
        audit_file("original_jpeg:" + track["id"], track["original_jpg"], track["original_jpg_sha256"])
    audit_file("validator", __file__)
    audit_file("strict_reference_validator_retained", ROOT / "experiments/qwen/validate_reference_outputs.py")

    report["input_counts_per_forward"] = {"target": 2304, "source_reference_images": 1024, "text": 173,
                                          "joint_queries": 3501, "packed_image_context_rows": 3328}
    report["limitations"] = [
        "Actual saved source, prompt, slots, noise, structural contexts and support compare exactly with reference40; every model forward was not independently replayed.",
        "NFE, block totals, per-call counts and per-forward prefix equality are internally consistent saved runtime records and assertions, not a separately collected execution trace.",
        "Effective gate arrays are independently verified. Their application after all 16 hints is a recorded runtime claim backed by source provenance; no real-model per-layer hint trace is saved in this artifact suite.",
        "Zero direct hints on known target cells do not prove preservation of intermediate hidden states. Only final known latents and composited source pixels are independently verified here.",
        "The source65 comparison is with saved reference40 contexts. Unsparsified baseline contexts are not saved, so source65 preservation during context construction is a recorded assertion.",
        "Model weight payload hashes and model forwards were not reverified; no GPU or inference runtime was used.",
        "Recorded denoising durations are descriptive single-run values and do not establish a speedup. Hint localization retains full joint computation.",
        "No visual quality or feature continuation acceptance follows from passing integrity checks.",
    ]
    report["failures"] = [name for name, passed in checks.items() if not passed]
    report["status"] = "passed" if not report["failures"] else "failed"
    report["check_count"] = len(checks)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.suite.resolve(), args.reference.resolve(), args.output.resolve())
    print(json.dumps({key: report[key] for key in ("status", "check_count", "failures", "hint_geometry", "input_counts_per_forward")}, indent=2))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
