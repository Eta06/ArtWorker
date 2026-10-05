#!/usr/bin/env python3
"""CPU-only integrity validation of completed six-step growing localized runs.

No inference helper, sampler, MLX or model is imported. Incomplete suites are
refused before any report is written. Saved-array checks, recorded hashes and
runtime assertions have explicitly different evidence limits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
PORT = ROOT / "experiments/qwen/controlnet_port"
HEIGHTS = (640, 896, 1152, 1152, 1152, 1152)
SCALES = (0.0, 0.5, 1.0)
MODE = "tangent-canny"


class IncompleteSuite(ValueError):
    pass


def read_json(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def pixels(path, mode="RGB"):
    with Image.open(path) as image:
        return np.asarray(image.convert(mode))


def require_complete(suite):
    path = suite / "growing_localized_metrics.json"
    if not path.is_file():
        raise IncompleteSuite("Growing metrics are absent")
    s = read_json(path)
    if s.get("status") != "success" or s.get("phase") != "complete":
        raise IncompleteSuite(f"Suite unfinished: status={s.get('status')}, phase={s.get('phase')}")
    expected = [f"{MODE}/scale{scale:g}" for scale in SCALES]
    if list(s.get("runs", {})) != expected or any(s["runs"][arm].get("status") != "success" for arm in expected):
        raise IncompleteSuite("All three scale arms must be complete")
    return s


def ids_window(height):
    top = (1152 - height) // 2
    ids = np.arange(top // 16 * 32, (top + height) // 16 * 32, dtype=np.int32)
    return ids, [0, top, 512, top + height]


def sigmas_formula():
    nodes = np.array([1, .9375, .875, .75, .5, .25], dtype=np.float64)
    shift = math.exp(.5 + .4 * (2304 - 256) / (8192 - 256))
    return np.concatenate([shift / (shift + (1 / nodes - 1)), [0]]).astype(np.float32)


def bf16_rne(value):
    bits = np.ascontiguousarray(value, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def geometry(guide):
    known = np.zeros((1152, 512), dtype=bool)
    known[320:832] = True
    lines = np.any(guide != 0, axis=-1) & ~known
    distance = cv2.distanceTransform((~lines).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    desired = np.zeros((1152, 512), dtype=np.float32)
    inside = distance < 64
    desired[inside] = .5 * (1 + np.cos(np.pi * distance[inside] / np.float32(64)))
    desired[lines] = 1
    desired[known] = 0
    support = known | ((distance <= 64) & ~known)
    # Floor/nearest samples at exact 16-pixel coordinates, in y-major order.
    rows, cols = np.arange(72) * 16, np.arange(32) * 16
    return desired, desired[np.ix_(rows, cols)].reshape(1, 2304, 1), support, support[np.ix_(rows, cols)].reshape(1, 2304, 1), lines


def difference(left, right):
    delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
    return {"exact": bool(np.array_equal(left, right)), "max_abs": float(delta.max()),
            "mean_abs": float(delta.mean()), "differing_values": int(np.count_nonzero(delta))}


class Audit:
    def __init__(self, suite, baseline):
        self.report = {"status": "running", "scope": "CPU-only saved artifact integrity; no quality conclusion",
                       "suite": str(suite), "baseline": str(baseline), "checks": {},
                       "file_hashes": {}, "array_hashes": {}, "active_gathers": {},
                       "arms": {}, "baseline_inputs": {}, "cpu_tests": {}}

    def check(self, name, value):
        self.report["checks"][name] = bool(value)

    def file(self, name, path, expected=None):
        digest = sha(path)
        item = {"path": str(Path(path).resolve()), "sha256": digest}
        if expected is not None:
            item.update(recorded_sha256=expected, matches_recorded=digest == expected)
            self.check("file_hash:" + name, digest == expected)
        self.report["file_hashes"][name] = item
        return digest

    def array(self, name, value, shape=None, dtype=np.float32, expected=None):
        if shape is not None:
            self.check("array_shape:" + name, value.shape == shape)
            self.check("array_dtype:" + name, value.dtype == dtype)
            self.check("array_finite:" + name, np.isfinite(value).all())
        digest = array_sha(value)
        item = {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": digest}
        if expected is not None:
            item.update(recorded_sha256=expected, matches_recorded=digest == expected)
            self.check("array_hash:" + name, digest == expected)
        self.report["array_hashes"][name] = item
        return digest


def validate(suite, baseline, output):
    s = require_complete(suite)
    baseline = baseline or Path(s["cached_geometry_baseline"]).resolve()
    b = read_json(baseline / "metrics.json")
    a = Audit(suite, baseline)
    a.file("suite_metrics", suite / "growing_localized_metrics.json")
    a.file("baseline_metrics", baseline / "metrics.json")
    a.check("six_steps_three_scales", s["steps"] == 6 and s["control_scales"] == list(SCALES))
    a.check("active_heights", s["active_heights"] == list(HEIGHTS))
    a.check("target_geometry", (s["width"], s["height"], s["source_rect_xyxy"]) == (512, 1152, [0, 320, 512, 832]))
    a.check("adapter_recipe", s["adapter_rank"] == 256 and s["adapter_scale"] == 1 and not s["adapter_baked"])
    a.check("full_prefix_no_cache", s["growing_compute"] and not any(s[key] for key in ("prefix_cache_enabled", "base_prefix_cache_enabled", "control_prefix_cache_enabled")))
    a.check("hint_recipe", s["hint_shape"] == "cosine" and s["hint_radius_pixels"] == 64 and s["structural_support"] == "tangent" and not s["hint_localization_changes_compute"])
    a.check("edge_known_codec_recorded", s["known_target_encoding_context"] == "edge-padded-full-canvas-center-crop" and s["known_target_and_control_source_codec_contexts_distinct"])
    a.check("four_cpu_gates_recorded_passed", len(s["cpu_gate_status"]) == 4 and all(s["cpu_gate_status"].values()))
    sigmas = np.asarray(s["sigmas"], dtype=np.float32)
    a.check("sigma_formula_exact", np.array_equal(sigmas, sigmas_formula()))
    a.array("sigmas", sigmas, (7,))
    a.check("layout", s["layout"] == {"reference_image_tokens": 1024, "target_tokens": 2304,
        "joint_prefix_tokens": 1197, "prefix_text_tokens": 173, "context_rows": 3328,
        "control_text_padding_rows": 0, "full_prefix_hint_gate_uses_target_mask": True})

    shared = suite / "shared"
    reference = arrays(shared / "reference_conditioning.npz")
    prefix, prompt, slots = (reference[key] for key in ("source_latents", "prompt_embeds", "image_slots"))
    noise = arrays(shared / "target_noise.npz")["noise"]
    edge = arrays(shared / "edge_known_target.npz")
    known_edge, edge_guess = edge["known"], edge["edge_guess"]
    context = arrays(shared / f"{MODE}_contexts.npz")
    target, padded, support = (context[key] for key in ("target_context", "padded_image_context", "structural_support"))
    info = s["control_conditioning"][MODE]
    for name, value, shape, digest, dtype in (
        ("source_latents", prefix, (1, 1024, 64), s["reference_encoding"]["source_prefix_sha256"], np.float32),
        ("prompt_embeds", prompt, (1, 429, 4096), s["prompt_embeds_sha256"], np.float32),
        ("image_slots", slots, (429,), s["image_slots_bool_sha256"], np.bool_),
        ("noise", noise, (1, 2304, 64), s["noise_sha256"], np.float32),
        ("edge_known", known_edge, (1, 1024, 64), s["edge_known_target_sha256"], np.float32),
        ("edge_guess", edge_guess, (1, 2304, 64), None, np.float32),
        ("target_context", target, (1, 2304, 129), info["target_context_sha256_float32"], np.float32),
        ("padded_context", padded, (1, 3328, 129), info["padded_image_context_sha256_float32"], np.float32),
        ("support", support, (1, 2304, 1), s["structural_support_metadata"]["support_tokens_bool_sha256"], np.bool_),
    ):
        a.array(name, value, shape, dtype, digest)
    known = np.zeros(2304, dtype=bool)
    known[640:1664] = True
    a.check("slots_256_image_173_text", slots.sum() == 256 and (~slots).sum() == 173)
    a.check("known_mask_exact", np.array_equal(target[0, :, 64], known.astype(np.float32)))
    a.check("reference129_zero_target_after_padding_exact", np.all(padded[:, :1024] == 0) and np.array_equal(padded[:, 1024:], target))
    a.check("unsupported_control64_zero", np.all(target[:, ~support.reshape(-1), :64] == 0))
    a.check("known_support_all_true", support.reshape(-1)[known].all())
    repeat = ((np.clip(np.arange(72), 20, 51) - 20)[:, None] * 32 + np.arange(32)[None, :]).reshape(-1)
    a.check("edge_guess_absolute_clamped_rows_exact", np.array_equal(edge_guess, known_edge[:, repeat]))
    a.check("edge_codec_distinct_from_control_source", not np.array_equal(known_edge, target[:, known, 65:]))
    source, rgba = pixels(s["source_512_path"]), pixels(suite / "source_rgba.png", "RGBA")
    a.array("source_rgb", source, (512, 512, 3), np.uint8, s["source_uint8_sha256"])
    a.array("source_rgba", rgba, (512, 512, 4), np.uint8, s["source_rgba_uint8_sha256"])
    a.check("actual_square_opaque_source_exact", np.all(rgba[:, :, 3] == 255) and np.array_equal(rgba[:, :, :3], source))
    a.file("source512", s["source_512_path"], s["source_512_sha256"])
    a.file("source_rgba_png", suite / "source_rgba.png")
    a.file("control_map", s["map_paths"][MODE], s["map_sha256"][MODE])

    hints = arrays(shared / "effective_hint_weights.npz")
    effective, joint, requested = (hints[key] for key in ("target_weights", "joint_weights", "requested_target_weights"))
    h = s["localized_hint_weights"]
    guide = pixels(s["guide_path"])
    desired_pixels, desired, support_pixels, expected_support, lines = geometry(guide)
    saved_pixels = np.load(shared / "hint_pixel_weights_float32.npy", allow_pickle=False)
    saved_desired = np.load(shared / "hint_target_weights_float32.npy", allow_pickle=False)
    saved_support = np.load(shared / "support_tokens.npy", allow_pickle=False)
    for name, value, shape, digest in (
        ("desired_hint_pixels", saved_pixels, (1152, 512), h["pixel_weights_float32_sha256"]),
        ("desired_hint_target_npy", saved_desired, (1, 2304, 1), h["target_weights_float32_sha256"]),
        ("requested_hint_target", requested, (1, 2304, 1), h["target_weights_float32_sha256"]),
        ("effective_hint_target", effective, (1, 2304, 1), h["effective_target_weights_float32_sha256"]),
        ("effective_hint_joint", joint, (1, 3501, 1), h["effective_joint_weights_float32_sha256"]),
    ):
        a.array(name, value, shape, expected=digest)
    a.array("guide_rgb", guide, (1152, 512, 3), np.uint8, h["guide_rgb_uint8_sha256"])
    a.array("support_pixels_reconstructed", support_pixels, expected=s["structural_support_metadata"]["support_pixels_bool_sha256"])
    a.check("independent_cosine64_pixels_exact", np.array_equal(saved_pixels, desired_pixels))
    a.check("independent_nearest16_weights_exact", np.array_equal(requested, desired) and np.array_equal(saved_desired, desired))
    a.check("independent_bf16_rounding_exact", np.array_equal(effective, bf16_rne(requested)) and h["effective_dtype"] == "mlx.core.bfloat16")
    a.check("all1197_prefix_weights_zero", np.all(joint[:, :1197] == 0))
    a.check("joint_suffix_effective_exact", np.array_equal(joint[:, 1197:], effective))
    a.check("all1024_known_target_weights_zero", np.all(effective[:, known] == 0))
    a.check("97_positive_unknown_only", np.count_nonzero(effective) == np.count_nonzero(effective[:, ~known]) == 97)
    a.check("weights_bounded_positive_support_preserved", np.all((effective >= 0) & (effective <= 1)) and np.array_equal(effective > 0, requested > 0))
    a.check("support_reconstruction_exact", np.array_equal(support, expected_support) and np.array_equal(saved_support, expected_support))
    a.check("positive_hints_inside_support", np.all(support.reshape(-1)[(effective > 0).reshape(-1)]))
    a.check("support_and_guide_counts", support.sum() == 1121 and support_pixels.sum() == 290010 and lines.sum() == 2031 and np.count_nonzero(saved_pixels) == 27864)
    a.check("hint_display_exact", np.array_equal(pixels(shared / "hint_weights.png", "L"), np.clip(saved_pixels * 255, 0, 255).round().astype(np.uint8)))
    a.check("support_display_exact", np.array_equal(pixels(shared / "support.png", "L"), support_pixels.astype(np.uint8) * 255))
    a.file("guide_png", s["guide_path"], h["guide_file_sha256"])
    a.file("hint_display", shared / "hint_weights.png", h["display_png_sha256"])
    a.file("support_display", shared / "support.png", s["structural_support_metadata"]["support_png_sha256"])
    for path in sorted(shared.iterdir()):
        if path.is_file():
            a.file("shared:" + path.name, path)
    a.report["hint_geometry"] = {"positive_unknown_tokens": 97, "support_tokens": 1121,
        "desired_sum": float(requested.sum()), "effective_sum": float(effective.sum()),
        "bf16_rounded_cells": int(np.count_nonzero(effective != requested)),
        "maximum_bf16_rounding_error": float(np.max(np.abs(effective - requested)))}

    for step, height in enumerate(HEIGHTS, 1):
        ids, window = ids_window(height)
        active_context, active_hint, active_noise = target[:, ids], effective[:, ids], noise[:, ids]
        a.check(f"step{step}:reconstructed_absolute_window", len(ids) == height // 16 * 32 and np.all((ids // 32 >= window[1] // 16) & (ids // 32 < window[3] // 16)))
        a.check(f"step{step}:reconstructed_known_hint_zero", known[ids].sum() == 1024 and np.all(active_hint[:, known[ids]] == 0))
        a.report["active_gathers"][str(step)] = {"window_xyxy": window, "absolute_id_bounds_inclusive": [int(ids[0]), int(ids[-1])],
            "target_tokens": len(ids), "future_target_tokens": 2304 - len(ids), "context_sha256": array_sha(active_context),
            "hint_sha256": array_sha(active_hint), "absolute_noise_gather_sha256": array_sha(active_noise), "absolute_ids_sha256": array_sha(ids),
            "padded_image_rows": 1024 + len(ids), "joint_queries": 1197 + len(ids), "positive_hint_tokens": int(np.count_nonzero(active_hint)),
            "evidence_limit": "Context/hint digests compared with saved runtime records; active noise is independently reconstructed, with no saved actual per-step noise trace"}

    finals, clean6 = {}, {}
    for scale in SCALES:
        arm = f"{MODE}/scale{scale:g}"
        r, directory = s["runs"][arm], suite / arm
        rows = r["step_records"]
        a.check(arm + ":arm_metrics_match_suite", read_json(directory / "metrics.json") == r)
        a.check(arm + ":6nfe_192base_96control", len(rows) == r["transformer_calls"] == 6 and r["base_block_calls"] == 192 and r["control_block_calls"] == 96)
        a.check(arm + ":token_totals", r["target_token_forward_sum"] == 12288 and r["reference_image_token_forward_sum"] == 6144 and r["joint_query_token_forward_sum"] == 19470)
        a.check(arm + ":shared_input_hashes", r["source_prefix_sha256"] == array_sha(prefix) and r["noise_sha256"] == array_sha(noise) and r["known_target_sha256"] == array_sha(known_edge))
        a.check(arm + ":shared_prompt_sigmas_recipe", r["prompt"] == s["prompt"] and r["sigmas"] == s["sigmas"] and r["control_scale"] == scale and r["active_heights"] == list(HEIGHTS))
        a.check(arm + ":no_cache_recorded", not r["base_prefix_cache_enabled"] and not r["control_prefix_cache_enabled"])
        a.check(arm + ":forward_assertions_recorded", r["source_prefix_exact_each_forward"] and r["future_targets_absent_both_chains"])
        for i, (row, height) in enumerate(zip(rows, HEIGHTS)):
            step, ids_window_pair = i + 1, ids_window(height)
            ids, window = ids_window_pair
            gather, name = a.report["active_gathers"][str(step)], f"{arm}:step{step}:"
            a.check(name + "absolute_window_count", row["step"] == step and row["active_height"] == height and row["active_window_xyxy"] == window and row["target_tokens_forwarded"] == len(ids))
            a.check(name + "absent_and_new_counts", row["future_target_tokens_absent"] == 2304 - len(ids) and row["newly_activated_tokens"] == (1280, 512, 512, 0, 0, 0)[i])
            a.check(name + "base_control_image_rows", row["input_image_latent_tokens"] == row["control_image_rows"] == 1024 + len(ids))
            a.check(name + "both_chain_queries", row["base_joint_query_tokens"] == row["control_joint_query_tokens"] == 1197 + len(ids))
            a.check(name + "blocks", row["base_blocks"] == 32 and row["control_blocks"] == 16)
            a.check(name + "sigmas", row["sigma"] == float(sigmas[i]) and row["next_sigma"] == float(sigmas[i + 1]))
            a.check(name + "context_gather_hash", row["active_context_sha256"] == gather["context_sha256"])
            a.check(name + "hint_gather_hash", row["active_hint_weights_sha256"] == gather["hint_sha256"])
            a.check(name + "runtime_assertions_recorded", row["prefix_recomputed_both_chains"] and row["known_source_target_hint_weights_zero"] and row["source_prefix_input_exact"])
        for field, total in (("target_tokens_forwarded", "target_token_forward_sum"), ("base_blocks", "base_block_calls"),
                             ("control_blocks", "control_block_calls"), ("base_joint_query_tokens", "joint_query_token_forward_sum")):
            a.check(arm + ":sum:" + field, sum(row[field] for row in rows) == r[total])
        final = arrays(directory / "final_latents.npz")["latents"]
        a.array(arm + ":final_latents", final, (1, 2304, 64), expected=r["final_latents_sha256"])
        a.file(arm + ":final_latents_npz", directory / "final_latents.npz")
        a.check(arm + ":final_known_edge_exact", np.array_equal(final[:, known], known_edge))
        a.check(arm + ":saved_prefix_cpu_assembly_exact", np.array_equal(np.concatenate([prefix, final], axis=1)[:, :1024], prefix))
        raw, composite = pixels(directory / "raw.png"), pixels(directory / "composite.png")
        a.check(arm + ":final_image_shapes", raw.shape == composite.shape == (1152, 512, 3))
        a.check(arm + ":final_composite_source_exact", np.array_equal(composite[320:832], source))
        a.file(arm + ":raw_png", directory / "raw.png", r["raw_sha256"])
        a.file(arm + ":composite_png", directory / "composite.png", r["composite_sha256"])
        a.check(arm + ":preview_steps", [p["step"] for p in r["previews"]] == [2, 4, 6])
        for p in r["previews"]:
            step = p["step"]
            ids, window = ids_window(HEIGHTS[step - 1])
            path = directory / "previews" / f"step{step:02d}_predicted_clean.npz"
            z, name = arrays(path), f"{arm}:preview{step}:"
            clean = z["latents"]
            a.array(name + "latents", clean, (1, len(ids), 64))
            a.array(name + "absolute_ids", z["final_canvas_ids"], (len(ids),), np.int32)
            a.check(name + "absolute_ids_exact", np.array_equal(z["final_canvas_ids"], ids))
            a.check(name + "window_exact", np.array_equal(z["window_xyxy"], np.asarray(window)) and p["active_window_xyxy"] == window)
            a.check(name + "prestep_sigma", z["sigma"].shape == () and float(z["sigma"]) == float(sigmas[step - 1]))
            a.check(name + "known_edge_codec_exact", np.array_equal(clean[:, known[ids]], edge_guess[:, ids][:, known[ids]]))
            raw_path = directory / "previews" / f"step{step:02d}_predicted_clean_raw.png"
            composite_path = directory / "previews" / f"step{step:02d}_predicted_clean_composite.png"
            preview_raw, preview_composite = pixels(raw_path), pixels(composite_path)
            local_y = 320 - window[1]
            a.check(name + "image_shapes", preview_raw.shape == preview_composite.shape == (HEIGHTS[step - 1], 512, 3))
            a.check(name + "composite_source_exact", np.array_equal(preview_composite[local_y:local_y + 512], source))
            a.file(name + "npz", path)
            a.file(name + "raw_png", raw_path, p["raw_sha256"])
            a.file(name + "composite_png", composite_path, p["composite_sha256"])
            if step == 6:
                clean6[scale] = clean
        finals[scale] = final
        a.report["arms"][arm] = {"nfe": 6, "base_blocks": 192, "control_blocks": 96, "target_forwards": 12288,
            "reference_forwards": 6144, "joint_queries": 19470, "denoising_seconds_recorded": r["phases"]["denoising_seconds"],
            "step6_clean_vs_final": difference(clean6[scale], final), "scale_zero_control_cost_retained": scale == 0}
        for kind, final_pixels in (("raw", raw), ("composite", composite)):
            preview_path = directory / "previews" / f"step06_predicted_clean_{kind}.png"
            a.report["arms"][arm]["step6_" + kind + "_vs_final_pixels"] = difference(pixels(preview_path), final_pixels)
        a.file(arm + ":metrics", directory / "metrics.json")

    # Baseline input arrays were not saved; equality claims are limited accordingly.
    for name, digest, saved_array in (("source_prefix", b["source_prefix_sha256"], prefix),
                                    ("noise", b["noise_sha256"], noise), ("edge_known", b["known_target_sha256"], known_edge)):
        actual = array_sha(saved_array)
        a.check("baseline_recorded_hash:" + name, digest == actual)
        a.report["baseline_inputs"][name] = {"baseline_recorded_sha256": digest, "new_saved_array_sha256": actual,
            "matches": digest == actual, "evidence_kind": "baseline recorded hash; actual original baseline input array unavailable"}
    for old_key, new_key in (("seed", "seed"), ("prompt", "prompt"), ("sigmas", "sigmas"), ("steps", "steps"),
        ("model_revision", "model_revision"), ("adapter_revision", "adapter_revision"), ("adapter_rank", "adapter_rank"),
        ("width", "width"), ("height", "height"), ("active_heights_by_step", "active_heights"),
        ("known_target_encoding_context", "known_target_encoding_context")):
        a.check("baseline_metadata:" + old_key, b[old_key] == s[new_key])
    a.check("baseline_metadata:quantization", b["quantization"] == s["base_quantization"])
    a.check("baseline_metadata:successful_edge_context", b["status"] == "success" and b["spatial_mode"] == "edge-context")
    old_path = baseline / "previews/step06_predicted_clean.npz"
    old_clean = arrays(old_path)["latents"]
    a.array("baseline_step6_clean", old_clean, (1, 2304, 64))
    a.file("baseline_step6_clean_npz", old_path)
    a.check("baseline_actual_known_edge_block_exact", np.array_equal(old_clean[:, known], known_edge))
    comparisons = {"old_step6_clean_vs_new_step6_clean": difference(old_clean, clean6[0.0]),
                   "old_step6_clean_vs_new_final": difference(old_clean, finals[0.0]),
                   "identity_required_for_integrity": False,
                   "old_final_latents_file_present": (baseline / "final_latents.npz").is_file(),
                   "interpretation": "Observed comparisons; differences may involve arithmetic ordering and cached/recomputed prefix behavior, with causes not isolated"}
    recorded = s["runs"][f"{MODE}/scale0"]["cached_geometry_scale_zero_comparison"]
    # Reconcile runner bookkeeping in its original float32 reduction precision.
    original_delta = np.abs(old_clean - finals[0.0])
    recorded_precision = {"exact": bool(np.array_equal(old_clean, finals[0.0])),
                          "max_abs": float(original_delta.max()), "mean_abs": float(original_delta.mean())}
    comparisons["old_clean_vs_new_final_original_float32_statistics"] = recorded_precision
    a.check("recorded_baseline_comparison_matches_original_precision", all(recorded.get(key) == recorded_precision[key] for key in ("exact", "max_abs", "mean_abs")))
    a.check("recorded_baseline_comparison_provenance", recorded["status"] == "compared_same_provenance" and all(recorded["matching_provenance"].values()) and Path(recorded["path"]).resolve() == baseline)
    a.check("recorded_baseline_clean_file_hash", sha(old_path) == recorded["old_latents_file_sha256"])
    comparisons["same_stage_scale_zero_previews"] = {}
    for step in (2, 4, 6):
        old_preview = baseline / "previews" / f"step{step:02d}_predicted_clean.npz"
        new_preview = suite / MODE / "scale0/previews" / f"step{step:02d}_predicted_clean.npz"
        old_arrays, new_arrays = arrays(old_preview), arrays(new_preview)
        a.check(f"baseline_preview{step}:absolute_ids_match_new", np.array_equal(old_arrays["final_canvas_ids"], new_arrays["final_canvas_ids"]))
        a.check(f"baseline_preview{step}:window_sigma_match_new", np.array_equal(old_arrays["window_xyxy"], new_arrays["window_xyxy"]) and np.array_equal(old_arrays["sigma"], new_arrays["sigma"]))
        stage = {"clean_latents": difference(old_arrays["latents"], new_arrays["latents"])}
        for kind in ("raw", "composite"):
            old_png = baseline / "previews" / f"step{step:02d}_predicted_clean_{kind}.png"
            new_png = suite / MODE / "scale0/previews" / f"step{step:02d}_predicted_clean_{kind}.png"
            stage[kind + "_pixels"] = difference(pixels(old_png), pixels(new_png))
            a.file(f"baseline_preview{step}:{kind}_png", old_png)
        a.file(f"baseline_preview{step}:npz", old_preview)
        comparisons["same_stage_scale_zero_previews"][str(step)] = stage
    for kind in ("raw", "composite"):
        old_pixels, new_pixels = pixels(baseline / f"{kind}.png"), pixels(suite / MODE / "scale0" / f"{kind}.png")
        comparisons["final_" + kind + "_pixel_comparison"] = difference(old_pixels, new_pixels)
        a.file("baseline_" + kind + "_png", baseline / f"{kind}.png", b.get("output_sha256") if kind == "composite" else None)
    a.report["cached_scale_zero_comparison"] = comparisons
    old_queries = [row["attention_query_tokens"] for row in b["step_records"]]
    a.check("baseline_cached_query_total", sum(old_queries) == 13485)
    a.report["compute_counts"] = {"target_counts": [1280, 1792, 2304, 2304, 2304, 2304], "target_total_per_arm": 12288,
        "new_joint_queries": [2477, 2989, 3501, 3501, 3501, 3501], "new_joint_total_per_chain": 19470,
        "old_cached_base_queries": old_queries, "old_cached_base_query_total": sum(old_queries),
        "new_prefix_query_total": 7182, "old_prefix_initial_query_total": 1197, "old_baseline_has_no_control_branch": True}

    a.file("runner", PORT / "run_growing_localized_control_trial.py", s["runner_sha256"])
    for name, digest in s["source_code_sha256"].items():
        a.file("code:" + name, PORT / name, digest)
    for name, info in s["runtime_dependency_hashes"].items():
        a.file("runtime:" + name, info["path"], info["sha256"])
    a.file("old_geometry_runner", ROOT / "experiments/qwen/run_geometry_trial.py", s["old_geometry_runner_sha256"])
    a.file("old_spatial_runner", ROOT / "experiments/qwen/run_spatial_ablation.py", s["old_spatial_runner_sha256"])
    for name, hash_key, script in (("cpu_parity.json", "port_cpu_parity_sha256", "test_cpu_parity.py"),
        ("reference_cpu_validation.json", "reference_cpu_validation_sha256", "test_reference_control_cpu.py"),
        ("localized_cpu_validation.json", "localized_cpu_validation_sha256", "test_localized_reference_cpu.py"),
        ("growing_localized_cpu_validation.json", "growing_cpu_validation_sha256", "test_growing_localized_cpu.py")):
        a.file("cpu_report:" + name, PORT / name, s[hash_key])
        test = read_json(PORT / name)
        a.check("cpu_report_passed:" + name, test["status"] == "passed" and test["checkpoint_weights_used"] is False)
        a.file("cpu_script:" + script, PORT / script, test["script_sha256"])
        if "source_code_sha256" in test:
            a.check("cpu_report_sources_pinned:" + name, all(s["source_code_sha256"].get(key) == value for key, value in test["source_code_sha256"].items()))
        for test_key, code_name in (("control_port_sha256", "control.py"),
                                   ("reference_helper_sha256", "reference_control.py"),
                                   ("conditioning_sha256", "conditioning.py")):
            if test_key in test:
                a.check("cpu_report_source_pin:" + name + ":" + code_name, test[test_key] == s["source_code_sha256"][code_name])
        if "runtime_dependency_hashes" in test:
            a.check("cpu_report_runtime_pinned:" + name, test["runtime_dependency_hashes"] == s["runtime_dependency_hashes"])
        a.report["cpu_tests"][name] = {"status": test["status"], "device": test["device"], "reexecuted_in_this_audit": False}
    growth_test = read_json(PORT / "growing_localized_cpu_validation.json")
    a.check("growth_test_geometry_runner_pinned", growth_test["old_geometry_runner_sha256"] == s["old_geometry_runner_sha256"])
    a.check("growth_test_two_paired_toy_trajectories", len(growth_test["trajectories"]) == 2 and all(case["status"] == "passed" and case["scale_zero_paired_trajectory_exact"] and case["future_target_tokens_absent_both_chains"] for case in growth_test["trajectories"]))
    a.file("base_manifest", ROOT / "experiments/qwen/base-manifest.json", s["base_manifest_sha256"])
    a.file("adapter_manifest", ROOT / "experiments/qwen/viggle-manifest.json", s["adapter_manifest_sha256"])
    manifest_path = ROOT / "experiments/evaluation/inputs/inputs.json"
    a.file("input_manifest", manifest_path)
    for track in read_json(manifest_path)["tracks"]:
        a.file("original_jpeg:" + track["id"], track["original_jpg"], track["original_jpg_sha256"])
    a.file("validator", __file__)
    a.report["limitations"] = [
        "No inference runtime, model weights, sampler or GPU was loaded. Model/adapter payload hashes were not reverified.",
        "Full conditioning arrays and saved preview absolute IDs are directly verified. Context/hint gathers reproduce per-step recorded hashes; actual active-noise/model-input tensors are not saved.",
        "Future-row exclusion and unchanged prefix in actual forwards remain recorded assertions plus pinned toy-test evidence. Actual rotary positions, per-layer hints, Euler states and insertion traces are unsaved.",
        "Known-edge latents and composite source pixels are directly verified in all finals and saved previews. Zero direct hints do not prove preservation of intermediate hidden states.",
        "Baseline prefix/noise equality is a recorded-hash comparison; prompt/sigma equality uses saved metadata. Baseline known-edge equality additionally uses actual saved preview source latents.",
        "Old baseline retains step6 pre-step predicted-clean latents, not a separate final latent. Same-stage clean and cross-stage final comparisons are reported separately; tiny differences are not attributed solely to prefix caching.",
        "New target total12288 equals the old schedule, but new joint queries19470 exceed cached baseline base queries13485. Scale0 still computes all96 control blocks; single-run timings do not prove speedup.",
        "CPU reports are pinned passing tiny random float32 test evidence and were not rerun. Their uncached mechanics do not establish trained cached-model parity or BF16 trajectory equivalence.",
        "Unsparsified control context is not saved. Source65/supported structural64 preservation during sparsification remains a recorded assertion.",
        "This report establishes saved artifact integrity only and makes no image quality or feature continuation claim."
    ]
    failures = [name for name, passed in a.report["checks"].items() if not passed]
    a.report.update(status="failed" if failures else "passed", check_count=len(a.report["checks"]), failures=failures)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(a.report, indent=2) + "\n")
    return a.report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = validate(args.suite.resolve(), args.baseline.resolve() if args.baseline else None, args.output.resolve())
    except IncompleteSuite as error:
        print(json.dumps({"status": "refused_unfinished_suite", "reason": str(error), "report_written": False}, indent=2))
        raise SystemExit(2)
    print(json.dumps({key: report[key] for key in ("status", "check_count", "failures", "compute_counts", "cached_scale_zero_comparison")}, indent=2))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
