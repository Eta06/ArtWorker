"""Strict persisted-input audit for a top-collar-only matched diagnostic.

No weights are loaded. Stored F32 arrays contain the exact BF16 inference
values; the new runner restores them rather than re-encoding or drawing noise.
The old completed baseline is reused, with no new GPU baseline rerun claim.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def assert_disjoint_output(output, baseline):
    output, baseline = Path(output).resolve(), Path(baseline).resolve()
    if output == baseline or output in baseline.parents or baseline in output.parents:
        raise ValueError("Output and saved baseline must be separate directories without nesting")


def assert_saved_files_unchanged(report):
    for path, expected in report["persisted_file_sha256"].items():
        if file_sha(path) != expected:
            raise ValueError(f"Saved baseline changed during the matched trial: {path}")


def normalized_function(path, name):
    node = next(n for n in ast.parse(Path(path).read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    node.name = "matched_function"
    if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
        node.body.pop(0)
    return ast.dump(node, include_attributes=False)


def load_array(np, path, key, shape, dtype="float32"):
    with np.load(path, allow_pickle=False) as data:
        value = data[key].copy()
    if value.shape != shape or str(value.dtype) != dtype:
        raise ValueError(f"Invalid saved array {path}:{key}: {value.shape}/{value.dtype}")
    if dtype == "float32" and not np.isfinite(value).all():
        raise ValueError(f"Nonfinite saved array {path}:{key}")
    return value


def bf16_values(np, value):
    """F32 representation of round-to-nearest-even BF16; finite inputs only."""
    bits = np.ascontiguousarray(value, dtype=np.float32).view(np.uint32)
    rounded = (bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)
    return rounded.view(np.float32)


def validate_saved_baseline(np, Image, baseline_dir, current_suite, hint_fields, support):
    baseline_dir = Path(baseline_dir).resolve()
    metrics_path = baseline_dir / "localized_reference_metrics.json"
    old = json.loads(metrics_path.read_text())
    key = "tangent-canny/localized-hints"
    arm = old["runs"][key]
    if old.get("status") != "success" or arm.get("status") != "success" or not arm.get("source_pixels_exact"):
        raise ValueError("Only the completed exact-source localized40 baseline is accepted")
    if json.loads((baseline_dir / key / "metrics.json").read_text()) != arm:
        raise ValueError("Saved per-arm metrics differ from completed suite record")
    if not arm.get("prefix_hints_disabled") or arm.get("transformer_calls") != 40:
        raise ValueError("Saved baseline must have zero prefix hints and all 40 denoiser calls")
    for name in ("seed", "steps", "prompt", "prompt_style", "control_scale", "known_bridge_extension", "structural_support", "sigmas", "noise_sha256"):
        if arm.get(name) != old.get(name):
            raise ValueError(f"Saved suite/arm protocol disagreement: {name}")
    if len(arm.get("step_records", [])) != 40:
        raise ValueError("Saved baseline lacks all 40 per-step records")
    for step, record in enumerate(arm["step_records"]):
        if record["step"] != step+1 or record["sigma"] != old["sigmas"][step] or record["next_sigma"] != old["sigmas"][step+1]:
            raise ValueError("Saved baseline per-step schedule disagrees with retained sigmas")
    if current_suite["known_collar_arms"] != ["top32"]:
        raise ValueError("Saved-baseline runner supports top32 alone; no baseline rerun")
    if current_suite["known_collar_scalar_weight"] != 1:
        raise ValueError("Saved-baseline top32 probe has fixed scalar 1")
    recipe_keys = ("track", "seed", "steps", "width", "height", "source_rect_xyxy", "prompt", "prompt_style",
                   "source_512_sha256", "source_uint8_sha256", "source_rgba_uint8_sha256", "map_sha256",
                   "model_repo", "model_revision", "base_quantization", "control_quantization", "cfg", "adapter",
                   "hint_shape", "hint_radius_pixels", "control_scale", "structural_support", "known_bridge_extension",
                   "scheduler_resolution_basis", "prefix_cache_enabled", "compute_mode", "growing_compute",
                   "base_manifest_sha256", "base_config_sha256", "checkpoint_header", "control_revision", "upstream_revision")
    differences = {k: {"saved": old.get(k), "current": current_suite.get(k)} for k in recipe_keys
                   if old.get(k) != current_suite.get(k)}
    if differences:
        raise ValueError(f"Saved baseline recipe drift: {differences}")
    if current_suite["structural_modes"] != ["tangent-canny"] or not current_suite["known_bridge_extension"]:
        raise ValueError("The matched arm requires tangent-canny and the original known bridge")
    here = Path(__file__).parent
    frozen_runner = here / "run_localized_reference_control_trial.py"
    frozen_helper = here / "localized_reference_control.py"
    if file_sha(frozen_runner) != old["runner_sha256"] or file_sha(frozen_helper) != old["source_code_sha256"]["localized_reference_control.py"]:
        raise ValueError("Frozen baseline runner/helper changed since completed generation")
    for name, expected in old["source_code_sha256"].items():
        if file_sha(here / name) != expected:
            raise ValueError(f"Baseline source dependency changed: {name}")
    if old["runtime_dependency_hashes"] != current_suite["runtime_dependency_hashes"]:
        raise ValueError("Imported transformer runtime differs from the saved baseline")
    forward_equal = normalized_function(frozen_helper, "localized_reference_controlled_forward") == normalized_function(here / "known_collar_reference_control.py", "known_collar_reference_controlled_forward")
    builder_equal = normalized_function(frozen_helper, "prepare_local_hint_weights") == normalized_function(here / "known_collar_reference_control.py", "prepare_local_hint_weights")
    if not forward_equal or not builder_equal:
        raise ValueError("Copied baseline forward/unknown field builder AST changed")
    shared = baseline_dir / "shared"
    arrays = {
        "source_latents": load_array(np, shared / "reference_conditioning.npz", "source_latents", (1, 1024, 64)),
        "prompt_embeds": load_array(np, shared / "reference_conditioning.npz", "prompt_embeds", (1, 429, 4096)),
        "image_slots": load_array(np, shared / "reference_conditioning.npz", "image_slots", (429,), "bool"),
        "target_context": load_array(np, shared / "tangent-canny_contexts.npz", "target_context", (1, 2304, 129)),
        "padded_image_context": load_array(np, shared / "tangent-canny_contexts.npz", "padded_image_context", (1, 3328, 129)),
        "noise": load_array(np, shared / "target_noise.npz", "noise", (1, 2304, 64)),
        "old_effective_target_hint": load_array(np, shared / "effective_hint_weights.npz", "target_weights", (1, 2304, 1)),
        "old_effective_joint_hint": load_array(np, shared / "effective_hint_weights.npz", "joint_weights", (1, 3501, 1)),
        "old_requested_target_hint": load_array(np, shared / "effective_hint_weights.npz", "requested_target_weights", (1, 2304, 1)),
    }
    expected_hashes = {
        "source_latents": old["reference_encoding"]["source_prefix_sha256"],
        "prompt_embeds": old["prompt_embeds_sha256"], "image_slots": old["image_slots_bool_sha256"],
        "noise": old["noise_sha256"],
        "target_context": old["control_conditioning"]["tangent-canny"]["target_context_sha256_float32"],
        "padded_image_context": old["control_conditioning"]["tangent-canny"]["padded_image_context_sha256_float32"],
        "old_effective_target_hint": old["localized_hint_weights"]["effective_target_weights_float32_sha256"],
        "old_effective_joint_hint": old["localized_hint_weights"]["effective_joint_weights_float32_sha256"],
        "old_requested_target_hint": old["localized_hint_weights"]["target_weights_float32_sha256"],
    }
    for name, value in arrays.items():
        if array_sha(value) != expected_hashes[name]:
            raise ValueError(f"Persisted tensor hash mismatch: {name}")
        if name != "image_slots" and name != "old_requested_target_hint" and not np.array_equal(value, bf16_values(np, value)):
            raise ValueError(f"Saved inference values do not restore losslessly as BF16: {name}")
    if int(arrays["image_slots"].sum()) != 256:
        raise ValueError("Saved square image slots changed")
    if not np.array_equal(arrays["old_effective_joint_hint"][:, 1197:], arrays["old_effective_target_hint"]):
        raise ValueError("Saved joint hint suffix differs from saved target hint")
    context, padded = arrays["target_context"], arrays["padded_image_context"]
    expected_mask = np.zeros((72, 32), dtype=np.float32); expected_mask[20:52] = 1
    if not np.array_equal(context[0, :, 64].reshape(72, 32), expected_mask):
        raise ValueError("Saved known mask no longer covers exactly original source rows")
    if np.any(padded[:, :1024] != 0) or not np.array_equal(padded[:, 1024:], context):
        raise ValueError("Saved reference-image context padding changed")
    with np.load(shared / "tangent-canny_contexts.npz", allow_pickle=False) as data:
        stored_support = data["structural_support"].copy()
    if not np.array_equal(support, stored_support) or not np.array_equal(support, np.load(shared / "support_tokens.npy", allow_pickle=False)):
        raise ValueError("Structural support differs from persisted baseline")
    if np.any(context[:, ~support.reshape(-1), :64] != 0):
        raise ValueError("Unknown unsupported structural channels are nonzero")
    requested = hint_fields["top32"]
    old_requested = arrays["old_requested_target_hint"]
    changed = np.flatnonzero((requested != old_requested).reshape(-1))
    expected_changed = np.arange(20 * 32, 22 * 32)
    if not np.array_equal(changed, expected_changed) or np.any(requested[:, changed] != 1):
        raise ValueError("Hint difference must be exactly 64 top32 tokens at scalar 1")
    effective = bf16_values(np, requested)
    unchanged = np.ones(2304, dtype=bool); unchanged[expected_changed] = False
    if not np.array_equal(effective[:, unchanged], arrays["old_effective_target_hint"][:, unchanged]):
        raise ValueError("Effective BF16 hints changed outside intended known top collar")
    arrays["expected_effective_top32_hint"] = effective
    arrays["sigmas"] = np.asarray(old["sigmas"], dtype=np.float32)
    if arrays["sigmas"].shape != (41,) or not np.isfinite(arrays["sigmas"]).all() or arrays["sigmas"][0] != 1 or arrays["sigmas"][-1] != 0 or not np.all(np.diff(arrays["sigmas"]) < 0):
        raise ValueError("Persisted 40-step sigma schedule is invalid")
    directory = baseline_dir / key
    final = load_array(np, directory / "final_latents.npz", "latents", (1, 2304, 64))
    if array_sha(final) != arm["final_latents_sha256"] or not np.array_equal(final[:, 640:1664], context[:, 640:1664, 65:]):
        raise ValueError("Saved final baseline or sigma-zero known bridge is invalid")
    source = np.asarray(Image.open(current_suite["source_512_path"]).convert("RGB"))
    baseline_composite = np.asarray(Image.open(directory / "composite.png").convert("RGB"))
    if not np.array_equal(baseline_composite[320:832], source):
        raise ValueError("Saved baseline no longer contains the exact original source pixels")
    for name in ("raw", "composite"):
        if file_sha(directory / f"{name}.png") != arm[f"{name}_sha256"]:
            raise ValueError(f"Saved baseline {name} pixels/file changed")
    retained_files = [metrics_path, directory / "metrics.json", directory / "final_latents.npz", directory / "raw.png", directory / "composite.png",
                      shared / "reference_conditioning.npz", shared / "tangent-canny_contexts.npz", shared / "target_noise.npz", shared / "effective_hint_weights.npz", shared / "support_tokens.npy"]
    report = {
        "status": "passed", "baseline_directory": str(baseline_dir), "baseline_arm": key,
        "new_arms": ["top32"], "actual_gpu_baseline_rerun_performed": False,
        "comparison": "completed saved localized40 result versus new top32; common inference tensors restored exactly",
        "recipe_keys_exact": list(recipe_keys), "persisted_array_sha256": {k: array_sha(v) for k, v in arrays.items()},
        "persisted_file_sha256": {str(p): file_sha(p) for p in retained_files},
        "frozen_baseline_runner_sha256": file_sha(frozen_runner), "frozen_baseline_helper_sha256": file_sha(frozen_helper),
        "frozen_forward_ast_exact_after_name_docstring": forward_equal, "old_unknown_hint_builder_ast_exact": builder_equal,
        "bf16_inference_value_roundtrip_exact": True, "context129_restored_including_mask_and_masked_source": True,
        "source_reference_and_prompt_embeddings_restored": True, "noise_restored_without_rng_draw": True,
        "known_bridge_code_unchanged": True, "saved_final_known_bridge_exact": True,
        "source_pixels_exact": True, "far_baseline_image_retained": True, "all_other_effective_hint_tokens_exact": True,
        "hint_changed_token_ids": changed.tolist(), "hint_changed_packed_rows": sorted(set((changed // 32).tolist())),
        "hint_prefix_weights_zero": bool(np.all(arrays["old_effective_joint_hint"][:, :1197] == 0)),
        "frozen_baseline_final_latents_sha256": arm["final_latents_sha256"], "frozen_baseline_raw_sha256": arm["raw_sha256"],
        "inference_encodes_planned": 0, "inference_rng_draws_planned": 0,
        "gpu_determinism_of_repeated_baseline_unmeasured": True,
        "weight_payload_hash_reverified": False,
    }
    if not report["hint_prefix_weights_zero"]:
        raise ValueError("Saved baseline prefix hints were nonzero")
    return arrays, report, old
