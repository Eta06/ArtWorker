"""Strict saved-input audit for six-step growing top32 hint diagnostics.

No model weights, encodes, noise draws, or device selection occur here. Stored
FP32 representations of inference tensors must restore losslessly as BF16.
Only requested/effective hint IDs 640:704 change; the saved growing baseline
is retained rather than rerun, and its older cached comparison remains inexact.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from known_collar_saved_baseline import (array_sha, assert_disjoint_output,
    assert_saved_files_unchanged, bf16_values, file_sha, load_array,
    normalized_function)

HEIGHTS = (640, 896, 1152, 1152, 1152, 1152)
SCALES = (0.0, 0.5, 1.0)
RECIPE_KEYS = (
    "track", "seed", "steps", "width", "height", "source_rect_xyxy", "prompt", "prompt_style",
    "source_512_path", "source_512_sha256", "source_uint8_sha256", "source_rgba_uint8_sha256",
    "map_paths", "map_sha256", "guide_path", "model_repo", "model_revision", "base_quantization",
    "control_quantization", "cfg", "adapter", "adapter_revision", "adapter_rank", "adapter_scale",
    "adapter_baked", "adapter_bytes", "adapter_expected_sha256", "adapter_manifest_sha256",
    "hint_shape", "hint_radius_pixels", "control_scales", "structural_modes", "structural_support",
    "known_bridge_extension", "scheduler_resolution_basis", "scheduler_target_tokens", "sigmas",
    "active_heights", "known_target_encoding_context", "prefix_cache_enabled",
    "base_prefix_cache_enabled", "control_prefix_cache_enabled", "compute_mode", "growing_compute",
    "base_manifest_sha256", "base_config_sha256", "checkpoint_header", "control_revision", "upstream_revision",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_saved_growing_baseline(np, Image, baseline_dir, current_suite=None, output=None):
    """Return (arrays, immutable-file report, old suite), adding top32 only.

    Optional ``current_suite`` must carry all RECIPE_KEYS and the same actual
    runtime hashes. ``output`` must not equal, contain, or nest in the baseline.
    Array keys preserve the prior saved-baseline helper conventions, with
    edge_known_target, edge_guess, support, requested_top32_hint,
    expected_effective_top32_hint and expected_effective_top32_joint_hint added.
    """
    baseline_dir = Path(baseline_dir).resolve()
    if output is not None:
        assert_disjoint_output(output, baseline_dir)
    metrics_path = baseline_dir / "growing_localized_metrics.json"
    old = json.loads(metrics_path.read_text())
    fixed = {"track": "track3", "seed": 42, "steps": 6, "width": 512, "height": 1152,
        "source_rect_xyxy": [0, 320, 512, 832], "base_quantization": 4, "control_quantization": None,
        "cfg": 1.0, "adapter_rank": 256, "adapter_scale": 1.0, "adapter_baked": False,
        "active_heights": list(HEIGHTS), "hint_shape": "cosine", "hint_radius_pixels": 64,
        "control_scales": list(SCALES), "structural_modes": ["tangent-canny"], "structural_support": "tangent",
        "known_bridge_extension": True, "scheduler_target_tokens": 2304,
        "known_target_encoding_context": "edge-padded-full-canvas-center-crop",
        "prefix_cache_enabled": False, "base_prefix_cache_enabled": False,
        "control_prefix_cache_enabled": False, "growing_compute": True,
        "compute_mode": "active targets only in both chains; full fixed prefix recomputed each step",
        "scheduler_resolution_basis": "target512x1152 only; reference tokens excluded"}
    require(old.get("status") == "success" and old.get("phase") == "complete", "Saved growing suite is not completed successfully")
    for name, expected in fixed.items():
        require(old.get(name) == expected, f"Saved growing recipe is invalid: {name}")
    require(old.get("known_target_and_control_source_codec_contexts_distinct") is True,
            "Edge-known target codec must remain distinct from masked-source129 codec")
    require(old.get("source_prefix_unchanged_after_target_conditioning") is True,
            "Saved conditioning did not retain exact source prefix")
    if current_suite is not None:
        differences = {key: {"saved": old.get(key), "current": current_suite.get(key)}
                       for key in RECIPE_KEYS if key not in current_suite or old.get(key) != current_suite[key]}
        require(not differences, f"Saved growing recipe drift: {differences}")
        require(current_suite.get("known_collar_arms") == ["top32"], "Only top32 arms are supported")
        require(current_suite.get("known_collar_scalar_weight") == 1, "Top32 scalar weight must be 1")

    here = Path(__file__).resolve().parent
    root = here.parents[2]
    if str(here.parent) not in sys.path:
        sys.path.insert(0, str(here.parent))
    frozen_runner = here / "run_growing_localized_control_trial.py"
    require(file_sha(frozen_runner) == old["runner_sha256"], "Frozen growing runner changed")
    retained = [metrics_path, frozen_runner]
    for name, expected in old["source_code_sha256"].items():
        path = here / name
        require(file_sha(path) == expected, f"Frozen growing dependency changed: {name}")
        retained.append(path)
    for filename, key in (("run_geometry_trial.py", "old_geometry_runner_sha256"),
                          ("run_spatial_ablation.py", "old_spatial_runner_sha256")):
        path = here.parent / filename
        require(file_sha(path) == old[key], f"Frozen geometry dependency changed: {filename}")
        retained.append(path)
    from known_collar_reference_control import runtime_dependency_hashes
    actual_runtime = runtime_dependency_hashes()
    require(actual_runtime == old["runtime_dependency_hashes"], "Imported transformer runtime changed")
    if current_suite is not None:
        require(current_suite.get("runtime_dependency_hashes") == actual_runtime,
                "Current suite runtime hashes differ from actual imported runtime")
    retained += [Path(item["path"]) for item in actual_runtime.values()]
    forward_equal = normalized_function(here / "localized_reference_control.py", "localized_reference_controlled_forward") == normalized_function(here / "known_collar_reference_control.py", "known_collar_reference_controlled_forward")
    builder_equal = normalized_function(here / "localized_reference_control.py", "prepare_local_hint_weights") == normalized_function(here / "known_collar_reference_control.py", "prepare_local_hint_weights")
    require(forward_equal and builder_equal, "Known collar copied forward/unknown field AST differs")
    retained += [here / "known_collar_reference_control.py", here / "known_collar_saved_baseline.py"]

    inputs_path = root / "experiments/evaluation/inputs/inputs.json"
    inputs = json.loads(inputs_path.read_text())
    track = next(item for item in inputs["tracks"] if item["id"] == "track3")
    source_path = Path(old["source_512_path"])
    require(source_path.resolve() == Path(track["source_512"]).resolve()
            and inputs["canvas_size"] == [512, 1152]
            and inputs["source_rect_xyxy"] == [0, 320, 512, 832], "Saved source differs from actual shared Track3 binding")
    original_jpg = Path(track["original_jpg"])
    require(file_sha(original_jpg) == track["original_jpg_sha256"], "Original Track3 JPEG changed")
    retained += [inputs_path, original_jpg]
    require(file_sha(source_path) == old["source_512_sha256"], "Current source PNG file changed")
    with Image.open(source_path) as image:
        require(image.size == (512, 512), "Source must be actual 512 square")
        source_rgb = np.asarray(image.convert("RGB")).copy()
        source_rgba = np.asarray(image.convert("RGBA")).copy()
    require(array_sha(source_rgb) == old["source_uint8_sha256"], "Source RGB pixels changed")
    require(array_sha(source_rgba) == old["source_rgba_uint8_sha256"] and np.all(source_rgba[..., 3] == 255),
            "Source opaque RGBA pixels changed")
    with Image.open(baseline_dir / "source_rgba.png") as image:
        require(np.array_equal(np.asarray(image.convert("RGBA")), source_rgba), "Saved actual-square source changed")
    retained += [source_path, baseline_dir / "source_rgba.png"]
    map_path = Path(old["map_paths"]["tangent-canny"])
    guide_path = Path(old["guide_path"])
    require(file_sha(map_path) == old["map_sha256"]["tangent-canny"], "Actual control map changed")
    require(file_sha(guide_path) == old["localized_hint_weights"]["guide_file_sha256"], "Actual unknown hint guide changed")
    for path in (map_path, guide_path):
        with Image.open(path) as image:
            require(image.size == (512, 1152), f"Actual map/guide size changed: {path}")
    retained += [map_path, guide_path]
    base_manifest = root / "experiments/qwen/base-manifest.json"
    adapter_manifest = root / "experiments/qwen/viggle-manifest.json"
    require(file_sha(base_manifest) == old["base_manifest_sha256"], "Base manifest changed")
    require(file_sha(adapter_manifest) == old["adapter_manifest_sha256"], "Adapter manifest changed")
    retained += [base_manifest, adapter_manifest]
    for filename, key in (("growing_localized_cpu_validation.json", "growing_cpu_validation_sha256"),
                          ("localized_cpu_validation.json", "localized_cpu_validation_sha256"),
                          ("reference_cpu_validation.json", "reference_cpu_validation_sha256"),
                          ("cpu_parity.json", "port_cpu_parity_sha256")):
        path = here / filename
        require(file_sha(path) == old[key] and json.loads(path.read_text()).get("status") == "passed",
                f"Frozen passed CPU gate changed: {filename}")
        retained.append(path)
    from run_trial import MODEL, PROMPT, REVISION, TURBO_REVISION
    from run_geometry_trial import ADAPTER, active_ids, six_sigmas
    from control import CHECKPOINT_BYTES, CHECKPOINT_FILENAME, CHECKPOINT_REVISION, UPSTREAM_REVISION, checkpoint_header
    prompt_path = root / "experiments/flux2/prompts.json"
    expected_prompt = PROMPT + " " + json.loads(prompt_path.read_text())["track3"]
    require(old["prompt"] == expected_prompt and old["prompt_style"] == "outpaint", "Saved original outpaint prompt changed")
    retained.append(prompt_path)
    require(old["model_revision"] == REVISION and old["adapter_revision"] == TURBO_REVISION,
            "Pinned base/adapter revisions changed")
    require(old["adapter"] == str(ADAPTER) and ADAPTER.stat().st_size == old["adapter_bytes"], "Pinned adapter path/size changed")
    require(old["control_revision"] == CHECKPOINT_REVISION and old["upstream_revision"] == UPSTREAM_REVISION,
            "Pinned control/upstream revisions changed")
    checkpoint = root / ".build/models/qwen/controlnet_union" / CHECKPOINT_FILENAME
    require(checkpoint.stat().st_size == CHECKPOINT_BYTES, "Pinned control checkpoint size changed")
    header, header_sha = checkpoint_header(checkpoint)
    actual_header = {"sha256": header_sha, "tensor_count": len(header)-int("__metadata__" in header),
        "payload_bytes": max(value["data_offsets"][1] for key, value in header.items() if key != "__metadata__")}
    require(actual_header == old["checkpoint_header"], "Actual pinned control checkpoint header changed")
    for name, expected in old["base_config_sha256"].items():
        path = MODEL / name
        require(file_sha(path) == expected, f"Base configuration changed: {name}")
        retained.append(path)

    shared = baseline_dir / "shared"
    specs = {
        "source_latents": ("reference_conditioning.npz", "source_latents", (1, 1024, 64)),
        "prompt_embeds": ("reference_conditioning.npz", "prompt_embeds", (1, 429, 4096)),
        "image_slots": ("reference_conditioning.npz", "image_slots", (429,), "bool"),
        "target_context": ("tangent-canny_contexts.npz", "target_context", (1, 2304, 129)),
        "padded_image_context": ("tangent-canny_contexts.npz", "padded_image_context", (1, 3328, 129)),
        "support": ("tangent-canny_contexts.npz", "structural_support", (1, 2304, 1), "bool"),
        "noise": ("target_noise.npz", "noise", (1, 2304, 64)),
        "edge_known_target": ("edge_known_target.npz", "known", (1, 1024, 64)),
        "edge_guess": ("edge_known_target.npz", "edge_guess", (1, 2304, 64)),
        "old_effective_target_hint": ("effective_hint_weights.npz", "target_weights", (1, 2304, 1)),
        "old_effective_joint_hint": ("effective_hint_weights.npz", "joint_weights", (1, 3501, 1)),
        "old_requested_target_hint": ("effective_hint_weights.npz", "requested_target_weights", (1, 2304, 1)),
    }
    arrays = {name: load_array(np, shared / spec[0], *spec[1:]) for name, spec in specs.items()}
    expected_hashes = {
        "source_latents": old["reference_encoding"]["source_prefix_sha256"],
        "prompt_embeds": old["prompt_embeds_sha256"], "image_slots": old["image_slots_bool_sha256"],
        "noise": old["noise_sha256"], "edge_known_target": old["edge_known_target_sha256"],
        "target_context": old["control_conditioning"]["tangent-canny"]["target_context_sha256_float32"],
        "padded_image_context": old["control_conditioning"]["tangent-canny"]["padded_image_context_sha256_float32"],
        "support": old["structural_support_metadata"]["support_tokens_bool_sha256"],
        "old_effective_target_hint": old["localized_hint_weights"]["effective_target_weights_float32_sha256"],
        "old_effective_joint_hint": old["localized_hint_weights"]["effective_joint_weights_float32_sha256"],
        "old_requested_target_hint": old["localized_hint_weights"]["target_weights_float32_sha256"],
    }
    for name, expected in expected_hashes.items():
        require(array_sha(arrays[name]) == expected, f"Saved inference tensor hash changed: {name}")
    for name, value in arrays.items():
        if name not in ("image_slots", "support", "old_requested_target_hint"):
            require(np.array_equal(value, bf16_values(np, value)), f"Saved inference tensor cannot restore losslessly as BF16: {name}")
    require(int(arrays["image_slots"].sum()) == 256 and old["layout"]["joint_prefix_tokens"] == 1197,
            "Saved source vision slots/prefix geometry changed")
    context, padded = arrays["target_context"], arrays["padded_image_context"]
    known = np.zeros((1, 2304, 1), dtype=np.float32); known[:, 640:1664] = 1
    require(np.array_equal(context[:, :, 64:65], known), "Known mask must cover target rows20:52 only")
    require(np.all(padded[:, :1024] == 0) and np.array_equal(padded[:, 1024:], context), "Reference IMAGE zero129 padding changed")
    require(np.all(arrays["old_effective_joint_hint"][:, :1197] == 0) and np.array_equal(arrays["old_effective_joint_hint"][:, 1197:], arrays["old_effective_target_hint"]), "Saved full-prefix hint gate changed")
    require(np.all(arrays["old_requested_target_hint"][:, 640:1664] == 0), "Saved original known target hints must be zero")
    require(np.array_equal(bf16_values(np, arrays["old_requested_target_hint"]), arrays["old_effective_target_hint"]), "Saved requested/effective BF16 hint fields disagree")
    support = arrays["support"]
    require(np.array_equal(support, np.load(shared / "support_tokens.npy", allow_pickle=False)), "Saved structural support differs between files")
    require(np.all(support[:, 640:1664]), "Known-source structural support changed")
    require(np.all(context[:, ~support.reshape(-1), :64] == 0), "Unsupported structural64 is not literal zero")
    edge_ids = ((np.clip(np.arange(72), 20, 51) - 20)[:, None] * 32 + np.arange(32)[None, :]).reshape(-1)
    require(np.array_equal(arrays["edge_guess"], arrays["edge_known_target"][:, edge_ids]), "Saved edge/frontier seed guess changed")
    guide_rgb = Image.open(guide_path).convert("RGB")
    from known_collar_reference_control import prepare_known_collar_hint_weights
    from sparse_conditioning import prepare_sparse_support
    pixels, requested, hint_info = prepare_known_collar_hint_weights(np, Image, 512, 1152,
        [0, 320, 512, 832], guide_rgb, 64, "cosine", 32, 1.0)
    requested_stored = np.load(shared / "hint_target_weights_float32.npy", allow_pickle=False)
    pixels_stored = np.load(shared / "hint_pixel_weights_float32.npy", allow_pickle=False)
    require(np.array_equal(requested_stored, arrays["old_requested_target_hint"]), "Saved requested hint files disagree")
    require(pixels_stored.shape == (1152, 512) and str(pixels_stored.dtype) == "float32" and np.isfinite(pixels_stored).all(), "Invalid saved pixel hint field")
    require(array_sha(pixels_stored) == old["localized_hint_weights"]["pixel_weights_float32_sha256"], "Saved pixel hint field changed")
    expected_pixels = pixels_stored.copy(); expected_pixels[320:352] = 1
    require(np.array_equal(pixels, expected_pixels), "Actual top32 builder changed the original unknown cosine64 field")
    support_pixels, rebuilt_support, _ = prepare_sparse_support(np, Image, 512, 1152,
        [0, 320, 512, 832], guide_rgb, old["structural_support_metadata"]["dilation_pixels"])
    require(np.array_equal(rebuilt_support, support), "Actual structural support builder differs from saved support")
    require(array_sha(support_pixels) == old["structural_support_metadata"]["support_pixels_bool_sha256"], "Actual structural support pixels changed")
    changed = np.flatnonzero((requested != arrays["old_requested_target_hint"]).reshape(-1))
    require(np.array_equal(changed, np.arange(640, 704)) and np.all(requested[:, 640:704] == 1), "Top32 change must be IDs640:704 only at scalar1")
    effective = arrays["old_effective_target_hint"].copy(); effective[:, 640:704] = 1
    require(np.array_equal(effective, bf16_values(np, requested)), "Top32 requested/effective BF16 rounding disagreement")
    unchanged = np.ones(2304, dtype=bool); unchanged[640:704] = False
    require(np.array_equal(effective[:, unchanged], arrays["old_effective_target_hint"][:, unchanged]), "Effective hint field changed outside top32")
    arrays.update(requested_top32_hint=requested, expected_effective_top32_hint=effective,
        expected_effective_top32_joint_hint=np.concatenate([np.zeros((1, 1197, 1), dtype=np.float32), effective], axis=1),
        sigmas=np.asarray(old["sigmas"], dtype=np.float32))
    require(np.array_equal(arrays["sigmas"], six_sigmas(np, 2304)), "Saved six shifted target-resolution sigmas changed")
    retained += [shared / spec[0] for spec in specs.values()]
    retained += [shared / name for name in ("support_tokens.npy", "support.json", "support.png",
        "hint_target_weights_float32.npy", "hint_pixel_weights_float32.npy", "hint_weights.json", "hint_weights.png")]
    for metadata_name, expected in (("support.json", old["structural_support_metadata"]), ("hint_weights.json", old["localized_hint_weights"])):
        saved_metadata = json.loads((shared / metadata_name).read_text())
        require(all(expected.get(key) == value for key, value in saved_metadata.items()), f"Saved field metadata disagrees: {metadata_name}")

    arms_report = {}
    expected_arms = [f"tangent-canny/scale{scale:g}" for scale in SCALES]
    require(set(old["runs"]) == set(expected_arms), "Saved growing suite must contain exactly scales0/.5/1")
    for scale, key in zip(SCALES, expected_arms):
        arm = old["runs"][key]; directory = baseline_dir / key
        arm_path = directory / "metrics.json"
        for name in ("raw", "composite"):
            require(Path(arm[f"{name}_path"]).resolve() == (directory / f"{name}.png").resolve(), f"Saved arm path binding changed: {key}:{name}")
        require(json.loads(arm_path.read_text()) == arm, f"Saved arm differs from suite: {key}")
        require(arm.get("status") == "success" and arm.get("source_pixels_exact") is True and arm.get("known_target_exact_at_sigma_zero") is True, f"Saved arm is incomplete: {key}")
        for name in ("seed", "steps", "cfg", "adapter", "adapter_revision", "adapter_rank", "adapter_scale", "adapter_baked", "prompt", "active_heights", "sigmas", "known_target_encoding_context", "base_prefix_cache_enabled", "control_prefix_cache_enabled", "compute_mode"):
            require(arm.get(name) == old.get(name), f"Saved suite/arm recipe mismatch: {key}:{name}")
        for name, expected in {"mode": "tangent-canny", "control_scale": scale, "initializer": "predicted-clean-active-frontier",
            "source_prefix_sha256": expected_hashes["source_latents"], "noise_sha256": expected_hashes["noise"],
            "known_target_sha256": expected_hashes["edge_known_target"],
            "context_metadata": old["control_conditioning"]["tangent-canny"], "localized_hint_weights": old["localized_hint_weights"],
            "known_bridge": "original edge-context target before/after every step", "future_targets_absent_both_chains": True,
            "source_prefix_exact_each_forward": True, "transformer_calls": 6, "base_block_calls": 192,
            "control_block_calls": 96, "target_token_forward_sum": 12288, "reference_image_token_forward_sum": 6144,
            "joint_query_token_forward_sum": 19470}.items():
            require(arm.get(name) == expected, f"Saved growing arm audit mismatch: {key}:{name}")
        require(len(arm["step_records"]) == 6, f"Saved arm lacks six step records: {key}")
        previous_count = 0; step_rows = []
        for step, (height, record) in enumerate(zip(HEIGHTS, arm["step_records"])):
            ids, window = active_ids(np, 512, 1152, height)
            expected = {"step": step+1, "sigma": old["sigmas"][step], "next_sigma": old["sigmas"][step+1],
                "active_height": height, "active_window_xyxy": list(window), "target_tokens_forwarded": len(ids),
                "future_target_tokens_absent": 2304-len(ids), "newly_activated_tokens": len(ids)-previous_count,
                "input_image_latent_tokens": 1024+len(ids), "control_image_rows": 1024+len(ids),
                "base_joint_query_tokens": 1197+len(ids), "control_joint_query_tokens": 1197+len(ids),
                "prefix_recomputed_both_chains": True, "base_blocks": 32, "control_blocks": 16,
                "known_source_target_hint_weights_zero": True, "source_prefix_input_exact": True,
                "active_context_sha256": array_sha(context[:, ids]),
                "active_hint_weights_sha256": array_sha(arrays["old_effective_target_hint"][:, ids])}
            for name, value in expected.items():
                require(record.get(name) == value, f"Saved growing step mismatch: {key}:step{step+1}:{name}")
            step_rows.append({**expected, "absolute_target_rows_half_open": [int(ids[0]//32), int(ids[-1]//32+1)],
                              "absolute_target_ids_sha256": array_sha(ids)})
            previous_count = len(ids)
        final_path = directory / "final_latents.npz"
        final = load_array(np, final_path, "latents", (1, 2304, 64))
        require(array_sha(final) == arm["final_latents_sha256"] and np.array_equal(final, bf16_values(np, final)), f"Saved final latent hash/BF16 mismatch: {key}")
        require(np.array_equal(final[:, 640:1664], arrays["edge_known_target"]), f"Saved sigma-zero edge-known bridge mismatch: {key}")
        for name in ("raw", "composite"):
            image_path = directory / f"{name}.png"
            require(file_sha(image_path) == arm[f"{name}_sha256"], f"Saved {name} changed: {key}")
            with Image.open(image_path) as image:
                require(image.size == (512, 1152), f"Saved {name} dimensions changed: {key}")
                if name == "composite":
                    require(np.array_equal(np.asarray(image.convert("RGB"))[320:832], source_rgb), f"Saved source composite pixels changed: {key}")
            retained.append(image_path)
        with Image.open(directory / "raw.png") as raw_image, Image.open(directory / "composite.png") as composite_image:
            raw_rgb, composite_rgb = np.asarray(raw_image.convert("RGB")), np.asarray(composite_image.convert("RGB"))
            require(np.array_equal(raw_rgb[:320], composite_rgb[:320]) and np.array_equal(raw_rgb[832:], composite_rgb[832:]), f"Saved composite exterior differs from raw: {key}")
        require([p["step"] for p in arm["previews"]] == [2, 4, 6], f"Saved actual previews missing: {key}")
        for preview in arm["previews"]:
            step = preview["step"]-1; height = HEIGHTS[step]
            ids, window = active_ids(np, 512, 1152, height)
            path = directory / "previews" / f"step{step+1:02d}_predicted_clean.npz"
            require(Path(preview["latent_path"]).resolve() == path.resolve(), f"Saved preview path binding changed: {key}")
            clean = load_array(np, path, "latents", (1, len(ids), 64))
            with np.load(path, allow_pickle=False) as data:
                require(set(data.files) == {"latents", "final_canvas_ids", "sigma", "window_xyxy"}
                    and str(data["final_canvas_ids"].dtype) == "int32" and str(data["window_xyxy"].dtype) == "int64"
                    and str(data["sigma"].dtype) == "float32", f"Saved preview keys/dtypes changed: {key}:step{step+1}")
                require(np.array_equal(data["final_canvas_ids"], ids) and np.array_equal(data["window_xyxy"], window)
                    and data["sigma"].shape == () and float(data["sigma"]) == old["sigmas"][step], f"Saved preview absolute IDs/sigma changed: {key}:step{step+1}")
            known_positions = np.searchsorted(ids, np.arange(640, 1664))
            require(np.array_equal(clean[:, known_positions], arrays["edge_known_target"]), f"Saved preview known target changed: {key}:step{step+1}")
            require(np.array_equal(clean, bf16_values(np, clean)), f"Saved preview BF16 representation changed: {key}:step{step+1}")
            require(preview["active_height"] == height and preview["active_window_xyxy"] == list(window), f"Saved preview geometry changed: {key}")
            if step == 5:
                require(np.array_equal(clean, final), f"Saved terminal predicted-clean differs from final: {key}")
            retained.append(path)
            for name in ("raw", "composite"):
                image_path = path.with_name(path.stem + f"_{name}.png")
                require(Path(preview[f"{name}_path"]).resolve() == image_path.resolve(), f"Saved preview image path binding changed: {key}:{name}")
                require(file_sha(image_path) == preview[f"{name}_sha256"], f"Saved preview pixels changed: {key}:{name}")
                retained.append(image_path)
        retained += [arm_path, final_path]
        arms_report[key] = {"control_scale": scale, "steps": step_rows, "source_pixels_exact": True,
            "saved_final_edge_known_bridge_exact": True, "final_latents_sha256": arm["final_latents_sha256"],
            "raw_sha256": arm["raw_sha256"], "composite_sha256": arm["composite_sha256"]}

    report = {"status": "passed", "baseline_directory": str(baseline_dir), "baseline_arms": expected_arms,
        "new_arms": [f"{key}/top32" for key in expected_arms], "actual_gpu_baseline_rerun_performed": False,
        "comparison": "saved completed growing6 same scales versus top32; common inference tensors restored exactly",
        "recipe_keys_exact": list(RECIPE_KEYS), "persisted_array_sha256": {k: array_sha(v) for k, v in arrays.items()},
        "persisted_file_sha256": {str(p.resolve()): file_sha(p) for p in sorted(set(retained))},
        "saved_arms": arms_report, "frozen_baseline_runner_sha256": file_sha(frozen_runner),
        "runtime_dependency_hashes": actual_runtime, "frozen_forward_ast_exact_after_name_docstring": forward_equal,
        "old_unknown_hint_builder_ast_exact": builder_equal, "bf16_inference_value_roundtrip_exact": True,
        "context129_restored_including_mask_and_masked_source": True, "edge_known_codec_distinct_from_control129": True,
        "source_reference_and_prompt_embeddings_restored": True, "noise_restored_without_rng_draw": True,
        "known_bridge_code_unchanged": True, "frontier_insertion_code_unchanged": True,
        "hint_changed_token_ids": changed.tolist(), "hint_changed_packed_rows": [20, 21],
        "hint_prefix_weights_zero": True, "all_other_effective_hint_tokens_exact": True,
        "unknown_cosine64_field_exact": True, "source_pixels_exact": True,
        "hint_metadata": hint_info, "inference_encodes_planned": 0, "inference_rng_draws_planned": 0,
        "gpu_determinism_of_repeated_baseline_unmeasured": True, "weight_payload_hash_reverified": False,
        "older_cached_geometry_comparison": old["runs"][expected_arms[0]].get("cached_geometry_scale_zero_comparison"),
        "older_cached_pipeline_exact_claim": False, "original_jpeg_sha256": track["original_jpg_sha256"],
        "checkpoint_header_reverified": True,
        "saved_preview_latent_historical_hashes_available": False,
        "unsaved_gpu_trace_claim": False}
    return arrays, report, old
