#!/usr/bin/env python3
"""Saved-input EARLY-FRONTIER Turbo/control growing probe with source top32 hints.

Only completed ControlNet hints on absolute target IDs640:704 gain scalar1.
The saved unknown cosine64 field and zero prefix gate remain exact. Both chains
process active targets only at heights640,896,1152x4; no future targets leak in.
Original source/prompt/noise/context129/edge-known codec/sigmas are restored from
the completed growing trial. There are no inference encodes or new RNG draws.
This is an untrained Turbo/control combination, not a quality claim. The frozen
uncached growing baseline is reused without rerunning scale0 or any other arm.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_trial import MODEL, REVISION, TURBO_REVISION
from run_geometry_trial import ADAPTER, active_ids, active_layout, insertion_state
from run_growing_localized_control_trial import HEIGHTS, PREVIEW_STEPS
from known_collar_saved_baseline import assert_disjoint_output, assert_saved_files_unchanged


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sampling_math_ast_parity():
    """Compare the retained native growth arithmetic/calls, excluding audits."""
    retained = {"ids", "index", "added", "active", "active_context", "known_mask_np", "known_mask", "known_clean",
                "latents", "before", "model_input", "timestep", "prediction", "advanced", "clean", "previous_clean", "previous_ids"}
    def selected(path):
        tree = ast.parse(Path(path).read_text())
        loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For)
                    and isinstance(node.target, ast.Tuple)
                    and [getattr(item, "id", None) for item in node.target.elts] == ["step", "active_height"])
        nodes = []
        for node in loop.body:
            if isinstance(node, ast.Assign) and any(isinstance(item, ast.Name) and item.id in retained
                    for target in node.targets for item in ast.walk(target)):
                nodes.append(node)
            elif isinstance(node, ast.If) and any(isinstance(item, ast.Name) and item.id == "previous_ids" for item in ast.walk(node.test)):
                nodes.append(node)
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and ast.unparse(node.value) == "mx.eval(latents, clean)":
                nodes.append(node)
        return nodes
    before = selected(Path(__file__).with_name("run_growing_localized_control_trial.py"))
    after = selected(Path(__file__))
    return {"exact": ast.dump(ast.Module(body=before, type_ignores=[]), include_attributes=False)
                       == ast.dump(ast.Module(body=after, type_ignores=[]), include_attributes=False),
            "retained_sampling_statements": len(before),
            "scope": "activation/insertion, absolute layout/gathers, edge bridges, all16 forward, timestep, Euler and clean state"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=("track3",), default="track3")
    parser.add_argument("--steps", type=int, choices=(6,), default=6)
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    parser.add_argument("--control-scales", "--scales", type=float, nargs="+", choices=(0.5, 1.0), default=[0.5, 1.0])
    parser.add_argument("--saved-baseline", type=Path, default=ROOT / "experiments/qwen/controlnet_runs/2026-10-05/track3-growing-localized6")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--max-seconds", type=int, default=900, help="Cooperative checks before phases and steps")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if len(set(args.control_scales)) != len(args.control_scales): parser.error("Control scales must be unique")
    if not 1 <= args.max_seconds <= 3600: parser.error("Time bound must be between1 and3600 seconds")
    args.modes = ["tangent-canny"]
    baseline_directory = args.saved_baseline.expanduser().resolve()
    suite_dir = (args.output or ROOT / "experiments/qwen/controlnet_runs/2026-10-05/track3-growing-top32-6step").expanduser().resolve()
    assert_disjoint_output(suite_dir, baseline_directory)
    if suite_dir.exists() and any(path.name not in {"launch.json", "process.log"} for path in suite_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing trial artifacts: {suite_dir}")
    begun = time.perf_counter()
    suite = {"status": "preflight" if args.preflight else "starting", "track": args.track, "seed": 42, "steps": 6,
             "runs": {}, "phases": {}, "runner_sha256": file_sha256(__file__),
             "experimental_turbo_control_combination": True, "official_control_recipe": False,
             "known_collar_arms": ["top32"], "known_collar_scalar_weight": 1.0,
             "control_scales": args.control_scales, "max_seconds": args.max_seconds,
             "wall_bound_scope": "cooperative phase/step checks; a synchronous kernel may overrun one check",
             "actual_saved_gpu_baseline_reused": True, "actual_gpu_baseline_rerun_performed": False,
             "inference_encodes_planned": 0, "inference_rng_draws_planned": 0,
             "only_hint_field_changed": True, "model_weights_loaded": False, "offline_environment": True,
             "hint_localization_changes_compute": False, "known_target_hidden_trace_preserved": False,
             "visual_quality": "requires direct inspection; mechanical validation is not quality acceptance",
             "checkpoint_payload_hash_reverified": False, "base_weight_payload_hashes_reverified": False,
             "planned_total_nfe": 6 * len(args.control_scales), "planned_total_target_token_forwards": 12288 * len(args.control_scales)}

    def budget():
        if time.perf_counter() - begun > args.max_seconds: raise TimeoutError("Growing top32 cooperative time bound exceeded")

    def write(name="growing_top32_metrics.json"):
        suite["elapsed_seconds"] = time.perf_counter() - begun
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        suite["process_peak_rss_bytes"] = rss if sys.platform == "darwin" else rss * 1024
        suite_dir.mkdir(parents=True, exist_ok=True)
        temporary = (suite_dir / name).with_suffix(".tmp")
        temporary.write_text(json.dumps(suite, indent=2) + "\n"); temporary.replace(suite_dir / name)

    def log(phase, **data):
        budget(); suite["phase"] = phase; suite.update(data); write()
        print(json.dumps({"phase": phase, "elapsed_seconds": round(suite["elapsed_seconds"], 2), **data}), flush=True)

    try:
        import mlx.core as mx
        if args.preflight: mx.set_default_device(mx.cpu)
        import numpy as np
        from PIL import Image
        from mlx.utils import tree_flatten
        from mflux.models.common.vae.vae_util import VAEUtil
        from mflux.models.common.vae.tiling_config import TilingConfig
        from mflux.models.qwen21.reference import QwenImage21Edit
        from mflux.models.qwen21.reference.latent_creator.qwen_image21_latent_creator import QwenImage21LatentCreator
        from mflux.models.qwen21.reference.model.qwen_image21_transformer.layout import QwenImage21Layout
        from control import CHECKPOINT_BYTES, CHECKPOINT_FILENAME, CHECKPOINT_SHA256, checkpoint_header, load_control_branch
        from reference_control import assemble_reference_input, validate_reference_layout
        from growing_localized_control import gather_active_conditioning, edge_known_flow_bridge, predicted_clean_with_known
        from localized_reference_control import localized_reference_controlled_forward, runtime_dependency_hashes
        from growing_known_collar_saved_inputs import validate_saved_growing_baseline

        def tensor_array(value):
            mx.eval(value); return np.asarray(value.astype(mx.float32)).copy()

        def tensor_sha(value):
            return hashlib.sha256(tensor_array(value).tobytes()).hexdigest()

        arrays, saved_report, old_suite = validate_saved_growing_baseline(np, Image, baseline_directory, output=suite_dir)
        frozen_protocol = ("width", "height", "source_rect_xyxy", "prompt", "prompt_style", "source_512_path", "source_512_sha256",
            "source_uint8_sha256", "source_rgba_uint8_sha256", "model_repo", "model_revision", "base_quantization", "control_quantization",
            "cfg", "adapter", "adapter_revision", "adapter_rank", "adapter_scale", "adapter_baked", "active_heights",
            "known_target_encoding_context", "known_target_and_control_source_codec_contexts_distinct", "compute_mode", "growing_compute",
            "prefix_cache_enabled", "base_prefix_cache_enabled", "control_prefix_cache_enabled", "old_cached_implementation_approximation",
            "scheduler_resolution_basis", "sigmas", "noise_sha256", "edge_known_target_sha256", "map_paths", "map_sha256", "guide_path",
            "structural_modes", "structural_support", "structural_support_metadata", "hint_shape", "hint_radius_pixels", "known_bridge_extension",
            "base_manifest_sha256", "base_config_sha256", "checkpoint_header", "control_revision", "upstream_revision",
            "reference_encoding", "control_conditioning", "source_conditioning", "reference_control_rows", "layout")
        suite.update({key: old_suite[key] for key in frozen_protocol})
        suite["runtime_dependency_hashes"] = runtime_dependency_hashes()
        if suite["runtime_dependency_hashes"] != old_suite["runtime_dependency_hashes"]: raise ValueError("Actual runtime differs from saved baseline")
        suite["saved_baseline_validation"] = saved_report
        suite["sampling_math_ast_parity"] = sampling_math_ast_parity()
        if not suite["sampling_math_ast_parity"]["exact"]: raise AssertionError("Frozen growing sampling arithmetic/calls changed")
        suite["source_code_sha256"] = {name: file_sha256(Path(__file__).with_name(name)) for name in
            ("run_growing_localized_control_trial.py", "growing_localized_control.py", "localized_reference_control.py",
             "reference_control.py", "control.py", "conditioning.py", "growing_known_collar_saved_inputs.py", "test_growing_known_collar_cpu.py")}
        suite["untrained_distribution_changes"] = old_suite["untrained_distribution_changes"] + ["completed trained hints enabled on known source top32 rows only"]
        suite["known_collar_field"] = {"absolute_target_ids_half_open": [640, 704], "packed_rows": [20, 21],
            "pixel_rect_xyxy": [0, 320, 512, 352], "scalar_weight_before_global_scale": 1.0,
            "unknown_effective_weights_exact_against_saved": True, "prefix_weights_zero": True,
            "all_other_known_weights_zero": True, "gated_after_all16_control_hints": True, "sparse_compute": False}
        validation_path = Path(__file__).with_name("growing_known_collar_cpu_validation.json")
        validation = json.loads(validation_path.read_text()) if validation_path.is_file() else None
        gate_current = bool(validation and validation.get("status") == "passed"
            and validation.get("helper_sha256") == suite["source_code_sha256"]["growing_known_collar_saved_inputs.py"]
            and validation.get("script_sha256") == suite["source_code_sha256"]["test_growing_known_collar_cpu.py"]
            and validation.get("frozen_growing_helper_sha256") == suite["source_code_sha256"]["growing_localized_control.py"]
            and validation.get("frozen_localized_helper_sha256") == suite["source_code_sha256"]["localized_reference_control.py"]
            and validation.get("runtime_dependency_hashes") == suite["runtime_dependency_hashes"])
        suite["growing_top32_cpu_gate_current"] = gate_current
        suite["growing_top32_cpu_validation_sha256"] = file_sha256(validation_path) if validation_path.is_file() else None
        checkpoint = (args.checkpoint or ROOT / ".build/models/qwen/controlnet_union" / CHECKPOINT_FILENAME).expanduser().resolve()
        manifest_path = ROOT / "experiments/qwen/base-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        missing = [str(MODEL / item["name"]) for item in manifest["files"] if not (MODEL / item["name"]).is_file()
                   or (MODEL / item["name"]).stat().st_size != item["expected_bytes"]]
        if not ADAPTER.is_file(): missing.append(str(ADAPTER))
        if not checkpoint.is_file() or checkpoint.stat().st_size != CHECKPOINT_BYTES: missing.append(str(checkpoint))
        if manifest["revision"] != REVISION or suite["model_revision"] != REVISION or suite["adapter_revision"] != TURBO_REVISION:
            raise ValueError("Pinned base/Turbo revision drift")
        if file_sha256(manifest_path) != suite["base_manifest_sha256"]: raise ValueError("Base manifest differs from saved baseline")
        if checkpoint.is_file() and checkpoint.stat().st_size == CHECKPOINT_BYTES:
            header, header_sha = checkpoint_header(checkpoint)
            current_header = {"sha256": header_sha, "tensor_count": len(header) - int("__metadata__" in header),
                "payload_bytes": max(value["data_offsets"][1] for key, value in header.items() if key != "__metadata__")}
            if current_header != suite["checkpoint_header"]: raise ValueError("Control checkpoint header differs from saved baseline")
        suite["missing_files"] = missing
        suite_dir.mkdir(parents=True, exist_ok=True)
        support_dir = suite_dir / "shared"; support_dir.mkdir(exist_ok=True)
        width, height = suite["width"], suite["height"]; rect = suite["source_rect_xyxy"]; prompt = suite["prompt"]
        source_rgb = Image.open(suite["source_512_path"]).convert("RGB"); source = source_rgb.convert("RGBA")
        source_bytes = np.asarray(source_rgb).copy(); source.save(suite_dir / "source_rgba.png")
        np.savez_compressed(support_dir / "reference_conditioning.npz", source_latents=arrays["source_latents"], prompt_embeds=arrays["prompt_embeds"], image_slots=arrays["image_slots"])
        np.savez_compressed(support_dir / "edge_known_target.npz", known=arrays["edge_known_target"], edge_guess=arrays["edge_guess"])
        np.savez_compressed(support_dir / "target_noise.npz", noise=arrays["noise"])
        np.savez_compressed(support_dir / "tangent-canny_contexts.npz", target_context=arrays["target_context"], padded_image_context=arrays["padded_image_context"], structural_support=arrays["support"])
        np.savez_compressed(support_dir / "effective_hint_weights.npz", target_weights=arrays["expected_effective_top32_hint"],
            joint_weights=np.concatenate([np.zeros((1, old_suite["layout"]["joint_prefix_tokens"], 1), dtype=np.float32), arrays["expected_effective_top32_hint"]], axis=1),
            requested_target_weights=arrays["requested_top32_hint"], old_target_weights=arrays["old_effective_target_hint"])
        np.save(support_dir / "support_tokens.npy", arrays["support"])
        (support_dir / "support.png").write_bytes((baseline_directory / "shared/support.png").read_bytes())
        pixel_hints = np.load(baseline_directory / "shared/hint_pixel_weights_float32.npy", allow_pickle=False).copy()
        pixel_hints[320:352] = 1
        if not np.array_equal(pixel_hints[::16, ::16].reshape(1,2304,1), arrays["requested_top32_hint"]):
            raise AssertionError("Saved diagnostic pixel field does not reproduce the declared target hints")
        np.save(support_dir / "hint_pixel_weights_float32.npy", pixel_hints)
        np.save(support_dir / "hint_target_weights_float32.npy", arrays["requested_top32_hint"])
        Image.fromarray(np.clip(pixel_hints * 255, 0, 255).round().astype(np.uint8), mode="L").save(support_dir / "hint_weights.png")
        suite["prompt_embeds_sha256"] = hashlib.sha256(arrays["prompt_embeds"].tobytes()).hexdigest()
        suite["image_slots_bool_sha256"] = hashlib.sha256(arrays["image_slots"].tobytes()).hexdigest()
        suite["localized_hint_weights"] = dict(old_suite["localized_hint_weights"])
        suite["localized_hint_weights"].update({"source_roi_always_zero": False, "known_collar_top32_only": True,
            "geometry_source": "same saved input guide plus declared source top32; no output-derived geometry",
            "positive_hint_pixels": int((pixel_hints > 0).sum()),
            "pixel_weights_float32_sha256": hashlib.sha256(pixel_hints.tobytes()).hexdigest(),
            "display_png_sha256": file_sha256(support_dir / "hint_weights.png"),
            "positive_hint_target_tokens": int((arrays["expected_effective_top32_hint"] > 0).sum()),
            "target_hint_weight_sum": float(arrays["requested_top32_hint"].sum()),
            "effective_target_hint_weight_sum": float(arrays["expected_effective_top32_hint"].sum()),
            "effective_target_weights_float32_sha256": hashlib.sha256(arrays["expected_effective_top32_hint"].tobytes()).hexdigest(),
            "target_weights_float32_sha256": hashlib.sha256(arrays["requested_top32_hint"].tobytes()).hexdigest(),
            "effective_joint_weights_float32_sha256": hashlib.sha256(np.concatenate([np.zeros((1, old_suite["layout"]["joint_prefix_tokens"], 1), dtype=np.float32), arrays["expected_effective_top32_hint"]], axis=1).tobytes()).hexdigest(),
            "pixel_weights_metadata_scope": "saved unknown cosine64 unchanged plus declared source top32 collar",
            "unknown_field_saved_effective_exact": True, "top32_hint_absolute_ids": [640, 704]})
        (support_dir / "hint_weights.json").write_text(json.dumps(suite["localized_hint_weights"], indent=2) + "\n")
        (support_dir / "support.json").write_text(json.dumps(suite["structural_support_metadata"], indent=2) + "\n")
        suite["shared_file_sha256"] = {path.name: file_sha256(path) for path in support_dir.iterdir() if path.is_file()}
        if args.preflight:
            suite.update({"status": "preflight_files_ready" if gate_current and not missing else "preflight_files_incomplete", "device": "CPU only"})
            assert_saved_files_unchanged(saved_report); write("preflight.json")
            print(json.dumps({"status": suite["status"], "cpu_gate_current": gate_current, "missing": missing,
                "hint_changed_ids": [640,704], "planned_nfe_per_arm": 6, "target_tokens_per_arm": 12288, "output": str(suite_dir)}, indent=2), flush=True)
            return
        if not gate_current or missing: raise RuntimeError("Current passed growing top32 CPU gate and complete pinned local files required")
        assert_saved_files_unchanged(saved_report)
        mx.set_default_device(mx.gpu); mx.set_cache_limit(512 * 1024**2); mx.set_memory_limit(28 * 1024**3)
        suite["mlx_version"] = mx.__version__
        if suite["mlx_version"] != old_suite["mlx_version"]: raise ValueError("MLX version differs from saved baseline")
        suite["metal_device"] = mx.device_info(); log("loading_base", status="running"); start = time.perf_counter()
        model = QwenImage21Edit(model_path=str(MODEL), quantize=4, lora_paths=[str(ADAPTER)], lora_scales=[1.0], bake_lora=False)
        suite["model_weights_loaded"] = True
        del model.text_encoder; model.text_encoder = None; mx.clear_cache()
        suite["phases"]["base_load_seconds"] = time.perf_counter() - start
        def restore(name):
            value = mx.array(arrays[name]).astype(mx.bfloat16)
            if not np.array_equal(tensor_array(value), arrays[name]): raise AssertionError(f"Saved BF16 restoration changed {name}")
            return value
        source_latents, prompt_embeds = restore("source_latents"), restore("prompt_embeds")
        slots = mx.array(arrays["image_slots"], dtype=mx.bool_)
        if not np.array_equal(np.asarray(slots), arrays["image_slots"]): raise AssertionError("Saved slots changed")
        noise, edge_guess = restore("noise"), restore("edge_guess")
        edge_known_np = arrays["edge_known_target"]
        hint_weights = restore("expected_effective_top32_hint")
        contexts = {"tangent-canny": restore("target_context")}; context_info = suite["control_conditioning"]
        source_prefix_np = arrays["source_latents"]; source_prefix_sha = suite["reference_encoding"]["source_prefix_sha256"]
        sigmas_np = arrays["sigmas"]; sigmas = mx.array(sigmas_np, dtype=mx.float32)
        layout = QwenImage21Layout.create(slots, [(1,32,32),(1,72,32)], model.transformer.axes)
        if validate_reference_layout(np, layout, 1024, 2304) != old_suite["layout"]: raise ValueError("Restored full layout differs from saved baseline")
        suite["inference_encodes_performed"] = 0; suite["inference_rng_draws_performed"] = 0
        log("loading_control"); start = time.perf_counter()
        branch, branch_info = load_control_branch(checkpoint, quantize=None)
        if branch_info["expected_checkpoint_sha256"] != CHECKPOINT_SHA256: raise AssertionError("Pinned control identity changed")
        if len(model.transformer.transformer_blocks) != 32 or len(branch.control_blocks) != 16: raise AssertionError("Expected official32base/16control blocks")
        suite["control_checkpoint"] = branch_info; suite["phases"]["control_load_seconds"] = time.perf_counter() - start
        suite["base_component_parameter_bytes"] = {name: sum(value.nbytes for _,value in tree_flatten(getattr(model,name).parameters())) for name in ("transformer","vae")}
        mx.set_memory_limit(25 * 1024**3)
        outputs = {}
        previews_by_arm = {}
        baseline_dir = baseline_directory
        suite["saved_growing_baseline"] = str(baseline_dir)
        assert_saved_files_unchanged(saved_report)
        for mode in args.modes:
            for scale in args.control_scales:
                arm = f"scale{scale:g}"
                key = f"{mode}/{arm}"
                directory = suite_dir / mode / arm
                (directory / "previews").mkdir(parents=True, exist_ok=True)
                latents = previous_ids = previous_clean = None
                captured = []
                record = {"status": "running", "mode": mode, "control_scale": scale,
                    "steps": 6, "seed": args.seed, "cfg": 1.0, "adapter": str(ADAPTER),
                    "adapter_revision": TURBO_REVISION, "adapter_rank": 256, "adapter_scale": 1.0, "adapter_baked": False,
                    "prompt": prompt, "active_heights": list(HEIGHTS), "initializer": "predicted-clean-active-frontier",
                    "source_prefix_sha256": source_prefix_sha, "noise_sha256": suite["noise_sha256"],
                    "known_target_sha256": suite["edge_known_target_sha256"],
                    "known_target_encoding_context": "edge-padded-full-canvas-center-crop",
                    "context_metadata": context_info[mode], "localized_hint_weights": suite["localized_hint_weights"],
                    "sigmas": suite["sigmas"], "experimental_turbo_control_combination": True,
                    "official_control_recipe": False, "known_bridge": "original edge-context target before/after every step",
                    "base_prefix_cache_enabled": False, "control_prefix_cache_enabled": False,
                    "old_cached_implementation_approximation": True,
                    "known_collar_top32_only": True, "global_control_scale_applied_once_to_full_field": True,
                    "scale_zero_extra_branch_cost": False,
                    "compute_mode": suite["compute_mode"], "future_targets_absent_both_chains": True,
                    "transformer_calls": 0, "base_block_calls": 0, "control_block_calls": 0,
                    "target_token_forward_sum": 0, "reference_image_token_forward_sum": 0,
                    "joint_query_token_forward_sum": 0, "source_prefix_exact_each_forward": True,
                    "step_records": [], "previews": [], "phases": {}}
                suite["runs"][key] = record
                mx.reset_peak_memory(); log("denoising_start", current_arm=key)
                wall_start = time.perf_counter(); core_seconds = capture_seconds = 0.0
                for step, active_height in enumerate(HEIGHTS):
                    budget(); step_start = time.perf_counter()
                    ids, window = active_ids(np, width, height, active_height)
                    index = mx.array(ids, dtype=mx.int32)
                    added = len(ids) if previous_ids is None else len(ids) - len(previous_ids)
                    if previous_ids is None or len(ids) != len(previous_ids):
                        latents = insertion_state(mx, np, ids, previous_ids, latents, previous_clean,
                                                  edge_guess, noise, sigmas[step], True)
                    active = active_layout(mx, np, QwenImage21Layout, slots, (1, 32, 32),
                                           layout, ids, active_height, width, model.transformer.axes)
                    active_context, padded_active, active_hints = gather_active_conditioning(np,
                        contexts[mode], hint_weights, ids, reference_tokens=1024)
                    known_mask_np = (ids // 32 >= 20) & (ids // 32 < 52)
                    known_mask = mx.array(known_mask_np[None, :, None])
                    known_clean, active_noise = edge_guess[:, index], noise[:, index]
                    latents = edge_known_flow_bridge(latents, known_mask, known_clean, active_noise, sigmas[step])
                    before = latents
                    model_input = assemble_reference_input(source_latents, latents)
                    if model_input.shape[1] != 1024 + len(ids) or padded_active.shape[1] != model_input.shape[1]:
                        raise AssertionError("Inactive/future image rows leaked into a denoiser chain")
                    if not np.array_equal(tensor_array(model_input[:, :1024]), source_prefix_np):
                        raise AssertionError("Fixed reference source changed")
                    collar = (ids >= 640) & (ids < 704)
                    effective_active = tensor_array(active_hints)
                    if int(collar.sum()) != 64 or np.any(effective_active[0, collar, 0] != 1):
                        raise AssertionError("Active top32 collar must contain exactly64 scalar-one hints")
                    if np.any(effective_active[0, known_mask_np & ~collar, 0] != 0):
                        raise AssertionError("Known source hints outside top32 must remain literal zero")
                    if not np.array_equal(effective_active[:, ~collar], arrays["old_effective_target_hint"][:, ids[~collar]]):
                        raise AssertionError("Effective hints outside the intended absolute collar changed")
                    timestep = (sigmas[step:step + 1] * 1000).astype(latents.dtype) / 1000
                    record["transformer_calls"] += 1; record["base_block_calls"] += 32; record["control_block_calls"] += 16
                    record["target_token_forward_sum"] += len(ids)
                    record["reference_image_token_forward_sum"] += 1024
                    record["joint_query_token_forward_sum"] += active.prefix_length + len(ids)
                    prediction = localized_reference_controlled_forward(model.transformer, branch, model_input, prompt_embeds,
                        timestep, active, padded_active, target_hint_weights=active_hints, control_scale=scale, cache=None)
                    if prediction.shape != latents.shape:
                        raise AssertionError("Denoiser predicted inactive target rows")
                    advanced = (latents.astype(mx.float32) + (sigmas[step + 1] - sigmas[step]) * prediction.astype(mx.float32)).astype(prompt_embeds.dtype)
                    latents = edge_known_flow_bridge(advanced, known_mask, known_clean, active_noise, sigmas[step + 1])
                    clean = predicted_clean_with_known(before, prediction, known_mask, known_clean, sigmas[step])
                    mx.eval(latents, clean)
                    if not bool(mx.all(mx.isfinite(latents))) or not bool(mx.all(mx.isfinite(clean))):
                        raise FloatingPointError("Nonfinite growing target/clean state")
                    previous_clean = clean
                    step_seconds = time.perf_counter() - step_start; core_seconds += step_seconds
                    row = {"step": step + 1, "sigma": float(sigmas_np[step]), "next_sigma": float(sigmas_np[step + 1]),
                        "active_height": active_height, "active_window_xyxy": window, "target_tokens_forwarded": len(ids),
                        "future_target_tokens_absent": 2304 - len(ids), "newly_activated_tokens": added,
                        "input_image_latent_tokens": model_input.shape[1], "control_image_rows": padded_active.shape[1],
                        "base_joint_query_tokens": active.prefix_length + len(ids), "control_joint_query_tokens": active.prefix_length + len(ids),
                        "prefix_recomputed_both_chains": True, "base_blocks": 32, "control_blocks": 16,
                        "known_source_hint_zero_outside_top32": True, "top32_hint_scalar_one_tokens": 64,
                        "top32_hint_absolute_ids": [640, 704], "source_prefix_input_exact": True,
                        "active_context_sha256": tensor_sha(active_context), "active_hint_weights_sha256": tensor_sha(active_hints),
                        "core_seconds": step_seconds}
                    record["step_records"].append(row)
                    if step + 1 in PREVIEW_STEPS:
                        capture_start = time.perf_counter(); clean_np = tensor_array(clean)
                        path = directory / "previews" / f"step{step + 1:02d}_predicted_clean.npz"
                        np.savez_compressed(path, latents=clean_np, final_canvas_ids=ids, sigma=sigmas_np[step], window_xyxy=np.asarray(window))
                        captured.append((step + 1, active_height, window, clean_np))
                        record["previews"].append({"step": step + 1, "active_height": active_height,
                            "active_window_xyxy": window, "latent_path": str(path),
                            "definition": "pre-step z_sigma-sigma*velocity; edge-known clean target restored"})
                        capture_seconds += time.perf_counter() - capture_start
                    previous_ids = ids
                    (directory / "metrics.json").write_text(json.dumps(record, indent=2) + "\n")
                    print(json.dumps({"phase": "denoising", "arm": key, **row}), flush=True)
                final = tensor_array(latents)
                if not np.array_equal(final[:, 20 * 32:52 * 32], edge_known_np):
                    raise AssertionError("Sigma-zero known target differs from original edge-context codec")
                if not np.array_equal(tensor_array(source_latents), source_prefix_np):
                    raise AssertionError("Growth changed source reference latents")
                if record["target_token_forward_sum"] != 12288 or record["transformer_calls"] != 6:
                    raise AssertionError("Actual EARLY-FRONTIER NFE/token count mismatch")
                np.savez_compressed(directory / "final_latents.npz", latents=final)
                record["final_latents_sha256"] = hashlib.sha256(final.tobytes()).hexdigest()
                record["known_target_exact_at_sigma_zero"] = True
                record["phases"].update({"denoising_seconds": core_seconds,
                    "core_loop_wall_seconds": time.perf_counter() - wall_start, "preview_capture_seconds": capture_seconds})
                record["denoising_mlx_peak_bytes"] = mx.get_peak_memory()
                old_arm = old_suite["runs"][key]
                with np.load(baseline_directory / key / "final_latents.npz", allow_pickle=False) as retained:
                    old_final = retained["latents"].copy()
                difference = np.abs(old_final - final)
                record["saved_same_scale_baseline_comparison"] = {"baseline_directory": str(baseline_directory / key),
                    "actual_gpu_baseline_rerun_performed": False, "baseline_final_latents_sha256": old_arm["final_latents_sha256"],
                    "common_inference_inputs_restored_exact": True, "only_hint_field_changed": True,
                    "final_latents_exact": bool(np.array_equal(old_final, final)),
                    "max_abs": float(difference.max()), "mean_abs": float(difference.mean()),
                    "baseline_raw_sha256": old_arm["raw_sha256"], "baseline_composite_sha256": old_arm["composite_sha256"]}
                outputs[key], previews_by_arm[key] = final, captured
                record["status"] = "denoised"; mx.clear_cache()
                log("denoising_complete", current_arm=key, nfe=record["transformer_calls"], target_tokens=12288)
        assert_saved_files_unchanged(saved_report)
        del branch, contexts, latents, prediction, model_input, active_context, padded_active
        del model.transformer; model.transformer = None
        mx.clear_cache()
        tiling = TilingConfig(vae_decode_tiles_per_dim=2, vae_decode_tile_size=512, vae_decode_overlap=4)
        for key, final in outputs.items():
            record = suite["runs"][key]
            directory = suite_dir / key
            start = time.perf_counter()
            mx.reset_peak_memory()
            log("decoding", current_arm=key)
            packed = mx.array(final).astype(mx.float32)
            unpacked = QwenImage21LatentCreator.unpack_latents(packed, height, width)
            decoded = VAEUtil.decode(model.vae, unpacked, tiling)
            mx.eval(decoded)
            if decoded.ndim == 5:
                decoded = decoded[:, :, 0]
            rgb = np.asarray(decoded[0, :3].transpose(1, 2, 0))
            if rgb.shape != (height, width, 3) or not np.isfinite(rgb).all():
                raise FloatingPointError("Invalid reference-control decoded target")
            raw = Image.fromarray(np.clip((rgb + 1) * 127.5, 0, 255).round().astype(np.uint8), mode="RGB")
            raw.save(directory / "raw.png")
            error = np.abs(np.asarray(raw.crop(rect), dtype=np.int16) - source_bytes.astype(np.int16))
            composite = raw.copy()
            composite.paste(source_rgb, (rect[0], rect[1]))
            composite.save(directory / "composite.png")
            if not np.array_equal(np.asarray(composite.crop(rect)), source_bytes):
                raise AssertionError("Final hardpaste changed original512 source pixels")
            record.update({"status": "success", "source_pixels_exact": True,
                           "final_latents_finite": True, "decoded_pixels_finite": True,
                           "raw_source_top16_mae_255": float(error[:16].mean()),
                           "raw_source_bottom16_mae_255": float(error[-16:].mean()),
                           "raw_source_inner_mae_255": float(error[16:-16].mean()),
                           "raw_source_mae_255": float(error.mean()), "raw_path": str(directory / "raw.png"),
                           "composite_path": str(directory / "composite.png"),
                           "raw_sha256": file_sha256(directory / "raw.png"),
                           "composite_sha256": file_sha256(directory / "composite.png"),
                           "decoding_mlx_peak_bytes": mx.get_peak_memory()})
            record["phases"]["decode_seconds"] = time.perf_counter() - start
            preview_start = time.perf_counter()
            for preview_record, (preview_step, active_height, window, clean_np) in zip(record["previews"], previews_by_arm[key]):
                preview_packed = mx.array(clean_np).astype(mx.float32)
                preview_unpacked = QwenImage21LatentCreator.unpack_latents(preview_packed, active_height, width)
                preview_decoded = VAEUtil.decode(model.vae, preview_unpacked, tiling); mx.eval(preview_decoded)
                if preview_decoded.ndim == 5: preview_decoded = preview_decoded[:, :, 0]
                preview_rgb = np.asarray(preview_decoded[0, :3].transpose(1, 2, 0))
                if preview_rgb.shape != (active_height, width, 3) or not np.isfinite(preview_rgb).all():
                    raise FloatingPointError("Invalid genuine growing predicted-clean preview")
                preview_raw = Image.fromarray(np.clip((preview_rgb + 1) * 127.5, 0, 255).round().astype(np.uint8), mode="RGB")
                raw_path = directory / "previews" / f"step{preview_step:02d}_predicted_clean_raw.png"
                preview_raw.save(raw_path)
                local_y = rect[1] - window[1]
                preview_composite = preview_raw.copy(); preview_composite.paste(source_rgb, (0, local_y))
                preview_path = directory / "previews" / f"step{preview_step:02d}_predicted_clean_composite.png"
                preview_composite.save(preview_path)
                if not np.array_equal(np.asarray(preview_composite.crop((0, local_y, width, local_y + 512))), source_bytes):
                    raise AssertionError("Growing preview hardpaste changed source pixels")
                preview_record.update({"raw_path": str(raw_path), "composite_path": str(preview_path),
                    "source_pixels_exact": True, "raw_sha256": file_sha256(raw_path), "composite_sha256": file_sha256(preview_path)})
            record["phases"]["preview_decode_seconds"] = time.perf_counter() - preview_start
            (directory / "metrics.json").write_text(json.dumps(record, indent=2) + "\n")
            mx.clear_cache()
        if any(record["transformer_calls"] != args.steps for record in suite["runs"].values()):
            raise AssertionError("Growing localized six-step NFE mismatch")
        assert_saved_files_unchanged(saved_report)
        suite["saved_input_files_unchanged_after_sampling_and_decode"] = True
        suite["original_source_png_unchanged"] = file_sha256(suite["source_512_path"]) == old_suite["source_512_sha256"]
        if not suite["original_source_png_unchanged"]: raise AssertionError("Original source PNG changed")
        log("complete", status="success")
    except BaseException as exc:
        suite.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        write("preflight.json" if args.preflight else "growing_top32_metrics.json")
        (suite_dir / "traceback.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
