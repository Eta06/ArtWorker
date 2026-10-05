#!/usr/bin/env python3
"""Untrained Viggle v0.2.1 r256 six-step EARLY-FRONTIER + localized ControlNet.

The old actual-square reference, edge-context target bridge, shifted sigmas,
prompt and growth geometry remain. Control129 and input-guide cosine64 weights
are gathered by final absolute target IDs before each forward. Future target
rows are absent from both chains. All16 hints compute before scalar localization;
prefix/known/outside direct hints are zero. Full prefix recomputes in both chains
without caches; this approximates the old cached implementation until actual
scale-zero comparison. Scale0 still pays all16 control blocks on every step.
No installed runtime or previous runner is edited. --preflight is CPU-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_trial import MODEL, PROMPT, REVISION, TURBO_REVISION
from run_geometry_trial import ADAPTER, active_ids, active_layout, insertion_state, six_sigmas, padded_source_rgba

HEIGHTS = (640, 896, 1152, 1152, 1152, 1152)
PREVIEW_STEPS = (2, 4, 6)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=("track1", "track2", "track3"), default="track3")
    parser.add_argument("--steps", type=int, choices=(6,), default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modes", nargs="+", choices=("mask-only", "source-canny", "tangent-canny"), default=["tangent-canny"])
    parser.add_argument("--hint-shape", choices=("cosine",), default="cosine")
    parser.add_argument("--hint-radius", type=int, choices=(64,), default=64)
    parser.add_argument("--control-scales", "--scales", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    parser.add_argument("--reference-baseline", type=Path, help="Optional cached edge-context arm for provenance-checked scale0 latent comparison")
    parser.add_argument("--source-canny", type=Path)
    parser.add_argument("--tangent-canny", type=Path)
    parser.add_argument("--structural-support", choices=("full", "known", "tangent"), default="tangent")
    parser.add_argument("--extrapolated-guide", type=Path)
    parser.add_argument("--support-dilation", type=int, default=64)
    parser.add_argument("--prompt-style", choices=("outpaint",), default="outpaint")
    parser.add_argument("--prompt", help="Complete text override; original outpaint prompt plus track description is default")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--known-bridge", action="store_true", default=True, help="Mandatory original edge-context target flow bridge; enabled by default")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(not 0 <= scale <= 10 for scale in args.control_scales) or len(set(args.control_scales)) != len(args.control_scales):
        parser.error("Control scales must be unique, finite and in [0,10]")
    args.prefix_hints = [f"scale{scale:g}" for scale in args.control_scales]
    if len(set(args.modes)) != len(args.modes) or len(set(args.prefix_hints)) != len(args.prefix_hints):
        parser.error("Modes and prefix-hint arms must be unique")
    if not 0 <= args.support_dilation <= 1152:
        parser.error("Support dilation must be in [0,1152]")
    suite_dir = args.output or ROOT / f"experiments/qwen/controlnet_runs/2026-10-05/{args.track}-growing-localized-v021r256-{args.structural_support}"
    if (suite_dir / "growing_localized_metrics.json").exists():
        raise FileExistsError(f"Refusing to overwrite prior trial: {suite_dir}")
    suite_dir.mkdir(parents=True, exist_ok=True)
    begun = time.perf_counter()
    source_paths = {name: Path(__file__).with_name(name) for name in
                    ("control.py", "conditioning.py", "reference_control.py", "sparse_conditioning.py",
                     "localized_reference_control.py", "growing_localized_control.py", "test_reference_control_cpu.py",
                     "test_localized_reference_cpu.py", "test_growing_localized_cpu.py")}
    suite = {"status": "preflight", "track": args.track, "seed": args.seed, "steps": args.steps,
             "runs": {}, "phases": {}, "runner_sha256": file_sha256(__file__),
             "source_code_sha256": {name: file_sha256(path) for name, path in source_paths.items()},
             "model_repo": "Qwen/Qwen-Image-2.1", "model_revision": REVISION,
             "base_quantization": 4, "control_quantization": None, "cfg": 1.0,
             "adapter": str(ADAPTER), "adapter_revision": TURBO_REVISION, "adapter_rank": 256, "adapter_scale": 1.0, "adapter_baked": False,
             "experimental_turbo_control_combination": True, "active_heights": list(HEIGHTS),
             "known_target_encoding_context": "edge-padded-full-canvas-center-crop",
             "known_target_and_control_source_codec_contexts_distinct": True,
             "experimental_reference_extension": True, "experimental_localized_hints": True,
             "hint_shape": args.hint_shape, "hint_radius_pixels": args.hint_radius, "control_scales": args.control_scales,
             "hint_localization_changes_compute": False, "known_target_hidden_trace_preserved": False,
             "official_control_recipe": False,
             "source_conditioning": "opaque actual-square prompt vision + fixed1024 image-reference latents + trained target masked-source channels",
             "reference_control_rows": "literal zero129 for reference IMAGE tokens; no text padding",
             "zero_reference_context_disables_prefix_hints": False,
             "prefix_hint_arms": args.prefix_hints, "structural_modes": args.modes,
             "structural_support": args.structural_support,
             "experimental_sparse_reference_combination": args.structural_support != "full",
             "untrained_distribution_changes": ["additional real square image-reference prefix", "zero129 reference image context",
                                                "Viggle r256 Turbo adapter and trained Union branch were not jointly trained",
                                                "dynamic early-frontier target activation with shared global sigmas is an untrained heuristic"],
             "known_bridge_extension": args.known_bridge,
             "compute_mode": "active targets only in both chains; full fixed prefix recomputed each step",
             "growing_compute": True, "prefix_cache_enabled": False,
             "base_prefix_cache_enabled": False, "control_prefix_cache_enabled": False,
             "old_cached_implementation_approximation": True,
             "scale_zero_extra_control_branch_cost": "All16 blocks on all6 steps; not a bare-base timing baseline",
             "scheduler_resolution_basis": "target512x1152 only; reference tokens excluded",
             "hash_representation": "float32 values converted from inference tensors; uint8 source pixels separately"}
    if args.control_scales:
        suite["untrained_distribution_changes"].append("post-chain scalar hint field: zero full prefix and known source target; localized unknown guide support")
    if args.structural_support != "full":
        suite["untrained_distribution_changes"].append("optional sparse structural64 support after full map VAE encode")
    if args.known_bridge:
        suite["untrained_distribution_changes"].append("original per-step edge-context target known-latent flow bridge")

    def write(name="growing_localized_metrics.json"):
        suite["elapsed_seconds"] = time.perf_counter() - begun
        suite["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (suite_dir / name).write_text(json.dumps(suite, indent=2) + "\n")

    def log(phase, **data):
        suite["phase"] = phase
        suite.update(data)
        write()
        print(json.dumps({"phase": phase, "elapsed_seconds": round(suite["elapsed_seconds"], 2), **data}), flush=True)

    try:
        import mlx.core as mx
        if args.preflight:
            mx.set_default_device(mx.cpu)
        import numpy as np
        from PIL import Image
        from mlx.utils import tree_flatten
        from mflux.models.common.vae.vae_util import VAEUtil
        from mflux.models.common.vae.tiling_config import TilingConfig
        from mflux.models.qwen21.reference import QwenImage21Edit
        from mflux.models.qwen21.reference.latent_creator.qwen_image21_latent_creator import QwenImage21LatentCreator
        from mflux.models.qwen21.reference.model.qwen_image21_transformer.layout import QwenImage21Layout
        from control import (CHECKPOINT_BYTES, CHECKPOINT_FILENAME, CHECKPOINT_REVISION,
                             CHECKPOINT_SHA256, UPSTREAM_REVISION, checkpoint_header, load_control_branch)
        from conditioning import control_prefix_padding, encode_control_context
        from reference_control import assemble_reference_input, validate_reference_layout
        from growing_localized_control import gather_active_conditioning, edge_known_flow_bridge, predicted_clean_with_known
        from localized_reference_control import (prepare_local_hint_weights, localized_reference_controlled_forward,
                                                 runtime_dependency_hashes)
        from sparse_conditioning import prepare_sparse_support, apply_sparse_structural_context

        def tensor_array(value):
            mx.eval(value)
            return np.asarray(value.astype(mx.float32)).copy()

        def tensor_sha(value):
            return hashlib.sha256(tensor_array(value).tobytes()).hexdigest()

        shared = json.loads((ROOT / "experiments/evaluation/inputs/inputs.json").read_text())
        track = next(item for item in shared["tracks"] if item["id"] == args.track)
        width, height = shared["canvas_size"]
        rect = shared["source_rect_xyxy"]
        source_rgb = Image.open(track["source_512"]).convert("RGB")
        source = source_rgb.convert("RGBA")
        source_bytes = np.asarray(source_rgb).copy()
        if (width, height) != (512, 1152) or source.size != (512, 512) or rect != [0, 320, 512, 832]:
            raise ValueError("Expected shared512 square at y320:832 on512x1152 target")
        if not np.all(np.asarray(source)[..., 3] == 255):
            raise ValueError("Actual-square reference must be opaque RGBA")
        source.save(suite_dir / "source_rgba.png")
        map_dir = ROOT / f"experiments/qwen/controlnet_inputs/2026-10-05/{args.track}"
        map_paths = {"mask-only": None, "source-canny": args.source_canny or map_dir / "canny_known.png",
                     "tangent-canny": args.tangent_canny or map_dir / "canny_extrapolated.png"}
        guide_path = args.extrapolated_guide or map_dir / "extrapolated_only.png"
        checkpoint = args.checkpoint or ROOT / ".build/models/qwen/controlnet_union" / CHECKPOINT_FILENAME
        manifest_path = ROOT / "experiments/qwen/base-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest["revision"] != REVISION:
            raise ValueError("Base manifest revision changed")
        missing = [str(MODEL / item["name"]) for item in manifest["files"]
                   if not (MODEL / item["name"]).is_file()
                   or (MODEL / item["name"]).stat().st_size != item["expected_bytes"]]
        missing += [str(map_paths[mode]) for mode in args.modes
                    if map_paths[mode] is not None and not map_paths[mode].is_file()]
        if not guide_path.is_file():
            missing.append(str(guide_path))
        if not checkpoint.is_file() or checkpoint.stat().st_size != CHECKPOINT_BYTES:
            missing.append(str(checkpoint))
        if not ADAPTER.is_file():
            missing.append(str(ADAPTER))
        for mode in args.modes:
            path = map_paths[mode]
            if path is not None and path.is_file():
                with Image.open(path) as check_map:
                    check_map.load()
                    if check_map.size != (width, height):
                        raise ValueError(f"Invalid control map dimensions before model loading: {path}")
        if guide_path.is_file():
            with Image.open(guide_path) as check_guide:
                check_guide.load()
                if check_guide.size != (width, height):
                    raise ValueError("Invalid input hint guide dimensions before model loading")
        descriptions = json.loads((ROOT / "experiments/flux2/prompts.json").read_text())
        prompt_prefix = "One coherent portrait music album artwork." if args.prompt_style == "native" else PROMPT
        prompt = args.prompt if args.prompt is not None else prompt_prefix + " " + descriptions[args.track]
        suite.update({"width": width, "height": height, "source_rect_xyxy": rect, "prompt": prompt,
                      "prompt_style": "custom" if args.prompt is not None else args.prompt_style,
                      "source_512_path": track["source_512"], "source_512_sha256": file_sha256(track["source_512"]),
                      "source_uint8_sha256": hashlib.sha256(source_bytes.tobytes()).hexdigest(),
                      "source_rgba_uint8_sha256": hashlib.sha256(np.asarray(source).tobytes()).hexdigest(),
                      "map_paths": {mode: str(map_paths[mode]) if map_paths[mode] else None for mode in args.modes},
                      "map_sha256": {mode: file_sha256(map_paths[mode]) if map_paths[mode] and map_paths[mode].is_file() else None for mode in args.modes},
                      "guide_path": str(guide_path), "missing_files": missing,
                      "control_repo": "alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union",
                      "control_revision": CHECKPOINT_REVISION, "upstream_revision": UPSTREAM_REVISION,
                      "checkpoint_expected_sha256": CHECKPOINT_SHA256,
                      "checkpoint_payload_hash_reverified": False,
                      "base_manifest_sha256": file_sha256(manifest_path),
                      "base_weight_expected_hashes": {item["name"]: item["lfs_sha256"] for item in manifest["files"] if item["name"].endswith(".safetensors")},
                      "base_weight_payload_hashes_reverified": False,
                      "base_config_sha256": {name: file_sha256(MODEL / name) for name in
                                             ("transformer/config.json", "vae/config.json", "text_encoder/config.json", "scheduler/scheduler_config.json")}})
        if checkpoint.is_file() and checkpoint.stat().st_size == CHECKPOINT_BYTES:
            header, header_sha = checkpoint_header(checkpoint)
            suite["checkpoint_header"] = {"sha256": header_sha, "tensor_count": len(header) - int("__metadata__" in header),
                                          "payload_bytes": max(value["data_offsets"][1] for key, value in header.items() if key != "__metadata__")}
        support_dir = suite_dir / "shared"
        support_dir.mkdir(exist_ok=True)
        if args.structural_support == "full":
            support_pixels = np.ones((height, width), dtype=bool)
            support = np.ones((1, 2304, 1), dtype=bool)
            support_info = {"experimental_sparse_control": False, "supported_target_tokens": 2304,
                            "total_target_tokens": 2304, "support_tokens_bool_sha256": hashlib.sha256(support.tobytes()).hexdigest()}
        elif args.structural_support != "tangent" or guide_path.is_file():
            guide = Image.open(guide_path).convert("RGB") if args.structural_support == "tangent" else None
            support_pixels, support, support_info = prepare_sparse_support(np, Image, width, height, rect, guide, args.support_dilation)
            support_info["guide_file_sha256"] = file_sha256(guide_path) if guide is not None else None
        else:
            support_pixels = support = support_info = None
        if support is not None:
            Image.fromarray(support_pixels.astype(np.uint8) * 255, mode="L").save(support_dir / "support.png")
            np.save(support_dir / "support_tokens.npy", support)
            support_info["support_png_sha256"] = file_sha256(support_dir / "support.png")
            (support_dir / "support.json").write_text(json.dumps(support_info, indent=2) + "\n")
        suite["structural_support_metadata"] = support_info
        hint_target_np = None
        if guide_path.is_file():
            hint_guide = Image.open(guide_path).convert("RGB")
            hint_pixels, hint_target_np, hint_info = prepare_local_hint_weights(np, Image, width, height, rect,
                hint_guide, args.hint_radius, args.hint_shape)
            hint_info["guide_file_sha256"] = file_sha256(guide_path)
            np.save(support_dir / "hint_pixel_weights_float32.npy", hint_pixels)
            np.save(support_dir / "hint_target_weights_float32.npy", hint_target_np)
            Image.fromarray(np.clip(hint_pixels * 255, 0, 255).round().astype(np.uint8), mode="L").save(support_dir / "hint_weights.png")
            hint_info["display_png_sha256"] = file_sha256(support_dir / "hint_weights.png")
            (support_dir / "hint_weights.json").write_text(json.dumps(hint_info, indent=2) + "\n")
            suite["localized_hint_weights"] = hint_info
        suite["runtime_dependency_hashes"] = runtime_dependency_hashes()
        suite["old_geometry_runner_sha256"] = file_sha256(Path(__file__).resolve().parents[1] / "run_geometry_trial.py")
        suite["old_spatial_runner_sha256"] = file_sha256(Path(__file__).resolve().parents[1] / "run_spatial_ablation.py")
        suite["adapter_bytes"] = ADAPTER.stat().st_size if ADAPTER.is_file() else None
        adapter_manifest_path = ROOT / "experiments/qwen/viggle-manifest.json"
        adapter_manifest = json.loads(adapter_manifest_path.read_text())
        adapter_entry = next(item for item in adapter_manifest["files"] if item["name"] == ADAPTER.name)
        if adapter_manifest["revision"] != TURBO_REVISION or suite["adapter_bytes"] != adapter_entry["expected_bytes"]:
            raise ValueError("Pinned r256 adapter manifest/size mismatch")
        suite["adapter_expected_sha256"] = adapter_entry["lfs_sha256"]
        suite["adapter_manifest_sha256"] = file_sha256(adapter_manifest_path)
        suite["adapter_payload_hash_reverified"] = False
        parity_path = Path(__file__).with_name("cpu_parity.json")
        validation_path = Path(__file__).with_name("reference_cpu_validation.json")
        parity = json.loads(parity_path.read_text()) if parity_path.is_file() else None
        validation = json.loads(validation_path.read_text()) if validation_path.is_file() else None
        validation_current = bool(validation and validation.get("status") == "passed"
            and validation.get("control_port_sha256") == suite["source_code_sha256"]["control.py"]
            and validation.get("reference_helper_sha256") == suite["source_code_sha256"]["reference_control.py"]
            and validation.get("conditioning_sha256") == suite["source_code_sha256"]["conditioning.py"]
            and validation.get("script_sha256") == file_sha256(Path(__file__).with_name("test_reference_control_cpu.py")))
        parity_current = bool(parity and parity.get("status") == "passed"
                              and parity.get("control_port_sha256") == suite["source_code_sha256"]["control.py"])
        localized_validation_path = Path(__file__).with_name("localized_cpu_validation.json")
        localized_validation = json.loads(localized_validation_path.read_text()) if localized_validation_path.is_file() else None
        localized_current = bool(localized_validation and localized_validation.get("status") == "passed"
            and all(localized_validation.get("source_code_sha256", {}).get(name) == suite["source_code_sha256"][name]
                    for name in ("control.py", "conditioning.py", "reference_control.py", "localized_reference_control.py", "sparse_conditioning.py", "test_reference_control_cpu.py"))
            and localized_validation.get("script_sha256") == suite["source_code_sha256"]["test_localized_reference_cpu.py"]
            and localized_validation.get("runtime_dependency_hashes") == suite["runtime_dependency_hashes"])
        growth_validation_path = Path(__file__).with_name("growing_localized_cpu_validation.json")
        growth_validation = json.loads(growth_validation_path.read_text()) if growth_validation_path.is_file() else None
        growth_current = bool(growth_validation and growth_validation.get("status") == "passed"
            and all(growth_validation.get("source_code_sha256", {}).get(name) == suite["source_code_sha256"][name]
                    for name in ("control.py", "conditioning.py", "reference_control.py", "localized_reference_control.py", "growing_localized_control.py", "sparse_conditioning.py", "test_reference_control_cpu.py"))
            and growth_validation.get("script_sha256") == suite["source_code_sha256"]["test_growing_localized_cpu.py"]
            and growth_validation.get("old_geometry_runner_sha256") == suite["old_geometry_runner_sha256"]
            and growth_validation.get("runtime_dependency_hashes") == suite["runtime_dependency_hashes"])
        suite["cpu_gate_status"] = {"existing_port_parity_current": parity_current,
                                    "reference_validation_current": validation_current,
                                    "localized_validation_current": localized_current,
                                    "growing_validation_current": growth_current}
        if args.preflight:
            # Real layout geometry, tiny random weights, and pixels only.
            from test_reference_control_cpu import geometry_case
            from test_localized_reference_cpu import field_case
            from test_growing_localized_cpu import gather_case, trajectory_case
            from conditioning import cpu_preflight
            from sparse_conditioning import cpu_preflight as sparse_cpu_preflight
            suite["reference_layout_cpu_test"] = geometry_case()
            suite["hint_field_cpu_test"] = field_case()
            suite["absolute_growth_gather_cpu_test"] = gather_case()
            suite["paired_six_step_growth_cpu_tests"] = [trajectory_case(False), trajectory_case(True)]
            suite["conditioning_cpu_test"] = cpu_preflight()
            if args.structural_support != "full":
                suite["sparse_cpu_test"] = sparse_cpu_preflight()
            suite.update({"status": "preflight_files_ready" if not missing and parity_current and validation_current and localized_current and growth_current else "preflight_files_incomplete",
                          "model_weights_loaded": False, "device": "CPU only",
                          "planned_nfe_per_arm": args.steps, "planned_target_token_forwards_per_arm": 12288,
                          "planned_reference_image_token_forwards_per_arm": args.steps * 1024,
                          "planned_base_block_forwards_per_arm": args.steps * 32,
                          "planned_control_block_forwards_per_arm": args.steps * 16})
            write("preflight.json")
            print(json.dumps({"status": suite["status"], "device": suite["device"], "missing_files": missing,
                              "cpu_gate_status": suite["cpu_gate_status"], "output": str(suite_dir)}, indent=2), flush=True)
            return
        if missing:
            raise FileNotFoundError(f"Trial files incomplete: {missing}")
        if not parity_current or not validation_current or not localized_current or not growth_current:
            raise RuntimeError("Matching current passed existing/reference/localized/growing CPU gates and runtime hashes are required")
        suite["growing_cpu_validation_sha256"] = file_sha256(growth_validation_path)
        suite["localized_cpu_validation_sha256"] = file_sha256(localized_validation_path)
        suite["reference_cpu_validation_sha256"] = file_sha256(validation_path)
        suite["port_cpu_parity_sha256"] = file_sha256(parity_path)
        mx.set_cache_limit(512 * 1024**2)
        mx.set_memory_limit(28 * 1024**3)
        suite["metal_device"] = mx.device_info()
        suite["mlx_version"] = mx.__version__
        log("loading_base", status="running")
        start = time.perf_counter()
        model = QwenImage21Edit(model_path=str(MODEL), quantize=4, lora_paths=[str(ADAPTER)],
                                lora_scales=[1.0], bake_lora=False)
        suite["phases"]["base_load_seconds"] = time.perf_counter() - start
        suite["base_load_mlx_peak_bytes"] = mx.get_peak_memory()
        mx.reset_peak_memory()
        log("reference_conditioning")
        start = time.perf_counter()
        prompt_embeds, slots = model._encode_prompt(prompt, [source])
        mx.eval(prompt_embeds, slots)
        if int(np.asarray(slots, dtype=bool).sum()) != 256:
            raise AssertionError("Actual-square reference must supply256 four-token image slots")
        del model.text_encoder
        model.text_encoder = None
        mx.clear_cache()
        pixels = mx.array(np.asarray(source).astype(np.float32) / 127.5 - 1).transpose(2, 0, 1)[None]
        source_latents = QwenImage21LatentCreator.pack_latents(model.vae.encode(pixels)).astype(prompt_embeds.dtype)
        mx.eval(source_latents)
        if source_latents.shape != (1, 1024, 64):
            raise AssertionError("Reference VAE must encode the actual512 square as1024 tokens")
        source_prefix_np = tensor_array(source_latents)
        source_prefix_sha = hashlib.sha256(source_prefix_np.tobytes()).hexdigest()
        layout = QwenImage21Layout.create(slots, [(1, 32, 32), (1, 72, 32)], model.transformer.axes)
        suite["layout"] = validate_reference_layout(np, layout, 1024, 2304)
        hint_weights = mx.array(hint_target_np).astype(prompt_embeds.dtype)
        effective_hint_np = tensor_array(hint_weights)
        joint_hint_np = np.concatenate([np.zeros((1, layout.prefix_length, 1), dtype=np.float32), effective_hint_np], axis=1)
        np.savez_compressed(support_dir / "effective_hint_weights.npz", target_weights=effective_hint_np,
                            joint_weights=joint_hint_np, requested_target_weights=hint_target_np)
        suite["localized_hint_weights"].update({"effective_dtype": str(prompt_embeds.dtype),
            "effective_target_weights_float32_sha256": hashlib.sha256(effective_hint_np.tobytes()).hexdigest(),
            "effective_joint_weights_float32_sha256": hashlib.sha256(joint_hint_np.tobytes()).hexdigest(),
            "joint_prefix_zero": bool(np.all(joint_hint_np[:, :layout.prefix_length] == 0)),
            "gate_applied_after_all16_hints": True})
        suite["reference_encoding"] = {"source_shape": [1, 32, 32], "source_latents_shape": [1, 1024, 64],
                                       "source_prefix_sha256": source_prefix_sha, "source_vae_encodes": 1,
                                       "actual_square_only": True, "opaque_rgba": True}
        suite["prompt_embeds_sha256"] = tensor_sha(prompt_embeds)
        suite["image_slots_bool_sha256"] = hashlib.sha256(np.asarray(slots, dtype=bool).tobytes()).hexdigest()
        np.savez_compressed(support_dir / "reference_conditioning.npz", source_latents=source_prefix_np,
                            prompt_embeds=tensor_array(prompt_embeds), image_slots=np.asarray(slots, dtype=bool))
        del pixels
        edge_rgba = padded_source_rgba(np, np.asarray(source), width, height, rect)
        edge_pixels = mx.array(edge_rgba.astype(np.float32) / 127.5 - 1).transpose(2, 0, 1)[None]
        edge_packed = QwenImage21LatentCreator.pack_latents(model.vae.encode(edge_pixels)).astype(prompt_embeds.dtype)
        mx.eval(edge_packed)
        if edge_packed.shape != (1, 2304, 64):
            raise AssertionError("Old edge-context full-canvas VAE geometry changed")
        edge_known = edge_packed[:, 20 * 32:52 * 32]
        edge_known_np = tensor_array(edge_known)
        edge_ids = ((np.clip(np.arange(72), 20, 51) - 20)[:, None] * 32 + np.arange(32)[None, :]).reshape(-1).astype(np.int32)
        edge_guess = edge_known[:, mx.array(edge_ids)]
        suite["edge_known_target_sha256"] = hashlib.sha256(edge_known_np.tobytes()).hexdigest()
        np.savez_compressed(support_dir / "edge_known_target.npz", known=edge_known_np, edge_guess=tensor_array(edge_guess))
        if not np.array_equal(tensor_array(source_latents), source_prefix_np):
            raise AssertionError("Edge target context encoding changed fixed source prefix")
        del edge_rgba, edge_pixels, edge_packed
        suite["phases"]["reference_conditioning_seconds"] = time.perf_counter() - start
        suite["text_encoder_deleted_before_control_load"] = True
        log("loading_control")
        start = time.perf_counter()
        branch, branch_info = load_control_branch(checkpoint, quantize=None)
        suite["control_checkpoint"] = branch_info
        suite["phases"]["control_load_seconds"] = time.perf_counter() - start
        suite["base_component_parameter_bytes"] = {name: sum(value.nbytes for _, value in tree_flatten(getattr(model, name).parameters())) for name in ("transformer", "vae")}
        if len(model.transformer.transformer_blocks) != 32 or len(branch.control_blocks) != 16:
            raise AssertionError("Pinned teacher must have32 base blocks and16 control hints")
        suite["control_load_mlx_peak_bytes"] = mx.get_peak_memory()
        mx.reset_peak_memory()
        mx.set_memory_limit(25 * 1024**3)
        log("target_control_conditioning")
        start = time.perf_counter()
        contexts, padded_contexts, context_info = {}, {}, {}
        for mode in args.modes:
            control_rgb = Image.open(map_paths[mode]).convert("RGB") if map_paths[mode] is not None else None
            if control_rgb is not None and control_rgb.size != (width, height):
                raise ValueError("Control map must preserve final target dimensions")
            context, info = encode_control_context(mx, np, Image, model.vae, QwenImage21LatentCreator,
                                                  source_rgb, rect, width, height, control_rgb=control_rgb)
            context = context.astype(prompt_embeds.dtype)
            baseline = tensor_array(context)
            info["baseline_target_context_sha256_float32"] = hashlib.sha256(baseline.tobytes()).hexdigest()
            if args.structural_support != "full":
                context = apply_sparse_structural_context(mx, np, context, support)
            target_np = tensor_array(context)
            if not np.array_equal(target_np[:, :, 64:], baseline[:, :, 64:]):
                raise AssertionError("Sparse option altered known-mask or masked-source channels")
            support_ids = support.reshape(-1)
            if not np.array_equal(target_np[:, support_ids, :64], baseline[:, support_ids, :64]):
                raise AssertionError("Supported structural64 changed")
            if np.any(target_np[:, ~support_ids, :64] != 0):
                raise AssertionError("Unsupported structural64 must be literal zero")
            padded = control_prefix_padding(mx, context, 1024)
            padded_np = tensor_array(padded)
            if padded.shape != (1, 3328, 129) or not np.all(padded_np[:, :1024] == 0) or not np.array_equal(padded_np[:, 1024:], target_np):
                raise AssertionError("Reference context padding must be literal zero129 and target context exact")
            expected_mask = np.zeros((72, 32), dtype=np.float32)
            expected_mask[20:52] = 1
            if not np.array_equal(target_np[0, :, 64].reshape(72, 32), expected_mask) or padded_np[:, :1024, 64].sum() != 0:
                raise AssertionError("Original source known mask must cover target rows only")
            info.update({"map_sha256": suite["map_sha256"][mode], "structural_support": args.structural_support,
                         "support_metadata": support_info, "source65_exact_against_baseline": True,
                         "reference_image_padding_tokens": 1024, "control_text_padding_rows": 0,
                         "reference_padding_literal_zero129": True, "reference_known_mask_zero": True,
                         "target_context_sha256_float32": hashlib.sha256(target_np.tobytes()).hexdigest(),
                         "padded_image_context_sha256_float32": hashlib.sha256(padded_np.tobytes()).hexdigest()})
            np.savez_compressed(support_dir / f"{mode}_contexts.npz", target_context=target_np,
                                padded_image_context=padded_np, structural_support=support)
            contexts[mode], padded_contexts[mode], context_info[mode] = context, padded, info
        if not np.array_equal(tensor_array(source_latents), source_prefix_np):
            raise AssertionError("Control conditioning changed fixed square reference latents")
        source_contexts = [tensor_array(value[:, :, 64:]) for value in contexts.values()]
        if any(not np.array_equal(value, source_contexts[0]) for value in source_contexts[1:]):
            raise AssertionError("Modes changed known-mask or masked-source conditioning")
        suite["control_conditioning"] = context_info
        suite["known_mask_and_source_context_matched"] = True
        suite["source_prefix_unchanged_after_target_conditioning"] = True
        suite["phases"]["target_control_conditioning_seconds"] = time.perf_counter() - start
        sigmas_np = six_sigmas(np, 2304)
        sigmas = mx.array(sigmas_np, dtype=mx.float32)
        suite["sigmas"] = sigmas_np.tolist()
        suite["sigma_recipe"] = "Old v0.2.1 six nodes shifted by final2304 target resolution, terminal0"
        suite["scheduler_target_tokens"] = 2304
        mx.random.seed(args.seed)
        noise = mx.random.normal((1, 2304, 64)).astype(prompt_embeds.dtype)
        mx.eval(noise)
        suite["noise_sha256"] = tensor_sha(noise)
        np.savez_compressed(support_dir / "target_noise.npz", noise=tensor_array(noise))
        outputs = {}
        previews_by_arm = {}
        baseline_dir = args.reference_baseline or ROOT / f"experiments/qwen/geometry_runs/2026-10-05/{args.track}/edge-context"
        suite["cached_geometry_baseline"] = str(baseline_dir)
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
                    "scale_zero_extra_branch_cost": scale == 0,
                    "compute_mode": suite["compute_mode"], "future_targets_absent_both_chains": True,
                    "transformer_calls": 0, "base_block_calls": 0, "control_block_calls": 0,
                    "target_token_forward_sum": 0, "reference_image_token_forward_sum": 0,
                    "joint_query_token_forward_sum": 0, "source_prefix_exact_each_forward": True,
                    "step_records": [], "previews": [], "phases": {}}
                suite["runs"][key] = record
                mx.reset_peak_memory(); log("denoising_start", current_arm=key)
                wall_start = time.perf_counter(); core_seconds = capture_seconds = 0.0
                for step, active_height in enumerate(HEIGHTS):
                    step_start = time.perf_counter()
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
                    if np.any(tensor_array(active_hints)[0, known_mask_np] != 0):
                        raise AssertionError("Known source target received direct localized hints")
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
                        "known_source_target_hint_weights_zero": True, "source_prefix_input_exact": True,
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
                if scale == 0:
                    old_metrics_path = baseline_dir / "metrics.json"
                    old_latents_path = baseline_dir / "previews/step06_predicted_clean.npz"
                    comparison = {"status": "unverified_missing_cached_baseline", "path": str(baseline_dir)}
                    if old_metrics_path.is_file() and old_latents_path.is_file():
                        old = json.loads(old_metrics_path.read_text())
                        matching = {"seed": old.get("seed") == args.seed, "prompt": old.get("prompt") == prompt,
                            "source_prefix": old.get("source_prefix_sha256") == source_prefix_sha,
                            "noise": old.get("noise_sha256") == suite["noise_sha256"],
                            "known_target": old.get("known_target_sha256") == suite["edge_known_target_sha256"],
                            "sigmas": old.get("sigmas") == suite["sigmas"], "adapter_rank": old.get("adapter_rank") == 256,
                            "adapter_revision": old.get("adapter_revision") == TURBO_REVISION,
                            "quantization": old.get("quantization") == 4, "model_revision": old.get("model_revision") == REVISION,
                            "width": old.get("width") == width, "height": old.get("height") == height,
                            "steps": old.get("steps") == 6, "spatial_mode": old.get("spatial_mode") == "edge-context",
                            "active_heights": old.get("active_heights_by_step") == list(HEIGHTS),
                            "initializer": old.get("initializer") == "predicted-clean-active-frontier",
                            "known_context": old.get("known_target_encoding_context") == "edge-padded-full-canvas-center-crop"}
                        comparison["matching_provenance"] = matching
                        if all(matching.values()):
                            with np.load(old_latents_path) as data: old_final = data["latents"]
                            if old_final.shape != final.shape: raise ValueError("Cached baseline latent shape mismatch")
                            difference = np.abs(old_final - final)
                            comparison.update({"status": "compared_same_provenance", "exact": bool(np.array_equal(old_final, final)),
                                "max_abs": float(difference.max()), "mean_abs": float(difference.mean()),
                                "old_latents_file_sha256": file_sha256(old_latents_path)})
                        else: comparison["status"] = "unverified_baseline_provenance_mismatch"
                    record["cached_geometry_scale_zero_comparison"] = comparison
                outputs[key], previews_by_arm[key] = final, captured
                record["status"] = "denoised"; mx.clear_cache()
                log("denoising_complete", current_arm=key, nfe=record["transformer_calls"], target_tokens=12288)
        del branch, contexts, padded_contexts, context, padded, latents, prediction, model_input, active_context, padded_active
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
        log("complete", status="success")
    except BaseException as exc:
        suite.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        write()
        (suite_dir / "traceback.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
