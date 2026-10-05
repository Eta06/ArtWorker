#!/usr/bin/env python3
"""Top32-only probe restoring exact persisted localized40 baseline inputs.

The original full joint control chain computes all16 hints before localization.
Completed unknown hints retain input-guide cosine64 support exactly. The none
arm keeps all known hints zero; top32/top64 retain a declared scalar only on
the first known source rows. Prefix hints always stay zero. Q4 base,
BF16 control, CFG1, no LoRA, target-resolution scheduler and no prefix cache.
Optional sparse129 context and target known bridge are separately declared.
No sparse compute claim; --preflight is CPU-only and loads no real weights.
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
from run_trial import MODEL, PROMPT, REVISION


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=("track1", "track2", "track3"), default="track3")
    parser.add_argument("--steps", type=int, choices=(40,), default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modes", nargs="+", choices=("mask-only", "source-canny", "tangent-canny"), default=["tangent-canny"])
    parser.add_argument("--hint-shape", choices=("cosine", "hard"), default="cosine")
    parser.add_argument("--hint-radius", type=int, default=64)
    parser.add_argument("--collar-arms", nargs="+", choices=("top32",), default=["top32"])
    parser.add_argument("--known-collar-weight", type=float, default=1.0)
    parser.add_argument("--control-scale", type=float, default=1.0)
    parser.add_argument("--source-canny", type=Path)
    parser.add_argument("--tangent-canny", type=Path)
    parser.add_argument("--structural-support", choices=("tangent",), default="tangent")
    parser.add_argument("--extrapolated-guide", type=Path)
    parser.add_argument("--support-dilation", type=int, default=64)
    parser.add_argument("--prompt-style", choices=("outpaint",), default="outpaint")
    parser.add_argument("--prompt", help="Complete text override; official baseline teacher prompt is default")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--known-bridge", action="store_true", default=True, help="Unchanged target-only known-latent flow bridge, active in this matched probe")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--saved-baseline", type=Path, default=ROOT / "experiments/qwen/controlnet_runs/2026-10-05/track3-localized40-outpaint")
    args = parser.parse_args()
    args.prefix_hints = args.collar_arms
    if not 0<=args.known_collar_weight<=1:
        parser.error("Known collar scalar must be finite in [0,1]")
    if args.hint_shape!="cosine" or args.hint_radius!=64 or args.support_dilation!=64:
        parser.error("This matched probe retains cosine64 unknown hints and tangent structural support64")
    if not 0 <= args.hint_radius <= 1152 or not 0 <= args.control_scale <= 10:
        parser.error("Hint radius must be in [0,1152] and control scale finite in [0,10]")
    if len(set(args.modes)) != len(args.modes) or len(set(args.prefix_hints)) != len(args.prefix_hints):
        parser.error("Modes and prefix-hint arms must be unique")
    if not 0 <= args.support_dilation <= 1152:
        parser.error("Support dilation must be in [0,1152]")
    suffix = "-knownbridge" if args.known_bridge else ""
    suite_dir = args.output or ROOT / f"experiments/qwen/controlnet_runs/2026-10-05/{args.track}-saved-top32-{args.steps}-outpaint"
    from known_collar_saved_baseline import assert_disjoint_output, assert_saved_files_unchanged
    assert_disjoint_output(suite_dir, args.saved_baseline)
    if (suite_dir / "saved_known_collar_reference_metrics.json").exists():
        raise FileExistsError(f"Refusing to overwrite prior trial: {suite_dir}")
    suite_dir.mkdir(parents=True, exist_ok=True)
    begun = time.perf_counter()
    source_paths = {name: Path(__file__).with_name(name) for name in
                    ("control.py", "conditioning.py", "reference_control.py", "sparse_conditioning.py",
                     "localized_reference_control.py", "known_collar_reference_control.py",
                     "test_reference_control_cpu.py", "test_localized_reference_cpu.py", "test_known_collar_reference_cpu.py")}
    suite = {"status": "preflight", "track": args.track, "seed": args.seed, "steps": args.steps,
             "runs": {}, "phases": {}, "runner_sha256": file_sha256(__file__),
             "source_code_sha256": {name: file_sha256(path) for name, path in source_paths.items()},
             "model_repo": "Qwen/Qwen-Image-2.1", "model_revision": REVISION,
             "base_quantization": 4, "control_quantization": None, "cfg": 1.0, "adapter": None,
             "experimental_reference_extension": True, "experimental_localized_hints": True,
             "experimental_known_top_collar_hints": True, "known_collar_arms": args.collar_arms,
             "known_collar_scalar_weight": args.known_collar_weight,
             "copied_from_runner": str(Path(__file__).with_name("run_localized_reference_control_trial.py")),
             "copied_from_runner_sha256": file_sha256(Path(__file__).with_name("run_localized_reference_control_trial.py")),
             "copied_from_helper_sha256": file_sha256(source_paths["localized_reference_control.py"]),
             "hint_shape": args.hint_shape, "hint_radius_pixels": args.hint_radius, "control_scale": args.control_scale,
             "hint_localization_changes_compute": False, "known_target_hidden_trace_preserved": False,
             "official_control_recipe": False,
             "source_conditioning": "opaque actual-square prompt vision + fixed1024 image-reference latents + trained target masked-source channels",
             "reference_control_rows": "literal zero129 for reference IMAGE tokens; no text padding",
             "zero_reference_context_disables_prefix_hints": False,
             "prefix_hint_arms": args.prefix_hints, "structural_modes": args.modes,
             "structural_support": args.structural_support,
             "experimental_sparse_reference_combination": args.structural_support != "full",
             "untrained_distribution_changes": ["additional real square image-reference prefix", "zero129 reference image context"],
             "known_bridge_extension": args.known_bridge,
             "compute_mode": "full reference+target joint every step in base and control; no growing compute",
             "growing_compute": False, "prefix_cache_enabled": False,
             "scheduler_resolution_basis": "target512x1152 only; reference tokens excluded",
             "hash_representation": "float32 values converted from inference tensors; uint8 source pixels separately"}
    suite["copied_from_helper_sha256"] = suite["source_code_sha256"]["localized_reference_control.py"]
    suite["untrained_distribution_changes"].append("post-chain hint field: prefix zero; unchanged localized unknown support; declared scalar on known top collar only")
    if args.structural_support != "full":
        suite["untrained_distribution_changes"].append("optional sparse structural64 support after full map VAE encode")
    if args.known_bridge:
        suite["untrained_distribution_changes"].append("optional per-step target known-latent flow bridge")

    def write(name="saved_known_collar_reference_metrics.json"):
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
        from mflux.models.common.config.config import Config
        from mflux.models.common.vae.vae_util import VAEUtil
        from mflux.models.common.vae.tiling_config import TilingConfig
        from mflux.models.qwen21.reference import QwenImage21Edit
        from mflux.models.qwen21.reference.latent_creator.qwen_image21_latent_creator import QwenImage21LatentCreator
        from mflux.models.qwen21.reference.model.qwen_image21_transformer.layout import QwenImage21Layout
        from control import (CHECKPOINT_BYTES, CHECKPOINT_FILENAME, CHECKPOINT_REVISION,
                             CHECKPOINT_SHA256, UPSTREAM_REVISION, checkpoint_header, load_control_branch)
        from conditioning import control_prefix_padding, encode_control_context
        from reference_control import (assemble_reference_input, target_known_bridge, validate_reference_layout)
        from known_collar_reference_control import (prepare_known_collar_hint_weights, known_collar_reference_controlled_forward,
                                                   runtime_dependency_hashes)
        from sparse_conditioning import prepare_sparse_support, apply_sparse_structural_context
        from known_collar_saved_baseline import validate_saved_baseline

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
        hint_target_np = {}
        hint_fields = {}
        if guide_path.is_file():
            hint_guide = Image.open(guide_path).convert("RGB")
            for arm in args.collar_arms:
                collar_pixels={"none":0,"top32":32,"top64":64}[arm]
                hint_pixels,field,info=prepare_known_collar_hint_weights(np, Image, width, height, rect,
                    hint_guide,args.hint_radius,args.hint_shape,collar_pixels,args.known_collar_weight)
                field_dir=support_dir/"hint_fields"/arm;field_dir.mkdir(parents=True,exist_ok=True)
                info["guide_file_sha256"]=file_sha256(guide_path)
                np.save(field_dir/"hint_pixel_weights_float32.npy",hint_pixels)
                np.save(field_dir/"hint_target_weights_float32.npy",field)
                Image.fromarray(np.clip(hint_pixels*255,0,255).round().astype(np.uint8),mode="L").save(field_dir/"hint_weights.png")
                info["display_png_sha256"]=file_sha256(field_dir/"hint_weights.png")
                (field_dir/"hint_weights.json").write_text(json.dumps(info,indent=2)+"\n")
                hint_target_np[arm]=field
                hint_fields[arm]=info
            suite["known_collar_hint_fields"]=hint_fields
        suite["runtime_dependency_hashes"] = runtime_dependency_hashes()
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
        suite["cpu_gate_status"] = {"existing_port_parity_current": parity_current,
                                    "reference_validation_current": validation_current,
                                    "localized_validation_current": localized_current}
        collar_validation_path=Path(__file__).with_name("known_collar_cpu_validation.json")
        collar_validation=json.loads(collar_validation_path.read_text()) if collar_validation_path.is_file() else None
        collar_current=bool(collar_validation and collar_validation.get("status")=="passed"
            and all(collar_validation.get("source_code_sha256",{}).get(name)==suite["source_code_sha256"][name]
                    for name in ("control.py","conditioning.py","reference_control.py","localized_reference_control.py",
                                 "known_collar_reference_control.py","sparse_conditioning.py","test_reference_control_cpu.py"))
            and collar_validation.get("script_sha256")==suite["source_code_sha256"]["test_known_collar_reference_cpu.py"]
            and collar_validation.get("runtime_dependency_hashes")==suite["runtime_dependency_hashes"])
        suite["cpu_gate_status"]["known_collar_validation_current"]=collar_current
        saved, matched_report, saved_metrics = validate_saved_baseline(np, Image, args.saved_baseline, suite, hint_target_np, support)
        suite["saved_baseline_validation"] = matched_report
        suite["saved_baseline_validator_sha256"] = file_sha256(Path(__file__).with_name("known_collar_saved_baseline.py"))
        suite["actual_gpu_baseline_rerun_performed"] = False
        suite["conditioning_mode"] = "restore exact persisted BF16-valued tensors; no encoder or RNG rerun"
        (suite_dir / "saved_baseline_validation.json").write_text(json.dumps(matched_report, indent=2) + "\n")
        saved_test_path = Path(__file__).with_name("saved_known_collar_cpu_validation.json")
        saved_test = json.loads(saved_test_path.read_text()) if saved_test_path.is_file() else None
        saved_test_current = bool(saved_test and saved_test.get("status") == "passed"
            and saved_test.get("script_sha256") == file_sha256(Path(__file__).with_name("test_saved_known_collar_baseline_cpu.py"))
            and all(saved_test.get("source_code_sha256", {}).get(name) == file_sha256(Path(__file__).with_name(name))
                    for name in ("known_collar_saved_baseline.py", "run_known_collar_saved_baseline_trial.py",
                                 "known_collar_reference_control.py", "localized_reference_control.py", "reference_control.py")))
        suite["cpu_gate_status"]["saved_baseline_validation_current"] = saved_test_current
        suite["saved_baseline_cpu_validation_sha256"] = file_sha256(saved_test_path) if saved_test_path.is_file() else None
        if args.preflight:
            # Real layout geometry, tiny random weights, and pixels only.
            from test_reference_control_cpu import geometry_case
            from test_localized_reference_cpu import field_case, tiny_case
            from test_known_collar_reference_cpu import full_canvas_geometry
            from conditioning import cpu_preflight
            from sparse_conditioning import cpu_preflight as sparse_cpu_preflight
            suite["reference_layout_cpu_test"] = geometry_case()
            suite["hint_field_cpu_test"] = field_case()
            suite["tiny_localized_cpu_tests"] = [tiny_case(False, "cosine"), tiny_case(True, "cosine"), tiny_case(False, "hard")]
            suite["known_collar_field_cpu_test"]=full_canvas_geometry()
            suite["conditioning_cpu_test"] = cpu_preflight()
            if args.structural_support != "full":
                suite["sparse_cpu_test"] = sparse_cpu_preflight()
            suite.update({"status": "preflight_files_ready" if not missing and parity_current and validation_current and localized_current and collar_current and saved_test_current else "preflight_files_incomplete",
                          "model_weights_loaded": False, "device": "CPU only",
                          "planned_nfe_per_arm": args.steps, "planned_target_token_forwards_per_arm": args.steps * 2304,
                          "planned_reference_image_token_forwards_per_arm": args.steps * 1024,
                          "planned_base_block_forwards_per_arm": args.steps * 32,
                          "planned_control_block_forwards_per_arm": args.steps * 16})
            write("preflight.json")
            print(json.dumps({"status": suite["status"], "device": suite["device"], "missing_files": missing,
                              "cpu_gate_status": suite["cpu_gate_status"], "output": str(suite_dir)}, indent=2), flush=True)
            return
        if missing:
            raise FileNotFoundError(f"Trial files incomplete: {missing}")
        if not parity_current or not validation_current or not localized_current or not collar_current:
            raise RuntimeError("Matching current existing/reference/localized/known-collar CPU gates and runtime hashes are required")
        if not saved_test_current:
            raise RuntimeError("Current saved-baseline restoration and rejection CPU tests are required")
        suite["known_collar_cpu_validation_sha256"]=file_sha256(collar_validation_path)
        suite["localized_cpu_validation_sha256"] = file_sha256(localized_validation_path)
        suite["reference_cpu_validation_sha256"] = file_sha256(validation_path)
        suite["port_cpu_parity_sha256"] = file_sha256(parity_path)
        mx.set_cache_limit(512 * 1024**2)
        mx.set_memory_limit(28 * 1024**3)
        suite["metal_device"] = mx.device_info()
        suite["mlx_version"] = mx.__version__
        log("loading_base", status="running")
        start = time.perf_counter()
        model = QwenImage21Edit(model_path=str(MODEL), quantize=4, lora_paths=None)
        suite["phases"]["base_load_seconds"] = time.perf_counter() - start
        suite["base_load_mlx_peak_bytes"] = mx.get_peak_memory()
        mx.reset_peak_memory()
        log("reference_conditioning")
        start = time.perf_counter()
        def restore_bf16(name):
            result = mx.array(saved[name]).astype(mx.bfloat16)
            if not np.array_equal(tensor_array(result), saved[name]):
                raise AssertionError(f"Restoring saved BF16 tensor changed values: {name}")
            return result
        prompt_embeds = restore_bf16("prompt_embeds")
        slots = mx.array(saved["image_slots"])
        source_latents = restore_bf16("source_latents")
        del model.text_encoder
        model.text_encoder = None
        mx.clear_cache()
        source_prefix_np = tensor_array(source_latents)
        source_prefix_sha = hashlib.sha256(source_prefix_np.tobytes()).hexdigest()
        layout = QwenImage21Layout.create(slots, [(1, 32, 32), (1, 72, 32)], model.transformer.axes)
        suite["layout"] = validate_reference_layout(np, layout, 1024, 2304)
        hint_weights_by_arm={}
        for arm in args.collar_arms:
            weights=mx.array(hint_target_np[arm]).astype(prompt_embeds.dtype)
            effective_hint_np=tensor_array(weights)
            if not np.array_equal(effective_hint_np, saved["expected_effective_top32_hint"]):
                raise AssertionError("Effective top32 hint differs from validated persisted-input comparison")
            joint_hint_np=np.concatenate([np.zeros((1,layout.prefix_length,1),dtype=np.float32),effective_hint_np],axis=1)
            if np.any(joint_hint_np[:,:layout.prefix_length]!=0):
                raise AssertionError("Reference/text prefix hints must remain zero")
            np.savez_compressed(support_dir/"hint_fields"/arm/"effective_hint_weights.npz",target_weights=effective_hint_np,
                                joint_weights=joint_hint_np,requested_target_weights=hint_target_np[arm])
            suite["known_collar_hint_fields"][arm].update({"effective_dtype":str(prompt_embeds.dtype),
                "effective_target_weights_float32_sha256":hashlib.sha256(effective_hint_np.tobytes()).hexdigest(),
                "effective_joint_weights_float32_sha256":hashlib.sha256(joint_hint_np.tobytes()).hexdigest(),
                "joint_prefix_zero":True,"gate_applied_after_all16_hints":True})
            hint_weights_by_arm[arm]=weights
        suite["reference_encoding"] = {"source_shape": [1, 32, 32], "source_latents_shape": [1, 1024, 64],
                                       "source_prefix_sha256": source_prefix_sha, "source_vae_encodes": 0, "saved_source_vae_encodes": 1, "restored_saved_reference": True,
                                       "actual_square_only": True, "opaque_rgba": True}
        suite["prompt_embeds_sha256"] = tensor_sha(prompt_embeds)
        suite["image_slots_bool_sha256"] = hashlib.sha256(np.asarray(slots, dtype=bool).tobytes()).hexdigest()
        np.savez_compressed(support_dir / "reference_conditioning.npz", source_latents=source_prefix_np,
                            prompt_embeds=tensor_array(prompt_embeds), image_slots=np.asarray(slots, dtype=bool))
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
            context = restore_bf16("target_context")
            padded = restore_bf16("padded_image_context")
            target_np = tensor_array(context)
            padded_np = tensor_array(padded)
            info = dict(saved_metrics["control_conditioning"][mode])
            info.update({"restored_saved_context": True, "new_control_encode_count": 0, "new_source_encode_count": 0})
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
        sigmas = mx.array(saved["sigmas"]).astype(mx.float32)
        mx.eval(sigmas)
        if not np.array_equal(tensor_array(sigmas), saved["sigmas"]):
            raise AssertionError("Persisted sigma schedule changed during restore")
        suite["sigmas"] = sigmas.tolist()
        suite["scheduler_target_tokens"] = 2304
        suite["sigma_schedule_source"] = "exact saved baseline 40-step target-only shifted schedule"
        noise = restore_bf16("noise")
        suite["noise_sha256"] = tensor_sha(noise)
        suite["rng_draws_for_noise"] = 0
        np.savez_compressed(support_dir / "target_noise.npz", noise=tensor_array(noise))
        assert_saved_files_unchanged(matched_report)
        outputs = {}
        for mode in args.modes:
            for arm in args.prefix_hints:
                key = f"{mode}/{arm}"
                directory = suite_dir / mode / arm
                directory.mkdir(parents=True, exist_ok=True)
                disabled = True
                hint_weights=hint_weights_by_arm[arm]
                latents = noise
                record = {"status": "running", "mode": mode, "prefix_hint_arm": arm,
                          "prefix_hints_disabled": disabled, "full_prefix_gate_after_all_hints": disabled,
                          "hint_gate_scope": "full text+reference prefix zero; declared known top collar only; unchanged input-guide-localized unknown targets",
                          "known_collar_hint_arm":arm,
                          "localized_hint_weights": suite["known_collar_hint_fields"][arm], "control_scale": args.control_scale,
                          "hint_localization_changes_compute": False,
                          "steps": args.steps, "seed": args.seed, "cfg": 1.0, "adapter": None,
                          "prompt": prompt, "prompt_style": suite["prompt_style"],
                          "source_prefix_sha256": source_prefix_sha, "noise_sha256": suite["noise_sha256"],
                          "context_metadata": context_info[mode], "sigmas": suite["sigmas"],
                          "experimental_reference_extension": True, "experimental_localized_hints": True,
                          "hint_shape": args.hint_shape, "hint_radius_pixels": args.hint_radius,
                          "known_target_hidden_trace_preserved": False, "official_control_recipe": False,
                          "known_bridge_extension": args.known_bridge, "structural_support": args.structural_support,
                          "prefix_cache_enabled": False, "compute_mode": suite["compute_mode"],
                          "transformer_calls": 0, "base_block_calls": 0, "control_block_calls": 0,
                          "target_token_forward_sum": 0, "reference_image_token_forward_sum": 0,
                          "joint_query_token_forward_sum": 0, "source_prefix_exact_each_forward": True,
                          "step_records": [], "phases": {}, "diagnostic_control": True,
                          "shared_canvas_comparable": False}
                suite["runs"][key] = record
                mx.reset_peak_memory()
                log("denoising_start", current_arm=key)
                start = time.perf_counter()
                for step in range(args.steps):
                    step_start = time.perf_counter()
                    if args.known_bridge:
                        latents = target_known_bridge(latents, noise, contexts[mode], sigmas[step])
                    model_input = assemble_reference_input(source_latents, latents)
                    if not np.array_equal(tensor_array(model_input[:, :1024]), source_prefix_np):
                        raise AssertionError("Transformer reference prefix changed")
                    timestep = (sigmas[step:step + 1] * 1000).astype(latents.dtype) / 1000
                    record["transformer_calls"] += 1
                    record["base_block_calls"] += 32
                    record["control_block_calls"] += 16
                    record["target_token_forward_sum"] += 2304
                    record["reference_image_token_forward_sum"] += 1024
                    record["joint_query_token_forward_sum"] += layout.prefix_length + 2304
                    velocity = known_collar_reference_controlled_forward(model.transformer, branch, model_input, prompt_embeds,
                        timestep, layout, padded_contexts[mode], target_hint_weights=hint_weights,
                        control_scale=args.control_scale, cache=None)
                    if velocity.shape != (1, 2304, 64):
                        raise AssertionError("Reference control must predict target tokens only")
                    latents = (latents.astype(mx.float32) + (sigmas[step + 1] - sigmas[step]) * velocity.astype(mx.float32)).astype(prompt_embeds.dtype)
                    if args.known_bridge:
                        latents = target_known_bridge(latents, noise, contexts[mode], sigmas[step + 1])
                    mx.eval(latents)
                    if not bool(mx.all(mx.isfinite(latents))):
                        raise FloatingPointError("Nonfinite experimental reference target state")
                    row = {"step": step + 1, "sigma": float(sigmas[step]), "next_sigma": float(sigmas[step + 1]),
                           "model_timestep": float(np.asarray(timestep.astype(mx.float32))[0]),
                           "target_tokens_forwarded": 2304, "reference_image_tokens_forwarded": 1024,
                           "joint_query_tokens": layout.prefix_length + 2304,
                           "base_blocks": 32, "control_blocks": 16, "source_prefix_input_exact": True,
                           "core_seconds": time.perf_counter() - step_start}
                    record["step_records"].append(row)
                    (directory / "metrics.json").write_text(json.dumps(record, indent=2) + "\n")
                    print(json.dumps({"phase": "denoising", "arm": key, **row}), flush=True)
                record["phases"]["denoising_seconds"] = time.perf_counter() - start
                record["denoising_mlx_peak_bytes"] = mx.get_peak_memory()
                final = tensor_array(latents)
                np.savez_compressed(directory / "final_latents.npz", latents=final, final_canvas_ids=np.arange(2304,dtype=np.int32),
                                    source_rect_xyxy=np.asarray(rect),sigmas=np.asarray(sigmas),source_prefix_latents=source_prefix_np)
                record["final_latents_sha256"] = hashlib.sha256(final.tobytes()).hexdigest()
                if args.known_bridge and not np.array_equal(final[:, 20 * 32:52 * 32], tensor_array(contexts[mode])[:, 20 * 32:52 * 32, 65:]):
                    raise AssertionError("Target known bridge failed sigma-zero source restoration")
                if not np.array_equal(tensor_array(source_latents), source_prefix_np):
                    raise AssertionError("Denoising altered the fixed reference source")
                outputs[key] = final
                record["status"] = "denoised"
                mx.clear_cache()
                log("denoising_complete", current_arm=key, nfe=record["transformer_calls"])
        del branch, contexts, padded_contexts, context, padded, latents, velocity, model_input
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
            (directory / "metrics.json").write_text(json.dumps(record, indent=2) + "\n")
            mx.clear_cache()
        if any(record["transformer_calls"] != args.steps for record in suite["runs"].values()):
            raise AssertionError("Localized reference NFE mismatch")
        assert_saved_files_unchanged(matched_report)
        suite["saved_baseline_files_unchanged_after_trial"] = True
        log("complete", status="success")
    except BaseException as exc:
        suite.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        write()
        (suite_dir / "traceback.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
