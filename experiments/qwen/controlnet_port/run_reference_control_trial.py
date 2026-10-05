#!/usr/bin/env python3
"""Experimental square-reference ControlNet trial with full-canvas denoising.

This extends the official text-only control distribution: real square image
tokens precede the target, with literal zero129 control rows for those image
tokens only. A separately labelled arm disables completed hints on the full
text/image prefix. Both arms recompute the full joint stream without caching.
Q4 base, original BF16 control, CFG1, no LoRA, and target-resolution scheduler.
Optional target known bridge and sparse structural support are extra declared
extensions. --preflight is CPU-only and never loads checkpoint/model weights.
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
    parser.add_argument("--steps", type=int, choices=(20, 40), default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modes", nargs="+", choices=("mask-only", "source-canny", "tangent-canny"), default=["tangent-canny"])
    parser.add_argument("--prefix-hints", nargs="+", choices=("full-control", "prefix-hints-disabled"),
                        default=["full-control", "prefix-hints-disabled"])
    parser.add_argument("--source-canny", type=Path)
    parser.add_argument("--tangent-canny", type=Path)
    parser.add_argument("--structural-support", choices=("full", "known", "tangent"), default="full")
    parser.add_argument("--extrapolated-guide", type=Path)
    parser.add_argument("--support-dilation", type=int, default=64)
    parser.add_argument("--prompt-style", choices=("native", "outpaint"), default="native")
    parser.add_argument("--prompt", help="Complete text override; official baseline teacher prompt is default")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--known-bridge", action="store_true", help="Additional target-only known-latent flow bridge")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if len(set(args.modes)) != len(args.modes) or len(set(args.prefix_hints)) != len(args.prefix_hints):
        parser.error("Modes and prefix-hint arms must be unique")
    if not 0 <= args.support_dilation <= 1152:
        parser.error("Support dilation must be in [0,1152]")
    suffix = "-knownbridge" if args.known_bridge else ""
    suite_dir = args.output or ROOT / f"experiments/qwen/controlnet_runs/2026-10-05/{args.track}-reference{args.steps}-{args.structural_support}{suffix}"
    if (suite_dir / "reference_control_metrics.json").exists():
        raise FileExistsError(f"Refusing to overwrite prior trial: {suite_dir}")
    suite_dir.mkdir(parents=True, exist_ok=True)
    begun = time.perf_counter()
    source_paths = {name: Path(__file__).with_name(name) for name in
                    ("control.py", "conditioning.py", "reference_control.py", "sparse_conditioning.py")}
    suite = {"status": "preflight", "track": args.track, "seed": args.seed, "steps": args.steps,
             "runs": {}, "phases": {}, "runner_sha256": file_sha256(__file__),
             "source_code_sha256": {name: file_sha256(path) for name, path in source_paths.items()},
             "model_repo": "Qwen/Qwen-Image-2.1", "model_revision": REVISION,
             "base_quantization": 4, "control_quantization": None, "cfg": 1.0, "adapter": None,
             "experimental_reference_extension": True, "official_control_recipe": False,
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
    if "prefix-hints-disabled" in args.prefix_hints:
        suite["untrained_distribution_changes"].append("optional post-chain hint gate on full text+reference prefix")
    if args.structural_support != "full":
        suite["untrained_distribution_changes"].append("optional sparse structural64 support after full map VAE encode")
    if args.known_bridge:
        suite["untrained_distribution_changes"].append("optional per-step target known-latent flow bridge")

    def write(name="reference_control_metrics.json"):
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
        from reference_control import (assemble_reference_input, reference_controlled_forward,
                                       target_known_bridge, validate_reference_layout)
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
        if args.structural_support == "tangent" and not guide_path.is_file():
            missing.append(str(guide_path))
        if not checkpoint.is_file() or checkpoint.stat().st_size != CHECKPOINT_BYTES:
            missing.append(str(checkpoint))
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
        suite["cpu_gate_status"] = {"existing_port_parity_current": parity_current, "reference_validation_current": validation_current}
        if args.preflight:
            # Real layout geometry, tiny random weights, and pixels only.
            from test_reference_control_cpu import geometry_case, tiny_case
            from conditioning import cpu_preflight
            from sparse_conditioning import cpu_preflight as sparse_cpu_preflight
            suite["reference_layout_cpu_test"] = geometry_case()
            suite["tiny_reference_cpu_tests"] = [tiny_case(False), tiny_case(True)]
            suite["conditioning_cpu_test"] = cpu_preflight()
            if args.structural_support != "full":
                suite["sparse_cpu_test"] = sparse_cpu_preflight()
            suite.update({"status": "preflight_files_ready" if not missing and parity_current and validation_current else "preflight_files_incomplete",
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
        if not parity_current or not validation_current:
            raise RuntimeError("Run existing port parity and test_reference_control_cpu.py; matching passed CPU gates are required")
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
        suite["reference_encoding"] = {"source_shape": [1, 32, 32], "source_latents_shape": [1, 1024, 64],
                                       "source_prefix_sha256": source_prefix_sha, "source_vae_encodes": 1,
                                       "actual_square_only": True, "opaque_rgba": True}
        suite["prompt_embeds_sha256"] = tensor_sha(prompt_embeds)
        suite["image_slots_bool_sha256"] = hashlib.sha256(np.asarray(slots, dtype=bool).tobytes()).hexdigest()
        np.savez_compressed(support_dir / "reference_conditioning.npz", source_latents=source_prefix_np,
                            prompt_embeds=tensor_array(prompt_embeds), image_slots=np.asarray(slots, dtype=bool))
        del pixels
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
        config = Config(model_config=model.model_config, num_inference_steps=args.steps,
                        width=width, height=height, guidance=1.0)
        sigmas = config.scheduler.sigmas
        suite["sigmas"] = sigmas.tolist()
        suite["scheduler_target_tokens"] = 2304
        mx.random.seed(args.seed)
        noise = mx.random.normal((1, 2304, 64)).astype(prompt_embeds.dtype)
        mx.eval(noise)
        suite["noise_sha256"] = tensor_sha(noise)
        np.savez_compressed(support_dir / "target_noise.npz", noise=tensor_array(noise))
        outputs = {}
        for mode in args.modes:
            for arm in args.prefix_hints:
                key = f"{mode}/{arm}"
                directory = suite_dir / mode / arm
                directory.mkdir(parents=True, exist_ok=True)
                disabled = arm == "prefix-hints-disabled"
                latents = noise
                record = {"status": "running", "mode": mode, "prefix_hint_arm": arm,
                          "prefix_hints_disabled": disabled, "full_prefix_gate_after_all_hints": disabled,
                          "hint_gate_scope": "full text+reference joint prefix" if disabled else "none; whole-joint trained-style hints",
                          "steps": args.steps, "seed": args.seed, "cfg": 1.0, "adapter": None,
                          "prompt": prompt, "prompt_style": suite["prompt_style"],
                          "source_prefix_sha256": source_prefix_sha, "noise_sha256": suite["noise_sha256"],
                          "context_metadata": context_info[mode], "sigmas": suite["sigmas"],
                          "experimental_reference_extension": True, "official_control_recipe": False,
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
                    velocity = reference_controlled_forward(model.transformer, branch, model_input, prompt_embeds,
                        timestep, layout, padded_contexts[mode], control_scale=1.0,
                        prefix_hints_disabled=disabled, cache=None)
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
                np.savez_compressed(directory / "final_latents.npz", latents=final)
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
            raise AssertionError("Experimental reference NFE mismatch")
        log("complete", status="success")
    except BaseException as exc:
        suite.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        write()
        (suite_dir / "traceback.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
