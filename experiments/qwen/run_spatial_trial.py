"""Paired FULL/GROW Qwen2.1 trials with actual target-token insertion.

One model load, the same 512-square reference prefix, seed, prompt and global
six-step schedule. GROW omits future target tokens from transformer forwards.
Late insertion uses an untrained latent-edge extrapolation initializer.
Inner target regions continue refining at the same global sigma.

No changes to the installed runtime or the baseline runner are required.
--preflight and --layout-smoke use CPU arrays and never load model weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import time
import traceback
from dataclasses import replace
from pathlib import Path

from run_trial import MODEL, PROMPT, REVISION, ROOT, TURBO_REVISION


HEIGHTS = (640, 640, 896, 896, 1152, 1152)
PREVIEW_STEPS = (2, 4, 6)
ADAPTER = ROOT / ".build/models/qwen/viggle/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors"


def active_ids(np, width: int, height: int, active_height: int):
    """Raster IDs in the final canvas, never local/recentered IDs."""
    if width % 32 or height % 32 or active_height % 32:
        raise ValueError("All dimensions must be multiples of 32 pixels")
    if active_height > height or (height - active_height) % 32:
        raise ValueError("Active window must be centered on the final latent grid")
    y0 = (height - active_height) // 2
    w = width // 16
    ids = np.arange((y0 // 16) * w, ((y0 + active_height) // 16) * w, dtype=np.int32)
    return ids, (0, y0, width, y0 + active_height)


def active_layout(mx, np, layout_type, slots, source_shape, full_layout, ids, active_height, width, axes):
    layout = layout_type.create(slots, [source_shape, (1, active_height // 16, width // 16)], axes)
    if layout.prefix_length != full_layout.prefix_length or layout.target_tokens != len(ids):
        raise AssertionError("Growing layout changed the fixed reference prefix")
    p = full_layout.prefix_length
    gather = mx.array(p + ids, dtype=mx.int32)
    rope = tuple(mx.concatenate([r[:p], r[gather]], axis=0) for r in full_layout.rope)
    return replace(layout, rope=rope)


def six_sigmas(np, final_tokens: int):
    nodes = np.array([1.0, 0.9375, 0.875, 0.75, 0.5, 0.25], dtype=np.float64)
    mu = 0.5 + (0.9 - 0.5) * (final_tokens - 256) / (8192 - 256)
    shifted = math.exp(mu) / (math.exp(mu) + (1.0 / nodes - 1.0))
    return np.concatenate([shifted, [0.0]]).astype(np.float32)


def cpu_layout_smoke(mx, np, layout_type):
    """Exercise real layout construction and target ID preservation on CPU."""
    mx.set_default_device(mx.cpu)
    slots = mx.array([False] * 7 + [True] * 256 + [False] * 10)
    axes = (16, 56, 56)
    source_shape = (1, 32, 32)
    full = layout_type.create(slots, [source_shape, (1, 72, 32)], axes)
    old_ids = None
    checks = []
    for h in (640, 896, 1152):
        ids, rect = active_ids(np, 512, 1152, h)
        layout = active_layout(mx, np, layout_type, slots, source_shape, full, ids, h, 512, axes)
        p = full.prefix_length
        for current, reference in zip(layout.rope, full.rope):
            if not np.array_equal(np.asarray(current[:p]), np.asarray(reference[:p])):
                raise AssertionError("Prefix rotary positions changed")
            if not np.array_equal(np.asarray(current[p:]), np.asarray(reference)[p + ids]):
                raise AssertionError("Target rotary positions do not match final raster IDs")
        if old_ids is not None:
            positions = np.searchsorted(ids, old_ids)
            if not np.array_equal(ids[positions], old_ids):
                raise AssertionError("Growth lost existing target IDs")
            # Indexed insertion must retain every existing state exactly.
            old = mx.array(old_ids.astype(np.float32)[None, :, None])
            grown = mx.zeros((1, len(ids), 1), dtype=mx.float32)
            grown[:, mx.array(positions, dtype=mx.int32)] = old
            if not np.array_equal(np.asarray(grown)[0, positions, 0], old_ids):
                raise AssertionError("Existing states changed during target insertion")
        checks.append({"height": h, "target_tokens": len(ids), "prefix_length": p, "window_xyxy": rect})
        old_ids = ids
    return {"device": "CPU", "model_weights_loaded": False, "checks": checks, "status": "passed"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=("track1", "track2", "track3"), default="track3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quantize", type=int, choices=(4, 8), default=4)
    parser.add_argument("--order", choices=("full-first", "grow-first"), default="full-first")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--layout-smoke", action="store_true")
    parser.add_argument("--output", type=Path, help="Optional suite directory containing full/ and grow/ outputs")
    args = parser.parse_args()
    suite_dir = args.output or ROOT / f"experiments/qwen/spatial_runs/q{args.quantize}/{args.track}"
    suite_dir.mkdir(parents=True, exist_ok=True)
    mode_dirs = {
        mode: (suite_dir / mode if args.output else ROOT / f"experiments/evaluation/results/qwen21-spatial-r256-{mode}-q{args.quantize}/{args.track}")
        for mode in ("full", "grow")
    }
    suite = {"status": "preflight", "track": args.track, "seed": args.seed, "runs": {}, "phases": {}}
    begun = time.perf_counter()

    def write_suite(name="paired_metrics.json"):
        suite["elapsed_seconds"] = time.perf_counter() - begun
        suite["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (suite_dir / name).write_text(json.dumps(suite, indent=2))

    def log(phase, **data):
        suite.update(data)
        suite["phase"] = phase
        write_suite()
        print(json.dumps({"phase": phase, "elapsed_seconds": round(suite["elapsed_seconds"], 3), **data}), flush=True)

    try:
        import mlx.core as mx
        import numpy as np
        from PIL import Image
        from mlx.utils import tree_flatten
        from mflux.models.common.vae.vae_util import VAEUtil
        from mflux.models.common.vae.tiling_config import TilingConfig
        from mflux.models.qwen21.reference import QwenImage21Edit
        from mflux.models.qwen21.reference.latent_creator.qwen_image21_latent_creator import QwenImage21LatentCreator
        from mflux.models.qwen21.reference.model.qwen_image21_transformer.layout import QwenImage21Layout

        if args.preflight or args.layout_smoke:
            # Explicitly avoid GPU work, including the layout/RoPE smoke arrays.
            mx.set_default_device(mx.cpu)
        shared = json.loads((ROOT / "experiments/evaluation/inputs/inputs.json").read_text())
        track = next(item for item in shared["tracks"] if item["id"] == args.track)
        width, height = shared["canvas_size"]
        rect = shared["source_rect_xyxy"]
        source = Image.open(track["source_512"]).convert("RGBA")
        if (width, height) != (512, 1152) or source.size != (512, 512) or rect != [0, 320, 512, 832]:
            raise ValueError("This controlled prototype expects the shared centered 512×1152 canvas")
        prompt = PROMPT
        descriptions_path = ROOT / "experiments/flux2/prompts.json"
        if descriptions_path.is_file():
            prompt += " " + json.loads(descriptions_path.read_text())[args.track]
        expected = [
            "transformer/config.json", "text_encoder/config.json", "vae/config.json",
            "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
            "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
            *[f"text_encoder/model-0000{i}-of-00004.safetensors" for i in range(1, 5)],
            "vae/diffusion_pytorch_model.safetensors", "processor/tokenizer.json",
        ]
        missing = [str(MODEL / item) for item in expected if not (MODEL / item).is_file()]
        if not ADAPTER.is_file():
            missing.append(str(ADAPTER))
        if missing:
            raise FileNotFoundError(f"Incomplete checkpoint: {missing}")
        suite.update({"prompt": prompt, "mlx_version": mx.__version__, "mode_output_directories": {k: str(v) for k, v in mode_dirs.items()}})
        if args.preflight or args.layout_smoke:
            suite["layout_smoke"] = cpu_layout_smoke(mx, np, QwenImage21Layout)
            suite["status"] = "preflight_passed"
            suite["model_weights_loaded"] = False
            write_suite("preflight.json")
            print(json.dumps(suite, indent=2), flush=True)
            return

        for directory in mode_dirs.values():
            (directory / "previews").mkdir(parents=True, exist_ok=True)
        mx.set_cache_limit(512 * 1024**2)
        mx.set_memory_limit(28 * 1024**3)
        suite["metal_device"] = mx.device_info()
        log("loading_weights", status="running")
        start = time.perf_counter()
        model = QwenImage21Edit(model_path=str(MODEL), quantize=args.quantize,
                               lora_paths=[str(ADAPTER)], lora_scales=[1.0], bake_lora=False)
        suite["phases"]["load_seconds"] = time.perf_counter() - start
        suite["loaded_component_parameter_bytes"] = {
            name: sum(v.nbytes for _, v in tree_flatten(getattr(model, name).parameters()))
            for name in ("transformer", "text_encoder", "vae")
        }
        suite["load_mlx_peak_bytes"] = mx.get_peak_memory()
        mx.reset_peak_memory()
        mx.set_memory_limit(23 * 1024**3)
        log("conditioning")
        start = time.perf_counter()
        # Both arms see the actual square source only: no future gray prefix tokens.
        prompt_embeds, slots = model._encode_prompt(prompt, [source])
        mx.eval(prompt_embeds)
        del model.text_encoder
        model.text_encoder = None
        mx.clear_cache()
        pixels = mx.array(np.asarray(source).astype(np.float32) / 127.5 - 1).transpose(2, 0, 1)[None]
        source_latents = QwenImage21LatentCreator.pack_latents(model.vae.encode(pixels)).astype(prompt_embeds.dtype)
        mx.eval(source_latents)
        source_shape = (1, 32, 32)
        if source_latents.shape != (1, 1024, 64):
            raise ValueError(f"Unexpected square-source latent shape: {source_latents.shape}")
        full_layout = QwenImage21Layout.create(slots, [source_shape, (1, 72, 32)], model.transformer.axes)
        full_tokens = full_layout.target_tokens
        mx.random.seed(args.seed)
        # One absolute-position noise table, shared by both arms. Future GROW
        # entries are never passed to its transformer until their activation.
        noise = mx.random.normal((1, full_tokens, 64)).astype(prompt_embeds.dtype)
        ys = np.arange(72).clip(20, 51) - 20
        edge_source_ids = (ys[:, None] * 32 + np.arange(32)[None, :]).reshape(-1).astype(np.int32)
        edge_guess = source_latents[:, mx.array(edge_source_ids)]
        sigmas_np = six_sigmas(np, full_tokens)
        sigmas = mx.array(sigmas_np, dtype=mx.float32)
        mx.eval(noise, edge_guess, *full_layout.rope)
        suite["phases"]["conditioning_seconds"] = time.perf_counter() - start
        suite["conditioning_mlx_peak_bytes"] = mx.get_peak_memory()
        suite.update({"sigmas": sigmas_np.tolist(), "source_prefix_latent_tokens": 1024,
                      "joint_prefix_tokens": full_layout.prefix_length, "full_target_tokens": full_tokens,
                      "source_prefix_sha256": hashlib.sha256(np.asarray(source_latents.astype(mx.float32)).tobytes()).hexdigest(),
                      "noise_sha256": hashlib.sha256(np.asarray(noise.astype(mx.float32)).tobytes()).hexdigest(),
                      "hash_representation": "float32 values converted from the shared inference tensors",
                      "growth_heights_by_step": list(HEIGHTS), "preview_steps": list(PREVIEW_STEPS)})
        source_np = np.asarray(source_latents.astype(mx.float32))
        order = ("full", "grow") if args.order == "full-first" else ("grow", "full")
        core_outputs = {}

        for order_index, mode in enumerate(order, 1):
            directory = mode_dirs[mode]
            record = {
                "model_repo": "Qwen/Qwen-Image-2.1", "model_revision": REVISION,
                "runtime": "mflux native MLX QwenImage21Edit; experimental actual target-token growth",
                "runtime_git_revision": json.loads((ROOT / "experiments/qwen/runtime.json").read_text())["mflux_git_revision"],
                "adapter_repo": "Viggle/Qwen-Image-2.1-viggle-turbo", "adapter_revision": TURBO_REVISION,
                "adapter_file": str(ADAPTER), "adapter_rank": 256, "adapter_baked": False,
                "quantization": args.quantize, "track": args.track, "seed": args.seed,
                "steps": 6, "prompt": prompt, "width": width, "height": height,
                "source_rect_xyxy": rect, "status": "running", "spatial_mode": mode,
                "diagnostic_control": True, "shared_canvas_comparable": False,
                "paired_suite": str(suite_dir / "paired_metrics.json"), "generation_order_within_process": order_index,
                "source_prefix": "actual 512-square source; fixed 1024 latent image tokens",
                "source_prefix_latent_tokens": 1024, "joint_prefix_tokens": full_layout.prefix_length,
                "source_prefix_sha256": suite["source_prefix_sha256"], "noise_sha256": suite["noise_sha256"],
                "source_encoding_shared_between_arms": True, "fresh_prefix_cache_per_arm": True,
                "sigmas": sigmas_np.tolist(), "schedule_shift_target_tokens": full_tokens,
                "target_tokens_future_absent_from_forward": mode == "grow",
                "late_activation_initializer": "(1-sigma)*source-latent edge extrapolation + sigma*shared noise; untrained heuristic" if mode == "grow" else None,
                "inner_region_denoising": "continues refining at the same global target sigma",
                "mask_method": "known square-source latent flow bridge restored before/after every step",
                "trained_spatial_time_schedule": False, "transformer_calls": 0,
                "preview_definition": "genuine pre-step z_sigma - sigma*predicted_velocity; source latent restored",
                "metal_device": suite["metal_device"], "mlx_version": mx.__version__,
                "phases": {"load_seconds": suite["phases"]["load_seconds"],
                           "conditioning_seconds": suite["phases"]["conditioning_seconds"]},
                "shared_phase_scope": "load and conditioning executed once for the entire pair; values attributed to both arms for context",
                "load_mlx_peak_bytes": suite["load_mlx_peak_bytes"],
                "conditioning_mlx_peak_bytes": suite["conditioning_mlx_peak_bytes"],
                "step_records": [], "previews": [],
            }
            suite["runs"][mode] = record
            cache = []  # Each arm pays exactly one prefix-cache construction.
            latents = None
            previous_ids = None
            previews = []
            core_seconds = capture_seconds = 0.0
            wall_start = time.perf_counter()
            mx.reset_peak_memory()
            log("denoising_start", current_mode=mode)
            for step in range(6):
                step_start = time.perf_counter()
                active_h = height if mode == "full" else HEIGHTS[step]
                ids, window = active_ids(np, width, height, active_h)
                index = mx.array(ids, dtype=mx.int32)
                added = len(ids) if previous_ids is None else len(ids) - len(previous_ids)
                if previous_ids is None or len(ids) != len(previous_ids):
                    candidate = ((1.0 - sigmas[step]) * edge_guess[:, index].astype(mx.float32)
                                 + sigmas[step] * noise[:, index].astype(mx.float32)).astype(prompt_embeds.dtype)
                    if previous_ids is not None:
                        positions = np.searchsorted(ids, previous_ids).astype(np.int32)
                        if not np.array_equal(ids[positions], previous_ids):
                            raise AssertionError("Growth changed or dropped existing absolute raster IDs")
                        candidate[:, mx.array(positions)] = latents
                    latents = candidate
                layout = active_layout(mx, np, QwenImage21Layout, slots, source_shape,
                                       full_layout, ids, active_h, width, model.transformer.axes)
                known_mask_np = ((ids // 32 >= 20) & (ids // 32 < 52))
                known_mask = mx.array(known_mask_np[None, :, None])
                known_clean = edge_guess[:, index]
                active_noise = noise[:, index]
                known_now = ((1.0 - sigmas[step]) * known_clean.astype(mx.float32)
                             + sigmas[step] * active_noise.astype(mx.float32)).astype(prompt_embeds.dtype)
                latents = mx.where(known_mask, known_now, latents)
                before = latents
                had_cache = bool(cache)
                # Actual input is square-reference prefix plus the active target.
                # No full target tensor is passed then hidden by an update mask.
                model_input = mx.concatenate([source_latents, latents], axis=1)
                if model_input.shape[1] != 1024 + len(ids):
                    raise AssertionError("Forward includes unexpected/future image tokens")
                timestep = (sigmas[step:step + 1] * 1000).astype(latents.dtype) / 1000
                prediction = model.transformer(model_input, prompt_embeds, timestep, layout, cache)
                if prediction.shape != latents.shape:
                    raise AssertionError("Transformer predicted tokens outside the active target")
                advanced = (latents.astype(mx.float32) + (sigmas[step + 1] - sigmas[step])
                            * prediction.astype(mx.float32)).astype(prompt_embeds.dtype)
                known_next = ((1.0 - sigmas[step + 1]) * known_clean.astype(mx.float32)
                              + sigmas[step + 1] * active_noise.astype(mx.float32)).astype(prompt_embeds.dtype)
                latents = mx.where(known_mask, known_next, advanced)
                mx.eval(latents)
                step_seconds = time.perf_counter() - step_start
                core_seconds += step_seconds
                record["transformer_calls"] += 1
                if len(cache) != len(model.transformer.transformer_blocks) or any(k.shape[2] != full_layout.prefix_length for k, _ in cache):
                    raise AssertionError("Prefix cache has the wrong layer/token count")
                step_record = {
                    "step": step + 1, "sigma": float(sigmas_np[step]), "next_sigma": float(sigmas_np[step + 1]),
                    "active_height": active_h, "active_window_xyxy": window, "target_tokens_forwarded": len(ids),
                    "future_target_tokens_absent": full_tokens - len(ids), "newly_activated_tokens": added,
                    "input_image_latent_tokens": int(model_input.shape[1]),
                    "attention_query_tokens": len(ids) if had_cache else full_layout.prefix_length + len(ids),
                    "attention_key_value_tokens": full_layout.prefix_length + len(ids),
                    "core_step_seconds": step_seconds,
                }
                record["step_records"].append(step_record)
                if step + 1 in PREVIEW_STEPS:
                    capture_start = time.perf_counter()
                    clean = (before.astype(mx.float32) - sigmas[step] * prediction.astype(mx.float32)).astype(prompt_embeds.dtype)
                    clean = mx.where(known_mask, known_clean, clean)
                    mx.eval(clean)
                    clean_np = np.asarray(clean.astype(mx.float32)).copy()
                    if not np.isfinite(clean_np).all():
                        raise FloatingPointError("Predicted-clean preview latent contains NaN/Inf")
                    latent_path = directory / "previews" / f"step{step + 1:02d}_predicted_clean.npz"
                    np.savez_compressed(latent_path, latents=clean_np, final_canvas_ids=ids, sigma=sigmas_np[step], window_xyxy=np.asarray(window))
                    previews.append((step + 1, active_h, window, clean_np))
                    record["previews"].append({"step": step + 1, "active_height": active_h, "active_window_xyxy": window,
                                               "latent_path": str(latent_path), "definition": record["preview_definition"]})
                    capture_seconds += time.perf_counter() - capture_start
                    del clean
                previous_ids = ids
                record["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                (directory / "metrics.json").write_text(json.dumps(record, indent=2))
                print(json.dumps({"phase": "denoising", "mode": mode, **step_record}), flush=True)
            final_np = np.asarray(latents.astype(mx.float32)).copy()
            if not np.array_equal(final_np[:, 20 * 32:52 * 32], source_np):
                raise AssertionError("Final known-source latent differs from its square encoding")
            record["known_source_latents_exact_at_sigma_zero"] = True
            record["phases"].update({"denoising_seconds": core_seconds, "core_loop_wall_seconds": time.perf_counter() - wall_start,
                                     "preview_capture_seconds": capture_seconds})
            record["denoising_timing_scope"] = "sum of six active-layout/state/transformer/Euler steps; preview capture and all VAE decoding excluded"
            record["denoising_mlx_peak_bytes"] = mx.get_peak_memory()
            record["status"] = "denoised"
            core_outputs[mode] = (final_np, previews)
            (directory / "metrics.json").write_text(json.dumps(record, indent=2))
            del cache, latents, before, prediction, advanced, model_input
            mx.clear_cache()
            log("denoising_complete", current_mode=mode, core_seconds=core_seconds, nfe=record["transformer_calls"])

        # Both core loops finish before any preview decode. This keeps speed
        # comparison independent of image preview work and releases the DiT.
        del model.transformer
        model.transformer = None
        mx.clear_cache()
        tiling = TilingConfig(vae_decode_tiles_per_dim=2, vae_decode_tile_size=512, vae_decode_overlap=4)

        def decode(array, active_h):
            packed = mx.array(array).astype(mx.float32)
            unpacked = QwenImage21LatentCreator.unpack_latents(packed, active_h, width)
            decoded = VAEUtil.decode(model.vae, unpacked, tiling)
            mx.eval(decoded)
            if decoded.ndim == 5:
                decoded = decoded[:, :, 0]
            rgb = np.asarray(decoded[0, :3].transpose(1, 2, 0))
            if rgb.shape != (active_h, width, 3) or not np.isfinite(rgb).all():
                raise FloatingPointError(f"Invalid decoded RGB shape/content: {rgb.shape}")
            return Image.fromarray(np.clip((rgb + 1) * 127.5, 0, 255).round().astype(np.uint8), mode="RGB")

        source_rgb = source.convert("RGB")
        for mode in order:
            record = suite["runs"][mode]
            directory = mode_dirs[mode]
            final_np, previews = core_outputs[mode]
            mx.reset_peak_memory()
            log("final_decode", current_mode=mode)
            start = time.perf_counter()
            raw = decode(final_np, height)
            raw.save(directory / "raw.png")
            raw_error = np.abs(np.asarray(raw.crop(rect), dtype=np.int16) - np.asarray(source_rgb, dtype=np.int16))
            record.update({"raw_source_region_mae_255": float(raw_error.mean()), "raw_source_region_max_error_255": int(raw_error.max()),
                           "raw_source_top16_mae_255": float(raw_error[:16].mean()), "raw_source_bottom16_mae_255": float(raw_error[-16:].mean()),
                           "raw_source_inner_mae_255": float(raw_error[16:-16].mean()), "decoded_pixels_finite": True})
            result = raw.copy()
            result.paste(source_rgb, (rect[0], rect[1]))
            result.save(directory / "composite.png")
            if not np.array_equal(np.asarray(result.crop(rect)), np.asarray(source_rgb)):
                raise AssertionError("Original resized source pixels changed during compositing")
            record["phases"]["decode_seconds"] = time.perf_counter() - start
            record["decoding_mlx_peak_bytes"] = mx.get_peak_memory()
            mx.reset_peak_memory()
            log("preview_decode", current_mode=mode)
            start = time.perf_counter()
            for preview_record, (step, active_h, window, array) in zip(record["previews"], previews):
                preview_start = time.perf_counter()
                preview_raw = decode(array, active_h)
                preview_raw_path = directory / "previews" / f"step{step:02d}_predicted_clean_raw.png"
                preview_raw.save(preview_raw_path)
                preview_composite = preview_raw.copy()
                local_source_y = rect[1] - window[1]
                preview_composite.paste(source_rgb, (0, local_source_y))
                preview_path = directory / "previews" / f"step{step:02d}_predicted_clean_composite.png"
                preview_composite.save(preview_path)
                preview_record.update({"raw_path": str(preview_raw_path), "composite_path": str(preview_path),
                                       "decode_seconds": time.perf_counter() - preview_start,
                                       "source_pixels_exact": np.array_equal(np.asarray(preview_composite.crop((0, local_source_y, width, local_source_y + 512))), np.asarray(source_rgb))})
            record["phases"]["preview_decode_seconds"] = time.perf_counter() - start
            record["preview_decoding_mlx_peak_bytes"] = mx.get_peak_memory()
            record.update({"status": "success", "source_pixels_exact": True,
                           "raw_path": str(directory / "raw.png"), "composite_path": str(directory / "composite.png"),
                           "elapsed_seconds": time.perf_counter() - begun,
                           "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                           "mlx_peak_bytes": max(record[k] for k in ("conditioning_mlx_peak_bytes", "denoising_mlx_peak_bytes", "decoding_mlx_peak_bytes")),
                           "output_sha256": hashlib.sha256((directory / "composite.png").read_bytes()).hexdigest()})
            record["generation_seconds_excluding_load_previews"] = sum(record["phases"][k] for k in ("conditioning_seconds", "denoising_seconds", "decode_seconds"))
            (directory / "metrics.json").write_text(json.dumps(record, indent=2))
            mx.clear_cache()
        suite["paired_comparison"] = {
            "nfe": {mode: suite["runs"][mode]["transformer_calls"] for mode in order},
            "target_token_forward_sum": {mode: sum(r["target_tokens_forwarded"] for r in suite["runs"][mode]["step_records"]) for mode in order},
            "core_seconds": {mode: suite["runs"][mode]["phases"]["denoising_seconds"] for mode in order},
            "source_prefix_matched": True, "noise_prompt_sigmas_matched": True,
            "timing_limit": "single process and seed; execution order/thermal/kernel warm-up can affect time; previews excluded",
            "quality_limit": "GROW late-token initialization is an untrained heuristic; no region is claimed finalized before the final step",
        }
        log("complete", status="success")
    except BaseException as exc:
        suite.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        write_suite()
        (suite_dir / "traceback.txt").write_text(traceback.format_exc())
        for mode, record in suite.get("runs", {}).items():
            if record.get("status") != "success":
                record.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
                (mode_dirs[mode] / "metrics.json").write_text(json.dumps(record, indent=2))
        raise


if __name__ == "__main__":
    main()
