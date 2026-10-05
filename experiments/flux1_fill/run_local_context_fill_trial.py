#!/usr/bin/env python3
"""Two independent native FLUX Fill local-context calls; no growth/streaming.

Upper512x448 has unknown320 then original source top128. Lower512x448 has
original source bottom128 then unknown320. Native384-channel inputs,50 steps,
guidance30 and seed42 per call. Both raw448 crops are retained before assembly.
Only generated external320 regions surround the untouched original512 square.
This changes crop geometry, context and prompts; not a single-factor comparison
against the full-canvas teacher. Local verified checkpoint only, no downloads.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
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

from run_fill_trial import ROOT, MODEL, MODEL_REVISION, checkpoint_inventory, verify_manifest, file_sha256
from local_context_inputs import (PATCH_SIZE, FINAL_SIZE, SOURCE_RECT, prepare_patches,
                                  assemble_composite, assemble_raw_context_diagnostic)

PROMPTS = {
    "upper": "Continue the adjacent lakeside background upward into the masked region: continuous lake water, low shrubs and a thin metal guardrail, seen from the same high camera angle. Muted twilight colors and coarse film grain. One continuous texture and perspective.",
    "lower": "Continue the adjacent dark road downward into the masked region: uninterrupted dark coarse asphalt with the same twilight lighting and film grain. Consistent high camera angle and texture.",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=("track1", "track2", "track3"), default="track3")
    parser.add_argument("--steps", type=int, choices=(50,), default=50)
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    parser.add_argument("--guidance", type=float, choices=(30.0,), default=30.0)
    parser.add_argument("--model-path", type=Path, default=MODEL)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--upper-prompt")
    parser.add_argument("--lower-prompt")
    parser.add_argument("--max-seconds", type=int, default=1800, help="Cooperative wall-time bound checked before phases and steps")
    parser.add_argument("--preflight", action="store_true", help="CPU/static/tokenizer/header audit only; no model weights")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.max_seconds <= 7200:
        parser.error("Wall-time bound must be between1 and7200 seconds")
    model_path = args.model_path.expanduser().resolve()
    manifest_path = (args.manifest or model_path.with_suffix(".manifest.json")).expanduser().resolve()
    destination = (args.output or ROOT / f"experiments/flux1_fill/runs/2026-10-05/{args.track}-local-context50-seed42").resolve()
    if (destination / "metrics.json").exists():
        raise FileExistsError(f"Refusing to overwrite a prior trial: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    metrics_path = destination / ("preflight.json" if args.preflight else "metrics.json")
    begun = time.perf_counter()
    prompts = {"upper": args.upper_prompt if args.upper_prompt is not None else PROMPTS["upper"],
               "lower": args.lower_prompt if args.lower_prompt is not None else PROMPTS["lower"]}
    metrics = {"status": "preflight" if args.preflight else "starting", "track": args.track,
               "steps_per_call": 50, "seed_per_call": 42, "guidance_per_call": 30.0,
               "calls": {}, "phases": {}, "planned_calls": 2, "planned_total_nfe": 100,
               "planned_target_tokens_per_call": 896, "planned_total_target_token_forwards": 89600,
               "patch_size": list(PATCH_SIZE), "final_size": list(FINAL_SIZE), "source_rect_xyxy": SOURCE_RECT,
               "model_path": str(model_path), "model_revision": MODEL_REVISION,
               "runner_sha256": file_sha256(Path(__file__)),
               "native_runner_sha256": file_sha256(Path(__file__).with_name("run_fill_trial.py")),
               "input_helper_sha256": file_sha256(Path(__file__).with_name("local_context_inputs.py")),
               "cpu_test_sha256": file_sha256(Path(__file__).with_name("test_local_context_cpu.py")),
               "native_cpu_oracle_sha256": file_sha256(Path(__file__).with_name("test_cpu_preflight.py")),
               "compute_mode": "two independent native full512x448 calls, original source context only",
               "true_growth_or_streaming": False, "growing_compute": False,
               "full_canvas_single_factor_comparison": False,
               "changed_factors": ["two local crops", "128 adjacent known source rows per call", "region-specific texture prompts", "896-token resolution schedule"],
               "known_bridge": False, "negative_branch": False, "adapters": [],
               "offline_environment": True, "model_weights_loaded": False,
               "research_use_only": True, "inherited_model_license": "flux-1-dev-non-commercial-license",
               "conditioning_channels": {"dynamic_noise": 64, "masked_source": 64, "pixel_mask": 256, "total": 384},
               "max_seconds": args.max_seconds, "wall_bound_scope": "cooperative checks before phases/steps; synchronous model kernels may overrun one check"}

    def write():
        metrics["elapsed_seconds"] = time.perf_counter() - begun
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        metrics["process_peak_rss_bytes"] = rss if sys.platform == "darwin" else rss * 1024
        temporary = metrics_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(metrics, indent=2) + "\n")
        temporary.replace(metrics_path)

    def budget():
        if time.perf_counter() - begun > args.max_seconds:
            raise TimeoutError("Local context Fill cooperative wall-time bound exceeded")

    def log(phase, **data):
        budget()
        metrics["phase"] = phase; metrics.update(data); write()
        print(json.dumps({"phase": phase, "elapsed_seconds": round(metrics["elapsed_seconds"], 2), **data}), flush=True)

    try:
        import mlx.core as mx
        if args.preflight: mx.set_default_device(mx.cpu)
        import numpy as np
        from PIL import Image
        from mflux.models.flux.variants.fill.mask_util import MaskUtil
        from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
        from mflux.models.common.config.config import Config
        from mflux.models.flux.variants.fill.flux_fill import Flux1Fill
        from mflux.utils.image_util import ImageUtil
        shared = json.loads((ROOT / "experiments/evaluation/inputs/inputs.json").read_text())
        track = next(item for item in shared["tracks"] if item["id"] == args.track)
        source_path, jpeg_path = Path(track["source_512"]), Path(track["original_jpg"])
        source = Image.open(source_path).convert("RGB")
        if source.size != (512, 512) or shared["canvas_size"] != [512, 1152] or shared["source_rect_xyxy"] != SOURCE_RECT:
            raise ValueError("Expected untouched shared source512 and centered512x1152 assembly")
        source_np = np.asarray(source).copy()
        source_sha, jpeg_sha = file_sha256(source_path), file_sha256(jpeg_path)
        source.save(destination / "source_512.png")
        patches = prepare_patches(np, Image, source)
        metrics.update({"source_512_path": str(source_path), "source_512_sha256": source_sha,
                        "source_pixels_uint8_sha256": hashlib.sha256(source_np.tobytes()).hexdigest(),
                        "original_jpeg_path": str(jpeg_path), "original_jpeg_sha256_before": jpeg_sha})
        for name, patch in patches.items():
            directory = destination / name; directory.mkdir(exist_ok=True)
            canvas_path, mask_path, strip_path = directory / "input_canvas.png", directory / "input_mask.png", directory / "source_strip.png"
            patch["canvas"].save(canvas_path); patch["mask"].save(mask_path); patch["source_strip"].save(strip_path)
            info = dict(patch["metadata"])
            info.update({"canvas_path": str(canvas_path), "mask_path": str(mask_path), "source_strip_path": str(strip_path),
                         "canvas_sha256": file_sha256(canvas_path), "mask_sha256": file_sha256(mask_path), "strip_file_sha256": file_sha256(strip_path),
                         "prompt": prompts[name], "prompt_sha256": hashlib.sha256(prompts[name].encode()).hexdigest(),
                         "status": "prepared", "nfe": 0, "target_token_forwards": 0,
                         "joint_block_forwards": 0, "single_block_forwards": 0, "step_metrics": [], "phases": {}})
            metrics["calls"][name] = info
            (directory / "inputs.json").write_text(json.dumps(info, indent=2) + "\n")
        metrics["checkpoint_inventory"] = checkpoint_inventory(model_path)
        metrics["checkpoint_manifest"] = verify_manifest(model_path, manifest_path)
        metrics["runtime_source_sha256"] = {name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in (("native_fill", Path(inspect.getsourcefile(Flux1Fill))), ("mask_util", Path(inspect.getsourcefile(MaskUtil))),
                               ("config", Path(inspect.getsourcefile(Config))), ("latent_creator", Path(inspect.getsourcefile(FluxLatentCreator))),
                               ("image_util", Path(inspect.getsourcefile(ImageUtil))))}
        validation_path = Path(__file__).with_name("local_context_cpu_validation.json")
        validation = json.loads(validation_path.read_text()) if validation_path.is_file() else None
        validation_current = bool(validation and validation.get("status") == "passed"
            and validation.get("helper_sha256") == metrics["input_helper_sha256"]
            and validation.get("script_sha256") == metrics["cpu_test_sha256"]
            and all(validation.get("runtime_source_sha256", {}).get(name) == metrics["runtime_source_sha256"][name]["sha256"]
                    for name in ("mask_util", "latent_creator", "config")))
        metrics["local_context_cpu_gate_current"] = validation_current
        if args.preflight:
            from test_local_context_cpu import run_local_context_cpu
            metrics["local_context_cpu_audit"] = run_local_context_cpu()
            from mflux.models.common.tokenizer.tokenizer_loader import TokenizerLoader
            from mflux.models.flux.weights.flux_weight_definition import FluxWeightDefinition
            if metrics["checkpoint_inventory"]["complete"]:
                tokenizers = TokenizerLoader.load_all(FluxWeightDefinition.get_tokenizers(), str(model_path), max_length_overrides={"t5": 512})
                metrics["tokenizer_preflight"] = {name: {tokenizer_name: {"max_length": tokenizer.max_length,
                    "token_shape": list(tokenizer.tokenize(prompts[name]).input_ids.shape)} for tokenizer_name, tokenizer in tokenizers.items()} for name in prompts}
            ready = metrics["checkpoint_inventory"]["complete"] and metrics["checkpoint_manifest"]["verified"] and validation_current
            metrics.update({"status": "preflight_files_ready" if ready else "preflight_files_incomplete", "device": "CPU only"})
            if file_sha256(source_path) != source_sha or file_sha256(jpeg_path) != jpeg_sha:
                raise AssertionError("Preflight changed an original source file")
            metrics["original_jpeg_unchanged"] = True; metrics["source_png_unchanged"] = True
            write()
            print(json.dumps({"status": metrics["status"], "device": metrics["device"], "cpu_gate_current": validation_current,
                              "checkpoint_verified": metrics["checkpoint_manifest"]["verified"], "output": str(destination)}, indent=2), flush=True)
            return
        if not validation_current or not metrics["checkpoint_inventory"]["complete"] or not metrics["checkpoint_manifest"]["verified"]:
            raise RuntimeError("Current passed CPU gate and complete pinned verified local checkpoint are required")
        # Only a requested real invocation chooses GPU. CPU preflight never gets here.
        mx.set_default_device(mx.gpu)
        log("checkpoint_hash_verification")
        metrics["checkpoint_manifest"] = verify_manifest(model_path, manifest_path, verify_payloads=True)
        if not metrics["checkpoint_manifest"]["verified"]:
            raise ValueError("Pinned checkpoint payload changed")
        from mlx.utils import tree_flatten
        from mflux.models.common.vae.vae_util import VAEUtil
        from mflux.models.flux.model.flux_text_encoder.prompt_encoder import PromptEncoder
        mx.set_cache_limit(512 * 1024**2); mx.set_memory_limit(25 * 1024**3)
        metrics["metal_device"] = mx.device_info(); metrics["mlx_version"] = mx.__version__
        log("model_load", status="running")
        phase_start = time.perf_counter()
        model = Flux1Fill(model_path=str(model_path), quantize=None, lora_paths=None)
        mx.eval(model.parameters())
        if model.bits != 4 or model.model_config.x_embedder_input_dim() != 384:
            raise ValueError("Expected stored Q4 native384-channel Fill model")
        blocks = {"joint": len(model.transformer.transformer_blocks), "single": len(model.transformer.single_transformer_blocks)}
        if blocks != {"joint": 19, "single": 38}: raise ValueError("Unexpected native Fill block layout")
        metrics.update({"model_weights_loaded": True, "stored_quantization": model.bits, "block_counts": blocks,
                        "component_parameter_bytes": {name: sum(value.nbytes for _, value in tree_flatten(getattr(model, name).parameters()))
                            for name in ("transformer", "vae", "t5_text_encoder", "clip_text_encoder")}})
        metrics["phases"]["model_load_seconds"] = time.perf_counter() - phase_start
        encoded = {}
        log("prompt_encode_both")
        for name in ("upper", "lower"):
            budget(); start = time.perf_counter()
            embeddings, pooled = PromptEncoder.encode_prompt(prompt=prompts[name], prompt_cache=model.prompt_cache,
                t5_tokenizer=model.tokenizers["t5"], clip_tokenizer=model.tokenizers["clip"],
                t5_text_encoder=model.t5_text_encoder, clip_text_encoder=model.clip_text_encoder)
            mx.eval(embeddings, pooled)
            if not bool(mx.all(mx.isfinite(embeddings))) or not bool(mx.all(mx.isfinite(pooled))): raise FloatingPointError("Nonfinite prompt")
            encoded[name] = embeddings, pooled
            metrics["calls"][name]["phases"]["prompt_encode_seconds"] = time.perf_counter() - start
            metrics["calls"][name]["prompt_tokens"] = embeddings.shape[1]
            np.savez_compressed(destination / name / "prompt_embeddings.npz", embeddings=np.asarray(embeddings.astype(mx.float32)), pooled=np.asarray(pooled.astype(mx.float32)))
        del model.t5_text_encoder, model.clip_text_encoder
        mx.clear_cache(); metrics["text_encoders_deleted_after_both_prompts"] = True
        finals = {}
        for name in ("upper", "lower"):
            directory = destination / name; record = metrics["calls"][name]
            config = Config(model_config=model.model_config, num_inference_steps=50, width=512, height=448, guidance=30.0,
                scheduler="linear", image_path=Path(record["canvas_path"]), masked_image_path=Path(record["mask_path"]))
            log("source_conditioning", current_call=name); start = time.perf_counter()
            static = MaskUtil.create_masked_latents(vae=model.vae, width=512, height=448,
                img_path=config.image_path, mask_path=config.masked_image_path)
            mx.eval(static)
            if static.shape != (1, 896, 320) or not bool(mx.all(mx.isfinite(static))): raise ValueError("Invalid native static320")
            static_np = np.asarray(static.astype(mx.float32)).copy()
            np.savez_compressed(directory / "static_conditioning.npz", static=static_np)
            record["static_shape"] = list(static.shape); record["static_sha256_float32"] = hashlib.sha256(static_np.tobytes()).hexdigest()
            record["phases"]["source_conditioning_seconds"] = time.perf_counter() - start
            latents = FluxLatentCreator.create_noise(seed=42, width=512, height=448); mx.eval(latents)
            initial = np.asarray(latents.astype(mx.float32)).copy()
            if initial.shape != (1, 896, 64) or not np.isfinite(initial).all(): raise ValueError("Invalid native noise64")
            np.savez_compressed(directory / "initial_noise.npz", latents=initial)
            record["initial_noise_dtype"] = str(latents.dtype); record["initial_noise_sha256_float32"] = hashlib.sha256(initial.tobytes()).hexdigest()
            record["scheduler_sigmas"] = np.asarray(config.scheduler.sigmas.astype(mx.float32)).tolist()
            log("denoise", current_call=name); start = time.perf_counter(); mx.reset_peak_memory()
            embeddings, pooled = encoded[name]
            for step in range(50):
                budget(); step_start = time.perf_counter()
                latents = config.scheduler.scale_model_input(latents, step)
                hidden = mx.concatenate([latents, static], axis=-1)
                if hidden.shape != (1, 896, 384): raise AssertionError("Native local Fill requires896 rows and384 channels")
                prediction = model.transformer(t=step, config=config, hidden_states=hidden,
                    prompt_embeds=embeddings, pooled_prompt_embeds=pooled)
                latents = config.scheduler.step(noise=prediction, timestep=step, latents=latents)
                mx.eval(prediction, latents)
                if prediction.shape != (1, 896, 64) or not bool(mx.all(mx.isfinite(prediction))) or not bool(mx.all(mx.isfinite(latents))):
                    raise FloatingPointError("Invalid native prediction/target state")
                record["nfe"] += 1; record["target_token_forwards"] += 896
                record["joint_block_forwards"] += 19; record["single_block_forwards"] += 38
                row = {"step": step + 1, "target_tokens": 896, "input_channels": 384,
                       "sigma": record["scheduler_sigmas"][step], "seconds": time.perf_counter() - step_start,
                       "prediction_finite": True, "target_finite": True}
                record["step_metrics"].append(row)
                (directory / "metrics.json").write_text(json.dumps(record, indent=2) + "\n")
                log("denoise", current_call=name, completed_steps=step + 1)
            record["phases"]["denoise_seconds"] = time.perf_counter() - start
            record["denoising_mlx_peak_bytes"] = mx.get_peak_memory()
            final = np.asarray(latents.astype(mx.float32)).copy()
            np.savez_compressed(directory / "final_latents.npz", latents=final)
            record["final_latents_sha256_float32"] = hashlib.sha256(final.tobytes()).hexdigest()
            record["final_latents_shape"] = list(final.shape); record["final_latents_dtype"] = str(latents.dtype)
            record["status"] = "denoised"; finals[name] = final
            mx.clear_cache()
        model.transformer = None; prediction = hidden = static = latents = None
        mx.clear_cache(); metrics["transformer_released_after_both_denoising_calls"] = True
        raw_crops = {}
        # Save both unmodified raw crops before any original-source assembly.
        for name in ("upper", "lower"):
            log("raw_decode", current_call=name); start = time.perf_counter()
            unpacked = FluxLatentCreator.unpack_latents(latents=mx.array(finals[name]), width=512, height=448)
            decoded = VAEUtil.decode(vae=model.vae, latent=unpacked, tiling_config=None); mx.eval(decoded)
            if not bool(mx.all(mx.isfinite(decoded))): raise FloatingPointError("Nonfinite raw crop decode")
            raw = ImageUtil.to_pil(decoded).convert("RGB")
            if raw.size != PATCH_SIZE: raise AssertionError("Invalid raw crop dimensions")
            raw_path = destination / name / "raw.png"; raw.save(raw_path); raw_crops[name] = raw
            record = metrics["calls"][name]; rect = record["known_patch_rect_xyxy"]
            error = np.abs(np.asarray(raw.crop(rect), dtype=np.float32) - np.asarray(patches[name]["source_strip"], dtype=np.float32))
            record.update({"raw_path": str(raw_path), "raw_sha256": file_sha256(raw_path),
                           "raw_known_source_mae_0_255": float(error.mean()), "raw_known_source_max_error_0_255": float(error.max()),
                           "raw_crop_saved_before_any_source_hardpaste": True, "status": "success"})
            record["phases"]["raw_decode_seconds"] = time.perf_counter() - start
            (destination / name / "metrics.json").write_text(json.dumps(record, indent=2) + "\n")
        log("assemble_original_source")
        composite = assemble_composite(np, Image, source, raw_crops["upper"], raw_crops["lower"])
        composite_path = destination / "composite.png"; composite.save(composite_path)
        diagnostic = assemble_raw_context_diagnostic(Image, source, raw_crops["upper"], raw_crops["lower"])
        diagnostic_path = destination / "raw_context_assembly.png"; diagnostic.save(diagnostic_path)
        edge_dir = destination / "edges"; edge_dir.mkdir(exist_ok=True)
        edges = {"composite_upper": composite.crop((0, 288, 512, 352)), "composite_lower": composite.crop((0, 800, 512, 864)),
                 "raw_upper": raw_crops["upper"].crop((0, 288, 512, 352)), "raw_lower": raw_crops["lower"].crop((0, 96, 512, 160))}
        metrics["edge_crops"] = {}
        for name, edge in edges.items():
            path = edge_dir / f"{name}.png"; edge.save(path)
            metrics["edge_crops"][name] = {"path": str(path), "sha256": file_sha256(path)}
        if file_sha256(source_path) != source_sha or file_sha256(jpeg_path) != jpeg_sha:
            raise AssertionError("Original source PNG/JPEG changed")
        totals = {field: sum(record[field] for record in metrics["calls"].values())
                  for field in ("nfe", "target_token_forwards", "joint_block_forwards", "single_block_forwards")}
        if totals != {"nfe": 100, "target_token_forwards": 89600, "joint_block_forwards": 1900, "single_block_forwards": 3800}:
            raise AssertionError("Two-call native Fill counters mismatch")
        metrics.update({"totals": totals, "source_exact_after_assembly": True, "generated_regions_raw_exact_after_assembly": True,
                        "original_jpeg_sha256_after": file_sha256(jpeg_path), "original_jpeg_unchanged": True, "source_png_unchanged": True,
                        "same_seed_independent_noise_hashes_match": metrics["calls"]["upper"]["initial_noise_sha256_float32"] == metrics["calls"]["lower"]["initial_noise_sha256_float32"],
                        "composite_path": str(composite_path), "composite_sha256": file_sha256(composite_path),
                        "raw_context_diagnostic_path": str(diagnostic_path), "raw_context_diagnostic_sha256": file_sha256(diagnostic_path),
                        "raw_context_diagnostic_definition": "both full raw448 crops with original center256; not a full-canvas model raw output",
                        "visual_quality": "requires direct inspection; execution success is not quality acceptance", "status": "success"})
        log("complete")
    except BaseException as exc:
        metrics.update({"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
        write(); raise


if __name__ == "__main__":
    main()
