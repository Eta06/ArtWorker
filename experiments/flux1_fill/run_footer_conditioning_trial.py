#!/usr/bin/env python3
"""Lower-only native50 Fill, one manual condition-only footer ROI change.

Original source, native mask, saved noise, original lower prompt, schedule,
resolution and native transformer/scheduler computations are retained. The
new full display composite reuses the baseline upper320; it is not one new
full-canvas raw generation. Actual unmodified lower512x448 raw is retained.
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

from run_fill_trial import ROOT, MODEL, MODEL_REVISION, file_sha256, checkpoint_inventory, verify_manifest
from footer_conditioning_inputs import remove_footer_for_condition, assemble_lower_only

BASELINE = ROOT / "experiments/flux1_fill/runs/2026-10-05/track3-local-context50-seed42"


def native_step(mx, model, config, latents, static, embeddings, pooled, step):
    # These native operations are AST-matched against the frozen local runner.
    latents = config.scheduler.scale_model_input(latents, step)
    hidden = mx.concatenate([latents, static], axis=-1)
    prediction = model.transformer(t=step, config=config, hidden_states=hidden,
        prompt_embeds=embeddings, pooled_prompt_embeds=pooled)
    latents = config.scheduler.step(noise=prediction, timestep=step, latents=latents)
    return prediction, latents, hidden


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=700)
    args = parser.parse_args()
    if not 1 <= args.max_seconds <= 1800:
        parser.error("Wall bound must be between 1 and 1800 seconds")
    destination = args.output.resolve()
    metrics_path = destination / ("preflight.json" if args.preflight else "metrics.json")
    if metrics_path.exists():
        raise FileExistsError(f"Refusing to overwrite prior experiment: {metrics_path}")
    destination.mkdir(parents=True, exist_ok=True)
    begun = time.perf_counter()
    metrics = {"status": "starting", "model_revision": MODEL_REVISION,
        "runner_sha256": file_sha256(Path(__file__)),
        "helper_sha256": file_sha256(Path(__file__).with_name("footer_conditioning_inputs.py")),
        "baseline_path": str(BASELINE), "manual_roi_experiment": True,
        "generic_detection_claim": False, "model_weights_loaded": False,
        "planned_calls": 1, "planned_nfe": 50, "planned_target_token_forwards": 44800,
        "guidance": 30.0, "seed": 42, "patch_size": [512, 448],
        "changed_factors": ["conditioning input pixels inside declared manual footer ROI"],
        "original_source_rendering_unchanged": True, "growing_compute": False,
        "known_bridge": False, "negative_branch": False,
        "nfe": 0, "target_token_forwards": 0, "joint_block_forwards": 0,
        "single_block_forwards": 0, "step_metrics": [], "phases": {},
        "max_seconds": args.max_seconds, "offline_environment": True,
        "quality": "pending direct image inspection; this is a hypothesis test",
        "inherited_model_license": "flux-1-dev-non-commercial-license"}

    def write():
        metrics["elapsed_seconds"] = time.perf_counter() - begun
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        metrics["process_peak_rss_bytes"] = rss if sys.platform == "darwin" else rss * 1024
        temporary = metrics_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(metrics, indent=2) + "\n")
        temporary.replace(metrics_path)

    def log(phase, **data):
        if time.perf_counter() - begun > args.max_seconds:
            raise TimeoutError("Footer condition cooperative wall bound exceeded")
        metrics.update({"phase": phase, **data}); write()
        print(json.dumps({"phase": phase, "elapsed_seconds": round(metrics["elapsed_seconds"], 3), **data}), flush=True)

    try:
        import numpy as np
        from PIL import Image
        import mlx.core as mx
        mx.set_default_device(mx.cpu if args.preflight else mx.gpu)
        from mflux.models.flux.variants.fill.mask_util import MaskUtil
        from mflux.models.flux.variants.fill.flux_fill import Flux1Fill
        from mflux.models.common.config.config import Config
        from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
        from mflux.models.flux.model.flux_text_encoder.prompt_encoder import PromptEncoder
        from mflux.models.common.vae.vae_util import VAEUtil
        from mflux.utils.image_util import ImageUtil
        from test_footer_conditioning_cpu import run_cpu_audit
        audit = run_cpu_audit()
        metrics["cpu_audit"] = audit
        # The CPU audit changes MLX default device only while running its fixtures.
        mx.set_default_device(mx.cpu if args.preflight else mx.gpu)
        baseline_lower = BASELINE / "lower"
        baseline_metrics = json.loads((baseline_lower / "metrics.json").read_text())
        baseline_suite = json.loads((BASELINE / "metrics.json").read_text())
        shared = json.loads((ROOT / "experiments/evaluation/inputs/inputs.json").read_text())
        track = next(item for item in shared["tracks"] if item["id"] == "track3")
        source_path, jpeg_path = Path(track["source_512"]), Path(track["original_jpg"])
        source_sha, jpeg_sha = file_sha256(source_path), file_sha256(jpeg_path)
        source = Image.open(source_path).convert("RGB")
        before = Image.open(baseline_lower / "input_canvas.png").convert("RGB")
        after, roi_mask, condition_info = remove_footer_for_condition(np, Image, source, before)
        before.save(destination / "condition_before.png")
        after.save(destination / "condition_after.png")
        roi_mask.save(destination / "condition_roi_mask.png")
        source.save(destination / "original_source_512.png")
        mask = Image.open(baseline_lower / "input_mask.png").convert("L")
        mask.save(destination / "input_mask.png")
        mask_path = destination / "input_mask.png"
        condition_path = destination / "condition_after.png"
        (destination / "condition_audit.json").write_text(json.dumps(condition_info, indent=2) + "\n")
        for name, im in (("before", before), ("after", after), ("roi", roi_mask)):
            im.crop((208, 66, 340, 128)).resize((792, 372), Image.Resampling.NEAREST).save(destination / f"footer_{name}_6x.png")
        metrics.update({"condition_audit": condition_info, "source_512_path": str(source_path),
            "source_512_sha256": source_sha, "original_jpeg_path": str(jpeg_path),
            "original_jpeg_sha256_before": jpeg_sha, "prompt": baseline_metrics["prompt"],
            "prompt_sha256": baseline_metrics["prompt_sha256"],
            "baseline_runner_sha256": baseline_suite["runner_sha256"],
            "baseline_unchanged_files_sha256": {str(p.relative_to(BASELINE)): file_sha256(p)
                for p in sorted(BASELINE.rglob("*")) if p.is_file()},
            "runtime_source_sha256": {name: {"path": str(path), "sha256": file_sha256(path)}
                for name, path in (("native_fill", Path(inspect.getsourcefile(Flux1Fill))),
                    ("mask_util", Path(inspect.getsourcefile(MaskUtil))),
                    ("config", Path(inspect.getsourcefile(Config))),
                    ("latent_creator", Path(inspect.getsourcefile(FluxLatentCreator))),
                    ("image_util", Path(inspect.getsourcefile(ImageUtil))))},
            "checkpoint_inventory": checkpoint_inventory(MODEL),
            "checkpoint_manifest": verify_manifest(MODEL, MODEL.with_suffix(".manifest.json"))})
        for name, record in baseline_suite["runtime_source_sha256"].items():
            if file_sha256(Path(record["path"])) != record["sha256"]:
                raise AssertionError(f"Baseline native runtime changed: {name}")
        if not metrics["checkpoint_inventory"]["complete"] or not metrics["checkpoint_manifest"]["verified"]:
            raise RuntimeError("Complete pinned local checkpoint required")
        if args.preflight:
            metrics.update({"status": "preflight_files_ready", "device": "CPU only",
                "gpu_exact_prompt_and_schedule_checks_pending": True})
            write(); print(json.dumps({"status": metrics["status"], "output": str(destination)}), flush=True)
            return
        log("checkpoint_hash_verification")
        metrics["checkpoint_manifest"] = verify_manifest(MODEL, MODEL.with_suffix(".manifest.json"), verify_payloads=True)
        if not metrics["checkpoint_manifest"]["verified"]:
            raise RuntimeError("Pinned checkpoint payload changed")
        from mlx.utils import tree_flatten
        mx.set_cache_limit(512 * 1024**2); mx.set_memory_limit(25 * 1024**3)
        metrics["metal_device"] = mx.device_info(); metrics["mlx_version"] = mx.__version__
        log("model_load", status="running")
        start = time.perf_counter()
        model = Flux1Fill(model_path=str(MODEL), quantize=None, lora_paths=None)
        mx.eval(model.parameters())
        if model.bits != 4 or model.model_config.x_embedder_input_dim() != 384:
            raise AssertionError("Expected unchanged stored native Q4 Fill")
        if (len(model.transformer.transformer_blocks), len(model.transformer.single_transformer_blocks)) != (19, 38):
            raise AssertionError("Native block layout changed")
        metrics["model_weights_loaded"] = True
        metrics["phases"]["model_load_seconds"] = time.perf_counter() - start
        metrics["component_parameter_bytes"] = {name: sum(value.nbytes for _, value in tree_flatten(getattr(model, name).parameters()))
            for name in ("transformer", "vae", "t5_text_encoder", "clip_text_encoder")}
        log("prompt_encode_exact_check")
        embeddings, pooled = PromptEncoder.encode_prompt(prompt=baseline_metrics["prompt"], prompt_cache=model.prompt_cache,
            t5_tokenizer=model.tokenizers["t5"], clip_tokenizer=model.tokenizers["clip"],
            t5_text_encoder=model.t5_text_encoder, clip_text_encoder=model.clip_text_encoder)
        mx.eval(embeddings, pooled)
        saved_prompt = np.load(baseline_lower / "prompt_embeddings.npz")
        for key, current in (("embeddings", embeddings), ("pooled", pooled)):
            current_np = np.asarray(current.astype(mx.float32))
            if not np.array_equal(current_np, saved_prompt[key]):
                raise AssertionError(f"Native same-prompt values differ from baseline: {key}")
        metrics["native_prompt_values_baseline_exact"] = True
        metrics["native_prompt_dtypes"] = {"embeddings": str(embeddings.dtype), "pooled": str(pooled.dtype)}
        np.savez_compressed(destination / "prompt_embeddings.npz", embeddings=np.asarray(embeddings.astype(mx.float32)), pooled=np.asarray(pooled.astype(mx.float32)))
        del model.t5_text_encoder, model.clip_text_encoder
        mx.clear_cache()
        config = Config(model_config=model.model_config, num_inference_steps=50, width=512, height=448,
            guidance=30.0, scheduler="linear", image_path=condition_path, masked_image_path=mask_path)
        sigmas = np.asarray(config.scheduler.sigmas.astype(mx.float32)).copy()
        expected_sigmas = np.asarray(baseline_metrics["scheduler_sigmas"], dtype=np.float32)
        if not np.array_equal(sigmas, expected_sigmas):
            raise AssertionError("Actual native schedule differs from saved baseline")
        metrics["scheduler_sigmas"] = sigmas.tolist(); metrics["scheduler_baseline_exact"] = True
        log("source_conditioning")
        start = time.perf_counter()
        static = MaskUtil.create_masked_latents(vae=model.vae, width=512, height=448,
            img_path=config.image_path, mask_path=config.masked_image_path)
        mx.eval(static)
        static_np = np.asarray(static.astype(mx.float32)).copy()
        baseline_static = np.load(baseline_lower / "static_conditioning.npz")["static"]
        if static.shape != (1, 896, 320) or not np.isfinite(static_np).all():
            raise AssertionError("Invalid native static320")
        if not np.array_equal(static_np[..., 64:], baseline_static[..., 64:]):
            raise AssertionError("Native256 mask conditioning changed")
        np.savez_compressed(destination / "static_conditioning.npz", static=static_np)
        metrics.update({"static_sha256_float32": hashlib.sha256(static_np.tobytes()).hexdigest(),
            "mask256_baseline_exact": True,
            "masked_source64_delta_mae": float(np.abs(static_np[..., :64] - baseline_static[..., :64]).mean()),
            "masked_source64_delta_max": float(np.abs(static_np[..., :64] - baseline_static[..., :64]).max()),
            "source_conditioning_delta_scope": "VAE features may change beyond ROI because its receptive field mixes pixels"})
        metrics["phases"]["source_conditioning_seconds"] = time.perf_counter() - start
        initial = np.load(baseline_lower / "initial_noise.npz")["latents"].copy()
        if hashlib.sha256(initial.tobytes()).hexdigest() != baseline_metrics["initial_noise_sha256_float32"]:
            raise AssertionError("Saved baseline noise changed")
        latents = mx.array(initial)
        np.savez_compressed(destination / "initial_noise.npz", latents=initial)
        metrics["initial_noise_sha256_float32"] = hashlib.sha256(initial.tobytes()).hexdigest()
        metrics["initial_noise_baseline_exact"] = True; metrics["noise_draws"] = 0
        log("denoise"); start = time.perf_counter(); mx.reset_peak_memory()
        for step in range(50):
            log("denoise", completed_steps=step)
            step_start = time.perf_counter()
            prediction, latents, hidden = native_step(mx, model, config, latents, static, embeddings, pooled, step)
            mx.eval(prediction, latents)
            if hidden.shape != (1, 896, 384) or prediction.shape != (1, 896, 64):
                raise AssertionError("Native Fill shape changed")
            if not bool(mx.all(mx.isfinite(prediction))) or not bool(mx.all(mx.isfinite(latents))):
                raise FloatingPointError("Nonfinite native state")
            metrics["nfe"] += 1; metrics["target_token_forwards"] += 896
            metrics["joint_block_forwards"] += 19; metrics["single_block_forwards"] += 38
            metrics["step_metrics"].append({"step": step + 1, "sigma": float(sigmas[step]),
                "seconds": time.perf_counter() - step_start, "target_tokens": 896, "input_channels": 384})
            write()
        metrics["phases"]["denoise_seconds"] = time.perf_counter() - start
        metrics["denoising_mlx_peak_bytes"] = mx.get_peak_memory()
        final = np.asarray(latents.astype(mx.float32)).copy()
        np.savez_compressed(destination / "final_latents.npz", latents=final)
        metrics["final_latents_sha256_float32"] = hashlib.sha256(final.tobytes()).hexdigest()
        model.transformer = None; prediction = hidden = static = latents = None; mx.clear_cache()
        log("raw_decode")
        unpacked = FluxLatentCreator.unpack_latents(latents=mx.array(final), width=512, height=448)
        decoded = VAEUtil.decode(vae=model.vae, latent=unpacked, tiling_config=None); mx.eval(decoded)
        if not bool(mx.all(mx.isfinite(decoded))):
            raise FloatingPointError("Nonfinite actual raw decode")
        raw = ImageUtil.to_pil(decoded).convert("RGB")
        if raw.size != (512, 448): raise AssertionError("Wrong actual raw size")
        raw.save(destination / "raw.png")
        metrics["raw_path"] = str(destination / "raw.png")
        metrics["raw_sha256"] = file_sha256(destination / "raw.png")
        metrics["raw_saved_before_original_source_hardpaste"] = True
        raw_error = np.abs(np.asarray(raw)[:128].astype(np.float32) - np.asarray(source)[384:].astype(np.float32))
        metrics["raw_known_original_mae_0_255"] = float(raw_error.mean())
        metrics["raw_known_original_max_error_0_255"] = float(raw_error.max())
        upper_path = BASELINE / "upper/raw.png"
        upper = Image.open(upper_path).convert("RGB")
        composite = assemble_lower_only(np, Image, source, upper, raw)
        composite.save(destination / "composite.png")
        raw.crop((0, 96, 512, 176)).save(destination / "raw_lower_boundary.png")
        composite.crop((0, 800, 512, 880)).save(destination / "composite_lower_boundary.png")
        for relative, expected in metrics["baseline_unchanged_files_sha256"].items():
            if file_sha256(BASELINE / relative) != expected:
                raise AssertionError(f"Baseline file changed: {relative}")
        if file_sha256(source_path) != source_sha or file_sha256(jpeg_path) != jpeg_sha:
            raise AssertionError("Original source changed")
        if (metrics["nfe"], metrics["target_token_forwards"], metrics["joint_block_forwards"], metrics["single_block_forwards"]) != (50, 44800, 950, 1900):
            raise AssertionError("Native lower-only counters mismatch")
        metrics.update({"status": "success", "baseline_files_unchanged": True,
            "source_png_unchanged": True, "original_jpeg_unchanged": True,
            "source_exact_after_assembly": True, "new_lower_generated_pixels_raw_exact": True,
            "reused_upper_generated_pixels_exact": True, "upper_reused_from": str(upper_path),
            "upper_reused_sha256": file_sha256(upper_path),
            "full_composite_definition": "original source + reused baseline upper320 + new lower-only generated320",
            "composite_path": str(destination / "composite.png"),
            "composite_sha256": file_sha256(destination / "composite.png")})
        log("complete")
    except BaseException as exc:
        metrics.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
        write(); raise


if __name__ == "__main__":
    main()
