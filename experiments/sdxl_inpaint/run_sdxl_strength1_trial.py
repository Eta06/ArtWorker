"""Frozen-copy SDXL strength-one pure-noise initialization probe.

Only the parent launches real inference. CPU preflight does not load a model.
The full canvas is processed throughout; no growing or mobile runtime claim.
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
from types import MethodType

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / ".build/models/sdxl-inpaint-fp16"
REPO = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
REVISION = "115134f363124c53c7d878647567d04daf26e41e"
DEFAULT_PROMPT = (
    "Continuous muted lake water, low green roadside shrubs and a thin metal guardrail extending through the upper area. "
    "Uninterrupted dark coarse asphalt continuing through the lower area. "
    "Matching twilight illumination, photographic grain and high-angle perspective across the whole scene."
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_check(model_path: Path, manifest_path: Path, payloads: bool = False) -> dict:
    if not manifest_path.is_file():
        return {"path": str(manifest_path), "verified": False, "issues": ["Manifest is absent"]}
    manifest = json.loads(manifest_path.read_text())
    issues = []
    result = {"path": str(manifest_path), "sha256": file_sha256(manifest_path), "status": manifest.get("status"),
              "repo": manifest.get("repo_id"), "revision": manifest.get("revision"),
              "payload_hashes_checked": payloads, "files": [], "issues": issues}
    if manifest.get("status") != "verified":
        issues.append("Downloader manifest is not verified")
    if manifest.get("repo_id") != REPO or manifest.get("revision") != REVISION:
        issues.append("Expected exact pinned native SDXL inpainting repository revision")
    if manifest.get("resolved_revision", REVISION) != REVISION:
        issues.append("Resolved checkpoint revision differs from the pinned commit")
    if Path(manifest.get("local_dir", "")).resolve() != model_path:
        issues.append("Manifest names another checkpoint directory")
    for entry in manifest.get("files", []):
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Manifest path escapes checkpoint directory")
        path = model_path / relative
        info = {"path": entry["path"], "expected_bytes": entry["bytes"], "exists": path.is_file()}
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            issues.append(f"Missing or incomplete: {relative}")
        elif payloads:
            if entry.get("sha256"):
                info["sha256"] = file_sha256(path)
                if info["sha256"] != entry["sha256"]:
                    issues.append(f"Payload SHA256 mismatch: {relative}")
            else:
                content = path.read_bytes()
                info["git_blob_sha1"] = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
                if info["git_blob_sha1"] != entry.get("git_blob_sha1"):
                    issues.append(f"Pinned Git blob mismatch: {relative}")
        result["files"].append(info)
    if not result["files"]:
        issues.append("Manifest has no files")
    result["verified"] = not issues
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=("track1", "track2", "track3"), default="track3")
    parser.add_argument("--steps", type=int, choices=(20, 30), default=20)
    parser.add_argument("--guidance", type=float, default=8.0)
    parser.add_argument("--strength", type=float, choices=(1.0,), default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt")
    parser.add_argument("--model-path", type=Path, default=MODEL)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if not 1 < args.guidance < 30:
        parser.error("Expected CFG guidance in (1,30)")
    model_path = args.model_path.expanduser().resolve()
    manifest_path = (args.manifest or model_path.with_suffix(".manifest.json")).resolve()
    output = (args.output or ROOT / f"experiments/sdxl_inpaint/runs/2026-10-05/{args.track}-{args.steps}-cfg{args.guidance:g}-strength{args.strength:g}-seed{args.seed}").resolve()
    metrics_path = output / ("preflight.json" if args.preflight else "metrics.json")
    if not args.preflight and metrics_path.exists():
        raise FileExistsError(f"Refusing to overwrite prior run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    begun = time.perf_counter()
    metrics = {"status": "preflight" if args.preflight else "starting", "track": args.track,
               "requested_steps": args.steps, "guidance": args.guidance, "strength": args.strength, "seed": args.seed,
               "model_repo": REPO, "model_revision": REVISION, "model_path": str(model_path),
               "runner_sha256": file_sha256(Path(__file__)), "phases": {}, "step_metrics": [],
               "model_weights_loaded": False, "offline": True, "compute_mode": "native full-canvas nine-channel inpainting",
               "growing_compute": False, "known_bridge": False, "padding_mask_crop": None,
               "watermark": False, "negative_prompt": None, "adapters": [], "research_use_only": True,
               "model_license": "CreativeML Open RAIL++-M", "unet_calls": 0, "cfg_sample_branch_evaluations": 0}
    baseline_path = Path(__file__).with_name("run_sdxl_trial.py")
    metrics.update({"variant": "native strength-one pure-noise initialization probe",
                    "baseline_runner": {"path": str(baseline_path), "sha256": file_sha256(baseline_path)},
                    "baseline_strength": 0.99,
                    "rng_path_difference_from_strength099": {
                        "strength1": ["initial noise", "masked-source VAE posterior"],
                        "strength099": ["full initial-image VAE posterior", "initial noise", "masked-source VAE posterior"],
                        "initial_noise_matched_by_same_seed": False}})

    def write() -> None:
        metrics["elapsed_seconds"] = time.perf_counter() - begun
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        metrics["process_peak_rss_bytes"] = rss if sys.platform == "darwin" else rss * 1024
        temporary = metrics_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(metrics, indent=2))
        temporary.replace(metrics_path)

    def log(phase: str, **data) -> None:
        metrics["phase"] = phase
        metrics.update(data)
        write()
        print(json.dumps({"phase": phase, "elapsed_seconds": round(metrics["elapsed_seconds"], 3), **data}), flush=True)

    try:
        import numpy as np
        import torch
        import diffusers
        from PIL import Image
        from diffusers import StableDiffusionXLInpaintPipeline
        from test_cpu_preflight import run_preflight
        from test_strength1_cpu_preflight import run_preflight as run_strength1_preflight

        metrics.update({"torch_version": torch.__version__, "diffusers_version": diffusers.__version__, "python_version": sys.version})
        source_file = Path(inspect.getsourcefile(StableDiffusionXLInpaintPipeline))
        metrics["pipeline_source"] = {"path": str(source_file), "sha256": file_sha256(source_file)}
        shared = json.loads((ROOT / "experiments/evaluation/inputs/inputs.json").read_text())
        track = next(item for item in shared["tracks"] if item["id"] == args.track)
        width, height = shared["canvas_size"]
        rect = shared["source_rect_xyxy"]
        source_path, canvas_path, mask_path = (Path(track[key]) for key in ("source_512", "canvas", "mask"))
        source, canvas, mask = (Image.open(path).convert(mode) for path, mode in ((source_path, "RGB"), (canvas_path, "RGB"), (mask_path, "L")))
        if (width, height) != (512, 1152) or rect != [0, 320, 512, 832] or source.size != (512, 512):
            raise ValueError("Expected unchanged shared 512-square source in512x1152 at[0,320,512,832]")
        if canvas.size != (width, height) or mask.size != canvas.size:
            raise ValueError("Frozen canvas/mask dimensions differ")
        source_array, canvas_array, mask_array = np.asarray(source), np.asarray(canvas), np.asarray(mask)
        x0, y0, x1, y1 = rect
        expected_mask = np.full((height, width), 255, dtype=np.uint8)
        expected_mask[y0:y1, x0:x1] = 0
        if not np.array_equal(source_array, canvas_array[y0:y1, x0:x1]) or not np.array_equal(mask_array, expected_mask):
            raise ValueError("Frozen source pixels or white-unknown mask changed")
        unknown_rgb_values = np.unique(canvas_array[expected_mask.astype(bool)], axis=0).tolist()
        metrics.update({"width": width, "height": height, "source_rect_xyxy": rect,
                        "input_files": {name: {"path": str(path), "sha256": file_sha256(path)}
                                        for name, path in (("source", source_path), ("canvas", canvas_path), ("mask", mask_path))},
                        "source_input_pixels_exact": True, "mask_convention": "white regenerates, black preserves",
                        "initial_canvas_unknown_rgb_values_0_255": unknown_rgb_values,
                        "masked_source_unknown_normalized_rgb": 0.0,
                        "native_initialization": "strength1 uses sampled noise times init_noise_sigma; complete initial-canvas VAE encoding is skipped",
                        "cross_model_comparison": {
                            "shared": "source pixels, regeneration mask and canvas geometry",
                            "full_canvas_bytes_matched_with_flux_fill": False,
                            "initial_noise_matched_with_flux_fill": False,
                            "flux_fill_initial_canvas_unknown_rgb": [128, 128, 128],
                            "note": "This strength1 probe skips complete initial-canvas encoding and separately encodes normalized-zero exterior masked source"},
                        "checkpoint_manifest": manifest_check(model_path, manifest_path)})
        prompt = args.prompt or (DEFAULT_PROMPT if args.track == "track3" else json.loads((ROOT / "experiments/flux2/prompts.json").read_text())[args.track])
        metrics["prompt"] = prompt
        metrics["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
        metrics["cpu_preflight"] = run_preflight()
        metrics["cpu_gate_sha256"] = file_sha256(Path(__file__).with_name("test_cpu_preflight.py"))
        metrics["strength1_cpu_preflight"] = run_strength1_preflight()
        metrics["strength1_cpu_gate_sha256"] = file_sha256(Path(__file__).with_name("test_strength1_cpu_preflight.py"))
        metrics["cpu_gate_scope"] = "Inherited source/mask and strength099 tests run separately; strength1 gate verifies this probe initialization and20 untrimmed timesteps"
        if metrics["cpu_preflight"]["status"] != "passed":
            raise RuntimeError("CPU conditioning gate failed")
        if metrics["strength1_cpu_preflight"]["status"] != "passed":
            raise RuntimeError("CPU strength1 initialization gate failed")
        if args.preflight:
            metrics.update({"device": "CPU", "status": "preflight_files_ready" if metrics["checkpoint_manifest"]["verified"] else "preflight_files_incomplete"})
            write()
            print(json.dumps(metrics, indent=2), flush=True)
            return
        if not metrics["checkpoint_manifest"]["verified"]:
            raise FileNotFoundError(f"Pinned checkpoint not ready: {metrics['checkpoint_manifest']['issues']}")
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable; parent must choose another explicit backend")
        # Preflight uses CPU, and this process explicitly chooses MPS for inference.
        torch.set_default_device("cpu")
        torch.set_num_threads(4)
        log("checkpoint_hash_verification", status="running")
        start = time.perf_counter()
        metrics["checkpoint_manifest"] = manifest_check(model_path, manifest_path, payloads=True)
        metrics["phases"]["checkpoint_hash_verification"] = time.perf_counter() - start
        if not metrics["checkpoint_manifest"]["verified"]:
            raise ValueError(f"Pinned checkpoint changed: {metrics['checkpoint_manifest']['issues']}")
        log("model_load")
        start = time.perf_counter()
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(str(model_path), local_files_only=True,
                    variant="fp16", use_safetensors=True, torch_dtype=torch.float16, add_watermarker=False).to("mps")
        pipe.set_progress_bar_config(disable=True)
        torch.mps.synchronize()
        metrics["phases"]["model_load"] = time.perf_counter() - start
        metrics["model_weights_loaded"] = True
        if pipe.unet.config.in_channels != 9 or pipe.vae.config.latent_channels != 4:
            raise ValueError("Expected native nine-channel SDXL inpainting model and four-channel VAE")
        metrics.update({"device": "mps", "dtype": "torch.float16", "unet_in_channels": 9,
                        "component_parameter_bytes": {name: sum(p.numel() * p.element_size() for p in getattr(pipe, name).parameters())
                                                      for name in ("unet", "vae", "text_encoder", "text_encoder_2")},
                        "scheduler_class": type(pipe.scheduler).__name__, "scheduler_config": dict(pipe.scheduler.config)})
        generator = torch.Generator(device="cpu").manual_seed(args.seed)

        def generator_hash() -> str:
            return hashlib.sha256(generator.get_state().cpu().numpy().tobytes()).hexdigest()

        def tensor_info(tensor, name: str, save: bool = False) -> dict:
            original_device, original_dtype = str(tensor.device), str(tensor.dtype)
            values = tensor.detach().float().cpu().numpy()
            info = {"shape": list(values.shape), "device": original_device, "dtype": original_dtype,
                    "finite": bool(np.isfinite(values).all()), "sha256_float32": hashlib.sha256(values.tobytes()).hexdigest()}
            if not info["finite"]:
                raise FloatingPointError(f"Non-finite {name}")
            if save:
                np.savez_compressed(output / f"{name}.npz", tensor=values)
            return info

        def memory_sample() -> dict:
            sample = {"current_allocated_bytes": torch.mps.current_allocated_memory(),
                      "driver_allocated_bytes": torch.mps.driver_allocated_memory()}
            peak = metrics.setdefault("mps_sampled_peaks", {"current_allocated_bytes": 0, "driver_allocated_bytes": 0})
            for key, value in sample.items():
                peak[key] = max(peak[key], value)
            return sample

        metrics["generator"] = {"device": "cpu", "seed": args.seed, "initial_state_sha256": generator_hash()}
        original_prepare = pipe.prepare_latents

        def observed_prepare(this, *call_args, **kwargs):
            state_before = generator_hash()
            encodes_before = len(metrics["vae_conditioning_encodes"])
            start = time.perf_counter()
            result = original_prepare(*call_args, **kwargs)
            torch.mps.synchronize()
            if len(metrics["vae_conditioning_encodes"]) != encodes_before:
                raise AssertionError("Strength1 prepare_latents must skip full initial-image VAE encoding")
            pure_noise_exact = bool(torch.equal(result[0], result[1] * pipe.scheduler.init_noise_sigma))
            if not pure_noise_exact:
                raise AssertionError("Strength1 initial latents must equal actual noise times native init_noise_sigma")
            metrics["prepare_latents"] = {"seconds": time.perf_counter() - start,
                "generator_state_before": state_before, "generator_state_after": generator_hash(),
                "initial_latents": tensor_info(result[0], "initial_latents", True),
                "actual_noise": tensor_info(result[1], "actual_noise", True),
                "initial_noise_mode": "native pure noise because strength1",
                "scheduler_init_noise_sigma": float(pipe.scheduler.init_noise_sigma),
                "initial_latents_equal_noise_times_sigma": pure_noise_exact,
                "full_initial_image_vae_encode_skipped": True}
            return result

        pipe.prepare_latents = MethodType(observed_prepare, pipe)
        original_mask = pipe.prepare_mask_latents

        def observed_mask(this, *call_args, **kwargs):
            bound = inspect.signature(original_mask).bind(*call_args, **kwargs)
            source_input = bound.arguments["masked_image"]
            mask_input = bound.arguments["mask"]
            known = mask_input < 0.5
            if torch.count_nonzero(source_input * (~known)).item() != 0:
                raise AssertionError("Unknown masked-source normalized RGB must be exactlyzero")
            state_before = generator_hash()
            start = time.perf_counter()
            result = original_mask(*call_args, **kwargs)
            torch.mps.synchronize()
            metrics["prepare_mask_latents"] = {"seconds": time.perf_counter() - start,
                "generator_state_before": state_before, "generator_state_after": generator_hash(),
                "normalized_masked_source_input": tensor_info(source_input, "normalized_masked_source_input", True),
                "mask_latents": tensor_info(result[0], "mask_latents", True),
                "masked_source_latents": tensor_info(result[1], "masked_source_latents", True),
                "unknown_normalized_rgb_exact_zero": True}
            return result

        pipe.prepare_mask_latents = MethodType(observed_mask, pipe)
        original_encode = pipe._encode_vae_image
        metrics["vae_conditioning_encodes"] = []

        def observed_encode(this, *call_args, **kwargs):
            bound = inspect.signature(original_encode).bind(*call_args, **kwargs)
            index = len(metrics["vae_conditioning_encodes"])
            state_before = generator_hash()
            start = time.perf_counter()
            result = original_encode(*call_args, **kwargs)
            torch.mps.synchronize()
            metrics["vae_conditioning_encodes"].append({"seconds": time.perf_counter() - start,
                "input": tensor_info(bound.arguments["image"], f"vae_encode_input{index}"),
                "output": tensor_info(result, f"vae_encode_latents{index}", True),
                "generator_state_before": state_before, "generator_state_after": generator_hash()})
            return result

        pipe._encode_vae_image = MethodType(observed_encode, pipe)
        original_timesteps = pipe.get_timesteps

        def observed_timesteps(this, *call_args, **kwargs):
            result = original_timesteps(*call_args, **kwargs)
            metrics["actual_timesteps"] = result[0].detach().float().cpu().tolist()
            metrics["actual_inference_steps"] = int(result[1])
            metrics["scheduler_timesteps_full"] = pipe.scheduler.timesteps.detach().float().cpu().tolist()
            if hasattr(pipe.scheduler, "sigmas"):
                metrics["scheduler_sigmas_full"] = pipe.scheduler.sigmas.detach().float().cpu().tolist()
            metrics["scheduler_begin_index"] = getattr(pipe.scheduler, "begin_index", None)
            return result

        pipe.get_timesteps = MethodType(observed_timesteps, pipe)
        unet_times = []
        unet_start = None
        denoise_start = None

        def unet_pre(module, call_args, kwargs):
            nonlocal unet_start, denoise_start
            sample = call_args[0] if call_args else kwargs["sample"]
            if sample.shape[0] != 2 or sample.shape[1] != 9:
                raise AssertionError(f"Expected one CFG UNet call with batch2 and9channels, got{list(sample.shape)}")
            if denoise_start is None:
                torch.mps.synchronize()
                denoise_start = time.perf_counter()
                metrics["phases"]["pipeline_pre_denoise"] = denoise_start - pipeline_start
                metrics["first_unet_input"] = tensor_info(sample, "first_unet_input", True)
            unet_start = time.perf_counter()

        def unet_post(module, call_args, kwargs, result):
            prediction = result[0] if isinstance(result, tuple) else result.sample
            if not bool(torch.isfinite(prediction).all().item()):
                raise FloatingPointError("Non-finite UNet prediction")
            torch.mps.synchronize()
            unet_times.append(time.perf_counter() - unet_start)
            metrics["unet_calls"] += 1
            metrics["cfg_sample_branch_evaluations"] += int(prediction.shape[0])

        handles = [pipe.unet.register_forward_pre_hook(unet_pre, with_kwargs=True),
                   pipe.unet.register_forward_hook(unet_post, with_kwargs=True)]
        original_decode = pipe.vae.decode

        def observed_decode(*call_args, **kwargs):
            log("raw_decode")
            start = time.perf_counter()
            result = original_decode(*call_args, **kwargs)
            torch.mps.synchronize()
            prediction = result[0] if isinstance(result, tuple) else result.sample
            metrics["raw_decoded_tensor"] = tensor_info(prediction, "raw_decoded_tensor", True)
            metrics["phases"]["raw_decode"] = time.perf_counter() - start
            memory_sample()
            return result

        pipe.vae.decode = observed_decode
        final_latents = None
        last_callback = None

        def callback(this, step, timestep, kwargs):
            nonlocal final_latents, last_callback
            latents = kwargs["latents"]
            if not bool(torch.isfinite(latents).all().item()):
                raise FloatingPointError(f"Non-finite denoised latents atstep{step+1}")
            torch.mps.synchronize()
            now = time.perf_counter()
            item = {"step": int(step + 1), "timestep": float(timestep.item()), "finite": True,
                    "unet_calls_so_far": metrics["unet_calls"], "last_unet_seconds": unet_times[-1],
                    "seconds_since_previous_callback": None if last_callback is None else now - last_callback,
                    "memory": memory_sample()}
            metrics["step_metrics"].append(item)
            last_callback = now
            final_latents = latents.detach().clone()
            log("denoise", completed_steps=step + 1, unet_calls=metrics["unet_calls"])
            return kwargs

        memory_sample()
        log("pipeline")
        pipeline_start = time.perf_counter()
        result = pipe(prompt=prompt, image=canvas, mask_image=mask, width=width, height=height,
                      num_inference_steps=args.steps, guidance_scale=args.guidance, strength=args.strength,
                      generator=generator, padding_mask_crop=None, output_type="pil",
                      callback_on_step_end=callback, callback_on_step_end_tensor_inputs=["latents"])
        torch.mps.synchronize()
        metrics["phases"]["pipeline_total"] = time.perf_counter() - pipeline_start
        metrics["phases"]["unet_seconds_sum"] = sum(unet_times)
        metrics["phases"]["denoise_with_observation"] = last_callback - denoise_start
        for handle in handles:
            handle.remove()
        if metrics["unet_calls"] != len(metrics["actual_timesteps"]) or len(metrics["step_metrics"]) != metrics["unet_calls"]:
            raise AssertionError("Observed UNet and callback counts differ from actual native timesteps")
        metrics["final_latents"] = tensor_info(final_latents, "final_latents", True)
        if metrics["unet_calls"] != args.steps or metrics["cfg_sample_branch_evaluations"] != 2 * args.steps:
            raise AssertionError("Strength1 must keep all requested Euler timesteps and both CFG branches")
        if len(metrics["vae_conditioning_encodes"]) != 1:
            raise AssertionError("Strength1 native9-channel conditioning must encode masked source exactly once")
        metrics["observed_rng_path"] = ["initial noise", "masked-source VAE posterior"]
        metrics["generator"]["final_state_sha256"] = generator_hash()
        raw = result.images[0].convert("RGB")
        if raw.size != (width, height):
            raise AssertionError("Unexpected raw output geometry")
        raw_path, composite_path = output / "raw.png", output / "composite.png"
        raw.save(raw_path)
        raw_array = np.asarray(raw)
        source_error = np.abs(raw_array[y0:y1, x0:x1].astype(np.float32) - source_array.astype(np.float32))
        metrics["raw_source_mae_0_255"] = {"all": float(source_error.mean()), "top16": float(source_error[:16].mean()),
                                          "bottom16": float(source_error[-16:].mean()), "inner": float(source_error[16:-16].mean())}
        composite = raw.copy()
        composite.paste(source, (x0, y0))
        composite.save(composite_path)
        composite_array = np.asarray(Image.open(composite_path).convert("RGB"))
        outside = expected_mask.astype(bool)
        metrics["source_exact_after_compositor"] = bool(np.array_equal(composite_array[y0:y1, x0:x1], source_array))
        metrics["generated_region_exact_raw_after_compositor"] = bool(np.array_equal(composite_array[outside], raw_array[outside]))
        if not metrics["source_exact_after_compositor"] or not metrics["generated_region_exact_raw_after_compositor"]:
            raise AssertionError("Source-only compositor failed")
        metrics.update({"raw_path": str(raw_path), "raw_sha256": file_sha256(raw_path),
                        "composite_path": str(composite_path), "composite_sha256": file_sha256(composite_path),
                        "unet_latent_spatial_sites_per_branch": height // 8 * (width // 8),
                        "unet_branch_spatial_site_evaluations": metrics["cfg_sample_branch_evaluations"] * (height // 8) * (width // 8),
                        "visual_quality": "requires independent direct inspection", "status": "success"})
        log("complete")
    except BaseException as exc:
        metrics.update({"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
        write()
        raise


if __name__ == "__main__":
    main()
