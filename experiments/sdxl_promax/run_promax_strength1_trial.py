"""Isolated strength-one ProMax pure-noise initialization probe on native Diffusers.

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
MODEL = ROOT / ".build/models/sdxl-promax-official"
REPOSITORIES = {
    "controlnet": ("xinsir/controlnet-union-sdxl-1.0", "801a4a3fa3d4c936f4feea95b98607bc6726f80c"),
    "base": ("stabilityai/stable-diffusion-xl-base-1.0", "462165984030d82259a11f4367a4eed129e94a7b"),
    "vae_fix": ("madebyollin/sdxl-vae-fp16-fix", "207b116dae70ace3637169f1ddd2434b91b3a8cd"),
}
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
              "repositories": manifest.get("repositories"),
              "payload_hashes_checked": payloads, "files": [], "issues": issues}
    if manifest.get("status") != "verified":
        issues.append("Downloader manifest is not verified")
    reported = {entry["component"]: (entry["repo_id"], entry["revision"]) for entry in manifest.get("repositories", [])}
    if reported != REPOSITORIES:
        issues.append("Expected exact pinned official ProMax/base/FP16-fix VAE repositories")
    if Path(manifest.get("local_dir", "")).resolve() != model_path:
        issues.append("Manifest names another checkpoint directory")
    for entry in manifest.get("files", []):
        relative = Path(entry["destination_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Manifest path escapes checkpoint directory")
        path = model_path / relative
        info = {"path": entry["destination_path"], "expected_bytes": entry["bytes"], "exists": path.is_file(),
                "repo_id": entry["repo_id"], "revision": entry["revision"], "action": entry["action"]}
        if (entry["repo_id"], entry["revision"]) not in REPOSITORIES.values():
            issues.append(f"Unpinned file repository: {relative}")
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
    parser.add_argument("--steps", type=int, choices=(30,), default=30)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--strength", type=float, choices=(1.0,), default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt")
    parser.add_argument("--model-path", type=Path, default=MODEL)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.strength != 1.0 or not 1 < args.guidance < 30:
        parser.error("This isolated probe requires strength=1 and CFG guidance in (1,30)")
    model_path = args.model_path.expanduser().resolve()
    manifest_path = (args.manifest or model_path.with_suffix(".manifest.json")).resolve()
    output = (args.output or ROOT / f"experiments/sdxl_promax/runs/2026-10-05/{args.track}-{args.steps}-cfg{args.guidance:g}-strength{args.strength:g}-seed{args.seed}").resolve()
    metrics_path = output / ("preflight.json" if args.preflight else "metrics.json")
    if not args.preflight and metrics_path.exists():
        raise FileExistsError(f"Refusing to overwrite prior run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    begun = time.perf_counter()
    metrics = {"status": "preflight" if args.preflight else "starting", "track": args.track,
               "requested_steps": args.steps, "expected_actual_steps": 30,
               "expected_cfg_branch_samples_per_model": 60,
               "guidance": args.guidance, "strength": args.strength, "seed": args.seed,
               "model_repositories": REPOSITORIES, "model_path": str(model_path),
               "runner_sha256": file_sha256(Path(__file__)), "phases": {}, "step_metrics": [],
               "model_weights_loaded": False, "offline": True, "compute_mode": "native full-canvas ordinary4-channel SDXL plus8-task Union repaint",
               "growing_compute": False, "native_known_region_latent_replacement": True, "padding_mask_crop": None,
               "watermark": False, "negative_prompt": None, "adapters": [], "research_use_only": True,
               "component_licenses": {"base": "OpenRAIL++", "controlnet": "Apache-2.0", "vae_fix": "MIT"},
               "unet_calls": 0, "cfg_sample_branch_evaluations": 0, "controlnet_calls": 0,
               "controlnet_branch_evaluations": 0, "scheduler_step_calls": 0,
               "control_mode": [7], "controlnet_conditioning_scale": 1.0, "guess_mode": False,
               "control_guidance_range": [0.0, 1.0], "author_demo_area_reference": "approximately1024-squared area",
               "resolution_matches_author_demo": False,"benchmark_pixels":512*1152,
               "benchmark_area_fraction_of_1024_squared":512*1152/(1024*1024)}

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
        from diffusers import (StableDiffusionXLControlNetUnionInpaintPipeline as StableDiffusionXLInpaintPipeline,
                               ControlNetUnionModel, AutoencoderKL, EulerAncestralDiscreteScheduler, UNet2DConditionModel)
        from test_strength1_cpu_preflight import run_preflight

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
        control_rgb_array = canvas_array.copy()
        control_rgb_array[expected_mask.astype(bool)] = 0
        control_rgb = Image.fromarray(control_rgb_array)
        control_rgb_path = output / "control_rgb.png"
        control_rgb.save(control_rgb_path)
        unknown_rgb_values = np.unique(canvas_array[expected_mask.astype(bool)], axis=0).tolist()
        metrics.update({"width": width, "height": height, "source_rect_xyxy": rect,
                        "input_files": {name: {"path": str(path), "sha256": file_sha256(path)}
                                        for name, path in (("source", source_path), ("canvas", canvas_path), ("mask", mask_path))},
                        "source_input_pixels_exact": True, "mask_convention": "white regenerates, black preserves",
                        "initial_canvas_unknown_rgb_values_0_255": unknown_rgb_values,
                        "masked_source_unknown_normalized_rgb": 0.0,
                        "control_rgb_path": str(control_rgb_path), "control_rgb_sha256": file_sha256(control_rgb_path),
                        "control_rgb_pixels_sha256": hashlib.sha256(control_rgb_array.tobytes()).hexdigest(),
                        "control_rgb_unknown_values": [0,0,0], "control_preprocessing_range": [0.0,1.0],
                        "native_initialization": "strength1 still encodes the complete canvas for the four-channel source anchor; initial full-canvas latents equal actual noise times scheduler init sigma",
                        "full_canvas_source_anchor_vae_encode_still_required": True,
                        "cross_model_comparison": {
                            "shared": "source pixels, regeneration mask and canvas geometry",
                            "full_canvas_bytes_matched_with_flux_fill": False,
                            "initial_noise_matched_with_flux_fill": False,
                            "flux_fill_initial_canvas_unknown_rgb": [128, 128, 128],
                            "note": "This strength1 probe uses pure-noise initialization, source-anchor encoding, learned repaint control and native per-step known-region latent replacement"},
                        "checkpoint_manifest": manifest_check(model_path, manifest_path)})
        prompt = args.prompt or (DEFAULT_PROMPT if args.track == "track3" else json.loads((ROOT / "experiments/flux2/prompts.json").read_text())[args.track])
        metrics["prompt"] = prompt
        metrics["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
        metrics["cpu_preflight"] = run_preflight()
        metrics["cpu_gate_sha256"] = file_sha256(Path(__file__).with_name("test_strength1_cpu_preflight.py"))
        if metrics["cpu_preflight"]["status"] != "passed":
            raise RuntimeError("CPU conditioning gate failed")
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
        metrics["strict_component_loading"] = {}
        def load_component(cls, path, name, **kwargs):
            module, info = cls.from_pretrained(str(path), local_files_only=True, use_safetensors=True,
                         torch_dtype=torch.float16, output_loading_info=True, **kwargs)
            metrics["strict_component_loading"][name] = info
            if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
                raise ValueError(f"Strict official component loading failed for{name}: {info}")
            return module
        controlnet = load_component(ControlNetUnionModel, model_path / "controlnet", "controlnet")
        vae = load_component(AutoencoderKL, model_path / "vae_fix", "vae_fix")
        unet = load_component(UNet2DConditionModel, model_path / "base", "base_unet", subfolder="unet", variant="fp16")
        scheduler = EulerAncestralDiscreteScheduler.from_pretrained(str(model_path / "base"), subfolder="scheduler", local_files_only=True)
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(str(model_path / "base"), controlnet=controlnet,
                    vae=vae, unet=unet, scheduler=scheduler, local_files_only=True, variant="fp16",
                    use_safetensors=True, torch_dtype=torch.float16, add_watermarker=False).to("mps")
        pipe.set_progress_bar_config(disable=True)
        torch.mps.synchronize()
        metrics["phases"]["model_load"] = time.perf_counter() - start
        metrics["model_weights_loaded"] = True
        if pipe.unet.config.in_channels != 4 or pipe.vae.config.latent_channels != 4 or pipe.controlnet.config.num_control_type != 8:
            raise ValueError("Expected ordinary4-channel SDXL,4-channel VAE and8-task Union")
        if list(pipe.controlnet.config.conditioning_embedding_out_channels) != [16,32,96,256]:
            raise ValueError("ProMax control embedding channels differ from the official checkpoint")
        metrics["native_model_classes"] = {name: {"class": type(getattr(pipe,name)).__name__, "dtype": str(getattr(pipe,name).dtype)}
                                          for name in ("unet", "controlnet", "vae")}
        metrics["runtime_controlnet_source"] = {"path": inspect.getsourcefile(ControlNetUnionModel), "sha256": file_sha256(Path(inspect.getsourcefile(ControlNetUnionModel)))}
        metrics.update({"device": "mps", "dtype": "torch.float16", "unet_in_channels": 4,
                        "component_parameter_bytes": {name: sum(p.numel() * p.element_size() for p in getattr(pipe, name).parameters())
                                                      for name in ("unet", "controlnet", "vae", "text_encoder", "text_encoder_2")},
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
        observed = {"image_latents": None, "noise": None, "mask": None, "solver_latents": None, "known_noised": None}
        original_prepare = pipe.prepare_latents

        def observed_prepare(this, *call_args, **kwargs):
            bound = inspect.signature(original_prepare).bind(*call_args, **kwargs)
            bound.apply_defaults()
            if not bound.arguments["is_strength_max"] or not bound.arguments["return_image_latents"]:
                raise AssertionError("Native four-channel strength1 requires pure-noise initialization and source anchor latents")
            if bound.arguments["latents"] is not None or not bound.arguments["add_noise"]:
                raise AssertionError("This probe must use actual native sampled initialization noise")
            encode_count_before = len(metrics["vae_conditioning_encodes"])
            state_before = generator_hash()
            start = time.perf_counter()
            result = original_prepare(*call_args, **kwargs)
            torch.mps.synchronize()
            if len(result) != 3:
                raise AssertionError("Ordinary4-channel ProMax must return original image latents")
            if len(metrics["vae_conditioning_encodes"]) != encode_count_before + 1:
                raise AssertionError("Native strength1 must still encode the full source canvas exactly once before noise")
            if not torch.equal(result[0], result[1] * pipe.scheduler.init_noise_sigma):
                raise AssertionError("Native strength1 initial latents differ from actual noise times init sigma")
            if metrics["native_add_noise_observations"]:
                raise AssertionError("Native strength1 must not call scheduler.add_noise during initialization")
            observed["image_latents"], observed["noise"] = result[2].detach(), result[1].detach()
            metrics["prepare_latents"] = {"seconds": time.perf_counter() - start,
                "generator_state_before": state_before, "generator_state_after": generator_hash(),
                "initial_latents": tensor_info(result[0], "initial_latents", True),
                "actual_noise": tensor_info(result[1], "actual_noise", True),
                "image_latents": tensor_info(result[2], "image_latents", True),
                "initial_noise_mode": "native pure noise times init sigma; full source-anchor posterior still encoded before noise",
                "scheduler_init_noise_sigma": float(pipe.scheduler.init_noise_sigma),
                "initial_full_canvas_exact_noise_times_sigma": True,
                "unknown_initial_latents_exact_noise_times_sigma": True,
                "full_canvas_source_anchor_encode_count": 1,
                "initialization_add_noise_calls": 0}
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
            observed["mask"] = result[0][:1].detach()
            metrics["prepare_mask_latents"] = {"seconds": time.perf_counter() - start,
                "generator_state_before": state_before, "generator_state_after": generator_hash(),
                "normalized_masked_source_input": tensor_info(source_input, "normalized_masked_source_input", True),
                "mask_latents": tensor_info(result[0], "mask_latents", True),
                "masked_source_latents": tensor_info(result[1], "masked_source_latents", True),
                "unknown_normalized_rgb_exact_zero": True}
            return result

        pipe.prepare_mask_latents = MethodType(observed_mask, pipe)
        original_control_prepare = pipe.prepare_control_image
        def observed_control_prepare(this, *call_args, **kwargs):
            start = time.perf_counter()
            result = original_control_prepare(*call_args, **kwargs)
            torch.mps.synchronize()
            if result.shape != (2,3,height,width) or float(result.min().item()) < 0 or float(result.max().item()) > 1:
                raise AssertionError("Official single repaint control must be CFGbatch2 RGB in[0,1]")
            metrics["prepared_control_rgb"] = {"seconds": time.perf_counter()-start,
                                                "tensor": tensor_info(result,"prepared_control_rgb",True)}
            return result
        pipe.prepare_control_image = MethodType(observed_control_prepare, pipe)
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
            if len(metrics["actual_timesteps"]) != args.steps or result[1] != args.steps or metrics["scheduler_begin_index"] != 0:
                raise AssertionError("Strength1 must retain all thirty native timesteps with begin index zero")
            return result

        pipe.get_timesteps = MethodType(observed_timesteps, pipe)
        original_solver_step, original_add_noise = pipe.scheduler.step, pipe.scheduler.add_noise
        metrics["ancestral_scheduler_steps"] = []
        metrics["native_add_noise_observations"] = []
        def observed_solver_step(*call_args, **kwargs):
            state_before = generator_hash()
            start = time.perf_counter()
            result = original_solver_step(*call_args, **kwargs)
            torch.mps.synchronize()
            observed["solver_latents"] = result[0].detach().clone()
            observed["known_noised"] = None
            metrics["scheduler_step_calls"] += 1
            state_after = generator_hash()
            metrics["ancestral_scheduler_steps"].append({"call":metrics["scheduler_step_calls"],
                "seconds":time.perf_counter()-start,"generator_state_before":state_before,
                "generator_state_after":state_after,"rng_state_changed":state_before!=state_after})
            return result
        def observed_add_noise(*call_args, **kwargs):
            bound = inspect.signature(original_add_noise).bind(*call_args, **kwargs)
            result = original_add_noise(*call_args, **kwargs)
            phase = "initialization" if metrics["scheduler_step_calls"] == 0 else "known_region_next_timestep"
            if metrics["scheduler_step_calls"] > 0:
                observed["known_noised"] = result.detach().clone()
            metrics["native_add_noise_observations"].append({"phase":phase,"after_step":metrics["scheduler_step_calls"],
                "timesteps":bound.arguments["timesteps"].detach().float().cpu().tolist()})
            return result
        # Native prepare_extra_step_kwargs inspects this signature to decide
        # whether to forward the CPU generator into the ancestral sampler.
        observed_solver_step.__signature__ = inspect.signature(original_solver_step)
        observed_add_noise.__signature__ = inspect.signature(original_add_noise)
        pipe.scheduler.step = observed_solver_step
        pipe.scheduler.add_noise = observed_add_noise
        sampler_kwargs = pipe.prepare_extra_step_kwargs(generator,0.0)
        if sampler_kwargs.get("generator") is not generator:
            raise AssertionError("Native ancestral sampler must receive the actual seeded CPU generator")
        metrics["native_sampler_generator_forwarding_verified"] = True
        metrics["ancestral_rng_path"] = ["full initial-image VAE posterior","initial noise",
                                          "masked-source VAE posterior","one ancestral noise draw per solver step"]
        metrics["comparison_with_strength09999"] = {
            "initial_rng_operation_order_shared": True,
            "strength1_skips_initial_scheduler_add_noise": True,
            "initial_scheduler_add_noise_consumes_rng": False,
            "actual_initial_noise_matching": "requires comparison of the actual saved noise tensors; same seed alone is not proof",
            "strength1_actual_solver_steps": 30,
            "strength09999_actual_solver_steps": 29,
            "note": "Both native four-channel paths encode the full source posterior before initial noise and then the masked-source posterior; strength1 has one additional ancestral solver noise draw"}
        unet_times = []
        control_times = []
        unet_start = None
        control_start = None
        denoise_start = None

        def control_pre(module, call_args, kwargs):
            nonlocal control_start,denoise_start
            sample = call_args[0] if call_args else kwargs["sample"]
            if sample.shape != (2,4,height//8,width//8) or kwargs["control_type_idx"] != [7] or kwargs["guess_mode"]:
                raise AssertionError("Expected single repaint control7 with CFGbatch2 and ordinary4channel latents")
            if len(kwargs["controlnet_cond"]) != 1:
                raise AssertionError("Official baseline supplies exactly one repaint RGB control")
            if denoise_start is None:
                torch.mps.synchronize()
                denoise_start = time.perf_counter()
                metrics["phases"]["pipeline_pre_denoise"] = denoise_start - pipeline_start
                metrics["first_controlnet_input"] = tensor_info(sample,"first_controlnet_input",True)
                type_values = kwargs["control_type"].detach().float().cpu().numpy()
                expected_type = np.zeros((2,8),dtype=np.float32); expected_type[:,7]=1
                if not np.array_equal(type_values,expected_type):
                    raise AssertionError("Native Union must receive8slot one-hot repaint7 for both CFG branches")
                metrics["control_type"] = tensor_info(kwargs["control_type"],"control_type",True)
            control_start = time.perf_counter()
        def control_post(module, call_args, kwargs, result):
            down,mid = result
            if any(not bool(torch.isfinite(value).all().item()) for value in [*down,mid]):
                raise FloatingPointError("Non-finite ProMax control residuals")
            torch.mps.synchronize()
            control_times.append(time.perf_counter()-control_start)
            metrics["controlnet_calls"] += 1
            metrics["controlnet_branch_evaluations"] += int(mid.shape[0])
            if metrics["controlnet_calls"] == 1:
                metrics["control_residual_shapes"] = {"down":[list(value.shape) for value in down],"mid":list(mid.shape)}

        def unet_pre(module, call_args, kwargs):
            nonlocal unet_start, denoise_start
            sample = call_args[0] if call_args else kwargs["sample"]
            if sample.shape[0] != 2 or sample.shape[1] != 4:
                raise AssertionError(f"Expected one CFG UNet call with batch2 and4channels, got{list(sample.shape)}")
            if denoise_start is None:
                torch.mps.synchronize()
                denoise_start = time.perf_counter()
                metrics["phases"]["pipeline_pre_denoise"] = denoise_start - pipeline_start
            if metrics["unet_calls"] == 0:
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
                   pipe.unet.register_forward_hook(unet_post, with_kwargs=True),
                   pipe.controlnet.register_forward_pre_hook(control_pre, with_kwargs=True),
                   pipe.controlnet.register_forward_hook(control_post, with_kwargs=True)]
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
            last_step = step == len(metrics["actual_timesteps"])-1
            known_target = observed["image_latents"] if last_step else observed["known_noised"]
            if known_target is None or observed["solver_latents"] is None or observed["mask"] is None:
                raise AssertionError("Missing observational tensors for native source-latent replacement")
            expected = (1-observed["mask"])*known_target + observed["mask"]*observed["solver_latents"]
            if not torch.equal(expected,latents):
                raise AssertionError("Native post-step source-latent replacement differs from observed equation")
            now = time.perf_counter()
            item = {"step": int(step + 1), "timestep": float(timestep.item()), "finite": True,
                    "unet_calls_so_far": metrics["unet_calls"], "last_unet_seconds": unet_times[-1],
                    "controlnet_calls_so_far":metrics["controlnet_calls"],"last_controlnet_seconds":control_times[-1],
                    "native_source_replacement_verified":True,"known_target":"clean_source_latents" if last_step else "source_noised_at_next_timestep",
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
                      control_image=[control_rgb],control_mode=[7],controlnet_conditioning_scale=1.0,
                      guess_mode=False,control_guidance_start=0.0,control_guidance_end=1.0,
                      generator=generator, padding_mask_crop=None, output_type="pil",
                      callback_on_step_end=callback, callback_on_step_end_tensor_inputs=["latents"])
        torch.mps.synchronize()
        metrics["phases"]["pipeline_total"] = time.perf_counter() - pipeline_start
        metrics["phases"]["unet_seconds_sum"] = sum(unet_times)
        metrics["phases"]["controlnet_seconds_sum"] = sum(control_times)
        metrics["phases"]["denoise_with_observation"] = last_callback - denoise_start
        for handle in handles:
            handle.remove()
        if metrics["unet_calls"] != len(metrics["actual_timesteps"]) or len(metrics["step_metrics"]) != metrics["unet_calls"]:
            raise AssertionError("Observed UNet and callback counts differ from actual native timesteps")
        if metrics["controlnet_calls"] != metrics["unet_calls"] or metrics["scheduler_step_calls"] != metrics["unet_calls"]:
            raise AssertionError("ControlNet/UNet/ancestral solver counts differ")
        if metrics["unet_calls"] != args.steps:
            raise AssertionError("Strength1 must execute all thirty model and ancestral solver calls")
        if len(metrics["native_add_noise_observations"]) != args.steps - 1 or any(item["phase"] != "known_region_next_timestep" for item in metrics["native_add_noise_observations"]):
            raise AssertionError("Strength1 must have exactly twenty-nine next-step source clamps and no initialization add_noise")
        if len(metrics["vae_conditioning_encodes"]) != 2:
            raise AssertionError("Expected full source-anchor posterior and masked-source posterior encodes")
        if metrics["cfg_sample_branch_evaluations"] != 2*metrics["unet_calls"] or metrics["controlnet_branch_evaluations"] != 2*metrics["controlnet_calls"]:
            raise AssertionError("Expected both CFG branches in each model call")
        metrics["native_source_replacement_all_steps_verified"] = True
        metrics["final_latents"] = tensor_info(final_latents, "final_latents", True)
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
