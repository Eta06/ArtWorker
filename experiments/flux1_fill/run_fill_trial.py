"""Isolated native FLUX.1 Fill outpainting teacher, with raw decode retained.

This follows the installed Flux1Fill loop and MaskUtil conditioning. The full
canvas is processed on every step. It does not implement growing computation,
streaming generation, latent known-region clamps, or a mobile runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import struct
import sys
import time
import traceback
from pathlib import Path

# Set these before importing transformers/huggingface_hub. The runner consumes
# an already downloaded local checkpoint and must never trigger a download.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / ".build/models/flux1-fill-q4"
MODEL_REVISION = "eebfbaa12c95107169452c7d22622e04771192a3"
EXPECTED_SIZE = (512, 1152)
EXPECTED_RECT = [0, 320, 512, 832]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_inventory(model_path: Path) -> dict:
    """Inspect headers only: never load tensor payloads in CPU preflight."""
    result = {"path": str(model_path), "complete": True, "components": {}, "missing": []}
    for component in ("transformer", "vae", "text_encoder", "text_encoder_2"):
        directory = model_path / component
        shards = sorted(path for path in directory.glob("*.safetensors") if not path.name.startswith("._"))
        index_path = directory / "model.safetensors.index.json"
        if index_path.is_file():
            index = json.loads(index_path.read_text())
            names = sorted(set(index.get("weight_map", {}).values()))
            if not names or any(not isinstance(name, str) or Path(name).name != name for name in names):
                raise ValueError(f"Invalid local shard index: {index_path}")
            shards = [directory / name for name in names]
        entries = []
        if not shards:
            result["missing"].append(f"{directory}/*.safetensors")
        for path in shards:
            if not path.is_file():
                result["missing"].append(str(path))
                continue
            with path.open("rb") as stream:
                prefix = stream.read(8)
                if len(prefix) != 8:
                    raise ValueError(f"Truncated checkpoint header: {path}")
                header_size = struct.unpack("<Q", prefix)[0]
                if not 2 <= header_size <= 32 * 1024 * 1024:
                    raise ValueError(f"Invalid checkpoint header length: {path}")
                encoded = stream.read(header_size)
            header = json.loads(encoded)
            tensors = {key: value for key, value in header.items() if key != "__metadata__"}
            payload_bytes = max((entry["data_offsets"][1] for entry in tensors.values()), default=0)
            expected_bytes = 8 + header_size + payload_bytes
            actual_bytes = path.stat().st_size
            if actual_bytes != expected_bytes:
                result["missing"].append(f"{path} size {actual_bytes}, expected {expected_bytes}")
            entries.append({"path": str(path), "bytes": actual_bytes, "expected_bytes": expected_bytes,
                            "header_sha256": hashlib.sha256(encoded).hexdigest(),
                            "tensor_count": len(tensors), "metadata": header.get("__metadata__", {}),
                            "x_embedder": {key: value for key, value in tensors.items()
                                           if key.startswith("x_embedder.")}})
        result["components"][component] = entries
    for component, required in (("tokenizer", ("tokenizer.json", "tokenizer_config.json")),
                                ("tokenizer_2", ("tokenizer.json", "tokenizer_config.json"))):
        for filename in required:
            path = model_path / component / filename
            if not path.is_file():
                result["missing"].append(str(path))
    result["complete"] = not result["missing"]
    result["total_weight_bytes"] = sum(entry["bytes"] for group in result["components"].values() for entry in group)
    return result


def verify_manifest(model_path: Path, manifest_path: Path, verify_payloads: bool = False) -> dict:
    if not manifest_path.is_file():
        return {"status": "missing", "verified": False, "path": str(manifest_path)}
    manifest = json.loads(manifest_path.read_text())
    result = {"status": manifest.get("status"), "verified": False, "path": str(manifest_path),
              "manifest_sha256": file_sha256(manifest_path), "repo": manifest.get("repo"),
              "revision": manifest.get("resolved_revision"), "payload_hashes_checked": verify_payloads,
              "issues": [], "files": []}
    if manifest.get("status") != "verified":
        result["issues"].append("Downloader manifest is not yet verified")
    if Path(manifest.get("local_dir", "")).resolve() != model_path:
        result["issues"].append("Downloader manifest names another checkpoint directory")
    if (manifest.get("repo") != "mflux-community/flux-1-dev-fill-mflux-q4"
            or manifest.get("resolved_revision") != MODEL_REVISION or manifest.get("revision") != MODEL_REVISION):
        result["issues"].append("Expected pinned public maintainer FLUX Fill Q4 repository")
    for entry in manifest.get("files", []):
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Manifest path escapes local checkpoint directory")
        path = model_path / relative
        expected_size = entry["expected_bytes"]
        info = {"path": entry["path"], "expected_bytes": expected_size,
                "expected_lfs_sha256": entry.get("expected_lfs_sha256"), "exists": path.is_file()}
        if not path.is_file() or path.stat().st_size != expected_size:
            result["issues"].append(f"Missing or incomplete manifest file: {relative}")
        elif verify_payloads:
            if entry.get("expected_lfs_sha256"):
                info["sha256"] = file_sha256(path)
                if info["sha256"] != entry["expected_lfs_sha256"]:
                    result["issues"].append(f"Payload SHA256 mismatch: {relative}")
            else:
                content = path.read_bytes()
                info["git_blob_sha1"] = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
                if info["git_blob_sha1"] != entry.get("git_blob_id"):
                    result["issues"].append(f"Pinned Git blob mismatch: {relative}")
        result["files"].append(info)
    if not result["files"]:
        result["issues"].append("Downloader manifest has no files")
    result["verified"] = not result["issues"]
    return result


def prepare_inputs(track_id: str, destination: Path) -> tuple:
    import numpy as np
    from PIL import Image

    shared = json.loads((ROOT / "experiments/evaluation/inputs/inputs.json").read_text())
    track = next(item for item in shared["tracks"] if item["id"] == track_id)
    width, height = shared["canvas_size"]
    rect = shared["source_rect_xyxy"]
    source_path = Path(track["source_512"])
    source = Image.open(source_path).convert("RGB")
    if (width, height) != EXPECTED_SIZE or source.size != (512, 512) or rect != EXPECTED_RECT:
        raise ValueError("Expected shared source512 centered at [0,320,512,832] in 512x1152")
    x0, y0, x1, y1 = rect
    canvas = Image.new("RGB", (width, height), (128, 128, 128))
    canvas.paste(source, (x0, y0))
    mask = Image.new("L", (width, height), 255)
    mask.paste(0, (x0, y0, x1, y1))
    assert np.array_equal(np.asarray(canvas)[y0:y1, x0:x1], np.asarray(source))
    mask_array = np.asarray(mask)
    assert np.all(mask_array[y0:y1, x0:x1] == 0)
    assert int(np.count_nonzero(mask_array)) == width * (height - 512)
    canvas_path, mask_path = destination / "input_canvas.png", destination / "input_mask.png"
    canvas.save(canvas_path)
    mask.save(mask_path)
    info = {"width": width, "height": height, "source_rect_xyxy": rect,
            "source_512_path": str(source_path), "source_512_sha256": file_sha256(source_path),
            "source_pixels_sha256": hashlib.sha256(np.asarray(source).tobytes()).hexdigest(),
            "canvas_path": str(canvas_path), "canvas_sha256": file_sha256(canvas_path),
            "mask_path": str(mask_path), "mask_sha256": file_sha256(mask_path),
            "mask_convention": "white 255 regenerates; black 0 preserves",
            "unknown_input_normalized_rgb": "exact zero after image * (1-mask)",
            "known_source_input_pixels_exact": True}
    return source, info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=("track1", "track2", "track3"), default="track3")
    parser.add_argument("--steps", type=int, choices=(30, 50), default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance", type=float, default=30.0)
    parser.add_argument("--model-path", type=Path, default=MODEL)
    parser.add_argument("--manifest", type=Path, help="Pinned verified downloader manifest (default model-path.manifest.json)")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prompt", help="Optional controlled prompt experiment; recorded verbatim")
    parser.add_argument("--preflight", action="store_true", help="CPU tests and checkpoint headers only; no model weights")
    args = parser.parse_args()
    if not 0 < args.guidance < 100:
        parser.error("Guidance must be between 0 and 100")
    model_path = args.model_path.expanduser().resolve()
    manifest_path = (args.manifest or model_path.with_suffix(".manifest.json")).expanduser().resolve()
    destination = (args.output or ROOT / f"experiments/flux1_fill/runs/2026-10-05/{args.track}-{args.steps}-seed{args.seed}").resolve()
    metrics_path = destination / ("preflight.json" if args.preflight else "metrics.json")
    if not args.preflight and metrics_path.exists():
        raise FileExistsError(f"Refusing to overwrite prior run: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    begun = time.perf_counter()
    metrics = {"status": "preflight" if args.preflight else "starting", "track": args.track,
               "seed": args.seed, "steps": args.steps, "guidance": args.guidance,
               "runner_sha256": file_sha256(Path(__file__)), "phases": {}, "step_metrics": [],
               "model_path": str(model_path), "official_model_repo": "black-forest-labs/FLUX.1-Fill-dev",
               "official_example": "50 steps, guidance 30, max sequence length 512",
               "compute_mode": "native full canvas every step", "growing_compute": False,
               "known_bridge": False, "negative_branch": False, "adapters": [],
               "offline_environment": True, "model_weights_loaded": False,
               "research_use_only": True, "inherited_model_license": "flux-1-dev-non-commercial-license"}

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
        import mlx.core as mx
        if args.preflight:
            mx.set_default_device(mx.cpu)
        import numpy as np
        from PIL import Image
        from mflux.models.flux.variants.fill.mask_util import MaskUtil
        from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator

        source, input_info = prepare_inputs(args.track, destination)
        metrics.update(input_info)
        metrics["checkpoint_inventory"] = checkpoint_inventory(model_path)
        metrics["checkpoint_manifest"] = verify_manifest(model_path, manifest_path)
        descriptions = json.loads((ROOT / "experiments/flux2/prompts.json").read_text())
        prompt = args.prompt or descriptions[args.track]
        metrics["prompt"] = prompt
        metrics["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
        width, height = EXPECTED_SIZE
        tokens = width // 16 * (height // 16)
        metrics.update({"target_tokens_per_forward": tokens, "planned_nfe": args.steps,
                        "planned_target_token_forwards": tokens * args.steps,
                        "conditioning_channels": {"noise": 64, "masked_source": 64, "pixel_mask": 256, "total": 384}})
        from test_cpu_preflight import run_preflight
        metrics["conditioning_cpu_preflight"] = run_preflight()
        if metrics["conditioning_cpu_preflight"]["status"] != "passed":
            raise RuntimeError("CPU conditioning gate failed")
        if args.preflight:
            files_ready = metrics["checkpoint_inventory"]["complete"] and metrics["checkpoint_manifest"]["verified"]
            tokenizer_files_present = all((model_path / directory / filename).is_file()
                                          for directory in ("tokenizer", "tokenizer_2")
                                          for filename in ("tokenizer.json", "tokenizer_config.json"))
            if tokenizer_files_present:
                from mflux.models.common.tokenizer.tokenizer_loader import TokenizerLoader
                from mflux.models.flux.weights.flux_weight_definition import FluxWeightDefinition
                tokenizers = TokenizerLoader.load_all(FluxWeightDefinition.get_tokenizers(), str(model_path),
                                                     max_length_overrides={"t5": 512})
                metrics["tokenizer_preflight"] = {name: {"max_length": tokenizer.max_length,
                                                         "token_shape": list(tokenizer.tokenize(prompt).input_ids.shape)}
                                                  for name, tokenizer in tokenizers.items()}
            metrics.update({"device": "CPU", "status": "preflight_files_ready" if files_ready else "preflight_files_incomplete"})
            write()
            print(json.dumps(metrics, indent=2), flush=True)
            return
        # The test deliberately selects CPU; restore GPU only in a requested real run.
        mx.set_default_device(mx.gpu)
        if not metrics["checkpoint_inventory"]["complete"]:
            raise FileNotFoundError(f"Local checkpoint incomplete: {metrics['checkpoint_inventory']['missing']}")
        if not metrics["checkpoint_manifest"]["verified"]:
            raise ValueError(f"Pinned downloader manifest not verified: {metrics['checkpoint_manifest']}")
        log("checkpoint_hash_verification")
        hash_start = time.perf_counter()
        metrics["checkpoint_manifest"] = verify_manifest(model_path, manifest_path, verify_payloads=True)
        metrics["phases"]["checkpoint_hash_verification"] = {"seconds": time.perf_counter() - hash_start}
        if not metrics["checkpoint_manifest"]["verified"]:
            raise ValueError(f"Pinned checkpoint changed: {metrics['checkpoint_manifest']['issues']}")
        from mlx.utils import tree_flatten
        from mflux.models.common.config.config import Config
        from mflux.models.common.vae.vae_util import VAEUtil
        from mflux.models.flux.model.flux_text_encoder.prompt_encoder import PromptEncoder
        from mflux.models.flux.variants.fill.flux_fill import Flux1Fill
        from mflux.utils.image_util import ImageUtil
        import inspect

        metrics["runtime_source_sha256"] = {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in (("native_fill", Path(inspect.getsourcefile(Flux1Fill))),
                               ("mask_util", Path(inspect.getsourcefile(MaskUtil))),
                               ("config", Path(inspect.getsourcefile(Config))),
                               ("latent_creator", Path(inspect.getsourcefile(FluxLatentCreator))))}
        metrics["cpu_gate_sha256"] = file_sha256(Path(__file__).with_name("test_cpu_preflight.py"))

        mx.set_cache_limit(512 * 1024 ** 2)
        mx.set_memory_limit(25 * 1024 ** 3)
        metrics["metal_device"] = mx.device_info()
        metrics["mlx_version"] = mx.__version__
        phase_peaks = []

        def phase_done(name: str, start: float) -> None:
            metrics["phases"][name] = {"seconds": time.perf_counter() - start, "mlx_peak_bytes": mx.get_peak_memory()}
            phase_peaks.append(mx.get_peak_memory())
            metrics["mlx_peak_bytes"] = max(phase_peaks)
            mx.reset_peak_memory()

        log("model_load", status="running")
        start = time.perf_counter()
        # Stored Q4 metadata is authoritative; this never converts a BF16 remote model.
        model = Flux1Fill(model_path=str(model_path), quantize=None, lora_paths=None)
        mx.eval(model.parameters())
        metrics["model_weights_loaded"] = True
        metrics["stored_quantization"] = model.bits
        if model.bits != 4 or model.model_config.x_embedder_input_dim() != 384:
            raise ValueError("Expected local Q4 FLUX Fill checkpoint and384-channel transformer")
        block_counts = {"joint": len(model.transformer.transformer_blocks), "single": len(model.transformer.single_transformer_blocks)}
        if block_counts != {"joint": 19, "single": 38}:
            raise ValueError(f"Unexpected native FLUX Fill block layout: {block_counts}")
        metrics["block_counts"] = block_counts
        metrics["component_parameter_bytes"] = {name: sum(value.nbytes for _, value in tree_flatten(getattr(model, name).parameters()))
                                                  for name in ("transformer", "vae", "t5_text_encoder", "clip_text_encoder")}
        phase_done("model_load", start)

        log("prompt_encode")
        start = time.perf_counter()
        prompt_embeds, pooled_prompt_embeds = PromptEncoder.encode_prompt(
            prompt=prompt, prompt_cache=model.prompt_cache, t5_tokenizer=model.tokenizers["t5"],
            clip_tokenizer=model.tokenizers["clip"], t5_text_encoder=model.t5_text_encoder,
            clip_text_encoder=model.clip_text_encoder)
        mx.eval(prompt_embeds, pooled_prompt_embeds)
        if not bool(mx.all(mx.isfinite(prompt_embeds))) or not bool(mx.all(mx.isfinite(pooled_prompt_embeds))):
            raise FloatingPointError("Non-finite prompt encoding")
        metrics["text_tokens_per_forward"] = int(prompt_embeds.shape[1])
        metrics["joint_query_tokens_per_forward"] = tokens + int(prompt_embeds.shape[1])
        del model.t5_text_encoder, model.clip_text_encoder
        mx.clear_cache()
        metrics["text_encoders_deleted_after_encoding"] = True
        phase_done("prompt_encode", start)

        config = Config(model_config=model.model_config, num_inference_steps=args.steps, width=width, height=height,
                        guidance=args.guidance, scheduler="linear", image_path=Path(input_info["canvas_path"]),
                        masked_image_path=Path(input_info["mask_path"]))
        log("source_conditioning")
        start = time.perf_counter()
        static = MaskUtil.create_masked_latents(vae=model.vae, width=width, height=height,
                                              img_path=config.image_path, mask_path=config.masked_image_path)
        mx.eval(static)
        if static.shape != (1, tokens, 320) or not bool(mx.all(mx.isfinite(static))):
            raise ValueError("Expected finite320-channel static Fill conditioning")
        metrics["static_conditioning_shape"] = list(static.shape)
        metrics["static_conditioning_finite"] = True
        metrics["static_conditioning_sha256_float32"] = hashlib.sha256(np.asarray(static.astype(mx.float32)).tobytes()).hexdigest()
        phase_done("source_conditioning", start)

        latents = FluxLatentCreator.create_noise(seed=args.seed, width=width, height=height)
        mx.eval(latents)
        initial = np.asarray(latents.astype(mx.float32))
        metrics["initial_noise_shape"] = list(initial.shape)
        metrics["initial_noise_dtype"] = str(latents.dtype)
        metrics["initial_noise_finite"] = bool(np.isfinite(initial).all())
        if not metrics["initial_noise_finite"]:
            raise FloatingPointError("Non-finite initial noise")
        metrics["initial_noise_sha256_float32"] = hashlib.sha256(initial.tobytes()).hexdigest()
        np.savez_compressed(destination / "initial_noise.npz", latents=initial)
        metrics["scheduler_sigmas"] = np.asarray(config.scheduler.sigmas.astype(mx.float32)).tolist()
        metrics.update({"scheduler": "native MFLUX LinearScheduler with FLUX resolution shift", "nfe": 0,
                        "target_token_forwards": 0, "joint_block_forwards": 0, "single_block_forwards": 0})
        log("denoise")
        start = time.perf_counter()
        for step in range(args.steps):
            step_start = time.perf_counter()
            latents = config.scheduler.scale_model_input(latents, step)
            hidden = mx.concatenate([latents, static], axis=-1)
            if hidden.shape != (1, tokens, 384):
                raise AssertionError("Native Fill input must have384 channels")
            prediction = model.transformer(t=step, config=config, hidden_states=hidden,
                                           prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds)
            latents = config.scheduler.step(noise=prediction, timestep=step, latents=latents)
            mx.eval(prediction, latents)
            if prediction.shape != (1, tokens, 64):
                raise AssertionError("Native Fill prediction must have64 channels")
            finite = bool(mx.all(mx.isfinite(prediction))) and bool(mx.all(mx.isfinite(latents)))
            if not finite:
                raise FloatingPointError(f"Non-finite prediction/latents at step {step + 1}")
            metrics["nfe"] += 1
            metrics["target_token_forwards"] += tokens
            metrics["joint_block_forwards"] += block_counts["joint"]
            metrics["single_block_forwards"] += block_counts["single"]
            metrics["step_metrics"].append({"step": step + 1, "seconds": time.perf_counter() - step_start,
                                             "sigma": metrics["scheduler_sigmas"][step], "finite": True,
                                             "target_tokens": tokens, "mlx_peak_bytes": mx.get_peak_memory()})
            log("denoise", completed_steps=step + 1, nfe=metrics["nfe"])
        phase_done("denoise", start)
        final = np.asarray(latents.astype(mx.float32))
        metrics["final_latents_shape"] = list(final.shape)
        metrics["final_latents_dtype"] = str(latents.dtype)
        metrics["final_latents_finite"] = bool(np.isfinite(final).all())
        metrics["final_latents_sha256_float32"] = hashlib.sha256(final.tobytes()).hexdigest()
        np.savez_compressed(destination / "final_latents.npz", latents=final)

        # A single-trial process does not need the denoiser again during decode.
        model.transformer = None
        prediction = None
        hidden = None
        mx.clear_cache()
        metrics["transformer_released_before_decode"] = True

        log("raw_decode")
        start = time.perf_counter()
        unpacked = FluxLatentCreator.unpack_latents(latents=latents, width=width, height=height)
        decoded = VAEUtil.decode(vae=model.vae, latent=unpacked, tiling_config=None)
        mx.eval(decoded)
        metrics["decoded_finite"] = bool(mx.all(mx.isfinite(decoded)))
        if not metrics["decoded_finite"]:
            raise FloatingPointError("Non-finite rawVAEdecode")
        raw = ImageUtil.to_pil(decoded).convert("RGB")
        if raw.size != EXPECTED_SIZE:
            raise AssertionError("Unexpected decode dimensions")
        raw_path = destination / "raw.png"
        raw.save(raw_path)
        phase_done("raw_decode", start)

        x0, y0, x1, y1 = EXPECTED_RECT
        raw_array, source_array = np.asarray(raw), np.asarray(source)
        error = np.abs(raw_array[y0:y1, x0:x1].astype(np.float32) - source_array.astype(np.float32))
        metrics["raw_source_mae_0_255"] = {"all": float(error.mean()), "top16": float(error[:16].mean()),
                                          "bottom16": float(error[-16:].mean()), "inner": float(error[16:-16].mean())}
        composite = raw.copy()
        composite.paste(source, (x0, y0))
        composite_path = destination / "composite.png"
        composite.save(composite_path)
        restored = np.asarray(Image.open(composite_path).convert("RGB"))
        metrics["source_exact_after_compositor"] = bool(np.array_equal(restored[y0:y1, x0:x1], source_array))
        outside = np.ones((height, width), dtype=bool)
        outside[y0:y1, x0:x1] = False
        metrics["generated_region_exact_raw_after_compositor"] = bool(np.array_equal(restored[outside], raw_array[outside]))
        if not metrics["source_exact_after_compositor"] or not metrics["generated_region_exact_raw_after_compositor"]:
            raise AssertionError("Exact source-only compositor failed")
        metrics.update({"raw_path": str(raw_path), "raw_sha256": file_sha256(raw_path),
                        "composite_path": str(composite_path), "composite_sha256": file_sha256(composite_path),
                        "visual_quality": "requires direct inspection; execution success is not quality acceptance",
                        "status": "success"})
        log("complete")
    except BaseException as exc:
        metrics.update({"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
        write()
        raise


if __name__ == "__main__":
    main()
