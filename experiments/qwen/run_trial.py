"""Local Qwen 2.1 masked outpaint comparison on the shared album-cover canvas.

Uses MFLUX's native MLX reference-edit model. Mask conditioning here is the
standard Euler flow known-latent blend, not LanPaint. White regenerates, black
preserves. Original resized source pixels are composited back after decoding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / ".build/models/qwen/base"
REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"
TURBO_REVISION = "bb26a0f38e5fe6c124aaccc9187a87eed5d9ed13"
PROMPT = (
    "Extend this album cover vertically above and below its existing central square "
    "to fill the entire portrait canvas. Seamlessly continue its existing colors, "
    "lighting, textures, scenery, and artistic style into the empty top and bottom "
    "regions. Keep the central artwork unchanged. Make one coherent expanded "
    "composition. Do not add new people, duplicated subjects, extra objects, text, "
    "letters, logos, frames, or borders."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=["track1", "track2", "track3"], default="track1")
    parser.add_argument("--turbo", action="store_true")
    parser.add_argument("--adapter-rank", type=int, choices=[128, 256], default=128)
    parser.add_argument("--known-padding", choices=["gray", "edge"], default="gray", help="Controlled VAE known-region border-context experiment; reference conditioning stays unchanged")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quantize", type=int, choices=[4, 8], default=4)
    parser.add_argument("--preflight", action="store_true", help="Check files and imports without loading any model weights")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    label = f"viggle-v021-r{args.adapter_rank}-6step" if args.turbo else f"base-{args.steps}step"
    if args.known_padding != "gray":
        label += f"-known{args.known_padding}"
    adapter = ROOT / f".build/models/qwen/viggle/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r{args.adapter_rank}.safetensors"
    outdir = args.output or ROOT / f"experiments/evaluation/results/qwen21-{label}-q{args.quantize}/{args.track}"
    outdir.mkdir(parents=True, exist_ok=True)
    record = {
        "model_repo": "Qwen/Qwen-Image-2.1",
        "model_revision": REVISION,
        "runtime": "mflux native MLX QwenImage21Edit",
        "runtime_git_revision": json.loads((ROOT / "experiments/qwen/runtime.json").read_text())["mflux_git_revision"],
        "quantization": args.quantize,
        "adapter_repo": "Viggle/Qwen-Image-2.1-viggle-turbo" if args.turbo else None,
        "adapter_revision": TURBO_REVISION if args.turbo else None,
        "adapter_file": str(adapter) if args.turbo else None,
        "adapter_rank": args.adapter_rank if args.turbo else None,
        "adapter_baked": False,
        "track": args.track,
        "seed": args.seed,
        "steps": 6 if args.turbo else args.steps,
        "prompt": PROMPT,
        "mask_method": "Flow Euler per-step known-latent preservation; not LanPaint",
        "known_region_encoding_padding": args.known_padding,
        "status": "preflight",
        "phases": {},
    }
    begun = time.perf_counter()

    def log(phase: str, **data: object) -> None:
        record.update(data)
        record["phase"] = phase
        record["elapsed_seconds"] = time.perf_counter() - begun
        record["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (outdir / "metrics.json").write_text(json.dumps(record, indent=2))
        print(json.dumps({"phase": phase, "elapsed_seconds": round(record["elapsed_seconds"], 3), **data}), flush=True)

    try:
        import mlx.core as mx
        from mlx.utils import tree_flatten
        import numpy as np
        from PIL import Image
        from mflux.models.common.config.config import Config
        from mflux.models.common.vae.vae_util import VAEUtil
        from mflux.models.common.vae.tiling_config import TilingConfig
        from mflux.models.qwen21.reference import QwenImage21Edit
        from mflux.models.qwen21.reference.latent_creator.qwen_image21_latent_creator import QwenImage21LatentCreator
        from mflux.models.qwen21.reference.model.qwen_image21_transformer.layout import QwenImage21Layout
        record["metal_device"] = mx.device_info()

        shared = json.loads((ROOT / "experiments/evaluation/inputs/inputs.json").read_text())
        track = next(t for t in shared["tracks"] if t["id"] == args.track)
        descriptions_path = ROOT / "experiments/flux2/prompts.json"
        prompt = PROMPT
        if descriptions_path.is_file():
            prompt += " " + json.loads(descriptions_path.read_text())[args.track]
        record["prompt"] = prompt
        width, height = shared["canvas_size"]
        rect = shared["source_rect_xyxy"]
        canvas = Image.open(track["canvas"]).convert("RGBA")
        mask_image = Image.open(track["mask"]).convert("L")
        source = Image.open(track["source_512"]).convert("RGB")
        expected = [
            "transformer/config.json", "text_encoder/config.json", "vae/config.json",
            "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
            "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
            *[f"text_encoder/model-0000{i}-of-00004.safetensors" for i in range(1, 5)],
            "vae/diffusion_pytorch_model.safetensors", "processor/tokenizer.json",
        ]
        missing = [p for p in expected if not (MODEL / p).is_file()]
        if args.turbo and not adapter.is_file():
            missing.append(str(adapter))
        record.update({"width": width, "height": height, "source_rect_xyxy": rect, "mlx_version": mx.__version__, "missing_files": missing})
        if missing:
            log("weights_incomplete", status="not_ready")
            if args.preflight:
                return
            raise FileNotFoundError(f"Incomplete checkpoint: {missing}")
        if args.preflight:
            log("ready", status="preflight_passed", model_bytes=sum((MODEL / p).stat().st_size for p in expected))
            return

        # Bound Metal caches; dense components are converted and released in sequence
        # by MFLUX's loader. This limits caching, not a guarantee about total process RSS.
        mx.set_cache_limit(512 * 1024**2)
        mx.set_memory_limit(28 * 1024**3)
        log("loading_weights", status="running")
        start = time.perf_counter()
        model = QwenImage21Edit(
            model_path=str(MODEL), quantize=args.quantize,
            lora_paths=[str(adapter)] if args.turbo else None,
            lora_scales=[1.0] if args.turbo else None,
            bake_lora=False,
        )
        record["phases"]["load_seconds"] = time.perf_counter() - start
        record["loaded_component_parameter_bytes"] = {
            name: sum(value.nbytes for _, value in tree_flatten(getattr(model, name).parameters()))
            for name in ("transformer", "text_encoder", "vae")
        }
        record["load_mlx_peak_bytes"] = mx.get_peak_memory()
        mx.reset_peak_memory()
        mx.set_memory_limit(23 * 1024**3)
        log("conditioning")
        start = time.perf_counter()
        # Keep the common portrait reference at its actual canvas geometry.
        prompt_embeds, slots = model._encode_prompt(prompt, [canvas])
        mx.eval(prompt_embeds)
        del model.text_encoder
        model.text_encoder = None
        mx.clear_cache()
        pixels = mx.array(np.asarray(canvas).astype(np.float32) / 127.5 - 1).transpose(2, 0, 1)[None]
        original_latent = QwenImage21LatentCreator.pack_latents(model.vae.encode(pixels)).astype(prompt_embeds.dtype)
        mx.eval(original_latent)
        known_latent = original_latent
        if args.known_padding == "edge":
            # Change only known-region latent encoding context, leaving both the
            # vision reference and its VAE prefix conditioned on the gray canvas.
            x1, y1, x2, y2 = rect
            known_pixels = np.pad(np.asarray(source.convert("RGBA")), ((y1, height - y2), (x1, width - x2), (0, 0)), mode="edge")
            edge_pixels = mx.array(known_pixels.astype(np.float32) / 127.5 - 1).transpose(2, 0, 1)[None]
            known_latent = QwenImage21LatentCreator.pack_latents(model.vae.encode(edge_pixels)).astype(prompt_embeds.dtype)
            mx.eval(known_latent)
            del edge_pixels
        mask = np.asarray(mask_image.resize((width // 16, height // 16), Image.Resampling.BOX), dtype=np.float32) / 255.0
        regenerate = mx.array(mask.reshape(1, -1, 1)).astype(prompt_embeds.dtype)
        shapes = [(1, height // 16, width // 16), (1, height // 16, width // 16)]
        layout = QwenImage21Layout.create(slots, shapes, model.transformer.axes)
        mx.random.seed(args.seed)
        noise_init = mx.random.normal(original_latent.shape).astype(prompt_embeds.dtype)
        latents = noise_init
        steps = 6 if args.turbo else args.steps
        config = Config(model_config=model.model_config, num_inference_steps=steps, width=width, height=height, guidance=1.0)
        if args.turbo:
            # Exact v0.2.1 six-node schedule, dynamically shifted without terminal stretch.
            nodes = np.array([1.0, 0.9375, 0.875, 0.75, 0.5, 0.25], dtype=np.float64)
            tokens = (height // 16) * (width // 16)
            mu = 0.5 + (0.9 - 0.5) * (tokens - 256) / (8192 - 256)
            shifted = math.exp(mu) / (math.exp(mu) + (1.0 / nodes - 1.0))
            sigmas = mx.array(np.concatenate([shifted, [0.0]]), dtype=mx.float32)
        else:
            sigmas = config.scheduler.sigmas
        record["sigmas"] = sigmas.tolist()
        record["vae_decode"] = {"tile_size_px": 512, "overlap_px": 64}
        record["phases"]["conditioning_seconds"] = time.perf_counter() - start
        record["conditioning_mlx_peak_bytes"] = mx.get_peak_memory()
        mx.reset_peak_memory()
        cache: list = []
        log("denoising_start", mlx_active_bytes=mx.get_active_memory())
        start = time.perf_counter()
        for step in range(steps):
            step_start = time.perf_counter()
            model_input = mx.concatenate([original_latent, latents], axis=1)
            timestep = (sigmas[step:step + 1] * 1000).astype(latents.dtype) / 1000
            prediction = model.transformer(model_input, prompt_embeds, timestep, layout, cache)
            advanced = (latents.astype(mx.float32) + (sigmas[step + 1] - sigmas[step]) * prediction.astype(mx.float32)).astype(latents.dtype)
            known_next = ((1.0 - sigmas[step + 1]) * known_latent + sigmas[step + 1] * noise_init).astype(latents.dtype)
            latents = (regenerate * advanced + (1.0 - regenerate) * known_next).astype(prompt_embeds.dtype)
            mx.eval(latents)
            log("denoising", completed_steps=step + 1, step_seconds=round(time.perf_counter() - step_start, 3), mlx_active_bytes=mx.get_active_memory(), mlx_peak_bytes=mx.get_peak_memory())
        record["phases"]["denoising_seconds"] = time.perf_counter() - start
        record["denoising_mlx_peak_bytes"] = mx.get_peak_memory()
        del cache, model_input, prediction, advanced
        log("decoding")
        start = time.perf_counter()
        del model.transformer
        model.transformer = None
        mx.clear_cache()
        mx.reset_peak_memory()
        unpacked = QwenImage21LatentCreator.unpack_latents(latents, height, width).astype(mx.float32)
        decoded = VAEUtil.decode(model.vae, unpacked, TilingConfig(vae_decode_tiles_per_dim=2, vae_decode_tile_size=512, vae_decode_overlap=4))
        mx.eval(decoded)
        if decoded.ndim == 5:
            decoded = decoded[:, :, 0]
        pixels_out = np.asarray(decoded[0, :3].transpose(1, 2, 0))
        record["decoded_pixels_finite"] = bool(np.isfinite(pixels_out).all())
        if not record["decoded_pixels_finite"]:
            raise FloatingPointError("Decoded image contains NaN or infinite pixels")
        if pixels_out.shape != (height, width, 3):
            raise ValueError(f"Unexpected decoded shape: {pixels_out.shape}")
        raw = Image.fromarray(np.clip((pixels_out + 1) * 127.5, 0, 255).round().astype(np.uint8), mode="RGB")
        raw.save(outdir / "raw.png")
        raw_center_error = np.abs(np.asarray(raw.crop(rect), dtype=np.int16) - np.asarray(source, dtype=np.int16))
        record["raw_source_region_mae_255"] = float(raw_center_error.mean())
        record["raw_source_region_max_error_255"] = int(raw_center_error.max())
        record["raw_source_top16_mae_255"] = float(raw_center_error[:16].mean())
        record["raw_source_bottom16_mae_255"] = float(raw_center_error[-16:].mean())
        record["raw_source_inner_mae_255"] = float(raw_center_error[16:-16].mean())
        result = raw.copy()
        result.paste(source, (rect[0], rect[1]))
        result.save(outdir / "composite.png")
        exact = np.array_equal(np.asarray(result.crop(rect)), np.asarray(source))
        record["phases"]["decode_seconds"] = time.perf_counter() - start
        record["decoding_mlx_peak_bytes"] = mx.get_peak_memory()
        overall_peak = max(record[k] for k in ("conditioning_mlx_peak_bytes", "denoising_mlx_peak_bytes", "decoding_mlx_peak_bytes"))
        log("complete", status="success", source_pixels_exact=exact, raw_path=str(outdir / "raw.png"), composite_path=str(outdir / "composite.png"), mlx_peak_bytes=overall_peak, output_sha256=hashlib.sha256((outdir / "composite.png").read_bytes()).hexdigest())
        if not exact:
            raise AssertionError("Source pixels changed during final compositing")
    except BaseException as exc:
        log("failed", status="failed", error_type=type(exc).__name__, error=str(exc))
        (outdir / "traceback.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
