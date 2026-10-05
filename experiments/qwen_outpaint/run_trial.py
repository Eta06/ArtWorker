"""Native Qwen 2.1 instruction outpaint; deliberately no known-latent clamp.

Use run_bounded_model.py for real runs. Public receipts exclude local artwork.
The raw model output and exact-source composite are separate evaluation arms.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import math
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / ".build/models/qwen/native-edit-q4"
ADAPTERS = ROOT / ".build/models/qwen/outpaint-lora"
TRIGGER = (
    "Outpaint the image: replace the solid gray areas with a seamless continuation "
    "of the scene, keeping the existing picture unchanged."
)


def header(path: Path) -> dict:
    with path.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        if length > 32 * 1024**2:
            raise ValueError("Unreasonably large safetensors header")
        return json.loads(stream.read(length))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=["track1", "track2", "track3"], default="track3")
    parser.add_argument("--adapter", choices=["v1", "v2", "none"], default="v1")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--turbo", choices=["none", "v03-r128"], default="none")
    parser.add_argument("--prompt-style", choices=["factual", "legacy"], default="factual")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--stream-encoder", action="store_true")
    parser.add_argument("--layerwise-encoder", action="store_true")
    parser.add_argument("--ondemand-encoder", action="store_true")
    parser.add_argument("--tiled-encode", action="store_true")
    parser.add_argument("--conditioning-only", action="store_true")
    parser.add_argument("--condition-cache", type=Path)
    parser.add_argument("--sequential-transformer", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    shared = json.loads((ROOT / "experiments/evaluation/inputs/inputs.json").read_text())
    track = next(t for t in shared["tracks"] if t["id"] == args.track)
    width, height = shared["canvas_size"]
    rect = shared["source_rect_xyxy"]
    if args.width is not None or args.height is not None:
        if args.width is None or args.height is None or args.width % 32 or args.height % 32 or args.height < args.width:
            parser.error("Supply a portrait width/height pair on the /32 grid")
        width, height = args.width, args.height
        if (height - width) % 2:
            parser.error("Centered source requires integer padding")
        top = (height - width) // 2
        rect = [0, top, width, top + width]
    adapter = ADAPTERS / ("qwen-image-2.1-outpaint-v2.safetensors" if args.adapter == "v2" else "qwen-image-2.1-outpaint.safetensors")
    description_file = ROOT / ("experiments/qwen_outpaint/captions.json" if args.prompt_style == "factual" else "experiments/flux2/prompts.json")
    descriptions = json.loads(description_file.read_text())
    prompt = TRIGGER + " Scene: " + descriptions[args.track]
    record = {
        "model_repo": "Qwen/Qwen-Image-2.1",
        "model_revision": "790c92633540aa0cb11d9abf19eb46d861714758",
        "checkpoint_export": "local tensor-wise affine q4, group64; complete visual encoder",
        "upstream_license": "qwen-research",
        "runtime": "mflux native reference QwenImage21Edit; no latent preservation",
        "runtime_revision": json.loads((ROOT / "experiments/qwen/runtime.json").read_text())["mflux_git_revision"],
        "adapter_repo": "ausboss/Qwen-Image-2.1-Outpaint-LoRA" if args.adapter != "none" else None,
        "adapter_revision": "449336db42ff074aee970ba0facc0ac0feb77863" if args.adapter != "none" else None,
        "adapter": args.adapter, "adapter_scale": args.scale, "bake_lora": False,
        "track": args.track, "width": width, "height": height, "source_rect": rect,
        "prompt": prompt, "steps": args.steps, "seed": args.seed, "guidance": 1.0,
        "prompt_style": args.prompt_style, "turbo": args.turbo, "jointly_trained_adapters": False,
        "reference_resize": False, "known_latent_clamp": False,
        "stream_encoder": args.stream_encoder,
        "layerwise_encoder": args.layerwise_encoder,
        "ondemand_encoder": args.ondemand_encoder,
        "tiled_encode": args.tiled_encode,
        "sequential_transformer": args.sequential_transformer,
        "scheduler": "MFLUX native flow Euler; not claimed identical to Comfy simple",
        "visual_quality_accepted": False, "status": "preflight", "phases": {},
    }
    begun = time.perf_counter()

    def log(phase: str, **values) -> None:
        record.update(values)
        record.update(phase=phase, elapsed_seconds=time.perf_counter() - begun)
        (args.output / "metrics.json").write_text(json.dumps(record, indent=2))
        print(json.dumps({"phase": phase, "seconds": round(record["elapsed_seconds"], 3), **values}), flush=True)

    try:
        from PIL import Image
        import numpy as np
        source = Image.open(track["source_square"]).convert("RGBA").resize((width, width), Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", (width, height), (128, 128, 128, 255))
        canvas.paste(source, (rect[0], rect[1]))
        canvas.save(args.output / "input.png")
        record["input_sha256"] = hashlib.sha256((args.output / "input.png").read_bytes()).hexdigest()
        expected = [f"{c}/config.json" for c in ("transformer", "text_encoder", "vae")]
        expected += [f"{c}/{i}.safetensors" for c, count in (("transformer", 2), ("text_encoder", 4), ("vae", 1)) for i in range(count)]
        expected += ["processor/tokenizer.json", "processor/preprocessor_config.json"]
        missing = [p for p in expected if not (MODEL / p).is_file()]
        if args.adapter != "none" and not adapter.is_file():
            missing.append(str(adapter))
        turbo = ROOT / ".build/models/qwen/viggle-v03/Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r128.safetensors"
        if args.turbo != "none" and (args.steps != 6 or not turbo.is_file()):
            raise ValueError("Turbo experiment needs the downloaded v0.3 rank128 adapter and exactly six steps")
        if missing:
            log("weights_incomplete", status="not_ready", missing=missing)
            if args.preflight:
                return
            raise FileNotFoundError(f"Missing checkpoint files: {missing}")
        export = json.loads((MODEL / "export_manifest.json").read_text())
        if export["status"] != "completed":
            raise ValueError("Native export is incomplete")
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from mflux.models.common.config.config import Config
        from mflux.models.common.vae.vae_util import VAEUtil
        from mflux.models.common.vae.tiling_config import TilingConfig
        from mflux.models.qwen21.reference import QwenImage21Edit
        from mflux.models.qwen21.reference.latent_creator.qwen_image21_latent_creator import QwenImage21LatentCreator as Latents
        from mflux.models.qwen21.reference.model.qwen_image21_transformer.layout import QwenImage21Layout
        from mflux.models.common.lora.mapping.lora_loader import LoRALoader
        from mflux.models.qwen21.weights.qwen21_lora_mapping import Qwen21LoRAMapping
        from mflux.models.common.config import ModelConfig
        from mflux.models.common.tokenizer import TokenizerLoader
        from mflux.models.qwen21.qwen21_initializer import Qwen21Initializer
        from mflux.models.qwen21.reference.weights.qwen_image21_weight_definition import QwenImage21WeightDefinition
        from mflux.models.qwen21.reference.model.qwen_image21_text_encoder.processor import QwenImage21Processor
        from mflux.models.qwen21.reference.model.qwen_image21_text_encoder.text_encoder import QwenImage21TextEncoder
        from mflux.models.qwen21.reference.model.qwen_image21_transformer.transformer import QwenImage21Transformer
        from mflux.models.qwen21.reference.model.qwen_image21_vae.vae import QwenImage21VAE
        from low_memory_loader import load_component
        import mlx.nn as nn
        for checked_adapter in ([adapter] if args.adapter != "none" else []) + ([turbo] if args.turbo != "none" else []):
            patterns = LoRALoader._build_pattern_mappings(Qwen21LoRAMapping.get_mapping())
            keys = set(header(checked_adapter)) - {"__metadata__"}
            unmatched = [key for key in keys if not any(LoRALoader._match_pattern(key, p.source_pattern) is not None for p in patterns)]
            record.setdefault("adapter_preflight", []).append({"file": checked_adapter.name, "tensor_count": len(keys), "unmapped_keys": unmatched})
            if unmatched:
                raise ValueError(f"Unsupported adapter tensors: {unmatched}")
        log("ready", status="preflight_passed", model_bytes=sum((MODEL / p).stat().st_size for p in expected), mlx_version=mx.__version__)
        if args.preflight:
            return
        mx.set_cache_limit(0)
        mx.set_memory_limit(10 * 1024**3)
        log("loading", status="running")
        start = time.perf_counter()
        model = QwenImage21Edit.__new__(QwenImage21Edit)
        nn.Module.__init__(model)
        Qwen21Initializer.init_config(model, ModelConfig.qwen_image_21())
        configs = {name: json.loads((MODEL / name / "config.json").read_text()) for name in ("text_encoder", "transformer", "vae")}
        record["component_bytes"] = {}
        if args.condition_cache:
            cache_manifest = json.loads(args.condition_cache.with_name("conditioning_manifest.json").read_text())
            for key in ("input_sha256", "prompt", "width", "height"):
                if cache_manifest[key] != record[key]:
                    raise ValueError(f"Conditioning cache mismatch: {key}")
            if hashlib.sha256(args.condition_cache.read_bytes()).hexdigest() != cache_manifest["cache_sha256"]:
                raise ValueError("Conditioning cache checksum mismatch")
            cached = mx.load(str(args.condition_cache))
            embeds, slots, condition = (cached[key] for key in ("embeds", "slots", "condition"))
            del cached
            mx.eval(embeds, slots, condition)
            record["conditioning_cache_sha256"] = cache_manifest["cache_sha256"]
            record["conditioning_tiled_encode"] = cache_manifest["tiled_encode"]
            record["phases"]["load_seconds"] = time.perf_counter() - start
            start = time.perf_counter()
        else:
            model.tokenizers = TokenizerLoader.load_all(QwenImage21WeightDefinition.get_tokenizers(), str(MODEL))
            model.processor = QwenImage21Processor(MODEL / "processor", model.tokenizers["qwen21"].tokenizer)
            encoder_directory = MODEL / ("text_encoder_layerwise" if args.layerwise_encoder else "text_encoder")
            if args.layerwise_encoder:
                manifest = json.loads((encoder_directory / "repack_manifest.json").read_text())
                if not args.stream_encoder or manifest["status"] != "completed":
                    raise ValueError("Layerwise encoder requires completed repack and --stream-encoder")
            if args.ondemand_encoder and not (args.layerwise_encoder and args.stream_encoder):
                raise ValueError("On-demand encoder requires streaming and layerwise layout")
            model.text_encoder = QwenImage21TextEncoder(configs["text_encoder"])
            if not args.ondemand_encoder:
                model.text_encoder = load_component(model.text_encoder, encoder_directory, materialize=not args.stream_encoder)
            if args.stream_encoder:
                from streaming_encoder import consume_encoder
                model.text_encoder = consume_encoder(model.text_encoder, encoder_directory if args.ondemand_encoder else None)
            record["phases"]["load_seconds"] = time.perf_counter() - start
            record["component_bytes"] = {"text_encoder": manifest["payload_bytes"] if args.ondemand_encoder else sum(v.nbytes for _, v in tree_flatten(model.text_encoder.parameters()))}
            log("conditioning", mlx_active_bytes=mx.get_active_memory())
            start = time.perf_counter()
            embeds, slots = model._encode_prompt(prompt, [canvas])
            mx.eval(embeds)
            model.text_encoder = None
            mx.clear_cache()
            log("reference_encoding", mlx_active_bytes=mx.get_active_memory())
            model.vae = load_component(QwenImage21VAE(configs["vae"]), MODEL / "vae")
            record["component_bytes"]["vae"] = sum(v.nbytes for _, v in tree_flatten(model.vae.parameters()))
            pixels = mx.array(np.asarray(canvas).astype(np.float32) / 127.5 - 1).transpose(2, 0, 1)[None]
            encoded = VAEUtil.encode(model.vae, pixels, TilingConfig(vae_encode_tiled=True, vae_encode_tile_size=256, vae_encode_tile_overlap=64)) if args.tiled_encode else model.vae.encode(pixels)
            condition = Latents.pack_latents(encoded).astype(embeds.dtype)
            mx.eval(condition)
            del pixels, encoded
            model.vae = None
            mx.clear_cache()
            if args.conditioning_only:
                cached = args.output / "conditioning.safetensors"
                mx.save_safetensors(str(cached), {"embeds": embeds, "slots": slots, "condition": condition})
                cache_manifest = {"input_sha256": record["input_sha256"], "prompt": prompt, "width": width, "height": height, "cache_sha256": hashlib.sha256(cached.read_bytes()).hexdigest(), "tiled_encode": args.tiled_encode}
                (args.output / "conditioning_manifest.json").write_text(json.dumps(cache_manifest, indent=2)+"\n")
                log("conditioning_completed", status="conditioning_only", visual_quality_accepted=False)
                return
        log("transformer_loading", mlx_active_bytes=mx.get_active_memory())
        model.transformer = load_component(QwenImage21Transformer(configs["transformer"]), MODEL / ("transformer_layerwise" if args.sequential_transformer else "transformer"), sequential=args.sequential_transformer)
        paths = ([str(adapter)] if args.adapter != "none" else []) + ([str(turbo)] if args.turbo != "none" else [])
        scales = ([args.scale] if args.adapter != "none" else []) + ([1.0] if args.turbo != "none" else [])
        Qwen21Initializer.apply_lora(model, paths or None, scales or None, bake_lora=False)
        mx.eval(model.transformer.parameters())
        record["component_bytes"]["transformer"] = sum(v.nbytes for _, v in tree_flatten(model.transformer.parameters()))
        layout = QwenImage21Layout.create(slots, [(1, height // 16, width // 16)] * 2, model.transformer.axes)
        mx.random.seed(args.seed)
        latents = Latents.pack_latents(mx.random.normal((1, 64, 1, height // 16, width // 16)).astype(embeds.dtype))
        config = Config(model_config=model.model_config, num_inference_steps=args.steps, width=width, height=height, guidance=1.0)
        sigmas = config.scheduler.sigmas
        if args.turbo != "none":
            nodes = np.array([1.0, 0.9375, 0.875, 0.75, 0.5, 0.25], dtype=np.float64)
            mu = 0.5 + (0.9 - 0.5) * ((height // 16) * (width // 16) - 256) / (8192 - 256)
            shifted = math.exp(mu) / (math.exp(mu) + (1.0 / nodes - 1.0))
            sigmas = mx.array(np.concatenate([shifted, [0.0]]), dtype=mx.float32)
            record["scheduler"] = "Viggle six custom nodes, native dynamic shift; no terminal stretch"
            record["turbo_revision"] = "009a44a895ef85f7e643c80fdca9543795248867"
        record["sigmas"] = sigmas.tolist()
        record["phases"]["conditioning_seconds"] = time.perf_counter() - start
        cache = []
        start = time.perf_counter()
        for step in range(args.steps):
            step_start = time.perf_counter()
            joined = mx.concatenate([condition, latents], axis=1)
            timestep = (sigmas[step:step + 1] * 1000).astype(latents.dtype) / 1000
            prediction = model.transformer(joined, embeds, timestep, layout, cache)
            sigma = sigmas
            latents = (latents.astype(mx.float32) + (sigma[step + 1] - sigma[step]) * prediction.astype(mx.float32)).astype(latents.dtype)
            mx.eval(latents)
            log("denoising", step=step + 1, step_seconds=time.perf_counter() - step_start, mlx_peak_bytes=mx.get_peak_memory())
        record["phases"]["denoise_seconds"] = time.perf_counter() - start
        del cache, condition, joined, prediction, embeds
        model.transformer = None
        mx.clear_cache()
        log("decoding")
        start = time.perf_counter()
        model.vae = load_component(QwenImage21VAE(configs["vae"]), MODEL / "vae")
        decoded = VAEUtil.decode(model.vae, Latents.unpack_latents(latents, height, width).astype(mx.float32),
            TilingConfig(vae_decode_tiles_per_dim=2, vae_decode_tile_size=512, vae_decode_overlap=4))
        mx.eval(decoded)
        if decoded.ndim == 5:
            decoded = decoded[:, :, 0]
        rgb = np.asarray(decoded[0, :3].transpose(1, 2, 0))
        if rgb.shape != (height, width, 3) or not np.isfinite(rgb).all():
            raise ValueError(f"Invalid decoded pixels: {rgb.shape}")
        raw = Image.fromarray(np.clip((rgb + 1) * 127.5, 0, 255).round().astype(np.uint8))
        raw.save(args.output / "raw.png")
        composite = raw.copy()
        composite.paste(source.convert("RGB"), (rect[0], rect[1]))
        composite.save(args.output / "composite.png")
        record["source_exact_in_composite"] = bool(np.array_equal(np.asarray(composite.crop(rect)), np.asarray(source.convert("RGB"))))
        record["raw_source_mae_255"] = float(np.abs(np.asarray(raw.crop(rect), dtype=np.float32) - np.asarray(source.convert("RGB"), dtype=np.float32)).mean())
        record["phases"]["decode_seconds"] = time.perf_counter() - start
        log("completed", status="completed", decoded_finite=True, mlx_peak_bytes=mx.get_peak_memory())
    except BaseException as exc:
        log("failed", status="failed", error_type=type(exc).__name__, error=str(exc)[:2000])
        raise


if __name__ == "__main__":
    main()
