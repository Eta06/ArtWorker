"""Local Mobile-O image-conditioned expansion trial, not a native mask model.

The public Python pipeline is CUDA-only. This adapter uses its released weights
and architecture on Apple MPS, initializes SANA from configuration to avoid
downloading duplicate base weights, and varies the latent height for a portrait
canvas. The unchanged source is composited back only after decoding.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / ".build/model-research/Mobile-O"
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", nargs="+", default=["track1", "track2", "track3"])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=1152)
    parser.add_argument("--latent-lock", action="store_true",
                        help="Experimental training-free known-latent restoration; not a native mask-trained model")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/mobile_o/results")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    metadata = {
        "model_id": "Amshaker/Mobile-O-0.5B",
        "model_revision": "92b5ef6a90a770c631153c3c32be0bbe049e2a8d",
        "repo_revision": "91c255080a0130846fe5b5cd54ff5af3c05ab2b9",
        "ios_model_revision": "56828ca7076d594dc38239627f0c4edb41a508b1",
        "method": "image-conditioned instruction expansion; no native mask or source-latent lock; exact source composited after decode",
        "runtime": "Python PyTorch MPS on macOS; not an iPhone benchmark",
        "seed": args.seed, "steps": args.steps,
        "working_size": [args.width, args.height], "results": [],
        "latent_lock": args.latent_lock,
    }
    if args.latent_lock:
        metadata["method"] = "image-conditioned instruction editing plus experimental training-free known-latent restoration at each next flow sigma; exact source composited after decode"
    try:
        import torch
        import numpy as np
        from PIL import Image
        from transformers import AutoTokenizer
        from diffusers import SanaTransformer2DModel, AutoencoderDC
        import mobileo.model.llava_arch as arch
        from mobileo.model.language_model.mobileo_inference import mobileoConfig, mobileoForInferenceLM
        from mobileo.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from mobileo.mm_utils import tokenizer_image_token, process_images
        from mobileo.conversation import conv_templates

        if not torch.backends.mps.is_available():
            raise RuntimeError("Apple MPS unavailable")
        device = torch.device("mps")
        dtype = torch.bfloat16
        torch.set_num_threads(4)
        base = ROOT / ".build/models/mobile-o/sana-config"
        checkpoint = ROOT / ".build/models/mobile-o/python"
        # All trained SANA and VAE tensors are in the unified Mobile-O checkpoint.
        arch.build_sana = lambda config, **kw: SanaTransformer2DModel.from_config(
            json.loads((base / "transformer/config.json").read_text()))
        arch.build_vae = lambda config, **kw: AutoencoderDC.from_config(
            json.loads((base / "vae/config.json").read_text()))
        config_dict = json.loads((checkpoint / "config.json").read_text())
        config_dict["diffusion_name_or_path"] = str(base)
        config = mobileoConfig.from_dict(config_dict)
        config._attn_implementation = "sdpa"
        # The released safetensors also contains 1,207 nested duplicate tensors
        # under a PEFT base_model.model prefix, with no unique or LoRA keys.
        # Load the complete top-level merged model, as the released loader does.
        mobileoForInferenceLM._keys_to_ignore_on_load_unexpected = [r"base_model\.model\..*"]
        metadata["checkpoint_note"] = "All 1207 nested base_model.model keys also exist at top level; load complete top-level model, ignore redundant PEFT-prefixed copy"
        load_start = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
        model, loading_info = mobileoForInferenceLM.from_pretrained(
            checkpoint, config=config, torch_dtype=dtype,
            low_cpu_mem_usage=True, output_loading_info=True,
        )
        if loading_info.get("missing_keys") or loading_info.get("unexpected_keys"):
            metadata["loading_info"] = loading_info
            raise RuntimeError("Released architecture/checkpoint mismatch; refusing random missing weights")
        model.eval().to(device)
        metadata["model_load_seconds"] = time.perf_counter() - load_start
        metadata["parameter_count"] = sum(p.numel() for p in model.parameters())
        metadata["package_versions"] = {
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "diffusers": __import__("diffusers").__version__,
        }
        print(f"READY load={metadata['model_load_seconds']:.2f}s parameters={metadata['parameter_count']}", flush=True)
        inputs = json.loads((ROOT / "experiments/evaluation/inputs/inputs.json").read_text())
        instruction = (
            "Extend this album artwork vertically to fill the blank top and bottom areas. "
            "Continue the existing scenery, colors, lighting and textures naturally. "
            "Keep the central square artwork unchanged in the same location. "
            "Do not add any new text, logos, borders or repeated people."
        )
        metadata["instruction"] = instruction
        for track in inputs["tracks"]:
            if track["id"] not in args.tracks:
                continue
            track_start = time.perf_counter()
            sampled_peak_driver = torch.mps.driver_allocated_memory()
            sampled_peak_current = torch.mps.current_allocated_memory()
            torch.manual_seed(args.seed)
            canvas = Image.open(track["canvas"]).convert("RGB")
            qs = DEFAULT_IMAGE_TOKEN + "\nPlease edit the provided image according to the following description: " + instruction
            conv = conv_templates["qwen_2"].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            input_ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")[None].to(device)
            image_tensor = process_images([canvas], model.get_vision_tower().image_processor, model.config)[0][None].to(device=device, dtype=dtype)
            with torch.inference_mode():
                prepared = model.prepare_inputs_labels_for_multimodal(
                    input_ids, None, None, None, None, und_images=image_tensor)
                encoded = model.model(
                    inputs_embeds=prepared[4], attention_mask=prepared[2],
                    output_hidden_states=True, return_dict=True)
                hidden = tuple(torch.cat([torch.zeros_like(layer), layer], dim=0)
                               for layer in encoded.hidden_states)
                condition = model.model.diffusion_connector(hidden).to(dtype)
                del encoded, hidden, image_tensor, prepared
                # SANA's convolutional/linear-attention model accepts rectangular latents;
                # the released sample_images helper hardcodes its 16x16 default.
                scale = 2 ** (len(model.model.vae.config.encoder_block_out_channels) - 1)
                if args.width % scale or args.height % scale:
                    raise ValueError(f"Working dimensions must be multiples of {scale}")
                generator = torch.Generator("cpu").manual_seed(args.seed)
                latents = torch.randn((1, model.model.dit.config.in_channels,
                                       args.height // scale, args.width // scale),
                                      generator=generator, dtype=torch.float32).to(device)
                scheduler = model.model.noise_scheduler
                scheduler.set_timesteps(args.steps, device=device)
                known_latents = preserve_mask = None
                if args.latent_lock:
                    pixels = torch.from_numpy(np.asarray(canvas).copy()).permute(2, 0, 1)[None].to(device=device, dtype=dtype)
                    pixels = pixels / 127.5 - 1
                    known_latents = model.model.vae.encode(pixels).latent.float() * model.model.vae.config.scaling_factor
                    mask = torch.from_numpy(np.asarray(Image.open(track["mask"]).convert("L")).copy())[None, None].to(device=device, dtype=torch.float32) / 255
                    preserve_mask = 1 - torch.nn.functional.interpolate(mask, size=known_latents.shape[-2:], mode="nearest")
                    initial_noise = latents.clone()
                    sigma = scheduler.sigmas[0].to(device)
                    latents = preserve_mask * ((1-sigma)*known_latents + sigma*initial_noise) + (1-preserve_mask)*latents
                    del pixels, mask
                sampled_peak_driver = max(sampled_peak_driver, torch.mps.driver_allocated_memory())
                sampled_peak_current = max(sampled_peak_current, torch.mps.current_allocated_memory())
                for i, timestep in enumerate(scheduler.timesteps):
                    latent_input = torch.cat([latents, latents])
                    if hasattr(scheduler, "scale_model_input"):
                        latent_input = scheduler.scale_model_input(latent_input, timestep)
                    noise = model.model.dit(
                        hidden_states=latent_input.to(dtype),
                        encoder_hidden_states=condition,
                        timestep=timestep[None].expand(2).to(device),
                        encoder_attention_mask=None).sample.float()
                    uncond, cond = noise.chunk(2)
                    noise = uncond + 1.5 * (cond - uncond)
                    latents = scheduler.step(noise, timestep, latents).prev_sample
                    if args.latent_lock:
                        next_sigma = scheduler.sigmas[i+1].to(device)
                        known_noised = (1-next_sigma)*known_latents + next_sigma*initial_noise
                        latents = preserve_mask*known_noised + (1-preserve_mask)*latents
                    torch.mps.synchronize()
                    sampled_peak_driver = max(sampled_peak_driver, torch.mps.driver_allocated_memory())
                    sampled_peak_current = max(sampled_peak_current, torch.mps.current_allocated_memory())
                    print(f"{track['id']} step {i+1}/{args.steps} elapsed={time.perf_counter()-track_start:.2f}s", flush=True)
                raw = model.decode_latents(latents.to(dtype))[0]
                sampled_peak_driver = max(sampled_peak_driver, torch.mps.driver_allocated_memory())
                sampled_peak_current = max(sampled_peak_current, torch.mps.current_allocated_memory())
            torch.mps.synchronize()
            if raw.size != canvas.size:
                raise RuntimeError(f"Decode size {raw.size} did not match shared canvas {canvas.size}")
            raw_path = args.output / f"{track['id']}-seed{args.seed}-raw.png"
            composite_path = args.output / f"{track['id']}-seed{args.seed}-composited.png"
            raw.save(raw_path)
            composite = raw.copy()
            source = Image.open(track["source_512"]).convert("RGB")
            x, y, _, _ = inputs["source_rect_xyxy"]
            composite.paste(source, (x, y))
            composite.save(composite_path)
            source_error = np.abs(np.asarray(composite)[y:y+source.height, x:x+source.width].astype(int) - np.asarray(source).astype(int))
            record = {
                "track": track["id"], "seconds": time.perf_counter()-track_start,
                "raw": str(raw_path), "composited": str(composite_path),
                "source_max_pixel_error": int(source_error.max()),
                "mps_current_allocated_bytes": torch.mps.current_allocated_memory(),
                "mps_driver_allocated_bytes": torch.mps.driver_allocated_memory(),
                "sampled_peak_mps_driver_bytes": sampled_peak_driver,
                "sampled_peak_mps_current_bytes": sampled_peak_current,
                "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            }
            metadata["results"].append(record)
            print(json.dumps(record), flush=True)
            (args.output / "trial.json").write_text(json.dumps(metadata, indent=2))
            del latents, latent_input, noise, uncond, cond, condition
            if args.latent_lock:
                del known_latents, preserve_mask, initial_noise, known_noised
            torch.mps.empty_cache()
    except Exception as exc:
        metadata["error"] = repr(exc)
        metadata["traceback"] = traceback.format_exc()
        raise
    finally:
        metadata["total_seconds"] = time.perf_counter() - started
        metadata["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (args.output / "trial.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
