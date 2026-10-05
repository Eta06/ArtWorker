"""Isolated standard latest MaskFlow teacher; defaults to weight-free CPU gate.

GPU execution is deliberately explicit and is reserved for the parent's serial
queue. This is a new 2511 RGB teacher, not the current 2.1/controlnet pipeline.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MASKFLOW = HERE.parent
RUNTIME_PIN = "99fb94dd3eaa9dd1931cd3cd8eae1ae3e20f2ef3"
OFFICIAL_PIN = "3125f1ecd72f5a4e068c5a1954e9d6a8df38d443"
GATES = ["math_cpu_validation.json", "transformer_cpu_validation.json",
         "adapter_cpu_validation.json", "conditioning_cpu_validation.json", "loading_cpu_validation.json",
         "checkpoint_cpu_validation.json", "runner_cpu_validation.json"]
PORT_FILES = ["run_maskflow_trial.py", "maskflow_math.py", "maskflow_transformer.py",
              "maskflow_adapter.py", "maskflow_conditioning.py", "maskflow_loading.py", "maskflow_vae.py",
              "maskflow_checkpoint.py"]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, data):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def protocol(args):
    from maskflow_math import prepare_canvas, prepare_condition_images, sigma_grid
    source_path = ROOT / "experiments/evaluation/inputs" / args.track / "source_512.png"
    source = np.array(Image.open(source_path).convert("RGB"))
    canvas = prepare_canvas(source, fill=args.fill)
    prepared = prepare_condition_images(canvas["source_canvas"], canvas["mask"], dtype="bfloat16")
    assert prepared["mask_latents"].shape == (1, 2304, 64)
    assert prepared["height"] == 1152 and prepared["width"] == 512
    original_canvas = np.rint(canvas["source_canvas"][0].transpose(1, 2, 0) * 255).astype(np.uint8)
    assert np.array_equal(original_canvas[320:832], source)
    if args.fill == "black":
        shared = ROOT / "experiments/evaluation/inputs" / args.track / "canvas.png"
        assert np.array_equal(np.array(Image.open(shared).convert("RGB")), original_canvas)
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text().strip()
    else:
        p = ROOT / "experiments/qwen/controlnet_runs/2026-10-05/track3-localized40-outpaint/localized_reference_metrics.json"
        if args.track != "track3":
            raise ValueError("Other tracks require an explicit --prompt-file")
        prompt = json.loads(p.read_text())["prompt"]
    schedule = sigma_grid(50, 2304)
    mask = prepared["mask"][0, 0]
    inside_soft_rows = np.flatnonzero(np.any(mask[320:832] > 0, axis=1)) + 320
    return source, original_canvas, prepared, prompt, {
        "source_file": str(source_path), "source_file_sha256": sha(source_path),
        "source_rgb_sha256": hashlib.sha256(source.tobytes()).hexdigest(),
        "canvas_rgb_sha256": hashlib.sha256(original_canvas.tobytes()).hexdigest(),
        "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "seed": args.seed, "noise_backend": "Torch CPU BF16 fresh2511 grid; no 2.1 noise reuse",
        "width": 512, "height": 1152, "known_box_xyxy": [0, 320, 512, 832], "unknown_fill": args.fill,
        "steps": 50, "text_cfg": 4.0, "mask_cfg": 1.0, "rescale_cfg": True,
        "transformer_forwards": 100, "vae_channels": 16, "vae_downsample": 8,
        "packed_tokens_per_image": 2304, "joint_image_tokens": 6912,
        "condition_order": ["target", "source", "processed_mask_VAE"],
        "native_blend_policy": "Official processed soft mask; known square boundary can change",
        "hard_paste_policy": "Separate PNG preserving complete original512 square exactly",
        "inside_source_soft_rows": inside_soft_rows.tolist(),
        "maximum_inside_boundary_softness_pixels": 24,
        "last_nonzero_sigma": float(schedule["sigmas"][-2]), "mu": schedule["mu"],
        "sigmas": schedule["sigmas"].tolist(), "official_revision": OFFICIAL_PIN,
        "runtime_revision": RUNTIME_PIN, "conditioning_geometry": prepared["metadata"],
        "estimated_static_after_vl_release_gib": 14.815, "peak_memory_unmeasured": True,
    }


def verify_cpu_gates():
    receipts = {}
    for name in GATES:
        path = HERE / name
        report = json.loads(path.read_text())
        if report.get("status") != "passed":
            raise ValueError(f"CPU gate not passed: {name}")
        hashes = report.get("source_sha256") or report.get("code_sha256")
        if not isinstance(hashes, dict) or not hashes:
            raise ValueError(f"CPU gate has no tested-source receipts: {name}")
        for file, expected in hashes.items():
            candidate = Path(file)
            if not candidate.is_absolute():
                candidate = ROOT / file
                if not candidate.exists():
                    candidate = HERE / file
            if sha(candidate) != expected:
                raise ValueError(f"CPU gate stale: {name}, {file}")
        receipts[name] = sha(path)
    checkout = MASKFLOW / "runtime/mlx-gen"
    commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(checkout), "status", "--porcelain"], text=True).strip()
    if commit != RUNTIME_PIN or dirty:
        raise ValueError("Pinned runtime checkout differs or is dirty")
    return receipts


def verify_download_receipts():
    manifest_path = MASKFLOW / "download_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("payload_verified") is not True:
        raise ValueError("Complete pinned payload checksum verification is required")
    files = manifest["files"]
    if not files:
        raise ValueError("Download manifest has no files")
    for item in files:
        file = Path(item["local_path"])
        if not file.is_absolute():
            file = ROOT / file
        if not item.get("verified") or file.stat().st_size != item["size"]:
            raise ValueError(f"Payload receipt is incomplete: {file}")
        # Rehash on explicit launch; header-only checks cannot authorize weights.
        if sha(file) != item["sha256"]:
            raise ValueError(f"Pinned payload changed: {file}")
    return {"manifest_sha256": sha(manifest_path), "verified_files": len(files)}


def preflight(args):
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    from maskflow_math import pack_latents, unpack_latents, scheduler_step, hard_paste_original
    source, canvas, prepared, prompt, report = protocol(args)
    grid = np.arange(16 * 144 * 64, dtype=np.float32).reshape(1, 16, 1, 144, 64)
    assert np.array_equal(unpack_latents(pack_latents(grid), 1152, 512), grid)
    noise = np.ones((1, 2304, 64), dtype=np.float32)
    source_latents = np.full_like(noise, 3)
    mask = prepared["mask_latents"]
    result = scheduler_step(noise, np.zeros_like(noise), 0.1, 0.0, source=source_latents,
                            mask=mask, noise=noise, dtype="bfloat16", mask_dtype="bfloat16")
    assert np.array_equal(result[mask == 0], source_latents[mask == 0])
    assert np.array_equal(hard_paste_original(np.zeros_like(canvas), source)[320:832], source)
    # Target/source/mask RoPE windows are separate grids with target-time only
    # on the first2304 tokens. There is no reused source-prefix or ControlNet.
    from maskflow_transformer import MaskFlowTransformer
    index = MaskFlowTransformer._compute_modulate_index(1, [(1, 72, 32)] * 3, 6912)
    mx.eval(index)
    index_np = np.array(index)
    assert np.all(index_np[:, :2304] == 0) and np.all(index_np[:, 2304:] == 1)
    report["port_sha256"] = {str(HERE / f): sha(HERE / f) for f in PORT_FILES if (HERE / f).exists()}
    report["cpu_gates"] = verify_cpu_gates() if args.require_gates else "not required for geometry-only preflight"
    report["downloads"] = verify_download_receipts() if args.require_full_payload else "not loaded"
    report.update(status="passed", gpu_execution=False, model_weights_loaded=False)
    dump(args.output_dir / "preflight.json", report)
    print(json.dumps({"status": "passed", "preflight": str(args.output_dir / "preflight.json"),
                      "gpu_execution": False, "payload_required": args.require_full_payload}), flush=True)


def execute(args):
    # All authorization gates precede device selection and real model loading.
    cpu_receipts = verify_cpu_gates()
    download_receipts = verify_download_receipts()
    source, canvas, prepared, prompt, metadata = protocol(args)
    metadata["port_sha256"] = {str(HERE / f): sha(HERE / f) for f in PORT_FILES}
    from maskflow_checkpoint import validate_checkpoint, restore_checkpoint, save_checkpoint
    resume = validate_checkpoint(args.resume_from, metadata) if args.resume_from else None
    import mlx.core as mx
    mx.set_default_device(mx.gpu)
    mx.set_memory_limit(int(args.memory_limit_gib * 1024**3))
    from mflux.utils.runtime_memory import RuntimeMemory
    RuntimeMemory.apply_mlx_cache_limit(args.cache_limit_gib * 1024**3 / 1000**3)
    from maskflow_math import (pack_latents, unpack_latents, sigma_grid, combine_predictions,
                               apply_poisson_to_prediction, scheduler_step, pixel_blend, hard_paste_original)
    from maskflow_loading import load_saved_component
    from maskflow_transformer import MaskFlowTransformer
    from maskflow_vae import MaskFlowVAE
    from maskflow_conditioning import encode_cfg_branches, MaskFlowEncoder
    from maskflow_adapter import apply_maskflow_adapter, read_safetensors_header
    from mflux.models.common.tokenizer import TokenizerLoader
    from mflux.models.qwen.weights.qwen_weight_definition import QwenWeightDefinition
    from mflux.models.qwen.model.qwen_text_encoder.qwen_text_encoder import QwenTextEncoder
    from mflux.models.qwen.model.qwen_text_encoder.qwen_vision_transformer import VisionTransformer
    from mflux.models.qwen.model.qwen_text_encoder.qwen_vision_language_encoder import QwenVisionLanguageEncoder
    from mflux.models.qwen.tokenizer.qwen_vision_language_processor import QwenVisionLanguageProcessor
    from mflux.models.qwen.tokenizer.qwen_vision_language_tokenizer import QwenVisionLanguageTokenizer
    import torch
    start = time.perf_counter()
    metadata.update(cpu_gates=cpu_receipts, downloads=download_receipts,
                    memory_limit_gib=args.memory_limit_gib, cache_limit_gib=args.cache_limit_gib,
                    memory_stages=[], components=[], step_records=[], status="running",
                    resume_receipt=str(args.resume_from) if args.resume_from else None)
    def snapshot(phase):
        metadata["memory_stages"].append(RuntimeMemory.snapshot(phase, synchronize=True).to_metadata())
        dump(args.output_dir / "metrics.json", metadata)
        print(json.dumps({"phase": phase, "mlx_active_bytes": mx.get_active_memory(),
                          "mlx_peak_bytes": mx.get_peak_memory()}), flush=True)
    snapshot("before_model_loading")
    base = MASKFLOW / "models/qwen-image-edit-2511-4bit"
    def text_constructor():
        text = QwenTextEncoder()
        text.encoder.__class__ = MaskFlowEncoder
        text.encoder.visual = VisionTransformer()
        return text
    if resume is None:
        text, receipt = load_saved_component(base, "text_encoder", text_constructor)
        metadata["components"].append(receipt)
        tokenizers = TokenizerLoader.load_all(QwenWeightDefinition.get_tokenizers(), str(base))
        tokenizer = QwenVisionLanguageTokenizer(
            processor=QwenVisionLanguageProcessor(tokenizer=tokenizers["qwen"].tokenizer),
            max_length=1024, use_picture_prefix=True)
        vl = QwenVisionLanguageEncoder(encoder=text.encoder)
        snapshot("vl_loaded")
        conditioning_start = time.perf_counter()
        branches = encode_cfg_branches(vlm_conditions=prepared["vlm_conditions"], prompt=prompt,
                                      negative_prompt="", qwen_vl_tokenizer=tokenizer, qwen_vl_encoder=vl)
        metadata["vlm"] = branches.pop("metadata")
        metadata["conditioning_seconds"] = time.perf_counter() - conditioning_start
        for name in ("pm", "nm"):
            branches[name] = {k: mx.stop_gradient(v) for k, v in branches[name].items()}
            mx.eval(*branches[name].values())
        del vl, text, tokenizer, tokenizers
        gc.collect()
        mx.clear_cache()
        snapshot("vl_released_after_both_cfg_conditions")
    else:
        restored = restore_checkpoint(resume)
        branches = {name: {key: restored[f"{name}.{key}"] for key in ("prompt_embeds", "prompt_embeds_mask")}
                    for name in ("pm", "nm")}
        metadata["vlm"] = resume["vlm"]
        snapshot("cached_both_cfg_conditions_restored_without_vl_load")
    vae, receipt = load_saved_component(base, "vae", MaskFlowVAE)
    metadata["components"].append(receipt)
    if resume is None:
        packed_conditions = []
        metadata["vae_condition_seconds"] = {}
        for name in ("source", "mask"):
            condition_start = time.perf_counter()
            image = mx.array(prepared["dit_conditions"][name]).astype(mx.bfloat16)
            encoded = pack_latents(vae.encode(image), backend="mlx").astype(mx.bfloat16)
            mx.eval(encoded)
            assert encoded.shape == (1, 2304, 64)
            packed_conditions.append(mx.stop_gradient(encoded))
            metadata["vae_condition_seconds"][name] = time.perf_counter()-condition_start
        source_latents, mask_image_latents = packed_conditions
        mask_latents = mx.array(prepared["mask_latents"]).astype(mx.bfloat16)
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        noise_cpu = torch.randn((1, 16, 1, 144, 64), generator=generator, dtype=torch.bfloat16).float().numpy()
        np.save(args.output_dir / "initial_noise_2511.npy", noise_cpu)
        metadata["initial_noise_sha256"] = sha(args.output_dir / "initial_noise_2511.npy")
        noise = pack_latents(mx.array(noise_cpu).astype(mx.bfloat16), backend="mlx")
        xt = noise
        starting_step = 0
    else:
        source_latents = restored["source_latents"]
        mask_image_latents = restored["mask_image_latents"]
        mask_latents = restored["mask_latents"]
        noise, xt = restored["initial_noise"], restored["latents"]
        metadata["initial_noise_sha256"] = resume["initial_noise_sha256"]
        starting_step = resume["completed_steps"]
        del restored
    def checkpoint(step):
        arrays = {"latents": xt, "initial_noise": noise, "source_latents": source_latents,
                  "mask_image_latents": mask_image_latents, "mask_latents": mask_latents}
        arrays.update({f"{name}.{key}": value for name in ("pm", "nm") for key, value in branches[name].items()})
        metadata["checkpoint"] = save_checkpoint(args.output_dir, step, arrays, metadata)
    checkpoint(starting_step)
    snapshot("conditions_encoded_before_dit")
    sidecar = MASKFLOW / "models/original-sidecar/norm_out.linear.bias.safetensors"
    transformer, receipt = load_saved_component(base, "transformer",
        lambda: MaskFlowTransformer(zero_cond_t=True), sidecar=sidecar if sidecar.exists() else None)
    metadata["components"].append(receipt)
    adapter_path = MASKFLOW / "models/MaskFlow/latest.safetensors"
    adapter_weights = mx.load(str(adapter_path))
    metadata["adapter"] = apply_maskflow_adapter(transformer, adapter_weights, read_safetensors_header(adapter_path))
    mx.eval(transformer.parameters())
    del adapter_weights
    gc.collect()
    mx.clear_cache()
    snapshot("teacher_and_adapter_loaded_vl_absent")
    sigmas = sigma_grid(50, 2304, backend="mlx")
    config = SimpleNamespace(height=1152, width=512)
    conditions = mx.concatenate([source_latents, mask_image_latents], axis=1)
    cond_grid = [(1, 72, 32), (1, 72, 32)]
    loop_start = time.perf_counter()
    for i in range(starting_step, 50):
        step_start = time.perf_counter()
        forward_seconds = {}
        hidden = mx.concatenate([xt, conditions], axis=1)
        predictions = {}
        for name in ("pm", "nm"):
            forward_start = time.perf_counter()
            predictions[name] = transformer(t=float(sigmas["sigmas"][i].item()), config=config,
                hidden_states=hidden, encoder_hidden_states=branches[name]["prompt_embeds"],
                encoder_hidden_states_mask=branches[name]["prompt_embeds_mask"], cond_image_grid=cond_grid)[:, :2304]
            mx.eval(predictions[name])
            forward_seconds[name] = time.perf_counter() - forward_start
        velocity = combine_predictions(predictions, backend="mlx")
        velocity = apply_poisson_to_prediction(xt, velocity, sigmas["sigmas"][i], source_latents,
            mask_latents, noise, 1152, 512, backend="mlx")
        xt = scheduler_step(xt, velocity, sigmas["sigmas"][i], sigmas["sigmas"][i+1],
            sigmas["d_sigmas_dt"][i], source_latents, mask_latents, noise, backend="mlx")
        mx.eval(xt)
        metadata["step_records"].append({"step": i+1, "seconds": time.perf_counter()-step_start,
                                         "forward_seconds": forward_seconds,
                                         "mlx_active_bytes": mx.get_active_memory(), "mlx_peak_bytes": mx.get_peak_memory()})
        if i == 0 or (i + 1) % 5 == 0:
            metadata["completed_steps"] = i + 1
            metadata["loop_seconds_so_far"] = time.perf_counter() - loop_start
            checkpoint(i+1)
            snapshot(f"step_{i+1:02}")
            print(json.dumps({"step": i+1, "steps":50, "loop_seconds": metadata["loop_seconds_so_far"]}), flush=True)
        del hidden, predictions, velocity
    metadata["loop_seconds"] = time.perf_counter() - loop_start
    known = mask_latents == 0
    assert bool(mx.all(mx.where(known, xt == source_latents, True)).item())
    metadata["processed_mask_zero_latents_exact"] = True
    del transformer, conditions, branches
    gc.collect()
    mx.clear_cache()
    snapshot("dit_released_before_vae_decode")
    decoded = vae.decode(unpack_latents(xt, 1152, 512, backend="mlx"))[:, :, 0]
    # Official VAE postprocess performs these operations in model dtype.
    half_decoded = (decoded / 2.0).astype(mx.bfloat16)
    output = mx.clip((half_decoded + 0.5).astype(mx.bfloat16), 0.0, 1.0)
    mx.eval(output)
    raw_float = np.array(output.astype(mx.float32)).transpose(0, 2, 3, 1)[0]
    raw_pixels = np.rint(raw_float * 255).astype(np.uint8)
    Image.fromarray(raw_pixels).save(args.output_dir / "raw_decoded.png")
    native = pixel_blend(output, mx.array(prepared["raw_source"]).astype(mx.bfloat16),
                         mx.array(prepared["mask"]).astype(mx.bfloat16), backend="mlx")
    mx.eval(native)
    native_pixels = np.rint(np.clip(np.array(native.astype(mx.float32)).transpose(0, 2, 3, 1)[0], 0, 1) * 255).astype(np.uint8)
    Image.fromarray(native_pixels).save(args.output_dir / "native_softblend.png")
    composite = hard_paste_original(native_pixels, source)
    assert np.array_equal(composite[320:832], source)
    Image.fromarray(composite).save(args.output_dir / "composite.png")
    Image.fromarray(np.rint(prepared["mask"][0, 0] * 255).clip(0, 255).astype(np.uint8)).save(args.output_dir / "processed_mask.png")
    metadata.update(status="complete", total_seconds=time.perf_counter()-start,
                    source_exact_composite=True, native_original_square_exact=False,
                    output_sha256={name:sha(args.output_dir/name) for name in ["raw_decoded.png", "native_softblend.png", "composite.png", "processed_mask.png"]},
                    dependency_versions={n:importlib.metadata.version(n) for n in ["mlx-gen","mlx","torch","diffusers","transformers","numpy"]})
    snapshot("complete")
    print(json.dumps({"status":"complete", "output_dir":str(args.output_dir), "total_seconds":metadata["total_seconds"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--track", default="track3", choices=["track1","track2","track3"])
    parser.add_argument("--fill", default="black", choices=["black","gray"])
    parser.add_argument("--prompt-file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--execute-gpu", action="store_true")
    parser.add_argument("--resume-from", type=Path, help="Retained checkpoint.json; validates identical protocol/code before GPU")
    parser.add_argument("--require-gates", action="store_true")
    parser.add_argument("--require-full-payload", action="store_true")
    parser.add_argument("--memory-limit-gib", type=float, default=26.0)
    parser.add_argument("--cache-limit-gib", type=float, default=1.0)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    if args.execute_gpu:
        execute(args)
    else:
        preflight(args)


if __name__ == "__main__":
    main()
