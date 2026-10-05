"""Qwen2.1 v0.2.1/v0.3 and official seven-turbo/two-base-tail controls.

One base model load, fixed actual-square reference prefix, edge-context known
target latents, seed/absolute noise and early-frontier real target growth.
The nine-step arm costs three additional forwards and re-extracts the prefix
cache when the adapter's actual LoRALinear.scale becomes zero. No runtime files
are modified. CPU preflight/mock checks never load a checkpoint or use Metal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import time
import traceback
from pathlib import Path

from run_trial import MODEL, PROMPT, REVISION, ROOT, TURBO_REVISION
from run_geometry_trial import (active_ids, active_layout, cpu_geometry_smoke,
                                cpu_insertion_smoke, cpu_layout_smoke,
                                insertion_state, padded_source_rgba)


V03_REVISION = "009a44a895ef85f7e643c80fdca9543795248867"
V03_SHA256 = "f06c266e04438b5272bdfb99410421d52a65d7a37f6f42aabc3cb1faf0142644"
ADAPTERS = {
    "v021": ROOT / ".build/models/qwen/viggle/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors",
    "v03": ROOT / ".build/models/qwen/viggle/Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r256.safetensors",
}
NODES6 = (1.0, 0.9375, 0.875, 0.75, 0.5, 0.25)
NODES9 = (1.0, 0.9583, 0.9167, 0.875, 0.75, 0.5, 0.25, 1 / 6, 1 / 12)
ARMS = {
    "v021-six": {"adapter": "v021", "nodes": NODES6, "base_tail_step_zero_based": None},
    "v03-six": {"adapter": "v03", "nodes": NODES6, "base_tail_step_zero_based": None},
    "v03-nine-base-tail": {"adapter": "v03", "nodes": NODES9, "base_tail_step_zero_based": 7},
}


def file_sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def shifted_sigmas(np, nodes, final_tokens):
    nodes = np.asarray(nodes, dtype=np.float64)
    mu = 0.5 + (0.9 - 0.5) * (final_tokens - 256) / (8192 - 256)
    shifted = math.exp(mu) / (math.exp(mu) + (1.0 / nodes - 1.0))
    return np.concatenate([shifted, [0.0]]).astype(np.float32)


def lora_modules(transformer, layer_type):
    return [(name, module) for name, module in transformer.named_modules()
            if isinstance(module, layer_type)]


def set_adapter_scale(transformer, layer_type, scale):
    modules = lora_modules(transformer, layer_type)
    if not modules:
        raise AssertionError("No actual LoRALinear modules found")
    previous = {float(module.scale) for _, module in modules}
    for _, module in modules:
        module.scale = float(scale)
    if any(float(module.scale) != float(scale) for _, module in modules):
        raise AssertionError("Actual adapter scales did not change")
    return {"module_count": len(modules), "previous_scales": sorted(previous),
            "new_scale": float(scale)}


def unwrap_adapter(transformer, layer_type, loader):
    """Restore identical base objects; never bake updates or fuse two adapters."""
    modules = lora_modules(transformer, layer_type)
    if not modules:
        raise AssertionError("No adapter to unwrap")
    bases = {name: module.linear for name, module in modules}
    for name, base in bases.items():
        loader._replace_target_module(transformer, name, base)
    if lora_modules(transformer, layer_type):
        raise AssertionError("Adapter wrappers survived unwrapping")
    if any(loader._get_target_module(transformer, name) is not base
           for name, base in bases.items()):
        raise AssertionError("Unwrapping changed a base module object")
    return bases


def replace_adapter(model, adapter_path, layer_type, loader, initializer, mx):
    bases = unwrap_adapter(model.transformer, layer_type, loader)
    mx.clear_cache()
    initializer.apply_lora(model, [str(adapter_path)], [1.0], False)
    modules = lora_modules(model.transformer, layer_type)
    if set(name for name, _ in modules) != set(bases):
        raise AssertionError("Adapter versions target different modules")
    if any(module.linear is not bases[name] for name, module in modules):
        raise AssertionError("Adapter swap changed/fused a base linear module")
    if any(float(module.scale) != 1.0 for _, module in modules):
        raise AssertionError("New adapter scale is not one")
    mx.eval([value for _, module in modules for value in (module.lora_A, module.lora_B)])
    return {"base_linear_objects_retained": True, "adapter_fusion_absent": True,
            "module_count": len(modules)}


def cpu_tail_smoke(mx, np, nn, layer_type, loader):
    """Exercise actual LoRA scale/wrapper behavior and the required cache switch."""
    mx.set_default_device(mx.cpu)
    base = nn.Linear(2, 2, bias=False)
    base.weight = mx.array([[1.0, 2.0], [3.0, 4.0]])
    adapter = layer_type.from_linear(base, r=1, scale=1.0)
    adapter.lora_A = mx.array([[2.0], [5.0]])
    adapter.lora_B = mx.array([[7.0, 11.0]])
    toy = nn.Sequential(adapter)
    x = mx.array([[1.0, 3.0]])
    turbo = np.asarray(toy(x))
    scale_event = set_adapter_scale(toy, layer_type, 0.0)
    if not np.array_equal(np.asarray(toy(x)), np.asarray(base(x))):
        raise AssertionError("Scale zero does not produce exact base output")
    if np.array_equal(turbo, np.asarray(base(x))):
        raise AssertionError("Toy adapter did not affect output")
    bases = unwrap_adapter(toy, layer_type, loader)
    if list(bases.values()) != [base] or toy.layers[0] is not base:
        raise AssertionError("Adapter unwrapping changed the base object")
    # Exercise the real loader replacement branch, which would fuse if an old
    # wrapper had accidentally remained. No checkpoint tensor is loaded here.
    for update in (3.0, 13.0):
        applied = loader._apply_adapter_to_target(
            toy, "layers.0", {"lora_A": mx.full((2,1), update),
                              "lora_B": mx.full((1,2), 1.0)}, 1.0, role=None)
        modules = lora_modules(toy, layer_type)
        if not applied or len(modules) != 1 or modules[0][0] != "layers.0" or modules[0][1].linear is not base:
            raise AssertionError("CPU adapter replacement fused or changed base")
        if float(modules[0][1].scale) != 1.0:
            raise AssertionError("CPU replacement did not reset actual scale")
        unwrap_adapter(toy,layer_type,loader)
    # A cache generated under scale one is intentionally stale after scale zero.
    # The mock records the same extract/reuse branches as forward_reference.
    cache = []
    cache_events = []
    scale = 1.0
    for step in range(9):
        if step == 7:
            if not cache:
                raise AssertionError("Expected turbo cache before base tail")
            cache.clear()
            scale = 0.0
        cached = bool(cache)
        if not cached:
            cache.append(scale)
        if cache[0] != scale:
            raise AssertionError("A stale turbo prefix was reused by base tail")
        cache_events.append({"step": step + 1, "scale": scale,
                             "cache_mode": "reuse" if cached else "extract"})
    if [e["step"] for e in cache_events if e["cache_mode"] == "extract"] != [1, 8]:
        raise AssertionError("Wrong cache re-extraction steps")
    for name, spec in ARMS.items():
        sigmas = shifted_sigmas(np, spec["nodes"], 2304)
        if len(sigmas) != len(spec["nodes"]) + 1 or sigmas[0] != 1 or sigmas[-1] != 0 or not np.all(np.diff(sigmas) < 0):
            raise AssertionError(f"Invalid sigma schedule: {name}")
    return {"status": "passed", "device": "CPU", "checkpoint_weights_loaded": False,
            "actual_zero_scale_equals_base": True, "base_object_unwrap_exact": True,
            "actual_loader_two_swaps_without_fusion": True,
            "scale_event": scale_event, "cache_events": cache_events,
            "cache_extract_steps": [1, 8], "nodes_six": NODES6, "nodes_nine": NODES9}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=("track1", "track2", "track3"), default="track3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quantize", type=int, choices=(4, 8), default=4)
    parser.add_argument("--arms", nargs="+", choices=tuple(ARMS), default=list(ARMS))
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if len(set(args.arms)) != len(args.arms):
        parser.error("Arms must be unique")
    outdir = args.output or ROOT / f"experiments/qwen/turbo_tail_runs/2026-10-05/{args.track}"
    metrics_path = outdir / "turbo_tail_metrics.json"
    if not args.preflight and metrics_path.exists():
        raise FileExistsError(f"Refusing to overwrite prior outputs: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    begun = time.perf_counter()
    suite = {"status": "preflight", "track": args.track, "seed": args.seed,
             "arms": args.arms, "runs": {}, "phases": {},
             "runner_sha256": file_sha256(Path(__file__))}

    def write(name="turbo_tail_metrics.json"):
        suite["elapsed_seconds"] = time.perf_counter() - begun
        suite["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (outdir / name).write_text(json.dumps(suite, indent=2))

    def log(phase, **data):
        suite.update(phase=phase, **data)
        write()
        print(json.dumps({"phase": phase, "elapsed_seconds": round(suite["elapsed_seconds"], 3), **data}), flush=True)

    try:
        import mlx.core as mx
        import mlx.nn as nn
        import numpy as np
        from PIL import Image, ImageDraw
        from mlx.utils import tree_flatten
        from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
        from mflux.models.common.lora.mapping.lora_loader import LoRALoader
        from mflux.models.common.vae.vae_util import VAEUtil
        from mflux.models.common.vae.tiling_config import TilingConfig
        from mflux.models.qwen21.qwen21_initializer import Qwen21Initializer
        from mflux.models.qwen21.reference import QwenImage21Edit
        from mflux.models.qwen21.reference.latent_creator.qwen_image21_latent_creator import QwenImage21LatentCreator
        from mflux.models.qwen21.reference.model.qwen_image21_transformer.layout import QwenImage21Layout

        if args.preflight:
            mx.set_default_device(mx.cpu)
        shared = json.loads((ROOT / "experiments/evaluation/inputs/inputs.json").read_text())
        track = next(t for t in shared["tracks"] if t["id"] == args.track)
        width, height = shared["canvas_size"]
        rect = shared["source_rect_xyxy"]
        source = Image.open(track["source_512"]).convert("RGBA")
        if (width, height) != (512, 1152) or source.size != (512, 512) or rect != [0, 320, 512, 832]:
            raise ValueError("Suite expects the common centered 512×1152 canvas")
        prompt = PROMPT
        prompts = ROOT / "experiments/flux2/prompts.json"
        if prompts.is_file():
            prompt += " " + json.loads(prompts.read_text())[args.track]
        expected = ["transformer/config.json", "text_encoder/config.json", "vae/config.json",
                    "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
                    "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
                    *[f"text_encoder/model-0000{i}-of-00004.safetensors" for i in range(1, 5)],
                    "vae/diffusion_pytorch_model.safetensors", "processor/tokenizer.json"]
        missing = [str(MODEL / item) for item in expected if not (MODEL / item).is_file()]
        needed = set(ARMS[arm]["adapter"] for arm in args.arms)
        missing += [str(ADAPTERS[key]) for key in sorted(needed) if not ADAPTERS[key].is_file()]
        suite.update(prompt=prompt, mlx_version=mx.__version__, missing_files=missing,
                     raw_nodes_by_arm={arm: ARMS[arm]["nodes"] for arm in args.arms})
        if args.preflight:
            suite["layout_smoke"] = cpu_layout_smoke(mx, np, QwenImage21Layout)
            suite["insertion_smoke"] = cpu_insertion_smoke(mx, np)
            suite["geometry_smoke"] = cpu_geometry_smoke(mx, np)
            suite["tail_scale_cache_smoke"] = cpu_tail_smoke(mx, np, nn, LoRALinear, LoRALoader)
            suite.update(status="preflight_passed" if not missing else "preflight_tests_passed_weights_pending",
                         model_weights_loaded=False, device="CPU")
            write("preflight.json")
            print(json.dumps(suite, indent=2), flush=True)
            return
        if missing:
            raise FileNotFoundError(f"Incomplete checkpoint: {missing}")
        suite["adapter_sha256"] = {key: file_sha256(ADAPTERS[key]) for key in sorted(needed)}
        if "v03" in needed and suite["adapter_sha256"]["v03"] != V03_SHA256:
            raise AssertionError("v0.3 checkpoint hash differs from pinned download")
        mx.set_cache_limit(512 * 1024**2)
        mx.set_memory_limit(28 * 1024**3)
        suite["metal_device"] = mx.device_info()
        adapter_key = ARMS[args.arms[0]]["adapter"]
        log("loading_weights", status="running")
        start = time.perf_counter()
        model = QwenImage21Edit(model_path=str(MODEL), quantize=args.quantize,
                               lora_paths=[str(ADAPTERS[adapter_key])], lora_scales=[1.0], bake_lora=False)
        suite["phases"]["load_seconds"] = time.perf_counter() - start
        suite["loaded_component_parameter_bytes"] = {
            name: sum(v.nbytes for _, v in tree_flatten(getattr(model, name).parameters()))
            for name in ("transformer", "text_encoder", "vae")}
        suite["load_mlx_peak_bytes"] = mx.get_peak_memory()
        mx.reset_peak_memory()
        mx.set_memory_limit(23 * 1024**3)
        log("conditioning")
        start = time.perf_counter()
        prompt_embeds, slots = model._encode_prompt(prompt, [source])
        mx.eval(prompt_embeds)
        del model.text_encoder
        model.text_encoder = None
        mx.clear_cache()
        pixels = mx.array(np.asarray(source).astype(np.float32) / 127.5 - 1).transpose(2, 0, 1)[None]
        source_latents = QwenImage21LatentCreator.pack_latents(model.vae.encode(pixels)).astype(prompt_embeds.dtype)
        mx.eval(source_latents)
        if source_latents.shape != (1, 1024, 64):
            raise AssertionError("Unexpected reference square latent geometry")
        source_np = np.asarray(source_latents.astype(mx.float32)).copy()
        encode_start = time.perf_counter()
        padded_rgba = padded_source_rgba(np, np.asarray(source), width, height, rect)
        padded_pixels = mx.array(padded_rgba.astype(np.float32) / 127.5 - 1).transpose(2, 0, 1)[None]
        padded_latents = QwenImage21LatentCreator.pack_latents(model.vae.encode(padded_pixels)).astype(prompt_embeds.dtype)
        known_latents = padded_latents[:,20*32:52*32]
        mx.eval(known_latents)
        if known_latents.shape != source_latents.shape or not np.array_equal(source_np, np.asarray(source_latents.astype(mx.float32))):
            raise AssertionError("Target context modified reference prefix/crop geometry")
        suite["target_context_encode_seconds"] = time.perf_counter() - encode_start
        known_np = np.asarray(known_latents.astype(mx.float32)).copy()
        del pixels, padded_pixels, padded_latents
        source_shape = (1, 32, 32)
        full_layout = QwenImage21Layout.create(slots, [source_shape, (1, 72, 32)], model.transformer.axes)
        full_tokens = full_layout.target_tokens
        mx.random.seed(args.seed)
        noise = mx.random.normal((1, full_tokens, 64)).astype(prompt_embeds.dtype)
        ys = np.arange(72).clip(20, 51) - 20
        source_ids = (ys[:,None] * 32 + np.arange(32)[None,:]).reshape(-1).astype(np.int32)
        edge_guess = known_latents[:,mx.array(source_ids)]
        mx.eval(noise, edge_guess, *full_layout.rope)
        suite["phases"]["conditioning_seconds"] = time.perf_counter() - start
        suite["conditioning_mlx_peak_bytes"] = mx.get_peak_memory()
        suite.update(source_prefix_sha256=hashlib.sha256(source_np.tobytes()).hexdigest(),
                     known_target_sha256=hashlib.sha256(known_np.tobytes()).hexdigest(),
                     noise_sha256=hashlib.sha256(np.asarray(noise.astype(mx.float32)).tobytes()).hexdigest(),
                     hash_representation="float32 inference values", source_prefix_tokens=1024,
                     joint_prefix_tokens=full_layout.prefix_length, full_target_tokens=full_tokens)
        outputs = {}
        for order, arm in enumerate(args.arms, 1):
            spec = ARMS[arm]
            directory = outdir / arm
            directory.mkdir(parents=True, exist_ok=True)
            swap_start = time.perf_counter()
            if spec["adapter"] != adapter_key:
                swap = replace_adapter(model, ADAPTERS[spec["adapter"]], LoRALinear, LoRALoader, Qwen21Initializer, mx)
                adapter_key = spec["adapter"]
            else:
                swap = {"adapter_reloaded": False, "same_version_reused": True}
            scale_reset = set_adapter_scale(model.transformer, LoRALinear, 1.0)
            swap_seconds = time.perf_counter() - swap_start
            sigmas_np = shifted_sigmas(np, spec["nodes"], full_tokens)
            sigmas = mx.array(sigmas_np, dtype=mx.float32)
            steps = len(spec["nodes"])
            heights = [640, 896] + [1152] * (steps - 2)
            record = {
                "status": "running", "track": args.track, "seed": args.seed, "arm": arm,
                "model_repo": "Qwen/Qwen-Image-2.1", "model_revision": REVISION,
                "runtime": "mflux native MLX QwenImage21Edit; true active target growth",
                "runtime_git_revision": json.loads((ROOT / "experiments/qwen/runtime.json").read_text())["mflux_git_revision"],
                "adapter_repo": "Viggle/Qwen-Image-2.1-viggle-turbo", "adapter_baked": False,
                "adapter_revision": TURBO_REVISION if adapter_key == "v021" else V03_REVISION,
                "adapter_file": str(ADAPTERS[adapter_key]), "adapter_sha256": suite["adapter_sha256"][adapter_key],
                "adapter_rank": 256, "adapter_switch": swap, "adapter_scale_reset": scale_reset,
                "quantization": args.quantize, "prompt": prompt, "width": width, "height": height,
                "source_rect_xyxy": rect, "source_prefix": "actual 512-square source; fixed 1024 image tokens",
                "known_target_encoding_context": "edge-padded full canvas cropped rows20:52",
                "source_prefix_and_known_target_separate": True, "reference_prefix_unchanged": True,
                "source_prefix_sha256": suite["source_prefix_sha256"], "known_target_sha256": suite["known_target_sha256"],
                "noise_sha256": suite["noise_sha256"], "generation_order_within_process": order,
                "steps": steps, "raw_sigma_nodes": spec["nodes"], "sigmas": sigmas_np.tolist(),
                "resolution_shift_target_tokens": full_tokens, "shift_terminal": None,
                "base_tail_step_zero_based": spec["base_tail_step_zero_based"], "cfg": 1.0,
                "base_tail_scale_zero_still_computes_lora_matmuls": spec["base_tail_step_zero_based"] is not None,
                "active_heights_by_step": heights, "target_tokens_future_absent_from_forward": True,
                "initializer": "previous predicted-clean active frontier with same absolute noise; untrained heuristic",
                "known_region_method": "known target flow bridge hard restored before/after every step",
                "source_region_final_method": "exact original source paste; no feather/warp",
                "diagnostic_control": True, "shared_canvas_comparable": False, "transformer_calls": 0,
                "step_records": [], "cache_extract_steps": [], "adapter_scale_events": [],
                "phases": {"load_seconds_shared": suite["phases"]["load_seconds"],
                           "conditioning_seconds_shared": suite["phases"]["conditioning_seconds"],
                           "adapter_switch_seconds": swap_seconds},
                "timing_limit": "nine-step arm has three extra forwards and a second prefix extraction; not matched six-step compute",
                "load_mlx_peak_bytes": suite["load_mlx_peak_bytes"],
                "conditioning_mlx_peak_bytes": suite["conditioning_mlx_peak_bytes"],
            }
            suite["runs"][arm] = record
            cache = []
            latents = previous_ids = previous_clean = None
            core_seconds = 0.0
            mx.reset_peak_memory()
            log("denoising_start", current_arm=arm)
            for step in range(steps):
                step_start = time.perf_counter()
                cache_invalidated = False
                if step == spec["base_tail_step_zero_based"]:
                    if not cache:
                        raise AssertionError("Missing turbo cache before base-tail transition")
                    event = set_adapter_scale(model.transformer, LoRALinear, 0.0)
                    if event["previous_scales"] != [1.0]:
                        raise AssertionError("Base tail did not switch from full turbo scale one")
                    event.update(step=step+1, cache_layers_discarded=len(cache))
                    record["adapter_scale_events"].append(event)
                    cache.clear()
                    cache_invalidated = True
                effective_scale = 0.0 if spec["base_tail_step_zero_based"] is not None and step >= spec["base_tail_step_zero_based"] else 1.0
                if any(float(layer.scale) != effective_scale for _, layer in lora_modules(model.transformer, LoRALinear)):
                    raise AssertionError("Adapter scale differs from requested turbo/base phase")
                active_h = heights[step]
                ids, window = active_ids(np, width, height, active_h)
                index = mx.array(ids, dtype=mx.int32)
                added = len(ids) if previous_ids is None else len(ids)-len(previous_ids)
                if previous_ids is None or len(ids) != len(previous_ids):
                    latents = insertion_state(mx,np,ids,previous_ids,latents,previous_clean,
                                              edge_guess,noise,sigmas[step],True)
                layout = active_layout(mx,np,QwenImage21Layout,slots,source_shape,full_layout,
                                       ids,active_h,width,model.transformer.axes)
                known_mask = mx.array(((ids//32 >= 20)&(ids//32 < 52))[None,:,None])
                known_clean, active_noise = edge_guess[:,index], noise[:,index]
                known_now = ((1-sigmas[step])*known_clean.astype(mx.float32)+sigmas[step]*active_noise.astype(mx.float32)).astype(prompt_embeds.dtype)
                latents = mx.where(known_mask,known_now,latents)
                before = latents
                had_cache = bool(cache)
                if cache_invalidated and had_cache:
                    raise AssertionError("Turbo cache survived the base-tail transition")
                model_input = mx.concatenate([source_latents,latents],axis=1)
                if model_input.shape[1] != 1024+len(ids):
                    raise AssertionError("Future target tokens were forwarded")
                timestep = (sigmas[step:step+1]*1000).astype(latents.dtype)/1000
                prediction = model.transformer(model_input,prompt_embeds,timestep,layout,cache)
                if prediction.shape != latents.shape:
                    raise AssertionError("Prediction includes tokens outside active target")
                advanced = (latents.astype(mx.float32)+(sigmas[step+1]-sigmas[step])*prediction.astype(mx.float32)).astype(prompt_embeds.dtype)
                known_next = ((1-sigmas[step+1])*known_clean.astype(mx.float32)+sigmas[step+1]*active_noise.astype(mx.float32)).astype(prompt_embeds.dtype)
                latents = mx.where(known_mask,known_next,advanced)
                clean = (before.astype(mx.float32)-sigmas[step]*prediction.astype(mx.float32)).astype(prompt_embeds.dtype)
                clean = mx.where(known_mask,known_clean,clean)
                mx.eval(latents,clean)
                previous_clean = clean
                if len(cache) != len(model.transformer.transformer_blocks) or any(k.shape[2] != full_layout.prefix_length for k,_ in cache):
                    raise AssertionError("Prefix cache does not match layer/token count")
                seconds = time.perf_counter()-step_start
                core_seconds += seconds
                record["transformer_calls"] += 1
                if not had_cache:
                    record["cache_extract_steps"].append(step+1)
                sr = {"step":step+1,"sigma":float(sigmas_np[step]),"next_sigma":float(sigmas_np[step+1]),
                      "active_height":active_h,"active_window_xyxy":window,"target_tokens_forwarded":len(ids),
                      "future_target_tokens_absent":full_tokens-len(ids),"newly_activated_tokens":added,
                      "input_image_latent_tokens":int(model_input.shape[1]),"adapter_scale":effective_scale,
                      "cache_mode":"reuse" if had_cache else "extract", "cache_invalidated":cache_invalidated,
                      "attention_query_tokens":len(ids) if had_cache else full_layout.prefix_length+len(ids),
                      "attention_key_value_tokens":full_layout.prefix_length+len(ids),"core_step_seconds":seconds}
                record["step_records"].append(sr)
                (directory/"metrics.json").write_text(json.dumps(record,indent=2))
                print(json.dumps({"phase":"denoising","arm":arm,**sr}),flush=True)
                previous_ids = ids
            final_np = np.asarray(latents.astype(mx.float32)).copy()
            if not np.isfinite(final_np).all() or not np.array_equal(final_np[:,20*32:52*32],known_np):
                raise AssertionError("Nonfinite final latents or changed known source at sigma zero")
            expected_extract = [1,8] if spec["base_tail_step_zero_based"] is not None else [1]
            if record["cache_extract_steps"] != expected_extract:
                raise AssertionError("Wrong actual prefix cache extraction steps")
            latent_path = directory/"final_latents.npz"
            np.savez_compressed(latent_path,latents=final_np,final_canvas_ids=ids,
                                sigmas=sigmas_np,source_rect_xyxy=np.asarray(rect))
            record.update(status="denoised",final_latents_finite=True,
                          known_source_latents_exact_at_sigma_zero=True,final_latents_path=str(latent_path),
                          denoising_mlx_peak_bytes=mx.get_peak_memory(),target_token_forward_sum=sum(r["target_tokens_forwarded"] for r in record["step_records"]))
            record["phases"]["denoising_seconds"] = core_seconds
            record["denoising_timing_scope"] = "active insertion/layout, scale/cache changes, transformer, Euler and clean prediction; load/swap/capture/decode excluded"
            outputs[arm]=final_np
            (directory/"metrics.json").write_text(json.dumps(record,indent=2))
            del cache,latents,before,prediction,advanced,model_input,clean,previous_clean
            mx.clear_cache()
            log("denoising_complete",current_arm=arm,nfe=record["transformer_calls"],core_seconds=core_seconds)
        del model.transformer
        model.transformer = None
        mx.clear_cache()
        tiling = TilingConfig(vae_decode_tiles_per_dim=2,vae_decode_tile_size=512,vae_decode_overlap=4)
        source_rgb = source.convert("RGB")
        montage = Image.new("RGB",(512*len(args.arms),1152+40),(18,20,25))
        draw=ImageDraw.Draw(montage)
        for column,arm in enumerate(args.arms):
            record=suite["runs"][arm]
            directory=outdir/arm
            mx.reset_peak_memory()
            log("final_decode",current_arm=arm)
            start=time.perf_counter()
            unpacked=QwenImage21LatentCreator.unpack_latents(mx.array(outputs[arm]).astype(mx.float32),height,width)
            decoded=VAEUtil.decode(model.vae,unpacked,tiling)
            mx.eval(decoded)
            if decoded.ndim==5:
                decoded=decoded[:,:,0]
            rgb=np.asarray(decoded[0,:3].transpose(1,2,0))
            if rgb.shape != (height,width,3) or not np.isfinite(rgb).all():
                raise AssertionError("Invalid final decoded RGB")
            raw=Image.fromarray(np.clip((rgb+1)*127.5,0,255).round().astype(np.uint8),mode="RGB")
            raw.save(directory/"raw.png")
            error=np.abs(np.asarray(raw.crop(rect),dtype=np.int16)-np.asarray(source_rgb,dtype=np.int16))
            composite=raw.copy()
            composite.paste(source_rgb,(rect[0],rect[1]))
            composite.save(directory/"composite.png")
            if not np.array_equal(np.asarray(composite.crop(rect)),np.asarray(source_rgb)):
                raise AssertionError("Compositing changed original source pixels")
            record.update(status="success",source_pixels_exact=True,decoded_pixels_finite=True,
                          raw_source_region_mae_255=float(error.mean()),raw_source_top16_mae_255=float(error[:16].mean()),
                          raw_source_bottom16_mae_255=float(error[-16:].mean()),raw_source_inner_mae_255=float(error[16:-16].mean()),
                          decoding_mlx_peak_bytes=mx.get_peak_memory(),raw_path=str(directory/"raw.png"),composite_path=str(directory/"composite.png"),
                          output_sha256=file_sha256(directory/"composite.png"))
            record["phases"]["decode_seconds"]=time.perf_counter()-start
            (directory/"metrics.json").write_text(json.dumps(record,indent=2))
            montage.paste(composite,(column*512,40))
            draw.text((column*512+8,12),arm,fill=(245,245,245))
            mx.clear_cache()
        montage.save(outdir/"final-comparison.png")
        suite["comparison"]={"source_prefix_and_known_target_and_noise_matched":True,
                             "six_step_sigma_schedule_matched":True,"nine_step_compute_matched":False,
                             "base_tail_exact_cache_extract_steps":[1,8],
                             "no_early_region_finalization":True,"no_live_ui_stream_implementation":True,
                             "limit":"single seed/device; visual quality needs direct image inspection; nine-step extra compute and warmup prevent reliable speed ranking"}
        log("complete",status="success")
    except BaseException as exc:
        suite.update(status="failed",error_type=type(exc).__name__,error=str(exc))
        write()
        (outdir/"traceback.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
