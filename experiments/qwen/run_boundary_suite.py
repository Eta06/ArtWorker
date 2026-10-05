"""Matched Qwen2.1 known-target border-context/source-enforcement probe.

One model load, identical actual-square reference prefix, seed/noise, prompt,
sigmas and EARLY-FRONTIER target growth. Control uses hard source enforcement
with edge-context known target encoding. Two controls soften only the first
two packed source rows at each border: 0.25, 0.75, then a fully hard interior.
They use isolated-square or edge-context known target encoding respectively.
The source/text reference prefix stays fixed. No outside pixels are clamped.
The fourth edge-dynamic arm instead re-encodes the known target source from
previous clean generated context before steps 5 and 6, with exact original
source pixels/opaque alpha. Its two decode/encode pairs are separately timed.
Future target tokens are absent from forwards. Frontier insertion is heuristic.
Inner target regions continue refining at the same global sigma.

No changes to the installed runtime or the baseline runner are required.
--preflight and --layout-smoke use CPU arrays and never load model weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import time
import traceback
from dataclasses import replace
from pathlib import Path

from run_trial import MODEL, PROMPT, REVISION, ROOT, TURBO_REVISION


ARMS = {
    "edge-context": {"heights": (640, 896, 1152, 1152, 1152, 1152), "initializer": "predicted-clean-active-frontier", "known_context": "edge-padded-full-canvas-center-crop", "enforcement": "hard"},
    "square-soft": {"heights": (640, 896, 1152, 1152, 1152, 1152), "initializer": "predicted-clean-active-frontier", "known_context": "isolated-square", "enforcement": "soft-two-packed-rows"},
    "edge-soft": {"heights": (640, 896, 1152, 1152, 1152, 1152), "initializer": "predicted-clean-active-frontier", "known_context": "edge-padded-full-canvas-center-crop", "enforcement": "soft-two-packed-rows"},
    "edge-dynamic": {"heights": (640, 896, 1152, 1152, 1152, 1152), "initializer": "predicted-clean-active-frontier", "known_context": "edge-padded-initially-then-generated-context-before-steps5and6", "enforcement": "hard", "dynamic_before_steps": (5,6)},
}
PREVIEW_STEPS = (2, 4, 6)
ADAPTER = ROOT / ".build/models/qwen/viggle/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors"


def active_ids(np, width: int, height: int, active_height: int):
    """Raster IDs in the final canvas, never local/recentered IDs."""
    if width % 32 or height % 32 or active_height % 32:
        raise ValueError("All dimensions must be multiples of 32 pixels")
    if active_height > height or (height - active_height) % 32:
        raise ValueError("Active window must be centered on the final latent grid")
    y0 = (height - active_height) // 2
    w = width // 16
    ids = np.arange((y0 // 16) * w, ((y0 + active_height) // 16) * w, dtype=np.int32)
    return ids, (0, y0, width, y0 + active_height)


def active_layout(mx, np, layout_type, slots, source_shape, full_layout, ids, active_height, width, axes):
    layout = layout_type.create(slots, [source_shape, (1, active_height // 16, width // 16)], axes)
    if layout.prefix_length != full_layout.prefix_length or layout.target_tokens != len(ids):
        raise AssertionError("Growing layout changed the fixed reference prefix")
    p = full_layout.prefix_length
    gather = mx.array(p + ids, dtype=mx.int32)
    rope = tuple(mx.concatenate([r[:p], r[gather]], axis=0) for r in full_layout.rope)
    return replace(layout, rope=rope)


def six_sigmas(np, final_tokens: int):
    nodes = np.array([1.0, 0.9375, 0.875, 0.75, 0.5, 0.25], dtype=np.float64)
    mu = 0.5 + (0.9 - 0.5) * (final_tokens - 256) / (8192 - 256)
    shifted = math.exp(mu) / (math.exp(mu) + (1.0 / nodes - 1.0))
    return np.concatenate([shifted, [0.0]]).astype(np.float32)


def cpu_layout_smoke(mx, np, layout_type):
    """Exercise real layout construction and target ID preservation on CPU."""
    mx.set_default_device(mx.cpu)
    slots = mx.array([False] * 7 + [True] * 256 + [False] * 10)
    axes = (16, 56, 56)
    source_shape = (1, 32, 32)
    full = layout_type.create(slots, [source_shape, (1, 72, 32)], axes)
    old_ids = None
    checks = []
    for h in (640, 896, 1152):
        ids, rect = active_ids(np, 512, 1152, h)
        layout = active_layout(mx, np, layout_type, slots, source_shape, full, ids, h, 512, axes)
        p = full.prefix_length
        for current, reference in zip(layout.rope, full.rope):
            if not np.array_equal(np.asarray(current[:p]), np.asarray(reference[:p])):
                raise AssertionError("Prefix rotary positions changed")
            if not np.array_equal(np.asarray(current[p:]), np.asarray(reference)[p + ids]):
                raise AssertionError("Target rotary positions do not match final raster IDs")
        if old_ids is not None:
            positions = np.searchsorted(ids, old_ids)
            if not np.array_equal(ids[positions], old_ids):
                raise AssertionError("Growth lost existing target IDs")
            # Indexed insertion must retain every existing state exactly.
            old = mx.array(old_ids.astype(np.float32)[None, :, None])
            grown = mx.zeros((1, len(ids), 1), dtype=mx.float32)
            grown[:, mx.array(positions, dtype=mx.int32)] = old
            if not np.array_equal(np.asarray(grown)[0, positions, 0], old_ids):
                raise AssertionError("Existing states changed during target insertion")
        checks.append({"height": h, "target_tokens": len(ids), "prefix_length": p, "window_xyxy": rect})
        old_ids = ids
    return {"device": "CPU", "model_weights_loaded": False, "checks": checks, "status": "passed"}



def insertion_state(mx, np, ids, previous_ids, latents, previous_clean,
                    edge_guess, noise, sigma, use_frontier):
    """Insert only new absolute raster tokens, retaining old state bit-for-bit.

    Source-edge and active-frontier guesses use the same final-position noise.
    Frontier extension repeats the most recent clean estimate's top/bottom rows;
    it is not an extra denoiser call, a learned predictor, or a fresh noise draw.
    """
    index = mx.array(ids, dtype=mx.int32)
    guess = edge_guess[:, index]
    if previous_ids is not None and use_frontier:
        if previous_clean is None:
            raise AssertionError("Active-frontier insertion has no prior clean estimate")
        first_row, last_row = int(previous_ids[0] // 32), int(previous_ids[-1] // 32)
        clamped_ids = np.clip(ids // 32, first_row, last_row) * 32 + ids % 32
        positions = np.searchsorted(previous_ids, clamped_ids).astype(np.int32)
        if not np.array_equal(previous_ids[positions], clamped_ids):
            raise AssertionError("Frontier guess changed absolute columns or lost rows")
        guess = previous_clean[:, mx.array(positions)]
    candidate = ((1.0 - sigma) * guess.astype(mx.float32)
                 + sigma * noise[:, index].astype(mx.float32)).astype(noise.dtype)
    if previous_ids is not None:
        positions = np.searchsorted(ids, previous_ids).astype(np.int32)
        if not np.array_equal(ids[positions], previous_ids):
            raise AssertionError("Growth changed or dropped existing absolute raster IDs")
        candidate[:, mx.array(positions)] = latents
    return candidate


def cpu_insertion_smoke(mx, np):
    """Test retention, absolute noise and boundary mapping with distinguishable values."""
    mx.set_default_device(mx.cpu)
    old_ids, _ = active_ids(np, 512, 1152, 640)
    ids, _ = active_ids(np, 512, 1152, 896)
    edge = mx.array(np.arange(2304, dtype=np.float32)[None, :, None])
    noise = mx.array((10000 + np.arange(2304, dtype=np.float32))[None, :, None])
    old_state = mx.array((-3000 - old_ids.astype(np.float32))[None, :, None])
    old_clean = mx.array((40000 + old_ids.astype(np.float32))[None, :, None])
    old_positions = np.searchsorted(ids, old_ids)
    new_mask = ~np.isin(ids, old_ids)
    checks = []
    for frontier in (False, True):
        state = insertion_state(mx, np, ids, old_ids, old_state, old_clean, edge, noise, 0.5, frontier)
        actual = np.asarray(state)[0, :, 0]
        if not np.array_equal(actual[old_positions], np.asarray(old_state)[0, :, 0]):
            raise AssertionError("Insertion modified existing latents")
        if frontier:
            mapped = np.clip(ids // 32, old_ids[0] // 32, old_ids[-1] // 32) * 32 + ids % 32
            guess = 40000 + mapped.astype(np.float32)
        else:
            guess = ids.astype(np.float32)
        expected = 0.5 * guess + 0.5 * (10000 + ids.astype(np.float32))
        if not np.array_equal(actual[new_mask], expected[new_mask]):
            raise AssertionError("Inserted tokens used the wrong clean frontier/noise positions")
        checks.append({"initializer": "frontier" if frontier else "source-edge",
                       "previous_tokens_exact": True, "new_absolute_noise_matched": True,
                       "new_frontier_mapping_exact": True, "inserted_tokens": int(new_mask.sum())})
    return {"status": "passed", "device": "CPU", "model_weights_loaded": False, "checks": checks}



def padded_source_rgba(np, source_rgba, width, height, rect):
    """Same RGB content with opaque alpha; no source pixel is modified."""
    x1, y1, x2, y2 = rect
    if source_rgba.shape != (y2-y1, x2-x1, 4):
        raise ValueError("Expected the original opaque RGBA source square")
    if not np.all(source_rgba[..., 3] == 255):
        raise ValueError("Reference VAE encoding requires opaque source alpha here")
    padded = np.pad(source_rgba, ((y1, height-y2), (x1, width-x2), (0, 0)), mode="edge")
    if padded.shape != (height, width, 4) or not np.array_equal(padded[y1:y2, x1:x2], source_rgba):
        raise AssertionError("Edge padding changed source content or geometry")
    return padded


def cpu_geometry_smoke(mx, np):
    """Ensure distinct target context never alters fixed prefix/noise geometry."""
    mx.set_default_device(mx.cpu)
    source = np.empty((512, 512, 4), dtype=np.uint8)
    source[..., 0] = np.arange(512, dtype=np.uint16)[:, None] % 256
    source[..., 1] = np.arange(512, dtype=np.uint16)[None, :] % 256
    source[..., 2] = 73
    source[..., 3] = 255
    before = source.copy()
    padded = padded_source_rgba(np, source, 512, 1152, [0,320,512,832])
    if not np.array_equal(padded[:320], np.broadcast_to(source[:1], (320,512,4))):
        raise AssertionError("Upper padding did not repeat original first RGB row")
    if not np.array_equal(padded[832:], np.broadcast_to(source[-1:], (320,512,4))):
        raise AssertionError("Lower padding did not repeat original last RGB row")
    if not np.array_equal(before, source):
        raise AssertionError("Padding mutated source")
    prefix = mx.array(np.arange(1024, dtype=np.float32)[None,:,None])
    frozen_prefix = np.asarray(prefix).copy()
    full_context = mx.array((10000+np.arange(2304,dtype=np.float32))[None,:,None])
    known_context = full_context[:,20*32:52*32]
    if known_context.shape != prefix.shape or not np.array_equal(np.asarray(known_context)[0,:,0], 10000+np.arange(640,1664,dtype=np.float32)):
        raise AssertionError("Known target crop shifted latent rows")
    ids, _ = active_ids(np,512,1152,640)
    known_mask = (ids//32>=20)&(ids//32<52)
    ys = np.arange(72).clip(20,51)-20
    source_ids=(ys[:,None]*32+np.arange(32)[None,:]).reshape(-1).astype(np.int32)
    target_guess=known_context[:,mx.array(source_ids)]
    target=target_guess[:,mx.array(ids)]
    model_input=mx.concatenate([prefix,target],axis=1)
    if not np.array_equal(np.asarray(model_input[:,:1024]),frozen_prefix):
        raise AssertionError("Changed target-known context altered reference prefix")
    if not np.array_equal(np.asarray(target)[0,known_mask,0],np.asarray(known_context)[0,:,0]):
        raise AssertionError("Known mask/absolute indexing shifted source crop")
    return {"status":"passed","device":"CPU","model_weights_loaded":False,
            "source_rgba_unchanged":True,"opaque_alpha_constant":True,
            "center_crop_rows_exact":[20,52],"changed_known_target_keeps_prefix_exact":True}


def source_enforcement_weights(np, ids, enforcement):
    """Weights on absolute packed rows: source is [20,52), 16 pixels/row.

    Each packed row contains a 2x2 block of VAE cells; "two rows" therefore
    means 32 original source pixels, not two individual VAE cells. Outside
    source rows are always zero. No target exterior context is enforced.
    """
    rows = ids // 32
    weights = ((rows >= 20) & (rows < 52)).astype(np.float32)
    if enforcement == "soft-two-packed-rows":
        weights[(rows == 20) | (rows == 51)] = 0.25
        weights[(rows == 21) | (rows == 50)] = 0.75
    elif enforcement != "hard":
        raise ValueError(f"Unknown source enforcement: {enforcement}")
    return weights


def enforce_source(mx, state, known, weights):
    """Blend sigma-matched trajectories while retaining hard/outside exactly.

    The two soft rows use w*known + (1-w)*state in float32, cast once to the
    state dtype. Explicit 0/1 branches avoid changing unenforced values and
    retain the hard known source exactly. The prefix is never an argument.
    """
    blended = (weights.astype(mx.float32) * known.astype(mx.float32)
               + (1.0 - weights.astype(mx.float32)) * state.astype(mx.float32)).astype(state.dtype)
    return mx.where(weights == 1, known, mx.where(weights == 0, state, blended))


def cpu_enforcement_smoke(mx, np):
    """Check exact absolute-row masks, blend formula and immutable prefix."""
    mx.set_default_device(mx.cpu)
    full_ids = np.arange(2304, dtype=np.int32)
    checks = []
    for enforcement in ("hard", "soft-two-packed-rows"):
        weights = source_enforcement_weights(np, full_ids, enforcement)
        expected_rows = np.zeros(72, dtype=np.float32)
        expected_rows[20:52] = 1
        if enforcement == "soft-two-packed-rows":
            expected_rows[[20, 51]] = 0.25
            expected_rows[[21, 50]] = 0.75
        if not np.array_equal(weights.reshape(72,32), np.repeat(expected_rows[:,None],32,axis=1)):
            raise AssertionError("Source enforcement mask moved or clamped exterior rows")
        for h in (640,896,1152):
            ids,_ = active_ids(np,512,1152,h)
            if not np.array_equal(source_enforcement_weights(np,ids,enforcement),weights[ids]):
                raise AssertionError("Enforcement mask changed with active-window indexing")
        prefix = mx.array(np.arange(1024,dtype=np.float32)[None,:,None])
        prefix_before = np.asarray(prefix).copy()
        state = mx.array((-12000+full_ids.astype(np.float32))[None,:,None])
        known = mx.array((40000+full_ids.astype(np.float32))[None,:,None])
        w = mx.array(weights[None,:,None])
        actual = enforce_source(mx,state,known,w)
        expected = weights*(40000+full_ids) + (1-weights)*(-12000+full_ids)
        if not np.array_equal(np.asarray(actual)[0,:,0],expected):
            raise AssertionError("Soft enforcement does not follow w*known+(1-w)*state")
        for sigma in (1.0,0.5,0.0):
            clean = known
            noise = state
            known_sigma = ((1-sigma)*clean+sigma*noise).astype(mx.float32)
            output = enforce_source(mx,state,known_sigma,w)
            output_np = np.asarray(output)[0,:,0]
            if not np.array_equal(output_np[weights==0],np.asarray(state)[0,weights==0,0]):
                raise AssertionError("Enforcement changed unmasked exterior")
            if not np.array_equal(output_np[weights==1],np.asarray(known_sigma)[0,weights==1,0]):
                raise AssertionError("Enforcement changed hard source trajectory")
            model_input = mx.concatenate([prefix,output],axis=1)
            if not np.array_equal(np.asarray(model_input[:,:1024]),prefix_before):
                raise AssertionError("Source enforcement altered reference prefix")
        checks.append({"enforcement":enforcement,"full_absolute_row_weights":expected_rows.tolist(),
                       "soft_formula_exact":True,"prefix_preserved":True,
                       "outside_source_untouched":True,"hard_interior_exact":True})
    return {"status":"passed","device":"CPU","model_weights_loaded":False,"checks":checks}


def generated_context_rgba(np, decoded_rgb, source_rgba, width, height, rect):
    """Build normalized generated context without quantizing or moving source.

    Only decoded exterior RGB is clipped into encoder range. Full-canvas alpha
    is opaque; source RGBA is then overwritten from its exact uint8 original.
    This returns an encoding input; it never receives or edits target latents.
    """
    if decoded_rgb.shape != (height,width,3) or not np.isfinite(decoded_rgb).all():
        raise ValueError("Dynamic context must be a finite full-canvas RGB decode")
    x1,y1,x2,y2 = rect
    if source_rgba.shape != (y2-y1,x2-x1,4) or not np.all(source_rgba[...,3]==255):
        raise ValueError("Dynamic context requires the exact original opaque source RGBA")
    context = np.ones((height,width,4),dtype=np.float32)
    context[...,:3] = np.clip(decoded_rgb,-1,1)
    normalized_source = source_rgba.astype(np.float32)/127.5-1
    context[y1:y2,x1:x2] = normalized_source
    if not np.array_equal(context[y1:y2,x1:x2],normalized_source) or not np.all(context[...,3]==1):
        raise AssertionError("Dynamic context changed the source or its opaque alpha")
    return context


def cpu_dynamic_context_smoke(mx,np):
    """Check source placement, normalized range and opaque alpha with CPU arrays."""
    mx.set_default_device(mx.cpu)
    source=np.empty((512,512,4),dtype=np.uint8)
    source[...,:3]=[17,121,243]
    source[...,3]=255
    before=source.copy()
    rgb=np.empty((1152,512,3),dtype=np.float32)
    rgb[...,0]=-1.25
    rgb[...,1]=0.375
    rgb[...,2]=1.25
    context=generated_context_rgba(np,rgb,source,512,1152,[0,320,512,832])
    outside=np.ones((1152,512),dtype=np.bool_)
    outside[320:832]=False
    if not np.array_equal(context[outside,:3],np.broadcast_to(np.array([-1,0.375,1],dtype=np.float32),(outside.sum(),3))):
        raise AssertionError("Dynamic context RGB exterior range/clipping incorrect")
    if not np.array_equal(source,before) or not np.all(context[...,3]==1):
        raise AssertionError("Dynamic context mutated original source or alpha")
    if not np.array_equal(context[320:832],before.astype(np.float32)/127.5-1):
        raise AssertionError("Dynamic context source pixels moved or were quantized")
    return {"status":"passed","device":"CPU","model_weights_loaded":False,
            "original_source_unchanged":True,"exact_normalized_source_placement":True,
            "encoder_rgb_range":[-1,1],"whole_canvas_alpha_opaque":True,
            "uint8_quantization_before_encode":False,"dynamic_before_steps":[5,6],
            "source_or_unknown_target_latents_not_an_argument":True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=("track1", "track2", "track3"), default="track3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quantize", type=int, choices=(4, 8), default=4)
    parser.add_argument("--arms", nargs="+", choices=tuple(ARMS), default=list(ARMS), help="Execution order; default includes all four controlled arms")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--layout-smoke", action="store_true")
    parser.add_argument("--output", type=Path, help="Optional suite directory containing one directory per arm")
    args = parser.parse_args()
    suite_dir = args.output or ROOT / f"experiments/qwen/boundary_runs/2026-10-05/{args.track}"
    if len(set(args.arms)) != len(args.arms):
        parser.error("Arms must be unique")
    if not (args.preflight or args.layout_smoke) and (suite_dir / "boundary_metrics.json").is_file():
        raise FileExistsError(f"Refusing to overwrite prior ablation outputs: {suite_dir}")
    suite_dir.mkdir(parents=True, exist_ok=True)
    mode_dirs = {mode: suite_dir / mode for mode in args.arms}
    suite = {"status": "preflight", "track": args.track, "seed": args.seed, "runs": {}, "phases": {}}
    begun = time.perf_counter()

    def write_suite(name="boundary_metrics.json"):
        suite["elapsed_seconds"] = time.perf_counter() - begun
        suite["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (suite_dir / name).write_text(json.dumps(suite, indent=2))

    def log(phase, **data):
        suite.update(data)
        suite["phase"] = phase
        write_suite()
        print(json.dumps({"phase": phase, "elapsed_seconds": round(suite["elapsed_seconds"], 3), **data}), flush=True)

    try:
        import mlx.core as mx
        import numpy as np
        from PIL import Image
        from mlx.utils import tree_flatten
        from mflux.models.common.vae.vae_util import VAEUtil
        from mflux.models.common.vae.tiling_config import TilingConfig
        from mflux.models.qwen21.reference import QwenImage21Edit
        from mflux.models.qwen21.reference.latent_creator.qwen_image21_latent_creator import QwenImage21LatentCreator
        from mflux.models.qwen21.reference.model.qwen_image21_transformer.layout import QwenImage21Layout

        if args.preflight or args.layout_smoke:
            # Explicitly avoid GPU work, including the layout/RoPE smoke arrays.
            mx.set_default_device(mx.cpu)
        shared = json.loads((ROOT / "experiments/evaluation/inputs/inputs.json").read_text())
        track = next(item for item in shared["tracks"] if item["id"] == args.track)
        width, height = shared["canvas_size"]
        rect = shared["source_rect_xyxy"]
        source = Image.open(track["source_512"]).convert("RGBA")
        if (width, height) != (512, 1152) or source.size != (512, 512) or rect != [0, 320, 512, 832]:
            raise ValueError("This controlled prototype expects the shared centered 512×1152 canvas")
        prompt = PROMPT
        descriptions_path = ROOT / "experiments/flux2/prompts.json"
        if descriptions_path.is_file():
            prompt += " " + json.loads(descriptions_path.read_text())[args.track]
        expected = [
            "transformer/config.json", "text_encoder/config.json", "vae/config.json",
            "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
            "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
            *[f"text_encoder/model-0000{i}-of-00004.safetensors" for i in range(1, 5)],
            "vae/diffusion_pytorch_model.safetensors", "processor/tokenizer.json",
        ]
        missing = [str(MODEL / item) for item in expected if not (MODEL / item).is_file()]
        if not ADAPTER.is_file():
            missing.append(str(ADAPTER))
        if missing:
            raise FileNotFoundError(f"Incomplete checkpoint: {missing}")
        suite.update({"prompt": prompt, "mlx_version": mx.__version__, "mode_output_directories": {k: str(v) for k, v in mode_dirs.items()}})
        if args.preflight or args.layout_smoke:
            suite["layout_smoke"] = cpu_layout_smoke(mx, np, QwenImage21Layout)
            suite["insertion_smoke"] = cpu_insertion_smoke(mx, np)
            suite["geometry_smoke"] = cpu_geometry_smoke(mx, np)
            suite["enforcement_smoke"] = cpu_enforcement_smoke(mx, np)
            suite["dynamic_context_smoke"] = cpu_dynamic_context_smoke(mx,np)
            suite["status"] = "preflight_passed"
            suite["model_weights_loaded"] = False
            write_suite("preflight.json")
            print(json.dumps(suite, indent=2), flush=True)
            return

        for directory in mode_dirs.values():
            (directory / "previews").mkdir(parents=True, exist_ok=True)
        mx.set_cache_limit(512 * 1024**2)
        mx.set_memory_limit(28 * 1024**3)
        suite["metal_device"] = mx.device_info()
        log("loading_weights", status="running")
        start = time.perf_counter()
        model = QwenImage21Edit(model_path=str(MODEL), quantize=args.quantize,
                               lora_paths=[str(ADAPTER)], lora_scales=[1.0], bake_lora=False)
        suite["phases"]["load_seconds"] = time.perf_counter() - start
        suite["loaded_component_parameter_bytes"] = {
            name: sum(v.nbytes for _, v in tree_flatten(getattr(model, name).parameters()))
            for name in ("transformer", "text_encoder", "vae")
        }
        suite["load_mlx_peak_bytes"] = mx.get_peak_memory()
        mx.reset_peak_memory()
        mx.set_memory_limit(23 * 1024**3)
        log("conditioning")
        start = time.perf_counter()
        # All arms see the actual square source only: no future gray prefix tokens.
        prompt_embeds, slots = model._encode_prompt(prompt, [source])
        mx.eval(prompt_embeds)
        del model.text_encoder
        model.text_encoder = None
        mx.clear_cache()
        pixels = mx.array(np.asarray(source).astype(np.float32) / 127.5 - 1).transpose(2, 0, 1)[None]
        source_latents = QwenImage21LatentCreator.pack_latents(model.vae.encode(pixels)).astype(prompt_embeds.dtype)
        mx.eval(source_latents)
        source_shape = (1, 32, 32)
        if source_latents.shape != (1, 1024, 64):
            raise ValueError(f"Unexpected square-source latent shape: {source_latents.shape}")
        # This changes known TARGET encoding only. The original square latent
        # remains the exact image prefix in every transformer forward.
        prefix_before_context = np.asarray(source_latents.astype(mx.float32)).copy()
        context_encode_start = time.perf_counter()
        padded_rgba = padded_source_rgba(np, np.asarray(source), width, height, rect)
        padded_pixels = mx.array(padded_rgba.astype(np.float32) / 127.5 - 1).transpose(2,0,1)[None]
        padded_latents = QwenImage21LatentCreator.pack_latents(model.vae.encode(padded_pixels)).astype(prompt_embeds.dtype)
        mx.eval(padded_latents)
        if padded_latents.shape != (1,2304,64):
            raise AssertionError("Unexpected edge-padded latent geometry")
        edge_known_latents = padded_latents[:,20*32:52*32]
        mx.eval(edge_known_latents)
        if not np.array_equal(prefix_before_context,np.asarray(source_latents.astype(mx.float32))):
            raise AssertionError("Target-context encoding modified square reference prefix")
        known_sources = {mode:(source_latents if spec["known_context"] == "isolated-square" else edge_known_latents)
                         for mode,spec in ARMS.items()}
        known_source_hashes = {key:hashlib.sha256(np.asarray(value.astype(mx.float32)).tobytes()).hexdigest()
                               for key,value in known_sources.items()}
        suite["target_context_encode_seconds"] = time.perf_counter()-context_encode_start
        suite["known_target_sha256_by_arm"] = known_source_hashes
        suite["prefix_unchanged_by_context_encoding"] = True
        del padded_pixels,padded_latents
        full_layout = QwenImage21Layout.create(slots, [source_shape, (1, 72, 32)], model.transformer.axes)
        full_tokens = full_layout.target_tokens
        mx.random.seed(args.seed)
        # One absolute-position noise table, shared by all arms. Future GROW
        # entries are never passed to its transformer until their activation.
        noise = mx.random.normal((1, full_tokens, 64)).astype(prompt_embeds.dtype)
        ys = np.arange(72).clip(20, 51) - 20
        edge_source_ids = (ys[:, None] * 32 + np.arange(32)[None, :]).reshape(-1).astype(np.int32)
        source_edge_ids = mx.array(edge_source_ids)
        edge_guess = source_latents[:, source_edge_ids]
        sigmas_np = six_sigmas(np, full_tokens)
        sigmas = mx.array(sigmas_np, dtype=mx.float32)
        mx.eval(noise, edge_guess, *full_layout.rope)
        suite["phases"]["conditioning_seconds"] = time.perf_counter() - start
        suite["conditioning_mlx_peak_bytes"] = mx.get_peak_memory()
        suite.update({"sigmas": sigmas_np.tolist(), "source_prefix_latent_tokens": 1024,
                      "joint_prefix_tokens": full_layout.prefix_length, "full_target_tokens": full_tokens,
                      "source_prefix_sha256": hashlib.sha256(np.asarray(source_latents.astype(mx.float32)).tobytes()).hexdigest(),
                      "noise_sha256": hashlib.sha256(np.asarray(noise.astype(mx.float32)).tobytes()).hexdigest(),
                      "hash_representation": "float32 values converted from the shared inference tensors",
                      "arm_specifications": ARMS, "preview_steps": list(PREVIEW_STEPS)})
        order = args.arms
        core_outputs = {}
        dynamic_tiling = TilingConfig(vae_decode_tiles_per_dim=2,vae_decode_tile_size=512,vae_decode_overlap=4)

        def dynamic_known_context(previous_clean, directory, step):
            """VAE-only context update; prefix/cache/noise/target state are absent.

            Only the returned target-known crop is installed by the caller.
            Two independent timings keep VAE cost visible beside DiT core cost.
            """
            started = time.perf_counter()
            unpacked = QwenImage21LatentCreator.unpack_latents(previous_clean.astype(mx.float32),height,width)
            decoded = VAEUtil.decode(model.vae,unpacked,dynamic_tiling)
            mx.eval(decoded)
            if decoded.ndim == 5:
                decoded = decoded[:,:,0]
            decoded_rgb = np.asarray(decoded[0,:3].transpose(1,2,0),dtype=np.float32)
            decode_seconds = time.perf_counter()-started
            context = generated_context_rgba(np,decoded_rgb,np.asarray(source),width,height,rect)
            context_sha = hashlib.sha256(context.tobytes()).hexdigest()
            start_encode = time.perf_counter()
            pixels = mx.array(context).transpose(2,0,1)[None]
            encoded = QwenImage21LatentCreator.pack_latents(model.vae.encode(pixels)).astype(prompt_embeds.dtype)
            mx.eval(encoded)
            if encoded.shape != (1,2304,64):
                raise AssertionError("Dynamic context encode changed absolute target geometry")
            known = encoded[:,20*32:52*32]
            mx.eval(known)
            known_np = np.asarray(known.astype(mx.float32)).copy()
            if not np.isfinite(known_np).all():
                raise FloatingPointError("Dynamic known target source contains NaN/Inf")
            encode_seconds = time.perf_counter()-start_encode
            artifact_start = time.perf_counter()
            artifact_path = directory/f"dynamic_context_before_step{step:02d}.npz"
            np.savez_compressed(artifact_path,normalized_rgba_context=context,known_target_source_latents=known_np,
                                source_rect_xyxy=np.asarray(rect),source_pixels_original_rgba=np.asarray(source))
            artifact_seconds = time.perf_counter()-artifact_start
            return known,known_np,{"before_step":step,"input_normalized_rgba_sha256":context_sha,
                                  "known_target_sha256":hashlib.sha256(known_np.tobytes()).hexdigest(),
                                  "decode_seconds":decode_seconds,"encode_seconds":encode_seconds,
                                  "artifact_seconds":artifact_seconds,"wall_seconds":time.perf_counter()-started,
                                  "artifact_path":str(artifact_path),"vae_decode_calls":1,"vae_encode_calls":1,
                                  "original_source_pixels_exact_in_encoding_input":True,
                                  "whole_canvas_alpha_opaque":True,"encoder_rgb_range":[-1,1],
                                  "full_canvas_encoding_then_known_crop_rows":[20,52],
                                  "noise_table_unchanged":True,"prefix_and_target_state_not_function_arguments":True}

        for order_index, mode in enumerate(order, 1):
            directory = mode_dirs[mode]
            known_source_latents = known_sources[mode]
            known_source_np = np.asarray(known_source_latents.astype(mx.float32))
            edge_guess = known_source_latents[:, source_edge_ids]
            mx.eval(edge_guess)
            if not np.array_equal(prefix_before_context,np.asarray(source_latents.astype(mx.float32))):
                raise AssertionError("Reference prefix changed between matched arms")
            record = {
                "model_repo": "Qwen/Qwen-Image-2.1", "model_revision": REVISION,
                "runtime": "mflux native MLX QwenImage21Edit; experimental actual target-token growth",
                "runtime_git_revision": json.loads((ROOT / "experiments/qwen/runtime.json").read_text())["mflux_git_revision"],
                "adapter_repo": "Viggle/Qwen-Image-2.1-viggle-turbo", "adapter_revision": TURBO_REVISION,
                "adapter_file": str(ADAPTER), "adapter_rank": 256, "adapter_baked": False,
                "quantization": args.quantize, "track": args.track, "seed": args.seed,
                "steps": 6, "prompt": prompt, "width": width, "height": height,
                "source_rect_xyxy": rect, "status": "running", "spatial_mode": mode,
                "diagnostic_control": True, "shared_canvas_comparable": False,
                "boundary_suite": str(suite_dir / "boundary_metrics.json"), "generation_order_within_process": order_index,
                "source_prefix": "actual 512-square source; fixed 1024 latent image tokens",
                "source_prefix_latent_tokens": 1024, "joint_prefix_tokens": full_layout.prefix_length,
                "source_prefix_sha256": suite["source_prefix_sha256"], "noise_sha256": suite["noise_sha256"],
                "source_encoding_shared_between_arms": True, "fresh_prefix_cache_per_arm": True,
                "known_target_encoding_context":ARMS[mode]["known_context"],
                "source_enforcement":ARMS[mode]["enforcement"],
                "source_enforcement_absolute_row_weights":source_enforcement_weights(np,np.arange(full_tokens,dtype=np.int32),ARMS[mode]["enforcement"]).reshape(72,32)[:,0].tolist(),
                "source_enforcement_soft_width_pixels_each_edge":0 if ARMS[mode]["enforcement"] == "hard" else 32,
                "soft_enforcement_formula":"w*known_sigma + (1-w)*state; applied before and after every Euler step; clean previews use same w with clean known target",
                "soft_source_latents_exact_required":False,
                "all_source_latents_required_exact":ARMS[mode]["enforcement"] == "hard",
                "hard_interior_source_rows_required_exact":[20,52] if ARMS[mode]["enforcement"] == "hard" else [22,50],
                "source_exterior_enforcement_weight":0,
                "dynamic_known_context_before_steps":list(ARMS[mode].get("dynamic_before_steps",())),
                "dynamic_context_updates":[],
                "dynamic_vae_decode_calls":0,"dynamic_vae_encode_calls":0,
                "known_target_sha256":known_source_hashes[mode],
                "reference_prefix_unchanged":True,
                "target_context_encode_seconds_shared":suite["target_context_encode_seconds"],
                "source_prefix_and_known_target_separate":True,
                "sigmas": sigmas_np.tolist(), "schedule_shift_target_tokens": full_tokens,
                "target_tokens_future_absent_from_forward": mode != "full",
                "active_heights_by_step": ARMS[mode]["heights"],
                "initializer": ARMS[mode]["initializer"],
                "late_activation_initializer": "(1-sigma)*" + ARMS[mode]["initializer"] + " extrapolation + sigma*same absolute noise; untrained heuristic" if mode != "full" else None,
                "inner_region_denoising": "continues refining at the same global target sigma",
                "mask_method": "weighted known TARGET source latent flow bridge before/after every step; reference prefix unchanged; no outside source clamping",
                "trained_spatial_time_schedule": False, "transformer_calls": 0,
                "preview_definition": "genuine pre-step z_sigma - sigma*predicted_velocity; same weighted clean-source enforcement applied",
                "metal_device": suite["metal_device"], "mlx_version": mx.__version__,
                "phases": {"load_seconds": suite["phases"]["load_seconds"],
                           "conditioning_seconds": suite["phases"]["conditioning_seconds"]},
                "shared_phase_scope": "load and conditioning executed once for entire suite; values attributed to each arm for context",
                "load_mlx_peak_bytes": suite["load_mlx_peak_bytes"],
                "conditioning_mlx_peak_bytes": suite["conditioning_mlx_peak_bytes"],
                "step_records": [], "previews": [],
            }
            suite["runs"][mode] = record
            cache = []  # Each arm pays exactly one prefix-cache construction.
            latents = None
            previous_ids = None
            previous_clean = None
            previews = []
            core_seconds = capture_seconds = dynamic_seconds = dynamic_vae_seconds = 0.0
            wall_start = time.perf_counter()
            mx.reset_peak_memory()
            log("denoising_start", current_mode=mode)
            for step in range(6):
                if step+1 in ARMS[mode].get("dynamic_before_steps",()):
                    if previous_clean is None or len(previous_ids) != full_tokens:
                        raise AssertionError("Dynamic context update requires an existing full-canvas clean estimate")
                    target_before = np.asarray(latents.astype(mx.float32)).copy()
                    cache_ids_before = (id(cache),[(id(k),id(v)) for k,v in cache])
                    known_source_latents,known_source_np,update = dynamic_known_context(previous_clean,directory,step+1)
                    edge_guess = known_source_latents[:,source_edge_ids]
                    mx.eval(edge_guess)
                    if not np.array_equal(target_before,np.asarray(latents.astype(mx.float32))):
                        raise AssertionError("Dynamic context update modified existing target state")
                    if not np.array_equal(prefix_before_context,np.asarray(source_latents.astype(mx.float32))):
                        raise AssertionError("Dynamic context update modified reference prefix")
                    if cache_ids_before != (id(cache),[(id(k),id(v)) for k,v in cache]):
                        raise AssertionError("Dynamic context update replaced reference prefix cache")
                    update.update({"existing_target_state_unchanged":True,
                                   "source_reference_prefix_unchanged":True,
                                   "prefix_cache_objects_unchanged":True})
                    record["dynamic_context_updates"].append(update)
                    record["dynamic_vae_decode_calls"] += 1
                    record["dynamic_vae_encode_calls"] += 1
                    dynamic_seconds += update["wall_seconds"]
                    dynamic_vae_seconds += update["decode_seconds"]+update["encode_seconds"]
                    print(json.dumps({"phase":"dynamic_known_context","mode":mode,**update}),flush=True)
                step_start = time.perf_counter()
                active_h = ARMS[mode]["heights"][step]
                ids, window = active_ids(np, width, height, active_h)
                index = mx.array(ids, dtype=mx.int32)
                added = len(ids) if previous_ids is None else len(ids) - len(previous_ids)
                if previous_ids is None or len(ids) != len(previous_ids):
                    latents = insertion_state(mx, np, ids, previous_ids, latents, previous_clean,
                                              edge_guess, noise, sigmas[step],
                                              ARMS[mode]["initializer"] == "predicted-clean-active-frontier")
                layout = active_layout(mx, np, QwenImage21Layout, slots, source_shape,
                                       full_layout, ids, active_h, width, model.transformer.axes)
                known_weights_np = source_enforcement_weights(np,ids,ARMS[mode]["enforcement"])
                known_weights = mx.array(known_weights_np[None,:,None],dtype=mx.float32)
                known_clean = edge_guess[:, index]
                active_noise = noise[:, index]
                known_now = ((1.0 - sigmas[step]) * known_clean.astype(mx.float32)
                             + sigmas[step] * active_noise.astype(mx.float32)).astype(prompt_embeds.dtype)
                latents = enforce_source(mx,latents,known_now,known_weights)
                before = latents
                had_cache = bool(cache)
                # Actual input is square-reference prefix plus the active target.
                # No full target tensor is passed then hidden by an update mask.
                model_input = mx.concatenate([source_latents, latents], axis=1)
                if model_input.shape[1] != 1024 + len(ids):
                    raise AssertionError("Forward includes unexpected/future image tokens")
                timestep = (sigmas[step:step + 1] * 1000).astype(latents.dtype) / 1000
                prediction = model.transformer(model_input, prompt_embeds, timestep, layout, cache)
                if prediction.shape != latents.shape:
                    raise AssertionError("Transformer predicted tokens outside the active target")
                advanced = (latents.astype(mx.float32) + (sigmas[step + 1] - sigmas[step])
                            * prediction.astype(mx.float32)).astype(prompt_embeds.dtype)
                known_next = ((1.0 - sigmas[step + 1]) * known_clean.astype(mx.float32)
                              + sigmas[step + 1] * active_noise.astype(mx.float32)).astype(prompt_embeds.dtype)
                latents = enforce_source(mx,advanced,known_next,known_weights)
                # All arms form the same clean estimate each step. FRONTIER uses
                # it at the next insertion; others pay the same small tensor cost.
                clean = (before.astype(mx.float32) - sigmas[step] * prediction.astype(mx.float32)).astype(prompt_embeds.dtype)
                clean = enforce_source(mx,clean,known_clean,known_weights)
                mx.eval(latents, clean)
                previous_clean = clean
                step_seconds = time.perf_counter() - step_start
                core_seconds += step_seconds
                record["transformer_calls"] += 1
                if len(cache) != len(model.transformer.transformer_blocks) or any(k.shape[2] != full_layout.prefix_length for k, _ in cache):
                    raise AssertionError("Prefix cache has the wrong layer/token count")
                step_record = {
                    "step": step + 1, "sigma": float(sigmas_np[step]), "next_sigma": float(sigmas_np[step + 1]),
                    "active_height": active_h, "active_window_xyxy": window, "target_tokens_forwarded": len(ids),
                    "future_target_tokens_absent": full_tokens - len(ids), "newly_activated_tokens": added,
                    "input_image_latent_tokens": int(model_input.shape[1]),
                    "attention_query_tokens": len(ids) if had_cache else full_layout.prefix_length + len(ids),
                    "attention_key_value_tokens": full_layout.prefix_length + len(ids),
                    "hard_enforced_source_tokens":int((known_weights_np==1).sum()),
                    "soft_enforced_source_tokens":int(((known_weights_np>0)&(known_weights_np<1)).sum()),
                    "exterior_source_weight_zero":bool(np.all(known_weights_np[(ids//32<20)|(ids//32>=52)]==0)),
                    "core_step_seconds": step_seconds,
                }
                record["step_records"].append(step_record)
                if step + 1 in PREVIEW_STEPS:
                    capture_start = time.perf_counter()
                    clean_np = np.asarray(clean.astype(mx.float32)).copy()
                    if not np.isfinite(clean_np).all():
                        raise FloatingPointError("Predicted-clean preview latent contains NaN/Inf")
                    latent_path = directory / "previews" / f"step{step + 1:02d}_predicted_clean.npz"
                    np.savez_compressed(latent_path, latents=clean_np, final_canvas_ids=ids, sigma=sigmas_np[step], window_xyxy=np.asarray(window))
                    previews.append((step + 1, active_h, window, clean_np))
                    record["previews"].append({"step": step + 1, "active_height": active_h, "active_window_xyxy": window,
                                               "latent_path": str(latent_path), "definition": record["preview_definition"]})
                    capture_seconds += time.perf_counter() - capture_start
                previous_ids = ids
                record["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                (directory / "metrics.json").write_text(json.dumps(record, indent=2))
                print(json.dumps({"phase": "denoising", "mode": mode, **step_record}), flush=True)
            final_np = np.asarray(latents.astype(mx.float32)).copy()
            if not np.isfinite(final_np).all():
                raise FloatingPointError("Final latent contains NaN/Inf")
            record["final_latents_finite"] = True
            hard_y1,hard_y2 = record["hard_interior_source_rows_required_exact"]
            if not np.array_equal(final_np[:,hard_y1*32:hard_y2*32],known_source_np[:,(hard_y1-20)*32:(hard_y2-20)*32]):
                raise AssertionError("Final hard interior TARGET latent differs from chosen context encoding")
            record["known_source_latents_exact_at_sigma_zero"] = bool(np.array_equal(final_np[:,20*32:52*32],known_source_np))
            record["hard_interior_latents_exact_at_sigma_zero"] = True
            record["known_source_soft_boundary_latent_mae"] = float(np.abs(final_np[:,20*32:52*32]-known_source_np).mean())
            record["final_known_target_sha256"] = hashlib.sha256(known_source_np.tobytes()).hexdigest()
            core_wall_seconds = time.perf_counter()-wall_start
            latent_artifact_start = time.perf_counter()
            final_latents_path = directory / "final_latents.npz"
            np.savez_compressed(final_latents_path,latents=final_np,source_prefix_latents=np.asarray(source_latents.astype(mx.float32)),
                                known_target_source_latents=known_source_np,final_canvas_ids=previous_ids,
                                source_enforcement_weights=source_enforcement_weights(np,previous_ids,ARMS[mode]["enforcement"]),
                                source_rect_xyxy=np.asarray(rect),sigmas=sigmas_np)
            record["final_latents_path"] = str(final_latents_path)
            record["phases"].update({"denoising_seconds": core_seconds, "core_loop_wall_seconds": core_wall_seconds,
                                     "preview_capture_seconds": capture_seconds,
                                     "dynamic_context_seconds":dynamic_seconds,"dynamic_vae_decode_encode_seconds":dynamic_vae_seconds,
                                     "final_latent_artifact_capture_seconds":time.perf_counter()-latent_artifact_start})
            record["denoising_timing_scope"] = "sum of six active-layout/insertion/transformer/Euler/clean-prediction steps; preview capture and all VAE decoding excluded"
            record["core_loop_wall_timing_scope"] = "includes dynamic context decode/encode/artifact work and preview latent capture; excludes final latent artifact capture and all final/preview decoding"
            record["denoising_mlx_peak_bytes"] = mx.get_peak_memory()
            record["status"] = "denoised"
            core_outputs[mode] = (final_np, previews)
            (directory / "metrics.json").write_text(json.dumps(record, indent=2))
            del cache, latents, before, prediction, advanced, model_input, clean, previous_clean
            mx.clear_cache()
            log("denoising_complete", current_mode=mode, core_seconds=core_seconds, nfe=record["transformer_calls"])

        # All core loops finish before any preview decode. This keeps speed
        # comparison independent of image preview work and releases the DiT.
        del model.transformer
        model.transformer = None
        mx.clear_cache()
        tiling = TilingConfig(vae_decode_tiles_per_dim=2, vae_decode_tile_size=512, vae_decode_overlap=4)

        def decode(array, active_h, include_float=False):
            packed = mx.array(array).astype(mx.float32)
            unpacked = QwenImage21LatentCreator.unpack_latents(packed, active_h, width)
            decoded = VAEUtil.decode(model.vae, unpacked, tiling)
            mx.eval(decoded)
            if decoded.ndim == 5:
                decoded = decoded[:, :, 0]
            rgb = np.asarray(decoded[0, :3].transpose(1, 2, 0),dtype=np.float32)
            if rgb.shape != (active_h, width, 3) or not np.isfinite(rgb).all():
                raise FloatingPointError(f"Invalid decoded RGB shape/content: {rgb.shape}")
            image = Image.fromarray(np.clip((rgb + 1) * 127.5, 0, 255).round().astype(np.uint8), mode="RGB")
            return (image,rgb.copy()) if include_float else image

        source_rgb = source.convert("RGB")
        for mode in order:
            record = suite["runs"][mode]
            directory = mode_dirs[mode]
            final_np, previews = core_outputs[mode]
            mx.reset_peak_memory()
            log("final_decode", current_mode=mode)
            start = time.perf_counter()
            raw,raw_float = decode(final_np,height,include_float=True)
            raw.save(directory / "raw.png")
            raw_error = np.abs(np.asarray(raw.crop(rect), dtype=np.int16) - np.asarray(source_rgb, dtype=np.int16))
            record.update({"raw_source_region_mae_255": float(raw_error.mean()), "raw_source_region_max_error_255": int(raw_error.max()),
                           "raw_source_top16_mae_255": float(raw_error[:16].mean()), "raw_source_bottom16_mae_255": float(raw_error[-16:].mean()),
                           "raw_source_inner_mae_255": float(raw_error[16:-16].mean()), "decoded_pixels_finite": True})
            result = raw.copy()
            result.paste(source_rgb, (rect[0], rect[1]))
            result.save(directory / "composite.png")
            if not np.array_equal(np.asarray(result.crop(rect)), np.asarray(source_rgb)):
                raise AssertionError("Original resized source pixels changed during compositing")
            final_arrays_path = directory / "final_arrays.npz"
            np.savez_compressed(final_arrays_path,latents=final_np,raw_rgb_float32=raw_float,
                                raw_rgb_uint8=np.asarray(raw),composite_rgb_uint8=np.asarray(result),
                                source_rgb_uint8=np.asarray(source_rgb),source_rect_xyxy=np.asarray(rect),
                                source_enforcement_weights=source_enforcement_weights(np,np.arange(full_tokens,dtype=np.int32),ARMS[mode]["enforcement"]),
                                sigmas=sigmas_np)
            record["final_arrays_path"] = str(final_arrays_path)
            record["raw_rgb_float_range_definition"] = "unclipped decoded RGB in nominal [-1,1] range before conversion to displayed uint8"
            record["phases"]["decode_seconds"] = time.perf_counter() - start
            record["decoding_mlx_peak_bytes"] = mx.get_peak_memory()
            mx.reset_peak_memory()
            log("preview_decode", current_mode=mode)
            start = time.perf_counter()
            for preview_record, (step, active_h, window, array) in zip(record["previews"], previews):
                preview_start = time.perf_counter()
                preview_raw = decode(array, active_h)
                preview_raw_path = directory / "previews" / f"step{step:02d}_predicted_clean_raw.png"
                preview_raw.save(preview_raw_path)
                preview_composite = preview_raw.copy()
                local_source_y = rect[1] - window[1]
                preview_composite.paste(source_rgb, (0, local_source_y))
                preview_path = directory / "previews" / f"step{step:02d}_predicted_clean_composite.png"
                preview_composite.save(preview_path)
                preview_record.update({"raw_path": str(preview_raw_path), "composite_path": str(preview_path),
                                       "decode_seconds": time.perf_counter() - preview_start,
                                       "source_pixels_exact": np.array_equal(np.asarray(preview_composite.crop((0, local_source_y, width, local_source_y + 512))), np.asarray(source_rgb))})
            record["phases"]["preview_decode_seconds"] = time.perf_counter() - start
            record["preview_decoding_mlx_peak_bytes"] = mx.get_peak_memory()
            record.update({"status": "success", "source_pixels_exact": True,
                           "raw_path": str(directory / "raw.png"), "composite_path": str(directory / "composite.png"),
                           "elapsed_seconds": time.perf_counter() - begun,
                           "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                           "mlx_peak_bytes": max(record[k] for k in ("conditioning_mlx_peak_bytes", "denoising_mlx_peak_bytes", "decoding_mlx_peak_bytes")),
                           "output_sha256": hashlib.sha256((directory / "composite.png").read_bytes()).hexdigest()})
            record["generation_seconds_excluding_load_previews"] = sum(record["phases"][k] for k in ("conditioning_seconds", "denoising_seconds", "dynamic_context_seconds", "decode_seconds"))
            (directory / "metrics.json").write_text(json.dumps(record, indent=2))
            mx.clear_cache()
        suite["boundary_comparison"] = {
            "nfe": {mode: suite["runs"][mode]["transformer_calls"] for mode in order},
            "target_token_forward_sum": {mode: sum(r["target_tokens_forwarded"] for r in suite["runs"][mode]["step_records"]) for mode in order},
            "core_seconds": {mode: suite["runs"][mode]["phases"]["denoising_seconds"] for mode in order},
            "source_prefix_matched": True, "noise_prompt_sigmas_matched": True,
            "controlled_arm_changes":["known target encoding context","two-row source enforcement strength","two VAE-only late generated-context updates in edge-dynamic"],
            "dynamic_update_dit_calls":0,
            "dynamic_vae_calls_by_arm":{mode:{"decode":suite["runs"][mode]["dynamic_vae_decode_calls"],"encode":suite["runs"][mode]["dynamic_vae_encode_calls"]} for mode in order},
            "reference_prefix_unmodified":True,
            "no_source_exterior_clamping":True,
            "timing_limit": "single process and seed; execution order/thermal/kernel warm-up can affect time; previews excluded",
            "quality_limit": "Soft border permits source latent adaptation and final original-source hard paste can reveal residual mismatch; neither border-context nor soft enforcement guarantees rail perspective; no region is finalized early",
        }
        log("complete", status="success")
    except BaseException as exc:
        suite.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        write_suite()
        (suite_dir / "traceback.txt").write_text(traceback.format_exc())
        for mode, record in suite.get("runs", {}).items():
            if record.get("status") != "success":
                record.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
                (mode_dirs[mode] / "metrics.json").write_text(json.dumps(record, indent=2))
        raise


if __name__ == "__main__":
    main()
