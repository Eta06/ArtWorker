"""Matched Qwen2.1 growing-canvas LanPaint-style conditional correction probe.

One model load, identical actual-square prefix, edge-context known target,
seed/noise/prompt, six sigmas and EARLY-FRONTIER target growth. The only arm
change is zero/one/two requested conditional correction iterations at selected
outer steps. Every correction is a real extra target-only denoiser call.
Future target tokens are absent from forwards. Frontier insertion is heuristic.
Inner target regions continue refining at the same global sigma.

Algorithm provenance: LanPaint 2.2.0 (GPL-3.0), commit
2d7912f9a5efe5ece8de334c7ca18317b8288c39; see BOUNDARY_METHOD_RESEARCH.md.
This experimental independent MLX implementation is not the official ComfyUI
workflow: h defaults to 0.1 instead of 0.2; correction only at outer steps3–5;
no CFG, no negative or BIG-guidance model; chosen known target restored after
each outer Euler step. No mobile performance or seamless-geometry guarantee.
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
    "edge-control": {"heights": (640, 896, 1152, 1152, 1152, 1152), "initializer": "predicted-clean-active-frontier", "known_context": "edge-padded-full-canvas-center-crop", "inner_requested": 0},
    "think1": {"heights": (640, 896, 1152, 1152, 1152, 1152), "initializer": "predicted-clean-active-frontier", "known_context": "edge-padded-full-canvas-center-crop", "inner_requested": 1},
    "think2": {"heights": (640, 896, 1152, 1152, 1152, 1152), "initializer": "predicted-clean-active-frontier", "known_context": "edge-padded-full-canvas-center-crop", "inner_requested": 2},
}
CORRECTION_STEPS = (3, 4, 5)  # One-based; sigma remains unchanged inside correction.
LANPAINT_REVISION = "2d7912f9a5efe5ece8de334c7ca18317b8288c39"

PREVIEW_STEPS = (2, 4, 6)
ADAPTER = ROOT / ".build/models/qwen/viggle/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors"


def flow_vp_parameters(t):
    """Rectified-flow time -> variance-preserving time; no VE infinity needed."""
    if not 0 < t <= 1:
        raise ValueError("Conditional correction requires a nonzero flow time")
    a = (1-t)**2 / ((1-t)**2+t*t)
    b = 1-a
    return a, b, math.sqrt(a)+math.sqrt(b)


def effective_inner_count(requested, t, outer_step):
    """Official remaining-variance rounding, restricted to this probe's steps."""
    if outer_step not in CORRECTION_STEPS or outer_step == 6:
        return 0
    _, b, _ = flow_vp_parameters(t)
    return max(0, round(requested*b))


def lanpaint_correct(mx, latents, known_clean, known_mask, t, count, h,
                     conditional_lambda, denoise, correction_noise):
    """Independent MLX implementation of the upstream active overdamped path.

    denoise(state,purpose) returns FLOW VELOCITY at the fixed outer timestep.
    q*state uses VP coordinates; x0 is always state-t*velocity. One NFE per
    inner iteration. Final outer NFE happens in the caller, at corrected state.
    No CFG means x0_BIG equals the same conditional x0; it is not a second call.
    The source reference/prefix is never modified. Known TARGET rows may move
    inside correction, then the caller restores its normal known flow bridge.
    """
    if count == 0:
        return latents, {"iterations":0,"sigma_unchanged":True,"prefix_unchanged":True}
    a, b, q = flow_vp_parameters(t)
    mask = known_mask.astype(mx.float32)
    y = known_clean.astype(mx.float32)
    u = latents.astype(mx.float32)*q
    A = (1+conditional_lambda*mask)/b
    previous_C = None

    def coefficient(current, iteration):
        state = current/q
        velocity = denoise(state, f"inner-{iteration+1}-coefficient")
        x0 = state-t*velocity.astype(mx.float32)
        unknown_score = -(current-x0)
        known_score = -(1+conditional_lambda)*(current-y)+conditional_lambda*(current-x0)
        score = mx.where(known_mask,known_score,unknown_score)
        conditional_x0 = current+score
        return (math.sqrt(a)*conditional_x0-current)/b+A*current

    def advance(current, C, interval, iteration, phase):
        # No singular coefficient is present for 0<t<=1 and lambda>0.
        exponent = mx.exp(-A*interval)
        k = -mx.expm1(-A*interval)/A
        k2 = -mx.expm1(-2*A*interval)/(2*A)
        eta = correction_noise(iteration,phase).astype(mx.float32)
        return exponent*current+k*C+mx.sqrt(mx.maximum(2*k2,0))*eta

    for iteration in range(count):
        if previous_C is None:
            C = coefficient(u,iteration)
            u = advance(u,C,h,iteration,0)
        else:
            u = advance(u,previous_C,h/2,iteration,0)
            C = coefficient(u,iteration)
            u = u+(C-previous_C)*h
            u = advance(u,C,h/2,iteration,1)
        previous_C = C
    result = (u/q).astype(latents.dtype)
    mx.eval(result)
    if not bool(mx.all(mx.isfinite(result))):
        raise FloatingPointError("Conditional correction produced NaN/Inf")
    return result, {"iterations":count,"flow_time":t,"vp_alpha_bar":a,"vp_noise_fraction":b,
                    "step_size":h,"conditional_lambda":conditional_lambda,"scheme":"overdamped",
                    "source_target_changes_inside_correction":True,
                    "known_target_restored_after_outer_update":True,
                    "sigma_unchanged":True,"prefix_unchanged":True}


def cpu_lanpaint_smoke(mx, np):
    """Compare MLX CPU correction to independent scalar-coefficient NumPy math."""
    mx.set_default_device(mx.cpu)
    rng=np.random.default_rng(8093)
    state=rng.standard_normal((1,12,4)).astype(np.float32)
    y=rng.standard_normal((1,12,4)).astype(np.float32)
    mask=np.asarray([True]*6+[False]*6)[None,:,None]
    fields={(i,p):rng.standard_normal(state.shape).astype(np.float32) for i in range(2) for p in range(2)}
    checks=[]
    for t in (1.0,0.92751545,0.64639395,0.37862548):
        for count in (0,1,2):
            calls=[]
            def denoise(value,purpose):
                calls.append(purpose)
                return 0.125*value+0.05
            actual,_=lanpaint_correct(mx,mx.array(state),mx.array(y),mx.array(mask),t,count,
                                     0.1,5.0,denoise,lambda i,p:mx.array(fields[(i,p)]))
            a,b,q=flow_vp_parameters(t)
            u=state.astype(np.float64)*q
            A=np.where(mask,6.0,1.0)/b
            previous_C=None
            def coefficient(current):
                x=current/q
                x0=x-t*(0.125*x+0.05)
                score=np.where(mask,-6*(current-y)+5*(current-x0),-(current-x0))
                return (math.sqrt(a)*(current+score)-current)/b+A*current
            def advance(current,C,h,iteration,phase):
                k=-np.expm1(-A*h)/A
                k2=-np.expm1(-2*A*h)/(2*A)
                return np.exp(-A*h)*current+k*C+np.sqrt(2*k2)*fields[(iteration,phase)]
            for iteration in range(count):
                if previous_C is None:
                    C=coefficient(u)
                    u=advance(u,C,0.1,iteration,0)
                else:
                    u=advance(u,previous_C,0.05,iteration,0)
                    C=coefficient(u)
                    u=u+(C-previous_C)*0.1
                    u=advance(u,C,0.05,iteration,1)
                previous_C=C
            expected=state if count==0 else (u/q).astype(np.float32)
            actual_np=np.asarray(actual)
            error=float(np.max(np.abs(actual_np-expected)))
            if not np.isfinite(actual_np).all() or error>3e-6 or len(calls)!=count:
                raise AssertionError(f"LanPaint CPU equation/NFE mismatch {t=} {count=} {error=}")
            checks.append({"t":t,"count":count,"actual_inner_nfe":len(calls),"max_abs_error":error,
                           "finite":True,"zero_iteration_exact":bool(np.array_equal(actual_np,state)) if count==0 else None})
    sigmas=six_sigmas(np,2304)
    planned={mode:[effective_inner_count(spec["inner_requested"],float(sigmas[i]),i+1)
                   for i in range(6)] for mode,spec in ARMS.items()}
    if planned!={"edge-control":[0]*6,"think1":[0,0,1,1,1,0],"think2":[0,0,2,2,2,0]}:
        raise AssertionError("Unexpected inner count or final-step correction")
    return {"status":"passed","device":"CPU","model_weights_loaded":False,
            "checks":checks,"planned_effective_inner_iterations":planned,
            "planned_true_nfe":{mode:6+sum(counts) for mode,counts in planned.items()},
            "upstream_revision":LANPAINT_REVISION}


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=("track1", "track2", "track3"), default="track3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quantize", type=int, choices=(4, 8), default=4)
    parser.add_argument("--arms", nargs="+", choices=tuple(ARMS), default=list(ARMS), help="Execution order; default includes both controlled arms")
    parser.add_argument("--correction-step-size", type=float, default=0.1)
    parser.add_argument("--conditional-lambda", type=float, default=5.0)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--layout-smoke", action="store_true")
    parser.add_argument("--output", type=Path, help="Optional suite directory containing one directory per arm")
    args = parser.parse_args()
    suite_dir = args.output or ROOT / f"experiments/qwen/lanpaint_runs/2026-10-05/{args.track}"
    if not (0 < args.correction_step_size <= 0.2) or not (0 < args.conditional_lambda <= 10):
        parser.error("Require 0 < correction step size <= 0.2 and 0 < lambda <= 10")
    if len(set(args.arms)) != len(args.arms):
        parser.error("Arms must be unique")
    if not (args.preflight or args.layout_smoke) and (suite_dir / "lanpaint_metrics.json").is_file():
        raise FileExistsError(f"Refusing to overwrite prior ablation outputs: {suite_dir}")
    suite_dir.mkdir(parents=True, exist_ok=True)
    mode_dirs = {mode: suite_dir / mode for mode in args.arms}
    suite = {"status": "preflight", "track": args.track, "seed": args.seed, "runs": {}, "phases": {},
             "runner_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "lanpaint_upstream_revision":LANPAINT_REVISION}
    begun = time.perf_counter()

    def write_suite(name="lanpaint_metrics.json"):
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
            suite["lanpaint_smoke"] = cpu_lanpaint_smoke(mx, np)
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
        # Both arms see the actual square source only: no future gray prefix tokens.
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
        known_sources = {mode:edge_known_latents for mode in ARMS}
        known_source_hashes = {key:hashlib.sha256(np.asarray(value.astype(mx.float32)).tobytes()).hexdigest()
                               for key,value in known_sources.items()}
        suite["target_context_encode_seconds"] = time.perf_counter()-context_encode_start
        suite["known_target_sha256_by_arm"] = known_source_hashes
        suite["prefix_unchanged_by_context_encoding"] = True
        del padded_pixels,padded_latents
        full_layout = QwenImage21Layout.create(slots, [source_shape, (1, 72, 32)], model.transformer.axes)
        full_tokens = full_layout.target_tokens
        mx.random.seed(args.seed)
        # One absolute-position noise table, shared by both arms. Future GROW
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
                "lanpaint_suite": str(suite_dir / "lanpaint_metrics.json"), "generation_order_within_process": order_index,
                "source_prefix": "actual 512-square source; fixed 1024 latent image tokens",
                "source_prefix_latent_tokens": 1024, "joint_prefix_tokens": full_layout.prefix_length,
                "source_prefix_sha256": suite["source_prefix_sha256"], "noise_sha256": suite["noise_sha256"],
                "source_encoding_shared_between_arms": True, "fresh_prefix_cache_per_arm": True,
                "known_target_encoding_context":ARMS[mode]["known_context"],
                "known_target_sha256":known_source_hashes[mode],
                "reference_prefix_unchanged":True,
                "target_context_encode_seconds_shared":suite["target_context_encode_seconds"],
                "source_prefix_and_known_target_separate":True,
                "conditional_correction": {"implementation": "experimental LanPaint-style MLX overdamped port", "upstream_revision": LANPAINT_REVISION, "requested_inner_iterations": ARMS[mode]["inner_requested"], "selected_outer_steps": list(CORRECTION_STEPS), "step_size": args.correction_step_size, "conditional_lambda": args.conditional_lambda, "min_step_fraction": 1.0, "inner_count_rounds_with_remaining_variance": True, "extra_noise": "separate deterministic absolute-position stream; never changes original outer noise", "same_timestep_for_inner_and_final_forward": True, "cfg": 1.0, "big_guidance_is_same_conditional_prediction": True},
                "sigmas": sigmas_np.tolist(), "schedule_shift_target_tokens": full_tokens,
                "target_tokens_future_absent_from_forward": mode != "full",
                "active_heights_by_step": ARMS[mode]["heights"],
                "initializer": ARMS[mode]["initializer"],
                "late_activation_initializer": "(1-sigma)*" + ARMS[mode]["initializer"] + " extrapolation + sigma*same absolute noise; untrained heuristic" if mode != "full" else None,
                "inner_region_denoising": "continues refining at the same global target sigma",
                "mask_method": "separate known TARGET source latent flow bridge restored before/after every step; reference prefix unchanged",
                "trained_spatial_time_schedule": False, "transformer_calls": 0, "inner_transformer_calls": 0, "target_token_forward_sum": 0, "all_forward_records": [],
                "preview_definition": "genuine pre-step z_sigma - sigma*predicted_velocity; source latent restored",
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
            core_seconds = capture_seconds = 0.0
            wall_start = time.perf_counter()
            mx.reset_peak_memory()
            log("denoising_start", current_mode=mode)
            for step in range(6):
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
                known_mask_np = ((ids // 32 >= 20) & (ids // 32 < 52))
                known_mask = mx.array(known_mask_np[None, :, None])
                known_clean = edge_guess[:, index]
                active_noise = noise[:, index]
                known_now = ((1.0 - sigmas[step]) * known_clean.astype(mx.float32)
                             + sigmas[step] * active_noise.astype(mx.float32)).astype(prompt_embeds.dtype)
                latents = mx.where(known_mask, known_now, latents)
                had_cache = bool(cache)
                timestep = (sigmas[step:step + 1] * 1000).astype(latents.dtype) / 1000
                step_call_records = []

                def denoise(state, purpose):
                    # Every inner call passes only the source prefix and ACTIVE
                    # target. The same outer timestep/layout/cache apply.
                    input_for_call = mx.concatenate([source_latents, state.astype(prompt_embeds.dtype)], axis=1)
                    if input_for_call.shape[1] != 1024 + len(ids):
                        raise AssertionError("Forward includes unexpected/future image tokens")
                    cached_before_call = bool(cache)
                    item = {"outer_step": step + 1, "purpose": purpose, "sigma": float(sigmas_np[step]),
                            "model_timestep":float(np.asarray(timestep.astype(mx.float32))[0]),
                            "target_tokens_forwarded": len(ids), "future_tokens_absent": full_tokens-len(ids),
                            "source_prefix_tokens": 1024, "cached_prefix": cached_before_call,
                            "attention_query_tokens": len(ids) if cached_before_call else full_layout.prefix_length+len(ids),
                            "status":"started"}
                    step_call_records.append(item)
                    record["all_forward_records"].append(item)
                    record["transformer_calls"] += 1
                    record["target_token_forward_sum"] += len(ids)
                    if purpose.startswith("inner"):
                        record["inner_transformer_calls"] += 1
                    velocity = model.transformer(input_for_call, prompt_embeds, timestep, layout, cache)
                    mx.eval(velocity)
                    if velocity.shape != state.shape or not bool(mx.all(mx.isfinite(velocity))):
                        item["status"]="invalid_velocity"
                        raise FloatingPointError("Invalid active-target velocity")
                    if len(cache) != len(model.transformer.transformer_blocks) or any(k.shape[2] != full_layout.prefix_length for k,_ in cache):
                        item["status"]="invalid_prefix_cache"
                        raise AssertionError("Inner/final forward invalidated source/text prefix cache")
                    item["status"]="success"
                    return velocity.astype(mx.float32)

                correction_noise_records = []
                def correction_noise(iteration, phase):
                    # Generate an independent field at FINAL absolute positions,
                    # then gather only ACTIVE IDs. New noise is not new model work.
                    noise_seed = args.seed + 700000 + step*100 + iteration*3 + phase
                    array = np.random.default_rng(noise_seed).standard_normal((1,full_tokens,64)).astype(np.float32)
                    correction_noise_records.append({"iteration":iteration,"phase":phase,"seed":noise_seed,
                                                     "full_field_sha256":hashlib.sha256(array.tobytes()).hexdigest(),
                                                     "active_absolute_id_count":len(ids)})
                    return mx.array(array[:,ids])

                inner_count = effective_inner_count(ARMS[mode]["inner_requested"], float(sigmas_np[step]), step+1)
                latents, correction_details = lanpaint_correct(
                    mx, latents, known_clean, known_mask, float(sigmas_np[step]), inner_count,
                    args.correction_step_size, args.conditional_lambda, denoise, correction_noise)
                before = latents
                prediction = denoise(latents, "outer-final")
                model_input_token_count = 1024 + len(ids)
                advanced = (latents.astype(mx.float32) + (sigmas[step + 1] - sigmas[step])
                            * prediction.astype(mx.float32)).astype(prompt_embeds.dtype)
                known_next = ((1.0 - sigmas[step + 1]) * known_clean.astype(mx.float32)
                              + sigmas[step + 1] * active_noise.astype(mx.float32)).astype(prompt_embeds.dtype)
                latents = mx.where(known_mask, known_next, advanced)
                # All arms form the same clean estimate each step. FRONTIER uses
                # it at the next insertion; others pay the same small tensor cost.
                clean = (before.astype(mx.float32) - sigmas[step] * prediction.astype(mx.float32)).astype(prompt_embeds.dtype)
                clean = mx.where(known_mask, known_clean, clean)
                mx.eval(latents, clean)
                previous_clean = clean
                step_seconds = time.perf_counter() - step_start
                core_seconds += step_seconds
                if len(cache) != len(model.transformer.transformer_blocks) or any(k.shape[2] != full_layout.prefix_length for k, _ in cache):
                    raise AssertionError("Prefix cache has the wrong layer/token count")
                step_record = {
                    "step": step + 1, "sigma": float(sigmas_np[step]), "next_sigma": float(sigmas_np[step + 1]),
                    "active_height": active_h, "active_window_xyxy": window, "target_tokens_forwarded": len(ids),
                    "future_target_tokens_absent": full_tokens - len(ids), "newly_activated_tokens": added,
                    "input_image_latent_tokens": model_input_token_count,
                    "effective_inner_iterations": inner_count, "transformer_calls_this_step": len(step_call_records),
                    "correction_details": correction_details, "correction_noise_records":correction_noise_records,
                    "target_token_forward_sum_this_step":len(ids)*len(step_call_records),
                    "attention_query_token_sum_this_step":sum(item["attention_query_tokens"] for item in step_call_records),
                    "attention_query_tokens": len(ids) if had_cache else full_layout.prefix_length + len(ids),
                    "attention_key_value_tokens": full_layout.prefix_length + len(ids),
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
            if not np.array_equal(final_np[:, 20 * 32:52 * 32], known_source_np):
                raise AssertionError("Final known TARGET latent differs from its chosen context encoding")
            record["known_source_latents_exact_at_sigma_zero"] = True
            record["phases"].update({"denoising_seconds": core_seconds, "core_loop_wall_seconds": time.perf_counter() - wall_start,
                                     "preview_capture_seconds": capture_seconds})
            record["denoising_timing_scope"] = "sum of six active outer steps including all conditional inner forwards, noise and Euler updates; preview capture and VAE decode excluded"
            record["denoising_mlx_peak_bytes"] = mx.get_peak_memory()
            record["status"] = "denoised"
            core_outputs[mode] = (final_np, previews)
            (directory / "metrics.json").write_text(json.dumps(record, indent=2))
            del cache, latents, before, prediction, advanced, clean, previous_clean
            mx.clear_cache()
            log("denoising_complete", current_mode=mode, core_seconds=core_seconds, nfe=record["transformer_calls"])

        # All core loops finish before any preview decode. This keeps speed
        # comparison independent of image preview work and releases the DiT.
        del model.transformer
        model.transformer = None
        mx.clear_cache()
        tiling = TilingConfig(vae_decode_tiles_per_dim=2, vae_decode_tile_size=512, vae_decode_overlap=4)

        def decode(array, active_h):
            packed = mx.array(array).astype(mx.float32)
            unpacked = QwenImage21LatentCreator.unpack_latents(packed, active_h, width)
            decoded = VAEUtil.decode(model.vae, unpacked, tiling)
            mx.eval(decoded)
            if decoded.ndim == 5:
                decoded = decoded[:, :, 0]
            rgb = np.asarray(decoded[0, :3].transpose(1, 2, 0))
            if rgb.shape != (active_h, width, 3) or not np.isfinite(rgb).all():
                raise FloatingPointError(f"Invalid decoded RGB shape/content: {rgb.shape}")
            return Image.fromarray(np.clip((rgb + 1) * 127.5, 0, 255).round().astype(np.uint8), mode="RGB")

        source_rgb = source.convert("RGB")
        for mode in order:
            record = suite["runs"][mode]
            directory = mode_dirs[mode]
            final_np, previews = core_outputs[mode]
            mx.reset_peak_memory()
            log("final_decode", current_mode=mode)
            start = time.perf_counter()
            raw = decode(final_np, height)
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
            record["generation_seconds_excluding_load_previews"] = sum(record["phases"][k] for k in ("conditioning_seconds", "denoising_seconds", "decode_seconds"))
            (directory / "metrics.json").write_text(json.dumps(record, indent=2))
            mx.clear_cache()
        suite["lanpaint_comparison"] = {
            "nfe": {mode: suite["runs"][mode]["transformer_calls"] for mode in order},
            "target_token_forward_sum": {mode: suite["runs"][mode]["target_token_forward_sum"] for mode in order},
            "core_seconds": {mode: suite["runs"][mode]["phases"]["denoising_seconds"] for mode in order},
            "source_prefix_matched": True, "noise_prompt_sigmas_matched": True,
            "conditional_inner_correction_is_only_arm_change":True,
            "timing_limit": "single process and seed; execution order/thermal/kernel warm-up can affect time; previews excluded",
            "quality_limit": "Experimental LanPaint-style correction has no growing-canvas quality guarantee; all extra forwards counted; inspect both boundaries and whole frame",
        }
        for mode in order:
            record=suite["runs"][mode]
            expected=sum(1+effective_inner_count(ARMS[mode]["inner_requested"],float(sigmas_np[i]),i+1) for i in range(6))
            if record["transformer_calls"] != expected or len(record["all_forward_records"]) != expected:
                raise AssertionError("True NFE accounting mismatch")
            if record["target_token_forward_sum"] != sum(item["target_tokens_forwarded"] for item in record["all_forward_records"]):
                raise AssertionError("All-forward target token accounting mismatch")
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
