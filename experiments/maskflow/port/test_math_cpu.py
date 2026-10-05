#!/usr/bin/env python3
"""Weight-free CPU parity tests against the pinned, executable PyTorch source.

Run with the experiment's Python environment, for example::

    experiments/maskflow/.venv/bin/python experiments/maskflow/port/test_math_cpu.py
    experiments/maskflow/.venv/bin/python experiments/maskflow/port/test_math_cpu.py --report result.json

The oracle is not a second handwritten implementation. Unmodified AST nodes are
compiled from the saved official source, with only its unused scheduler base and
progress display stubbed. Diffusers packing and image preprocessing are taken
from the installed package; their version and source hashes are recorded.
This validates mathematical plumbing on CPU, not model output quality.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Literal, Optional


HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent / "research" / "2026-10-05"
REPO = HERE.parents[2]
SEED = 20261005
HEIGHT, WIDTH = 1152, 512
SOURCE_Y0, SOURCE_Y1 = 320, 832


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute_nodes(path: Path, namespace: dict, names: set[str], *, owner: str | None = None) -> None:
    """Execute selected unchanged functions/classes, never import model modules."""
    tree = ast.parse(path.read_text(), filename=str(path))
    scope = tree.body
    if owner is not None:
        scope = next(node.body for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner)
    nodes = [node for node in scope if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    found = {node.name for node in nodes}
    if found != names:
        raise RuntimeError(f"Oracle source {path}: missing nodes {sorted(names - found)}")
    # A class's staticmethod decorator is unnecessary once extracted as a free
    # function. Retaining it would produce a descriptor, not an ordinary callable.
    if owner is not None:
        for node in nodes:
            node.decorator_list = [d for d in node.decorator_list if not (isinstance(d, ast.Name) and d.id == "staticmethod")]
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)


def extract_pixel_blend(path: Path, namespace: dict) -> None:
    """Compile the official eval_step's final blend expression unchanged."""
    tree = ast.parse(path.read_text(), filename=str(path))
    expressions = [node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "output" for target in node.targets)
                   and {"mask", "raw_source"}.issubset({name.id for name in ast.walk(node.value) if isinstance(name, ast.Name)})]
    if len(expressions) != 1:
        raise RuntimeError("Could not identify the unique official final pixel blend")
    function = ast.FunctionDef(name="official_pixel_blend",
                               args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=name) for name in ("output", "mask", "raw_source")],
                                                  vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
                               body=[ast.Return(value=expressions[0])], decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)


def load_oracle(report: dict) -> SimpleNamespace:
    import numpy as np
    import torch
    import torch.nn.functional as F
    import torchvision
    import torchvision.transforms.functional as T
    from PIL import Image
    from diffusers.image_processor import VaeImageProcessor

    torch.set_num_threads(2)
    manifest = json.loads((RESEARCH / "source_manifest.json").read_text())
    entries = {entry["path"]: entry for entry in manifest["files"]}
    source_paths = [
        RESEARCH / "code" / "pipelines" / "cfg.py",
        RESEARCH / "code" / "pipelines" / "maskflow_utils.py",
        RESEARCH / "code" / "schedulers" / "flow_matching.py",
        RESEARCH / "code" / "schedulers" / "mask_flow.py",
        RESEARCH / "code" / "pipelines" / "qwenimage" / "qwenimage_maskflow.py",
        RESEARCH / "code" / "pipelines" / "qwenimage" / "qwenimage_edit_plus.py",
        RESEARCH / "code" / "data_module" / "utils.py",
        RESEARCH / "base" / "vae" / "config.json",
    ]
    for path in source_paths:
        relative = str(path.relative_to(REPO))
        actual = sha256(path)
        expected = entries[relative]["sha256"]
        report["sources"].append({"path": relative, "sha256": actual, "manifest_sha256": expected, "hash_matches": actual == expected})
        if actual != expected:
            raise RuntimeError(f"Pinned source hash mismatch: {relative}")
    distribution = importlib.metadata.distribution("diffusers")
    pack_source = Path(distribution.locate_file("diffusers/pipelines/qwenimage/pipeline_qwenimage_edit_plus.py"))
    image_processor_source = Path(distribution.locate_file("diffusers/image_processor.py"))
    for path in (pack_source, image_processor_source):
        report["sources"].append({"path": str(path), "sha256": sha256(path), "provider": "installed diffusers", "version": distribution.version})
    report["official_revision"] = manifest["git_revision"]
    report["runtime"].update({"numpy": np.__version__, "torch": torch.__version__, "torchvision": torchvision.__version__, "diffusers": distribution.version, "torch_device": "cpu", "torch_threads": torch.get_num_threads()})

    oracle_module = ModuleType("maskflow_official_cpu_oracle")
    sys.modules[oracle_module.__name__] = oracle_module
    ns = oracle_module.__dict__
    ns.update(dict(np=np, torch=torch, F=F, T=T, Image=Image, math=math,
                   dataclasses=dataclasses, contextmanager=contextmanager,
                   Literal=Literal, Any=Any, Optional=Optional,
                   BaseScheduler=type("BaseScheduler", (), {}),
                   tqdm=lambda iterable, **kwargs: iterable,
                   BRANCHES=("pm", "pn", "nm", "nn")))
    execute_nodes(source_paths[0], ns, {"cfg_coefficients", "required_branches", "combine_predictions"})
    cfg = SimpleNamespace(**{name: ns[name] for name in ("cfg_coefficients", "required_branches", "combine_predictions")})
    execute_nodes(source_paths[1], ns, {"dilate_mask", "blur_mask", "neighbor_sum", "poisson_refine"})
    utils = SimpleNamespace(**{name: ns[name] for name in ("dilate_mask", "blur_mask", "neighbor_sum", "poisson_refine")})
    execute_nodes(source_paths[2], ns, {"RectifiedFlowMatchingScheduler"})
    execute_nodes(source_paths[3], ns, {"MaskFlowScheduler"})
    execute_nodes(pack_source, ns, {"_pack_latents", "_unpack_latents"}, owner="QwenImageEditPlusPipeline")
    codec = SimpleNamespace(_pack_latents=ns["_pack_latents"], _unpack_latents=ns["_unpack_latents"])
    ns["QwenImageEditPlusPipeline"] = codec
    ns["maskflow_utils"] = utils
    ns["QwenMaskFlowPreprocessOutput"] = SimpleNamespace
    ns["logger"] = SimpleNamespace(warning=lambda *args, **kwargs: None)
    execute_nodes(source_paths[6], ns, {"image_dimensions", "resize_image"})
    execute_nodes(source_paths[5], ns, {"resize_rgb"})
    execute_nodes(source_paths[5], ns, {"encode_image", "decode_image"}, owner="QwenImageEditPlus")
    execute_nodes(source_paths[4], ns, {"encode_mask", "preprocess_inputs", "apply_poisson_to_prediction"}, owner="QwenImageMaskFlow")
    extract_pixel_blend(source_paths[4], ns)
    vae_config = json.loads(source_paths[7].read_text())
    config = SimpleNamespace(vae_scale_factor=8, vae_channels=16, device=torch.device("cpu"), dtype=torch.bfloat16,
                             mask_dilation_kernel=25, mask_blur_kernel=25, mask_blur_sigma=25.,
                             max_condition_resolution=384 * 384, divisible_by=32,
                             image_processor=VaeImageProcessor(vae_scale_factor=16),
                             poisson_lambda_e=1., poisson_lambda_s=1., poisson_num_iter=50, poisson_momentum=.1)
    return SimpleNamespace(np=np, torch=torch, cfg=cfg, utils=utils, codec=codec, config=config,
                           scheduler=ns["MaskFlowScheduler"], flow_scheduler=ns["RectifiedFlowMatchingScheduler"],
                           encode_mask=ns["encode_mask"], preprocess_inputs=ns["preprocess_inputs"],
                           encode_image=ns["encode_image"], decode_image=ns["decode_image"], vae_config=vae_config,
                           pixel_blend=ns["official_pixel_blend"], apply_poisson=ns["apply_poisson_to_prediction"], ns=ns)


class Checks:
    def __init__(self, report: dict, oracle: SimpleNamespace):
        self.report = report
        self.oracle = oracle

    def run(self, name: str, function, *, backend: str = "shared") -> None:
        started = time.monotonic()
        entry = {"name": name, "backend": backend}
        try:
            metrics = function()
            entry.update(status="passed", metrics=metrics or {})
        except Exception as error:
            entry.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(limit=5))
        entry["seconds"] = round(time.monotonic() - started, 6)
        self.report["checks"].append(entry)

    def array(self, value):
        np, torch = self.oracle.np, self.oracle.torch
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().float().numpy()
        # NumPy has no BF16 buffer format; first convert MLX's native BF16
        # representation on the explicitly selected CPU backend.
        if str(getattr(value, "dtype", "")) == "mlx.core.bfloat16":
            import mlx.core as mx
            with mx.stream(mx.cpu):
                value = value.astype(mx.float32)
                mx.eval(value)
        return np.asarray(value, dtype=np.float32)

    def compare(self, actual, expected, *, atol=2e-6, rtol=2e-6) -> dict:
        np = self.oracle.np
        actual, expected = self.array(actual), self.array(expected)
        if actual.shape != expected.shape:
            raise AssertionError(f"shape {actual.shape} != official {expected.shape}")
        if not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise AssertionError("nonfinite values in actual or official result")
        difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        allowance = atol + rtol * np.abs(expected.astype(np.float64))
        metrics = {"shape": list(actual.shape), "max_abs_error": float(difference.max(initial=0)),
                   "mean_abs_error": float(difference.mean()) if difference.size else 0.,
                   "max_relative_error": float((difference / np.maximum(np.abs(expected), 1e-6)).max(initial=0)),
                   "atol": atol, "rtol": rtol, "mismatched_elements": int((difference > allowance).sum())}
        if metrics["mismatched_elements"]:
            raise AssertionError(json.dumps(metrics))
        return metrics


def make_canvas(np):
    """User-requested geometry: known 512-square centered in 512x1152."""
    yy, xx = np.indices((512, 512))
    source = np.stack(((xx % 31) / 30, (yy % 29) / 28, ((xx // 8 + yy // 8) % 2)), axis=0).astype(np.float32)
    canvas = np.zeros((1, 3, HEIGHT, WIDTH), np.float32)
    canvas[0, :, SOURCE_Y0:SOURCE_Y1] = source
    mask = np.ones_like(canvas)
    mask[:, :, SOURCE_Y0:SOURCE_Y1] = 0
    return source, canvas, mask


def run_backend(backend: str, port, oracle, checks: Checks) -> None:
    np, torch = oracle.np, oracle.torch
    rng = np.random.default_rng(SEED)
    mx = None
    if backend == "mlx":
        import mlx.core as mx
        mx.set_default_device(mx.cpu)

    def data(value, dtype="float32"):
        array = np.asarray(value, np.float32)
        if dtype == "bfloat16":
            array = torch.from_numpy(array.copy()).to(torch.bfloat16).float().numpy()
        if mx is not None:
            return mx.array(array, dtype=mx.bfloat16 if dtype == "bfloat16" else mx.float32)
        return array.copy()

    def ref(value, dtype="float32"):
        return torch.from_numpy(np.asarray(value, np.float32).copy()).to(torch.bfloat16 if dtype == "bfloat16" else torch.float32)

    def invoke(name, *args, **kwargs):
        if mx is not None:
            with mx.stream(mx.cpu):
                value = getattr(port, name)(*args, backend=backend, **kwargs)
                if isinstance(value, dict):
                    for item in value.values():
                        if isinstance(item, mx.array):
                            mx.eval(item)
                elif isinstance(value, mx.array):
                    mx.eval(value)
                return value
        return getattr(port, name)(*args, backend=backend, **kwargs)

    def check(name, function):
        checks.run(name, function, backend=backend)

    for steps, length in ((50, 2304), (16, 2304), (8, 256), (4, 8192), (1, 2304)):
        def schedule(steps=steps, length=length):
            scheduler = oracle.flow_scheduler()
            t = torch.from_numpy(np.linspace(1, 1 / steps, steps, endpoint=True)).float()
            sigma, derivative = scheduler.get_sigmas(t, length, return_d_sigmas_dt=True)
            expected = {"timesteps": t, "sigmas": torch.cat((sigma, torch.zeros(1))), "d_sigmas_dt": derivative}
            actual = invoke("sigma_grid", num_steps=steps, img_seq_len=length)
            metrics = {key: checks.compare(actual[key], value) for key, value in expected.items()}
            assert abs(float(actual["mu"]) - scheduler.calculate_shift_mu(length)) < 1e-12
            assert float(checks.array(actual["sigmas"])[-1]) == 0
            metrics["mu"] = float(actual["mu"])
            return metrics
        check(f"schedule_{steps}_steps_{length}_tokens", schedule)

    for shape in ((2, 3, 1, 6, 10), (1, 16, 1, HEIGHT // 8, WIDTH // 8)):
        def packing(shape=shape):
            x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
            expected = oracle.codec._pack_latents(ref(x), shape[0], shape[1], shape[-2], shape[-1])
            packed = invoke("pack_latents", data(x))
            metrics = {"pack": checks.compare(packed, expected, atol=0, rtol=0)}
            unpacked = invoke("unpack_latents", packed, height=shape[-2] * 8, width=shape[-1] * 8, vae_scale_factor=8)
            expected_unpacked = oracle.codec._unpack_latents(expected, shape[-2] * 8, shape[-1] * 8, 8)
            metrics["unpack"] = checks.compare(unpacked, expected_unpacked, atol=0, rtol=0)
            metrics["roundtrip"] = checks.compare(unpacked, x, atol=0, rtol=0)
            return metrics
        check("pack_unpack_" + "x".join(map(str, shape)), packing)

    def bf16_rounding():
        values = np.array([0., -0., 1., -1., 1.00390625, 1.01171875, -1.00390625, .33333334, 1e-10, 1025., -1025.], np.float32)
        return checks.compare(port.bfloat16_round(values), ref(values, "bfloat16"), atol=0, rtol=0)
    check("bfloat16_round_to_nearest_even", bf16_rounding)

    source, canvas, full_mask = make_canvas(np)
    small_mask = rng.uniform(0, 1, (1, 3, 33, 41)).astype(np.float32)
    small_mask[:, :, 8:25, 11:30] = 0
    small_mask[:, :, 14:19, 17:23] = 1
    checker = (np.indices((33, 41)).sum(axis=0) % 2).astype(np.float32)[None, None]
    for label, mask in (("nonbinary", small_mask), ("checkerboard", checker), ("full_canvas", full_mask)):
        for dtype in ("float32", "bfloat16"):
            def morphology(mask=mask, dtype=dtype):
                x = ref(mask, dtype)
                dilated = oracle.utils.dilate_mask(x, 25)
                blurred = oracle.utils.blur_mask(dilated, 25, 25.)
                atol = 1 / 128 if dtype == "bfloat16" else 2e-5
                return {
                    "dilate": checks.compare(invoke("dilate_mask", data(mask, dtype), kernel_size=25, dtype=dtype), dilated, atol=0, rtol=0),
                    "blur": checks.compare(invoke("blur_mask", data(dilated.float().numpy(), dtype), kernel_size=25, sigma=25., dtype=dtype), blurred, atol=atol, rtol=0),
                    "process": checks.compare(invoke("process_mask", data(mask), dtype=dtype), blurred, atol=atol, rtol=0),
                }
            check(f"morphology_{label}_{dtype}", morphology)
    for kernel in (2, 3, 4):
        def even_kernel(kernel=kernel):
            x = ref(checker)
            return {"dilate": checks.compare(invoke("dilate_mask", data(checker), kernel_size=kernel), oracle.utils.dilate_mask(x, kernel), atol=0, rtol=0),
                    "blur": checks.compare(invoke("blur_mask", data(checker), kernel_size=kernel, sigma=1.25), oracle.utils.blur_mask(x, kernel, 1.25), atol=2e-6, rtol=2e-6)}
        check(f"morphology_odd_kernel_normalization_{kernel}", even_kernel)

    for dtype in ("float32", "bfloat16"):
        def mask_encoding(dtype=dtype):
            processed = oracle.utils.blur_mask(oracle.utils.dilate_mask(ref(full_mask, dtype), 25), 25, 25.)
            expected = oracle.encode_mask(oracle.config, processed)
            actual = invoke("encode_mask", data(processed.float().numpy(), dtype), vae_scale_factor=8, vae_channels=16, dtype=dtype)
            metrics = checks.compare(actual, expected, atol=0, rtol=0)
            actual_packed = invoke("pack_latents", actual)
            expected_packed = oracle.codec._pack_latents(expected, 1, 16, 144, 64)
            assert checks.array(actual_packed).shape == (1, 2304, 64)
            return {"encode": metrics, "packed": checks.compare(actual_packed, expected_packed, atol=0, rtol=0)}
        check(f"encode_mask_full_canvas_{dtype}", mask_encoding)
        def multichannel_encoding(dtype=dtype):
            # Channel mean reduction can differ from repeating a binary mask.
            array = rng.uniform(-.25, 1.25, (2, 3, 32, 48)).astype(np.float32)
            expected = oracle.encode_mask(oracle.config, ref(array, dtype))
            return checks.compare(invoke("encode_mask", data(array, dtype), dtype=dtype), expected,
                                  atol=1 / 128 if dtype == "bfloat16" else 2e-6, rtol=0)
        check(f"encode_mask_nonidentical_channels_clamp_{dtype}", multichannel_encoding)

    def neighbors():
        x = rng.normal(size=(2, 3, 7, 9)).astype(np.float32)
        return checks.compare(invoke("neighbor_sum", data(x)), oracle.utils.neighbor_sum(ref(x)), atol=1e-6, rtol=1e-6)
    check("neighbor_sum_zero_padding_multichannel", neighbors)
    for mask_kind in ("empty", "full", "soft_checkerboard"):
        def poisson(mask_kind=mask_kind):
            g, source = (rng.normal(size=(2, 3, 9, 11)).astype(np.float32) for _ in range(2))
            soft = rng.uniform(0, 1, (2, 1, 9, 11)).astype(np.float32)
            mask = (np.indices((9, 11)).sum(axis=0) % 2)[None, None].astype(np.float32)
            if mask_kind == "empty":
                mask.fill(0)
            elif mask_kind == "full":
                mask.fill(1)
            kwargs = dict(poisson_lambda_e=1., poisson_lambda_s=1., poisson_num_iter=50, poisson_momentum=.1)
            expected = oracle.utils.poisson_refine(ref(g), ref(source), ref(mask), ref(soft), **kwargs, disable_progress_bar=True)
            actual = invoke("poisson_refine", data(g), data(source), data(mask), data(soft), **kwargs)
            metrics = checks.compare(actual, expected, atol=2e-5, rtol=2e-5)
            outside = np.broadcast_to(mask == 0, g.shape)
            if outside.any():
                metrics["outside_source_max_abs_error"] = float(np.abs(checks.array(actual)[outside] - source[outside]).max())
                assert metrics["outside_source_max_abs_error"] <= 2e-6
            return metrics
        check(f"poisson_refine_{mask_kind}_50_iterations", poisson)

    scales = ((1, 1, None), (4, 1, None), (0, 1, None), (0, 0, None), (2, 3, .5), (1.5, .5, 0))
    predictions = {branch: rng.normal(size=(2, 7, 8)).astype(np.float32) for branch in ("pm", "pn", "nm", "nn")}
    predictions["pm"][0, 0] = 0
    for text, mask_scale, interaction in scales:
        for rescale in (False, True):
            def cfg(text=text, mask_scale=mask_scale, interaction=interaction, rescale=rescale):
                kwargs = dict(text_scale=text, mask_scale=mask_scale, interaction_scale=interaction)
                assert port.cfg_coefficients(**kwargs) == oracle.cfg.cfg_coefficients(**kwargs)
                assert port.required_branches(**kwargs, rescale=rescale) == oracle.cfg.required_branches(**kwargs, rescale=rescale)
                needed = oracle.cfg.required_branches(**kwargs, rescale=rescale)
                results = {}
                for dtype in ("float32", "bfloat16"):
                    expected = oracle.cfg.combine_predictions({k: ref(predictions[k], dtype) for k in needed}, **kwargs, rescale=rescale)
                    actual = invoke("combine_predictions", {k: data(predictions[k], dtype) for k in needed}, **kwargs, rescale=rescale, dtype=dtype)
                    results[dtype] = checks.compare(actual, expected, atol=1 / 128 if dtype == "bfloat16" else 2e-6, rtol=1 / 128 if dtype == "bfloat16" else 2e-6)
                results["required_branches"] = needed
                return results
            check(f"cfg_text{text}_mask{mask_scale}_interaction{interaction}_rescale{rescale}", cfg)
    def missing_cfg_branch():
        try:
            invoke("combine_predictions", {"pm": data(predictions["pm"])}, text_scale=4, mask_scale=1)
        except ValueError:
            return {"raised": "ValueError"}
        raise AssertionError("missing nm branch did not raise ValueError")
    check("cfg_missing_branch_rejected", missing_cfg_branch)

    x0, noise, source_small, velocity = (rng.normal(size=(2, 7, 8)).astype(np.float32) for _ in range(4))
    mask = rng.uniform(0, 1, (2, 7, 1)).astype(np.float32)
    mask[0, 0] = 0
    mask[1, -1] = 1
    sigma = np.array([.75, .25], np.float32)
    for mode in ("target", "source", "noisy_target", "noisy_source"):
        for gamma in ((1., 2., 3.) if mode == "noisy_source" else (1.,)):
            for dtype in ("float32", "bfloat16"):
                def scheduler_math(mode=mode, gamma=gamma, dtype=dtype):
                    scheduler = oracle.scheduler(unmask_with=mode, background_noise_power=gamma)
                    rt_x, rt_n, rt_s, rt_m, rt_v = (ref(x, dtype) for x in (x0, noise, source_small, mask, velocity))
                    rt_sig = ref(sigma)
                    kwargs = dict(source=data(source_small, dtype), mask=data(mask, dtype), noise=data(noise, dtype), unmask_with=mode, background_noise_power=gamma, dtype=dtype)
                    addition_kwargs = {key: value for key, value in kwargs.items() if key != "noise"}
                    expected_xt = scheduler.add_noise_by_sigmas(rt_n, rt_x, rt_sig, rt_s, rt_m)
                    actual_xt = invoke("add_noise_by_sigmas", data(noise, dtype), data(x0, dtype), data(sigma), **addition_kwargs)
                    expected_pred = scheduler.predict_x0(expected_xt, rt_sig, ref(sigma * 0 + 7), rt_v, rt_s, rt_m, rt_n)
                    actual_pred = invoke("predict_x0", data(expected_xt.float().numpy(), dtype), data(sigma), data(sigma * 0 + 7), data(velocity, dtype), **kwargs)
                    # The official step uses scalar schedule entries; using batch
                    # entries here would not broadcast over [B,N,C].
                    cur, nxt, derivative = ref(np.array(.75)), ref(np.array(.5)), ref(np.array(99.))
                    expected_step = scheduler.step(expected_xt, rt_v, cur, nxt, derivative, rt_s, rt_m, rt_n)
                    actual_step = invoke("scheduler_step", data(expected_xt.float().numpy(), dtype), data(velocity, dtype), data(np.array(.75)), data(np.array(.5)), d_sigma_dt=data(np.array(99.)), **kwargs)
                    tolerance = dict(atol=1 / 64, rtol=1 / 128) if dtype == "bfloat16" else dict(atol=2e-6, rtol=2e-6)
                    addition_tolerance = dict(atol=0, rtol=0) if dtype == "bfloat16" else tolerance
                    return {"add_noise": checks.compare(actual_xt, expected_xt, **addition_tolerance), "predict_x0": checks.compare(actual_pred, expected_pred, **tolerance), "step": checks.compare(actual_step, expected_step, **tolerance)}
                check(f"scheduler_{mode}_gamma{gamma}_{dtype}", scheduler_math)
    def scheduler_fallback_and_derivative():
        scheduler = oracle.scheduler()
        expected = scheduler.add_noise_by_sigmas(ref(noise), ref(x0), ref(sigma))
        addition = invoke("add_noise_by_sigmas", data(noise), data(x0), data(sigma))
        step = invoke("scheduler_step", data(x0), data(velocity), .75, .5, d_sigma_dt=999.)
        step2 = invoke("scheduler_step", data(x0), data(velocity), .75, .5, d_sigma_dt=.001)
        return {"fallback": checks.compare(addition, expected), "step": checks.compare(step, scheduler.step(ref(x0), ref(velocity), .75, .5, 999.)), "derivative_ignored": checks.compare(step, step2, atol=0, rtol=0)}
    check("scheduler_unconditioned_fallback_derivative_ignored", scheduler_fallback_and_derivative)

    def precision_step_fixture(mode, *, current, following, shaped=False, mask_dtype="bfloat16"):
        # A separately seeded fixture exercises scalar-tensor promotion without
        # depending on earlier tests' random draws or the noise-addition path.
        fixture_rng = np.random.default_rng(SEED + 91)
        arrays = [fixture_rng.normal(0, .4, (1, 11, 8)).astype(np.float32) for _ in range(4)]
        fixture_xt, fixture_v, fixture_source, fixture_noise = arrays
        fixture_mask = fixture_rng.uniform(0, 1, (1, 11, 1)).astype(np.float32)
        fixture_mask[:, 0] = 0
        fixture_mask[:, -1] = 1
        tensor_shape = (1, 1, 1) if shaped else ()
        current_array = np.full(tensor_shape, current, np.float32)
        next_array = np.full(tensor_shape, following, np.float32)
        scheduler = oracle.scheduler(unmask_with=mode)
        expected = scheduler.step(ref(fixture_xt, "bfloat16"), ref(fixture_v, "bfloat16"), ref(current_array), ref(next_array),
                                  ref(np.array(1.)), ref(fixture_source, "bfloat16"), ref(fixture_mask, mask_dtype), ref(fixture_noise, "bfloat16"))
        kwargs = dict(dtype="bfloat16", unmask_with=mode, source=data(fixture_source, "bfloat16"),
                      mask=data(fixture_mask, mask_dtype), noise=data(fixture_noise, "bfloat16"))
        if backend == "numpy":
            kwargs["mask_dtype"] = mask_dtype
        actual = invoke("scheduler_step", data(fixture_xt, "bfloat16"), data(fixture_v, "bfloat16"),
                        data(current_array), data(next_array), **kwargs)
        output_dtype = "float32" if shaped or mask_dtype == "float32" else "bfloat16"
        official_dtype = torch.float32 if output_dtype == "float32" else torch.bfloat16
        actual_dtype = (mx.float32 if output_dtype == "float32" else mx.bfloat16) if mx is not None else np.dtype(np.float32)
        assert expected.dtype == official_dtype, f"official projection dtype: {expected.dtype} != {official_dtype}"
        assert actual.dtype == actual_dtype, f"port projection dtype: {actual.dtype} != {actual_dtype}"
        metrics = checks.compare(actual, expected, atol=0, rtol=0)
        metrics.update(official_dtype=str(expected.dtype), port_dtype=str(actual.dtype), sigma_shape=list(tensor_shape),
                       current=float(current_array.reshape(-1)[0]), next=float(next_array.reshape(-1)[0]), mask_dtype=mask_dtype)
        return metrics

    for mode in ("target", "source", "noisy_target", "noisy_source"):
        check(f"scheduler_scalar_tensor_coefficients_exact_{mode}",
              lambda mode=mode: precision_step_fixture(mode, current=.63, following=.521))
    for mode in ("noisy_target", "noisy_source"):
        check(f"scheduler_shaped_sigmas_partial_fp32_promotion_exact_{mode}",
              lambda mode=mode: precision_step_fixture(mode, current=.63, following=.521, shaped=True))
    for mode in ("target", "source", "noisy_target", "noisy_source"):
        check(f"scheduler_mixed_fp32_mask_native_inference_exact_{mode}",
              lambda mode=mode: precision_step_fixture(mode, current=.75, following=.625, mask_dtype="float32"))

    for dtype in ("float32", "bfloat16"):
        for gamma in (1., 2.):
            def poisson_velocity(dtype=dtype, gamma=gamma):
                # Full target geometry, 16 VAE channels and all 2304 packed tokens.
                latent_shape = (1, 16, 1, HEIGHT // 8, WIDTH // 8)
                packed = []
                for _ in range(4):
                    array = rng.normal(0, .3, latent_shape).astype(np.float32)
                    packed.append(oracle.codec._pack_latents(ref(array, dtype), 1, 16, 144, 64))
                xt, pred, latent_source, latent_noise = packed
                processed = oracle.utils.blur_mask(oracle.utils.dilate_mask(ref(full_mask, dtype), 25), 25, 25.)
                encoded = oracle.encode_mask(oracle.config, processed)
                latent_mask = oracle.codec._pack_latents(encoded, 1, 16, 144, 64)
                oracle.config.scheduler = oracle.scheduler(unmask_with="noisy_source", background_noise_power=gamma)
                oracle.config.poisson_num_iter = 50
                # A non-power-of-two sigma exposes inverse-division rounding;
                # sigma=.5 alone cannot distinguish FP32 from a BF16 quotient.
                sigma_value = .63 if dtype == "bfloat16" and gamma == 1. else .5
                curr_sigma = ref(np.array(sigma_value))
                expected = oracle.apply_poisson(oracle.config, xt, pred, curr_sigma, ref(np.array(1.)), latent_source, latent_mask, latent_noise, HEIGHT, WIDTH, disable_progress_bar=True)
                actual = invoke("apply_poisson_to_prediction", data(xt.float().numpy(), dtype), data(pred.float().numpy(), dtype), sigma_value,
                                data(latent_source.float().numpy(), dtype), data(latent_mask.float().numpy(), dtype), data(latent_noise.float().numpy(), dtype),
                                HEIGHT, WIDTH, dtype=dtype, background_noise_power=gamma, poisson_num_iter=50)
                assert expected.dtype == torch.float32, f"official Poisson velocity dtype: {expected.dtype}"
                expected_dtype = mx.float32 if mx is not None else np.dtype(np.float32)
                assert actual.dtype == expected_dtype, f"port Poisson velocity dtype: {actual.dtype} != {expected_dtype}"
                step_kwargs = dict(source=data(latent_source.float().numpy(), dtype), mask=data(latent_mask.float().numpy(), dtype),
                                   noise=data(latent_noise.float().numpy(), dtype), dtype=dtype, background_noise_power=gamma)
                if backend == "numpy":
                    # Native MLX must infer the actual FP32 velocity dtype.
                    # NumPy BF16 emulation needs the explicit mixed-type marker.
                    step_kwargs["velocity_dtype"] = "float32"
                expected_step = oracle.config.scheduler.step(xt, expected, curr_sigma, ref(np.array(.25)), ref(np.array(1.)),
                                                              latent_source, latent_mask, latent_noise)
                actual_step = invoke("scheduler_step", data(xt.float().numpy(), dtype), actual, sigma_value, .25, **step_kwargs)
                tolerance = dict(atol=1 / 32, rtol=1 / 128) if dtype == "bfloat16" else dict(atol=5e-5, rtol=2e-5)
                return {"velocity": checks.compare(actual, expected, **tolerance),
                        "euler_after_poisson": checks.compare(actual_step, expected_step, **tolerance),
                        "official_velocity_dtype": str(expected.dtype), "port_velocity_dtype": str(actual.dtype),
                        "curr_sigma": float(curr_sigma), "next_sigma": .25}
            check(f"poisson_prediction_full_canvas_gamma{gamma}_{dtype}_50_iterations", poisson_velocity)

    def canvas_geometry():
        result = port.prepare_canvas(source, fill="black")
        if isinstance(result, tuple):
            actual_canvas, actual_mask, metadata = result
        else:
            actual_canvas = result["source_canvas"]
            actual_mask = result.get("mask", result.get("pixel_mask"))
            metadata = result.get("metadata", {})
        return {"source_canvas": checks.compare(actual_canvas, canvas, atol=0, rtol=0),
                "pixel_mask": checks.compare(actual_mask, full_mask, atol=0, rtol=0),
                "known_source_rows": [SOURCE_Y0, SOURCE_Y1], "metadata": metadata}
    check("canvas_exact_512x1152_source_rows320_to832", canvas_geometry)

    def conditions():
        oracle.config.dtype = torch.bfloat16
        expected = oracle.preprocess_inputs(oracle.config, {"prompt": ["test"], "conditions": {"source": ref(canvas), "mask": ref(full_mask)}})
        actual = invoke("prepare_condition_images", data(canvas), data(full_mask), dtype="bfloat16")
        results = {"raw_source": checks.compare(actual["raw_source"], expected.raw_source, atol=0, rtol=0),
                   "mask": checks.compare(actual["mask"], expected.mask, atol=1 / 128, rtol=0)}
        for kind in ("dit_conditions", "vlm_conditions"):
            for key in ("source", "mask"):
                results[f"{kind}.{key}"] = checks.compare(actual[kind][key], getattr(expected, kind)[key], atol=1 / 64, rtol=0)
        encoded = oracle.encode_mask(oracle.config, expected.mask)
        packed = oracle.codec._pack_latents(encoded, 1, 16, 144, 64)
        results["mask_latents"] = checks.compare(actual["mask_latents"], packed, atol=1 / 128, rtol=0)
        assert checks.array(actual["vlm_conditions"]["source"]).shape == (1, 3, 576, 256)
        results["vlm_shape"] = [1, 3, 576, 256]
        results["packed_mask_shape"] = [1, 2304, 64]
        return results
    check("conditions_against_official_preprocess_full_canvas_bfloat16", conditions)

    for dtype in ("float32", "bfloat16"):
        def latent_normalization(dtype=dtype):
            x = ref(rng.normal(size=(2, 16, 1, 6, 10)).astype(np.float32), dtype)
            captured = {}
            def decode(array, return_dict=False):
                captured["latents"] = array
                return (array[:, :3],)
            config = SimpleNamespace(vae=SimpleNamespace(config=SimpleNamespace(**oracle.vae_config),
                                                        encode=lambda image: SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: x)),
                                                        decode=decode),
                                     text_pipeline=SimpleNamespace(image_processor=SimpleNamespace(postprocess=lambda array, **kwargs: array)))
            expected_normalized = oracle.encode_image(config, x, "argmax")
            actual_normalized = invoke("normalize_vae_latents", data(x.float().numpy(), dtype), dtype=dtype)
            oracle.decode_image(config, expected_normalized)
            actual_denormalized = invoke("denormalize_vae_latents", data(expected_normalized.float().numpy(), dtype), dtype=dtype)
            tolerance = dict(atol=1 / 128, rtol=1 / 128) if dtype == "bfloat16" else dict(atol=2e-6, rtol=2e-6)
            return {"normalize": checks.compare(actual_normalized, expected_normalized, **tolerance),
                    "denormalize": checks.compare(actual_denormalized, captured["latents"], **tolerance),
                    "vae_sample_mode": "argmax"}
        check(f"vae_normalization_against_official_encode_decode_{dtype}", latent_normalization)

    for output_dtype in ("float32", "bfloat16"):
        def final_pixel_blend(output_dtype=output_dtype):
            output = rng.uniform(0, 1, canvas.shape).astype(np.float32)
            source = ref(canvas, "bfloat16")
            processed = oracle.utils.blur_mask(oracle.utils.dilate_mask(ref(full_mask, "bfloat16"), 25), 25, 25.)
            expected = oracle.pixel_blend(ref(output, output_dtype), processed, source)
            actual = invoke("pixel_blend", data(output, output_dtype), data(canvas, "bfloat16"), data(processed.float().numpy(), "bfloat16"),
                            dtype="bfloat16", output_dtype=output_dtype)
            metrics = checks.compare(actual, expected, atol=1e-6 if output_dtype == "float32" else 0, rtol=1e-6 if output_dtype == "float32" else 0)
            preserved = np.broadcast_to(processed.float().numpy() == 0, output.shape)
            changed_boundary = np.broadcast_to((processed.float().numpy() > 0) & (full_mask == 0), output.shape)
            assert preserved.any() and changed_boundary.any()
            assert np.array_equal(checks.array(actual)[preserved], source.float().numpy()[preserved])
            metrics["exactly_preserved_elements_where_processed_mask_zero"] = int(preserved.sum())
            metrics["original_region_elements_with_nonzero_processed_mask"] = int(changed_boundary.sum())
            return metrics
        check(f"final_official_pixel_blend_{output_dtype}_preserves_processed_mask_zero", final_pixel_blend)

    def hard_paste():
        source_u8 = np.rint(source.transpose(1, 2, 0) * 255).astype(np.uint8)
        output_u8 = rng.integers(0, 256, (HEIGHT, WIDTH, 3), np.uint8)
        metrics = {}
        for layout in ("HWC", "CHW", "BCHW"):
            original, output = source_u8, output_u8
            if layout != "HWC":
                original, output = original.transpose(2, 0, 1), output.transpose(2, 0, 1)
            if layout == "BCHW":
                original, output = original[None], output[None]
            actual = port.hard_paste_original(output, original)
            actual_hwc = actual if layout == "HWC" else (actual[0] if layout == "BCHW" else actual).transpose(1, 2, 0)
            assert actual.dtype == np.uint8
            assert np.array_equal(actual_hwc[SOURCE_Y0:SOURCE_Y1], source_u8)
            assert np.array_equal(actual_hwc[:SOURCE_Y0], output_u8[:SOURCE_Y0])
            assert np.array_equal(actual_hwc[SOURCE_Y1:], output_u8[SOURCE_Y1:])
            assert np.array_equal(output, output_u8 if layout == "HWC" else output_u8.transpose(2, 0, 1)[None] if layout == "BCHW" else output_u8.transpose(2, 0, 1))
            metrics[layout] = {"known_source_exact": True, "expansion_unchanged": True, "input_unmodified": True}
        return metrics
    check("separate_hard_paste_policy_preserves_original_uint8_all_layouts", hard_paste)

    def gray_canvas():
        result = port.prepare_canvas(source.transpose(1, 2, 0), fill="gray")
        actual = result["source_canvas"]
        assert np.all(actual[:, :, :SOURCE_Y0] == .5)
        assert np.all(actual[:, :, SOURCE_Y1:] == .5)
        return {"known_source": checks.compare(actual[0, :, SOURCE_Y0:SOURCE_Y1], source, atol=0, rtol=0), "fill": .5}
    check("canvas_gray_fill_exact_known_region", gray_canvas)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Write machine-readable results to this exact path")
    parser.add_argument("--backend", choices=("numpy", "mlx", "all"), default="all")
    args = parser.parse_args()
    started = time.monotonic()
    report = {"schema_version": 1, "status": "running", "seed": SEED,
              "scope": "Weight-free CPU parity against official PyTorch mathematical source; no image-quality or GPU inference acceptance",
              "runtime": {"python": sys.version.split()[0], "executable": sys.executable, "platform": platform.platform(), "machine": platform.machine()},
              "sources": [], "checks": [], "skipped": []}
    try:
        oracle = load_oracle(report)
        import maskflow_math as port
        report["port_sha256"] = sha256(HERE / "maskflow_math.py")
        report["test_sha256"] = sha256(Path(__file__))
        report["source_sha256"] = {str(HERE / "maskflow_math.py"): report["port_sha256"],
                                   str(Path(__file__).resolve()): report["test_sha256"]}
        checks = Checks(report, oracle)
        if args.backend in ("numpy", "all"):
            run_backend("numpy", port, oracle, checks)
        if args.backend in ("mlx", "all"):
            try:
                import mlx.core as mx
            except ImportError as error:
                if args.backend == "mlx":
                    raise RuntimeError("Explicit MLX backend unavailable") from error
                report["skipped"].append({"backend": "mlx", "reason": "MLX is not installed; NumPy tests do not validate MLX"})
            else:
                mx.set_default_device(mx.cpu)
                report["runtime"].update(mlx=importlib.metadata.version("mlx"), mlx_device="cpu")
                run_backend("mlx", port, oracle, checks)
        failures = [check for check in report["checks"] if check["status"] != "passed"]
        report.update(status="failed" if failures else "passed", passed=len(report["checks"]) - len(failures), failed=len(failures))
        exit_code = 1 if failures else 0
    except Exception as error:
        report.update(status="blocked", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(limit=8))
        exit_code = 2
    report["seconds"] = round(time.monotonic() - started, 6)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
