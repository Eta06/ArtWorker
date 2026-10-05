#!/usr/bin/env python3
"""Seeded CPU parity of the isolated MaskFlow transformer against Diffusers.

Runs an actual 60-layer tiny float32 Qwen-Image-Edit-2511 transformer and
compares every dual-stream block. No pretrained checkpoint is opened. The
saved official Diffusers source executes with the real pinned library helpers;
the frozen MLX runtime executes unmodified, through the isolated compatibility
subclass when selected. Both implementations receive identical NumPy weights.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

# This must precede runtime imports, array creation and module initialization.
import mlx.core as mx
mx.set_default_device(mx.cpu)

import numpy as np
import torch
torch.set_default_device("cpu")
torch.set_num_threads(2)
torch.set_default_dtype(torch.float32)
from mlx.utils import tree_flatten

ROOT = Path(__file__).resolve().parents[3]
AREA = ROOT / "experiments/maskflow"
RUNTIME = AREA / "runtime/mlx-gen"
OFFICIAL = AREA / "research/2026-10-05/diffusers/transformer_qwenimage_v0.37.1.py"
RUNTIME_REVISION = "99fb94dd3eaa9dd1931cd3cd8eae1ae3e20f2ef3"
SEED = 2511
CONFIG = dict(in_channels=8, out_channels=2, num_layers=60,
              attention_head_dim=12, num_attention_heads=2,
              joint_attention_dim=10, patch_size=2,
              axes_dims_rope=(4, 4, 4), zero_cond_t=True)
GRIDS = [(1, 3, 5), (1, 2, 4), (1, 4, 3)]
LENGTHS = [int(np.prod(shape)) for shape in GRIDS]
TOLERANCES = dict(atol=5e-5, rtol=2e-5)
REPORT: dict = {}
sys.path.insert(0, str(RUNTIME / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array(value):
    if isinstance(value, torch.Tensor):
        assert value.device.type == "cpu"
        return value.detach().numpy()
    if isinstance(value, mx.array):
        mx.eval(value)
    return np.asarray(value)


def compare(label, actual, expected, *, atol=None, rtol=None):
    actual, expected = array(actual), array(expected)
    atol = TOLERANCES["atol"] if atol is None else atol
    rtol = TOLERANCES["rtol"] if rtol is None else rtol
    assert actual.shape == expected.shape, f"{label}: {actual.shape} != {expected.shape}"
    error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    limit = atol + rtol * np.abs(expected.astype(np.float64))
    passed = finite and bool(np.all(error <= limit))
    ratios = np.zeros_like(error)
    np.divide(error, limit, out=ratios, where=limit != 0)
    ratios[(limit == 0) & (error != 0)] = np.inf
    result = dict(shape=list(actual.shape), max_abs_error=float(error.max(initial=0)),
                  rms_error=float(np.sqrt(np.mean(error ** 2))),
                  max_tolerance_ratio=float(np.max(ratios, initial=0)),
                  atol=atol, rtol=rtol, finite=finite, passed=passed)
    REPORT.setdefault("comparisons", {})[label] = result
    assert passed, f"{label}: {result}"
    return result


def reference_module():
    assert importlib.metadata.version("diffusers") == "0.37.1", "Diffusers must be pinned to 0.37.1"
    import diffusers.models.transformers
    name = "diffusers.models.transformers._maskflow_frozen_reference_v0371_cpu"
    spec = importlib.util.spec_from_file_location(name, OFFICIAL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def mlx_name(torch_name):
    """Explicit mapping, with identical Linear weight layout [out, in]."""
    value = torch_name.replace(".img_mod.1.", ".img_mod_linear.")
    value = value.replace(".txt_mod.1.", ".txt_mod_linear.")
    value = value.replace(".attn.to_out.0.", ".attn.attn_to_out.0.")
    for stream in ("img", "txt"):
        value = value.replace(f".{stream}_mlp.net.0.proj.", f".{stream}_ff.mlp_in.")
        value = value.replace(f".{stream}_mlp.net.2.", f".{stream}_ff.mlp_out.")
    return value


def fixtures(reference):
    rng = np.random.default_rng(SEED)
    weights = {}
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, parameter in sorted(reference.named_parameters()):
            shape = tuple(parameter.shape)
            if parameter.ndim == 1 and name.endswith(".weight"):
                fixture = (1 + rng.normal(0, .025, shape)).astype(np.float32)
            else:
                std = .6 / np.sqrt(shape[-1]) if parameter.ndim == 2 else .04
                fixture = rng.normal(0, std, shape).astype(np.float32)
            parameter.copy_(torch.from_numpy(fixture))
            weights[name] = fixture
            digest.update(name.encode())
            digest.update(str(shape).encode())
            digest.update(fixture.tobytes())
    image = rng.normal(0, .6, (2, sum(LENGTHS), 8)).astype(np.float32)
    # Segment offsets are the production concatenation order target/source/mask.
    image[:, :LENGTHS[0]] += .1
    image[:, LENGTHS[0]:LENGTHS[0] + LENGTHS[1]] += .3
    image[:, LENGTHS[0] + LENGTHS[1]:] -= .2
    text = rng.normal(0, .6, (2, 5, 10)).astype(np.float32)
    mask = np.array([[1, 0, 1, 1, 0], [1, 1, 0, 1, 1]], dtype=np.bool_)
    REPORT["fixtures"] = dict(seed=SEED, numpy_weights_sha256=digest.hexdigest(),
                              tensor_count=len(weights), parameter_count=sum(v.size for v in weights.values()),
                              image_shape=list(image.shape), text_shape=list(text.shape),
                              text_mask=mask.tolist(), token_order=["target", "source", "mask"],
                              image_grids=GRIDS, segment_lengths=LENGTHS,
                              segment_offsets=[0, LENGTHS[0], LENGTHS[0]+LENGTHS[1]])
    return weights, image, text, mask


def load_weights(candidate, weights, *, allow_missing=False):
    destination = dict(tree_flatten(candidate.parameters()))
    updates, missing = [], []
    for name, fixture in weights.items():
        mapped = mlx_name(name)
        if mapped not in destination:
            missing.append(name)
            continue
        assert tuple(destination[mapped].shape) == fixture.shape, f"shape mismatch for {name} -> {mapped}"
        updates.append((mapped, mx.array(fixture, dtype=mx.float32)))
    mapped_names = {key for key, _ in updates}
    extra = sorted(set(destination) - mapped_names)
    result = dict(torch_count=len(weights), mlx_count=len(destination), loaded_count=len(updates),
                  missing_torch_parameters=missing, extra_mlx_parameters=extra,
                  layout="Both Linear weights are [out_features,in_features]; no transpose")
    assert not extra, result
    assert allow_missing or not missing, result
    candidate.load_weights(updates, strict=not allow_missing)
    mx.eval(candidate.parameters())
    for key, fixture in updates:
        assert np.array_equal(array(dict(tree_flatten(candidate.parameters()))[key]), array(fixture))
    return result


def make_config():
    return SimpleNamespace(height=GRIDS[0][1] * 16, width=GRIDS[0][2] * 16,
                           scheduler=SimpleNamespace(sigmas=[], timesteps=[]))


def run_mlx(candidate, image, text, mask, *, t=.37):
    result = candidate(t=float(t), config=make_config(), hidden_states=mx.array(image),
                       encoder_hidden_states=mx.array(text), encoder_hidden_states_mask=mx.array(mask),
                       cond_image_grid=GRIDS[1:])
    mx.eval(result)
    return result


def run_torch(reference, image, text, mask, *, t=.37):
    with torch.no_grad():
        return reference(hidden_states=torch.from_numpy(image),
                         encoder_hidden_states=torch.from_numpy(text),
                         encoder_hidden_states_mask=torch.from_numpy(mask),
                         timestep=torch.full((image.shape[0],), t, device="cpu"),
                         img_shapes=[GRIDS for _ in range(image.shape[0])], return_dict=False)[0]


def rope_and_time(reference, candidate, image, text, mask):
    torch_rope = reference.pos_embed([GRIDS, GRIDS], max_txt_seq_len=text.shape[1], device=torch.device("cpu"))
    mlx_rope = candidate._compute_rotary_embeddings(mx.array(mask), candidate.pos_embed, GRIDS)
    for stream, expected, actual in zip(("image", "text"), torch_rope, mlx_rope):
        compare(f"rope.{stream}.cos", actual[0], expected.real, atol=2e-6, rtol=2e-6)
        compare(f"rope.{stream}.sin", actual[1], expected.imag, atol=2e-6, rtol=2e-6)
    for idx, (start, count) in enumerate(zip((0, LENGTHS[0], LENGTHS[0]+LENGTHS[1]), LENGTHS)):
        # This checks all rows of each rectangular grid including both edges.
        compare(f"rope.image.segment_{idx}.cos", mlx_rope[0][0][start:start+count],
                torch_rope[0].real[start:start+count], atol=2e-6, rtol=2e-6)
    target_index = np.array([[0] * LENGTHS[0] + [1] * sum(LENGTHS[1:])] * 2, dtype=np.int32)
    actual_index = candidate._compute_modulate_index(2, GRIDS, sum(LENGTHS))
    compare("routing.target_source_mask_index", actual_index, target_index, atol=0, rtol=0)
    times = np.array([.37, .81, 0., 0.], dtype=np.float32)
    t_image = reference.img_in(torch.from_numpy(image))
    m_image = candidate.img_in(mx.array(image))
    t_time = reference.time_text_embed(torch.from_numpy(times), t_image)
    m_time = candidate.time_text_embed(mx.array(times), m_image)
    compare("time.distinct_batch_t_then_zero", m_time, t_time)
    t_x = reference.transformer_blocks[0].img_norm1(t_image)
    m_x = candidate.transformer_blocks[0].img_norm1(m_image)
    t_mod = reference.transformer_blocks[0].img_mod(t_time).chunk(2, dim=-1)[0]
    m_mod = mx.split(candidate.transformer_blocks[0].img_mod_linear(mx.sigmoid(m_time) * m_time), 2, axis=-1)[0]
    t_values, t_gates = reference.transformer_blocks[0]._modulate(t_x, t_mod, torch.from_numpy(target_index))
    m_values, m_gates = candidate.transformer_blocks[0]._modulate(m_x, m_mod, actual_index)
    compare("routing.selected_image_modulation", m_values, t_values)
    compare("routing.selected_image_gates", m_gates, t_gates)
    shift, scale, gate = array(m_mod).reshape(4, 3, -1).transpose(1, 0, 2)
    normed = array(m_x)
    for batch in range(2):
        for segment, start, count in zip(("target", "source", "mask"),
                                         (0, LENGTHS[0], LENGTHS[0]+LENGTHS[1]), LENGTHS):
            row = batch if segment == "target" else batch + 2
            expected = normed[batch, start:start+count] * (1+scale[row]) + shift[row]
            compare(f"routing.batch_{batch}.{segment}.values", m_values[batch, start:start+count], expected)
            compare(f"routing.batch_{batch}.{segment}.gates", m_gates[batch, start:start+count],
                    np.broadcast_to(gate[row], (count, gate.shape[-1])))
    t_txtmod = reference.transformer_blocks[0].txt_mod(t_time[:2])
    m_txtmod = candidate.transformer_blocks[0].txt_mod_linear(mx.sigmoid(m_time[:2]) * m_time[:2])
    compare("routing.text_target_time", m_txtmod, t_txtmod)
    REPORT["routing"] = dict(batch_time_fixture=times.tolist(),
                             image_temporal_routes=["target t", "source 0", "mask 0"],
                             text_temporal_route="target t", final_norm_temporal_route="target t")
    return torch_rope, mlx_rope, torch.from_numpy(target_index), actual_index


def compare_blocks(reference, candidate, image, text, mask, ropecache):
    t_rope, m_rope, t_index, m_index = ropecache
    t_image = reference.img_in(torch.from_numpy(image))
    t_text = reference.txt_in(reference.txt_norm(torch.from_numpy(text)))
    m_image = candidate.img_in(mx.array(image))
    m_text = candidate.txt_in(candidate.txt_norm(mx.array(text)))
    times = np.array([.37, .81, 0., 0.], dtype=np.float32)
    t_time = reference.time_text_embed(torch.from_numpy(times), t_image)
    m_time = candidate.time_text_embed(mx.array(times), m_image)
    joint_mask = torch.from_numpy(np.concatenate([mask, np.ones((2, sum(LENGTHS)), dtype=np.bool_)], axis=1))
    for idx, (t_block, m_block) in enumerate(zip(reference.transformer_blocks, candidate.transformer_blocks)):
        t_text, t_image = t_block(hidden_states=t_image, encoder_hidden_states=t_text,
                                 encoder_hidden_states_mask=None, temb=t_time,
                                 image_rotary_emb=t_rope, modulate_index=t_index,
                                 joint_attention_kwargs={"attention_mask": joint_mask})
        m_text, m_image = m_block(hidden_states=m_image, encoder_hidden_states=m_text,
                                 encoder_hidden_states_mask=mx.array(mask), text_embeddings=m_time,
                                 image_rotary_emb=m_rope, modulate_index=m_index, block_idx=idx)
        compare(f"block.{idx:02d}.image", m_image, t_image)
        compare(f"block.{idx:02d}.text", m_text, t_text)
    t_out = reference.proj_out(reference.norm_out(t_image, t_time[:2]))
    m_out = candidate.proj_out(candidate.norm_out(m_image, m_time[:2]))
    compare("transformer.distinct_batch_times_manual_chain", m_out, t_out)
    REPORT["layer_coverage"] = dict(actual_layers=60, every_image_and_text_block_output_compared=True,
                                    production_layer_count=60, hidden_dim=24,
                                    rationale="Same complete 60-layer graph at reduced channels/head width; tests float32 semantics, not pretrained quality")


def full_and_padding(reference, candidate, image, text, mask):
    t_base = run_torch(reference, image, text, mask)
    m_base = run_mlx(candidate, image, text, mask)
    compare("transformer.full_forward", m_base, t_base)
    altered = text.copy()
    altered[~mask] = np.random.default_rng(SEED + 1).normal(0, 5, altered[~mask].shape)
    t_altered = run_torch(reference, image, altered, mask)
    m_altered = run_mlx(candidate, image, altered, mask)
    compare("mask.perturbed_padding.torch_invariance", t_altered, t_base, atol=3e-6, rtol=3e-6)
    compare("mask.perturbed_padding.mlx_invariance", m_altered, m_base, atol=3e-6, rtol=3e-6)
    compare("mask.perturbed_padding.parity", m_altered, t_altered)
    appended = np.concatenate([text, np.random.default_rng(SEED+2).normal(0, 4, (2, 2, 10)).astype(np.float32)], axis=1)
    append_mask = np.concatenate([mask, np.zeros((2, 2), dtype=np.bool_)], axis=1)
    t_append = run_torch(reference, image, appended, append_mask)
    m_append = run_mlx(candidate, image, appended, append_mask)
    compare("mask.appended_padding.torch_invariance", t_append, t_base, atol=3e-6, rtol=3e-6)
    compare("mask.appended_padding.mlx_invariance", m_append, m_base, atol=3e-6, rtol=3e-6)
    compare("mask.appended_padding.parity", m_append, t_append)
    active_altered = text.copy()
    active_altered[mask] += .8
    t_active = run_torch(reference, image, active_altered, mask)
    m_active = run_mlx(candidate, image, active_altered, mask)
    compare("mask.active_perturbation.parity", m_active, t_active)
    active_effect = float(np.max(np.abs(array(m_active) - array(m_base))))
    assert active_effect > 1e-4, "Active-text negative control is insensitive"
    t_changed = run_torch(reference, image, text, mask, t=.81)
    m_changed = run_mlx(candidate, image, text, mask, t=.81)
    compare("time.changed_target_t.parity", m_changed, t_changed)
    time_effect = float(np.max(np.abs(array(m_changed) - array(m_base))))
    assert time_effect > 1e-4, "Time negative control is insensitive"
    REPORT["negative_controls"] = dict(active_text_max_output_change=active_effect,
                                       target_time_max_output_change=time_effect,
                                       minimum_sensitive_change=1e-4)


def provenance():
    revision = subprocess.check_output(["git", "-C", str(RUNTIME), "rev-parse", "HEAD"], text=True).strip()
    assert revision == RUNTIME_REVISION, revision
    dirty = subprocess.check_output(["git", "-C", str(RUNTIME), "status", "--porcelain"], text=True).strip()
    assert not dirty, f"Frozen runtime checkout changed: {dirty}"
    files = {}
    support_files = {}
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.suffix == ".py" and (path.is_relative_to(RUNTIME) or path == OFFICIAL or path.name == "maskflow_transformer.py"):
            files[str(path.relative_to(ROOT))] = sha(path)
        if path.suffix == ".py" and getattr(module, "__name__", "").startswith("diffusers.") and path != OFFICIAL:
            support_files[getattr(module, "__name__")] = dict(path=str(path), sha256=sha(path))
    return dict(runtime_git_revision=revision, runtime_clean=True,
                official_source_sha256=sha(OFFICIAL), executed_source_hashes=dict(sorted(files.items())),
                diffusers_support_sources=dict(sorted(support_files.items())),
                test_file_sha256=sha(__file__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("port", "native"), default="port")
    parser.add_argument("--output", type=Path, default=AREA / "port/transformer_cpu_validation.json")
    args = parser.parse_args()
    start = time.monotonic()
    REPORT.update(status="running", target=args.target, tiny_model_config=CONFIG,
                  dtype="float32", tolerances=TOLERANCES,
                  cpu_only=dict(mlx_default_device=str(mx.default_device()), torch_default_device=str(torch.get_default_device()),
                                pretrained_weights_loaded=False, model_downloads=False),
                  versions={"python":platform.python_version(), "torch":torch.__version__,
                            "mlx":importlib.metadata.version("mlx"), "numpy":np.__version__,
                            "diffusers":importlib.metadata.version("diffusers")})
    try:
        assert mx.default_device() == mx.cpu
        assert torch.get_default_device().type == "cpu"
        reference = reference_module().QwenImageTransformer2DModel(**CONFIG).to(device="cpu", dtype=torch.float32).eval()
        from mflux.models.qwen.model.qwen_transformer.qwen_transformer import QwenTransformer
        weights, image, text, mask = fixtures(reference)
        native = QwenTransformer(**{**CONFIG, "axes_dims_rope": list(CONFIG["axes_dims_rope"])})
        native_coverage = load_weights(native, weights, allow_missing=True)
        native_failure = None
        try:
            native_output = run_mlx(native, image, text, mask)
            native_numeric = compare("native_baseline.full_forward", native_output,
                                     run_torch(reference, image, text, mask))
        except Exception as exc:
            native_failure = f"{type(exc).__name__}: {exc}"
            native_numeric = None
        REPORT["native_runtime_baseline"] = dict(weight_coverage=native_coverage,
                                                 actual_forward_failure=native_failure,
                                                 numeric_result=native_numeric,
                                                 expected_text_rope_width=text.shape[1],
                                                 native_mask_sum_max=int(mask.sum(axis=1).max()))
        if args.target == "native":
            assert not native_coverage["missing_torch_parameters"], native_coverage
            assert native_failure is None, native_failure
            candidate = native
        else:
            from maskflow_transformer import MaskFlowTransformer
            candidate = MaskFlowTransformer(**{**CONFIG, "axes_dims_rope": list(CONFIG["axes_dims_rope"])})
        REPORT["weight_coverage"] = load_weights(candidate, weights)
        with torch.no_grad():
            ropecache = rope_and_time(reference, candidate, image, text, mask)
            compare_blocks(reference, candidate, image, text, mask, ropecache)
            full_and_padding(reference, candidate, image, text, mask)
        assert all(item["passed"] for key, item in REPORT["comparisons"].items() if not key.startswith("native_baseline."))
        REPORT["status"] = "passed"
        REPORT["limits"] = ["Random tiny weights; does not validate pretrained model or generation quality",
                            "Float32 CPU only; excludes fp16 clipping, quantization and Metal behavior",
                            "Production channel/head sizes and 2511 text encoder/VAE are not executed"]
    except Exception as exc:
        REPORT.update(status="failed", failure=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    finally:
        try:
            REPORT["provenance"] = provenance()
            REPORT["runtime_git_revision"] = REPORT["provenance"]["runtime_git_revision"]
            REPORT["source_sha256"] = {
                str(Path(__file__).resolve().relative_to(ROOT)): sha(__file__),
                **{path: digest for path, digest in REPORT["provenance"]["executed_source_hashes"].items()
                   if path.startswith("experiments/maskflow/port/")},
            }
            successful = [value for key, value in REPORT.get("comparisons", {}).items()
                          if not key.startswith("native_baseline.")]
            if successful:
                REPORT["numeric_summary"] = dict(comparison_count=len(successful),
                                                  max_abs_error=max(value["max_abs_error"] for value in successful),
                                                  max_tolerance_ratio=max(value["max_tolerance_ratio"] for value in successful),
                                                  all_passed=all(value["passed"] for value in successful))
        except Exception as exc:
            REPORT.update(status="failed", provenance_failure=f"{type(exc).__name__}: {exc}")
        REPORT["elapsed_seconds"] = time.monotonic() - start
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(REPORT, indent=2) + "\n")
        print(json.dumps({"status":REPORT["status"], "comparisons":len(REPORT.get("comparisons", {})),
                          "elapsed_seconds":round(REPORT["elapsed_seconds"], 3),
                          "report":str(args.output), "failure":REPORT.get("failure")}), flush=True)
    return 0 if REPORT["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
