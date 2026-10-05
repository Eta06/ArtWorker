#!/usr/bin/env python3
"""CPU-only parity against pinned, AST-extracted VideoX-Fun Torch code.

No checkpoint is read and neither Torch MPS/CUDA nor MLX Metal is selected.
The official control/model forward methods and attention/block functions run
unmodified. Three unavailable diffusers/repository helpers are substituted:
an equivalent CPU SDPA dispatcher, weight-bearing RMSNorm, and the two-linear
SiLU timestep embedder. Every fetched/extracted source SHA256 is recorded.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import sys
import types
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mlx.core as mx

mx.set_default_device(mx.cpu)
torch.set_num_threads(2)
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / ".build/model-research/mflux/src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mlx.utils import tree_flatten
from control import ControlBranch, controlled_forward
from mflux.models.qwen21.model.qwen21_transformer.qwen21_layout import QwenImage21Layout
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer


REVISION = "4b7b6402a1e0f0406bd6801fb66c0a00bd922621"
BASE_NAME = "qwenimage21_transformer2d.py"
CONTROL_NAME = "qwenimage21_transformer2d_control.py"
BASE_NAMES = [
    "apply_rotary_emb_qwen", "QwenImage21TemporalTimesteps",
    "QwenImage21TimestepProjEmbeddings", "QwenImage21ZeroCenterRMSNorm",
    "QwenImage21TextProjection", "QwenImage21SwiGLUFeedForward",
    "QwenImage21AdaLayerNormContinuous", "_select_modulation_rows",
    "_qwenimage21_prefix_segments", "_qwenimage21_project_qkv",
    "_qwenimage21_apply_cache", "_qwenimage21_prepare_qkv",
    "_qwenimage21_block_causal_attention", "QwenImage21AttnProcessor",
    "QwenImage21Attention", "QwenImage21TransformerBlock", "QwenImage21Rope",
]


class CPURMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        return (x * torch.rsqrt(torch.mean(x.square(), dim=-1, keepdim=True) + self.eps)
                * self.weight.float()).to(dtype)


class CPUTimestepEmbedding(nn.Module):
    def __init__(self, in_channels, time_embed_dim, sample_proj_bias=False):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=sample_proj_bias)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=sample_proj_bias)

    def forward(self, x):
        return self.linear_2(F.silu(self.linear_1(x)))


def cpu_attention(query, key, value, dropout_p=0.0, attn_mask=None):
    """VideoX helper takes B,S,H,D; Torch CPU SDPA takes B,H,S,D."""
    return F.scaled_dot_product_attention(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
        attn_mask=attn_mask, dropout_p=dropout_p,
    ).transpose(1, 2)


def upstream_namespace():
    metadata = {"revision": REVISION, "files": {}, "extracted": {}}
    namespace = dict(torch=torch, nn=nn, F=F, math=math, Any=Any, Dict=Dict,
                     List=List, Optional=Optional, Tuple=Tuple, Union=Union,
                     RMSNorm=CPURMSNorm, TimestepEmbedding=CPUTimestepEmbedding,
                     attention=cpu_attention, _IMG_TOKENS_PER_SLOT=4,
                     maybe_allow_in_graph=lambda value: value)

    def execute(file_name, class_names, method_owner=None, method_names=()):
        url = f"https://raw.githubusercontent.com/aigc-apps/VideoX-Fun/{REVISION}/videox_fun/models/{file_name}"
        raw = urllib.request.urlopen(url, timeout=30).read()
        source = raw.decode("utf-8")
        metadata["files"][file_name] = {"url": url, "sha256": hashlib.sha256(raw).hexdigest()}
        parsed = ast.parse(source)
        nodes = []
        for node in parsed.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in class_names:
                nodes.append(node)
                segment = ast.get_source_segment(source, node)
                metadata["extracted"][node.name] = hashlib.sha256(segment.encode()).hexdigest()
            if isinstance(node, ast.ClassDef) and node.name == method_owner:
                for method in node.body:
                    if isinstance(method, ast.FunctionDef) and method.name in method_names:
                        segment = ast.get_source_segment(source, method)
                        metadata["extracted"][f"{method_owner}.{method.name}"] = hashlib.sha256(segment.encode()).hexdigest()
                        method.decorator_list = []
                        method.name = f"official_{method_owner}_{method.name}"
                        nodes.append(method)
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), url, "exec"), namespace)

    execute(BASE_NAME, BASE_NAMES, "QwenImage21Transformer2DModel", ["build_token_metadata"])
    execute(CONTROL_NAME, ["QwenImage21ControlTransformerBlock", "BaseQwenImage21TransformerBlock"],
            "QwenImage21ControlTransformer2DModel", ["forward_control", "forward"])
    metadata["substitutions"] = {
        "attention": "VideoX attention dispatcher replaced by Torch CPU float32 scaled_dot_product_attention; B,S,H,D transposed to B,H,S,D and back; masks unchanged.",
        "RMSNorm": "diffusers RMSNorm replaced by explicit float32 x*rsqrt(mean(x*x)+eps)*learned_weight; no zero-centering.",
        "TimestepEmbedding": "diffusers TimestepEmbedding(in_channels=256,time_embed_dim=32,sample_proj_bias=False) replaced by bias-free Linear->SiLU->Linear with identically named keys.",
        "Transformer2DModelOutput": "Not needed: exact official forward is called with return_dict=False.",
        "sequence_parallel": "Official branches inactive because sp_world_size=1; no distributed substitute executes.",
        "maybe_allow_in_graph": "diffusers graph-registration decorator replaced by identity; eager CPU evaluation only.",
    }
    return namespace, metadata


def make_torch_model(upstream, control_layers, in_channels=8, context_dim=12, control_dim=129):
    model = nn.Module()
    model.inner_dim = 32
    model.config = types.SimpleNamespace(causal_condition=True)
    model.img_in = nn.Linear(in_channels, 32, bias=False)
    model.txt_in = upstream["QwenImage21TextProjection"](context_dim, 32, eps=1e-6)
    model.time_text_embed = upstream["QwenImage21TimestepProjEmbeddings"](32)
    model.modulation = nn.Sequential(nn.SiLU(), nn.Linear(32, 128, bias=False))
    model.pos_embed = upstream["QwenImage21Rope"](10000, [2, 2, 4])
    model.transformer_blocks = nn.ModuleList([
        upstream["BaseQwenImage21TransformerBlock"](32, 4, 8, mlp_ratio=3,
                block_id=control_layers.index(i) if i in control_layers else None)
        for i in range(4)
    ])
    model.norm_out = upstream["QwenImage21AdaLayerNormContinuous"](32, 32, eps=1e-6)
    model.proj_out = nn.Linear(32, in_channels, bias=False)
    model.control_img_in = nn.Linear(control_dim, 32, bias=True)
    model.control_blocks = nn.ModuleList([
        upstream["QwenImage21ControlTransformerBlock"](32, 4, 8, mlp_ratio=3, block_id=i)
        for i in control_layers
    ])
    model.control_layers = list(control_layers)
    model.control_layers_mapping = {i: n for n, i in enumerate(control_layers)}
    model.gradient_checkpointing = False
    model.sp_world_size = 1
    model.sp_world_rank = 0
    model.build_token_metadata = upstream["official_QwenImage21Transformer2DModel_build_token_metadata"]
    model.forward_control = types.MethodType(upstream["official_QwenImage21ControlTransformer2DModel_forward_control"], model)
    model.forward = types.MethodType(upstream["official_QwenImage21ControlTransformer2DModel_forward"], model)
    return model


def randomize(model, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for key, param in model.named_parameters():
            values = torch.randn(param.shape, generator=generator)
            if ".norm_q.weight" in key or ".norm_k.weight" in key:
                param.copy_(1.0 + values * 0.1)
            elif key == "txt_in.text_norm.weight":
                param.copy_(values * 0.1)
            else:
                param.copy_(values * (0.10 if param.ndim == 1 else 0.12))


def load_mlx(torch_model, mlx_base, mlx_branch):
    state = torch_model.state_dict()
    # MLX Sequential names its children under .layers; Torch uses bare indices.
    state = {key.replace("modulation.1.", "modulation.layers.1."): value for key, value in state.items()}
    base_keys = dict(tree_flatten(mlx_base.parameters()))
    branch_keys = dict(tree_flatten(mlx_branch.parameters()))
    assert not set(base_keys) & set(branch_keys)
    for module, keys in [(mlx_base, base_keys), (mlx_branch, branch_keys)]:
        assert all(key in state and tuple(state[key].shape) == tuple(value.shape) for key, value in keys.items())
        module.load_weights([(key, mx.array(state[key].detach().numpy())) for key in keys], strict=True)
    assert set(state) == set(base_keys) | set(branch_keys), set(state) ^ (set(base_keys) | set(branch_keys))
    mx.eval(mlx_base.parameters(), mlx_branch.parameters())
    return {"base_tensor_count": len(base_keys), "control_tensor_count": len(branch_keys),
            "all_torch_keys_loaded_strictly": True,
            "key_rename": {"modulation.1.weight": "modulation.layers.1.weight"}}


def numpy_mx(value):
    mx.eval(value)
    return np.asarray(value)


def error_record(expected, actual):
    expected = expected.detach().cpu().numpy() if isinstance(expected, torch.Tensor) else np.asarray(expected)
    actual = numpy_mx(actual) if isinstance(actual, mx.array) else np.asarray(actual)
    assert expected.shape == actual.shape
    assert np.isfinite(expected).all() and np.isfinite(actual).all()
    diff = np.abs(expected - actual)
    return {"max_abs": float(diff.max()), "mean_abs": float(diff.mean()), "shape": list(actual.shape), "finite": True}


def layout_case(name):
    if name == "target-only":
        return np.array([False, False, False, False]), [(1, 2, 4)]
    if name == "reference-interleaved":
        return np.array([False, True, False, False, False]), [(1, 2, 2), (1, 2, 4)]
    if name == "adjacent-two-references":
        return np.array([False, True, True, False, False]), [(1, 2, 2), (1, 2, 2), (1, 2, 4)]
    raise ValueError(name)


def run_case(upstream, name, layers, seed, padded=False):
    torch_model = make_torch_model(upstream, layers)
    randomize(torch_model, seed)
    mlx_base = Qwen21Transformer(in_channels=8, out_channels=8, num_layers=4,
            attention_head_dim=8, num_attention_heads=4, context_in_dim=12,
            axes_dims_rope=(2, 2, 4), mlp_ratio=3)
    mlx_branch = ControlBranch(dim=32, num_heads=4, head_dim=8, mlp_ratio=3,
            num_layers=4, control_layers=layers, in_channels=129)
    loading = load_mlx(torch_model, mlx_base, mlx_branch)
    slots, shapes = layout_case(name)
    layout = QwenImage21Layout.create(mx.array(slots), shapes, (2, 2, 4))
    rng = np.random.default_rng(seed + 100)
    image_len = sum(math.prod(shape) for shape in shapes)
    hidden = rng.standard_normal((1, image_len, 8), dtype=np.float32)
    prompt = rng.standard_normal((1, len(slots), 12), dtype=np.float32)
    context = rng.standard_normal((1, image_len, 129), dtype=np.float32)
    # Binary representable time keeps timestep trig parity distinguishable
    # from branch/injection math instead of inflating a drift source.
    time = np.array([0.5], dtype=np.float32)
    prompt_mask = np.ones((1, len(slots)), dtype=bool)
    if padded:
        prompt_mask[0, -1] = False
    mask = prompt_mask if padded else None
    img_mask = np.concatenate([slots, np.ones(layout.target_tokens // 4, dtype=bool)])[None]
    repeats = np.where(img_mask[0], 4, 1)
    image_pad_mask = torch.tensor(np.repeat(img_mask[0], repeats))
    image_ids, target_mask = torch_model.build_token_metadata(image_pad_mask, shapes)
    segments = upstream["_qwenimage21_prefix_segments"](image_ids, layout.prefix_length)
    assert segments == layout.segments
    np.testing.assert_array_equal(target_mask.numpy(), np.asarray(layout.target_mask))
    rope = torch_model.pos_embed(shapes, image_pad_mask, torch.device("cpu"))
    rope_error = max(error_record(rope.real, layout.rope[0])["max_abs"],
                     error_record(rope.imag, layout.rope[1])["max_abs"])
    args = (torch.tensor(hidden), torch.tensor(prompt), torch.tensor(time), [shapes], torch.tensor(img_mask))
    mask_t = torch.tensor(mask) if mask is not None else None
    mx_args = [mx.array(hidden), mx.array(prompt), mx.array(time)]
    mx_mask = mx.array(mask) if mask is not None else None
    results = {}
    with torch.no_grad():
        # Independent official block and stacked control chain comparison.
        images = torch_model.img_in(args[0])
        text = torch_model.txt_in(args[1])
        joint = torch.cat([text, torch.zeros((1, layout.target_tokens // 4, 32))], dim=1)
        joint = joint.repeat_interleave(torch.tensor(repeats), dim=1)
        joint[:, image_pad_mask] = images
        control_joint = torch.zeros_like(joint)
        control_joint[:, image_pad_mask] = torch_model.control_img_in(torch.tensor(context))
        times = torch.cat([args[2], torch.zeros(1)])
        temb = torch_model.time_text_embed(times, joint)
        modulation = torch_model.modulation(temb)
        local_images = mlx_base.img_in(mx_args[0])
        local_text = mlx_base.txt_in(mx_args[1])
        local_time = mlx_base.time_text_embed(mx.array(times.numpy()), mx.float32)
        results["image_projection"] = error_record(images, local_images)
        results["text_projection"] = error_record(text, local_text)
        results["time_projection"] = error_record(temb, local_time)
        results["shared_modulation"] = error_record(modulation, mlx_base.modulation(local_time))
        key_valid = None
        if mask_t is not None:
            key_valid = torch.ones((1, len(image_pad_mask)), dtype=torch.bool)
            key_valid[:, ~image_pad_mask] = mask_t[:, ~torch.tensor(slots)]
        kwargs = dict(modulation=modulation, rotary_emb=rope, target_token_mask=target_mask,
                segments=segments, key_valid=key_valid)
        mx_joint, mx_control = mx.array(joint.numpy()), mx.array(control_joint.numpy())
        mx_mod, mx_key_valid = mx.array(modulation.numpy()), mx.array(key_valid.numpy()) if key_valid is not None else None
        stacked = control_joint
        running = mx_control
        for index, (t_block, m_block) in enumerate(zip(torch_model.control_blocks, mlx_branch.control_blocks)):
            stacked = t_block(stacked, joint, **kwargs)
            hint, running = m_block.forward_control(running, mx_joint, mx_mod, layout.target_mask,
                    layout, layout.rope, mx_key_valid)
            results[f"control_block_{index}_hint"] = error_record(stacked[index], hint)
            results[f"control_block_{index}_running"] = error_record(stacked[-1], running)
        official_hints = torch_model.forward_control(control_joint, joint, kwargs)
        mlx_hints = mlx_branch.forward_hints(mx_control, mx_joint, mx_mod, layout.target_mask,
                layout, layout.rope, mx_key_valid)
        assert len(official_hints) == len(mlx_hints) == len(layers)
        for index, (t_hint, m_hint) in enumerate(zip(official_hints, mlx_hints)):
            results[f"branch_hint_{index}"] = error_record(t_hint, m_hint)
        assert all(float(h.abs().max()) > 0.01 for h in official_hints), "Random control skips must be nonzero"
        base_joint = joint.clone()
        m_joint = mx_joint
        for index, (t_block, m_block) in enumerate(zip(torch_model.transformer_blocks, mlx_base.transformer_blocks)):
            base_joint = t_block(base_joint, hints=official_hints, context_scale=0.7, **kwargs)
            m_joint, _ = m_block.forward_reference(m_joint, mx_mod, layout.target_mask,
                    layout, layout.rope, key_valid=mx_key_valid)
            if index in layers:
                m_joint = m_joint + mlx_hints[layers.index(index)] * 0.7
            results[f"base_block_{index}_after_injection"] = error_record(base_joint, m_joint)
        for scale in [0.0, 0.7, 1.0, -0.25]:
            official = torch_model(*args, encoder_hidden_states_mask=mask_t,
                    control_context=torch.tensor(context), control_context_scale=scale,
                    return_dict=False)[0][:, -layout.target_tokens:]
            actual = controlled_forward(mlx_base, mlx_branch, *mx_args, layout, mx.array(context),
                    control_scale=scale, encoder_hidden_states_mask=mx_mask)
            results[f"full_forward_scale_{scale}"] = error_record(official, actual)
            if scale == 0:
                mlx_base_result = mlx_base.forward_reference(mx_args[0], mx_args[1], mx_args[2], layout,
                        encoder_hidden_states_mask=mx_mask)
                official_base_result = torch_model(*args, encoder_hidden_states_mask=mask_t,
                        control_context=None, return_dict=False)[0][:, -layout.target_tokens:]
                results["scale_zero_vs_local_base"] = error_record(numpy_mx(mlx_base_result), actual)
                results["scale_zero_vs_official_base"] = error_record(official_base_result, actual)
            if scale == 1:
                nonzero_output = numpy_mx(actual)
        output_without_control = mlx_base.forward_reference(mx_args[0], mx_args[1], mx_args[2], layout,
                encoder_hidden_states_mask=mx_mask)
        control_effect = float(np.abs(nonzero_output - numpy_mx(output_without_control)).max())
        assert control_effect > 0.01, "Control must affect final target predictions"
        try:
            controlled_forward(mlx_base, mlx_branch, *mx_args, layout, mx.array(context), cache=[])
        except ValueError as exc:
            cache_rejected = "cache" in str(exc).lower()
        else:
            cache_rejected = False
        assert cache_rejected
    return {"layout": name, "padded_prompt": padded, "control_layers": layers,
            "prefix_length": layout.prefix_length, "target_tokens": layout.target_tokens,
            "image_tokens": image_len, "segments": segments, "rope_max_abs": rope_error,
            "loading": loading, "control_effect_max_abs": control_effect,
            "cache_rejected": cache_rejected, "comparisons": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("cpu_parity.json"))
    args = parser.parse_args()
    upstream, source = upstream_namespace()
    results = [
        run_case(upstream, "target-only", [0, 2], 129, False),
        run_case(upstream, "target-only", [0, 2], 130, True),
        run_case(upstream, "reference-interleaved", [0, 2], 131, True),
        run_case(upstream, "adjacent-two-references", [0, 3], 132, False),
    ]
    maximum = max(record["max_abs"] for case in results for record in case["comparisons"].values())
    block_maximum = max(record["max_abs"] for case in results
                        for name, record in case["comparisons"].items()
                        if name.startswith(("control_block_", "branch_hint_", "base_block_")))
    report = {"status": "passed" if maximum < 2e-5 else "failed", "device": "CPU only",
              "torch_version": torch.__version__, "dtype": "float32", "checkpoint_weights_used": False,
              "scope": "Exact upstream control/base blocks, chain, hints and model.forward; full joint execution only; random trained-like weights; no image-quality or quantized checkpoint validation.",
              "upstream": source, "cases": results, "max_abs_error": maximum,
              "block_chain_and_injection_max_abs_error": block_maximum,
              "threshold": 2e-5,
              "control_port_sha256": hashlib.sha256(Path(__file__).with_name("control.py").read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "cases": len(results), "max_abs_error": maximum,
                      "output": str(args.output.resolve()), "device": report["device"]}))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
