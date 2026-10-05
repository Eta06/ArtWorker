#!/usr/bin/env python3
"""Numerical author-Xinsir Union -> installed Diffusers Union CPU parity.

The exact 47 KB pinned author module is cached beside this test and imported
without modifications. Its hardcoded 320-channel/8-head task attention remains
intact, including batch_first=False applied directly to B,task,C tensors.
Model depth, spatial size and shared text-conditioning dimensions are reduced
in matching constructor configs; original RGB conditioning widths are kept.
All parameter keys and values are loaded strictly into the native model.
No real checkpoint, GPU, SDXL UNet, VAE, text encoder or runtime edit is used.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
import diffusers
from diffusers import ControlNetUnionModel


REVISION = "b48420576eac63c04388cb65fb74513cbd17405a"
AUTHOR_URL = f"https://raw.githubusercontent.com/xinsir6/ControlNetPlus/{REVISION}/models/controlnet_union.py"
AUTHOR_SHA256 = "26e2540b0c3ebde1a77dfbb95ca2e61504c10bed0f101703552e1d76a05832ca"
ROOT = Path(__file__).resolve().parent
torch.set_num_threads(2)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def import_author():
    path = ROOT / f"original_controlnet_union_{REVISION[:8]}.py"
    if not path.exists():
        raw = urllib.request.urlopen(AUTHOR_URL, timeout=30).read()
        if hashlib.sha256(raw).hexdigest() != AUTHOR_SHA256:
            raise ValueError("Pinned author source hash mismatch")
        path.write_bytes(raw)
    assert sha(path) == AUTHOR_SHA256
    spec = importlib.util.spec_from_file_location("xinsir_union_cpu_parity_original", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, path


def flatten_tensors(value):
    if torch.is_tensor(value):
        return [value.detach().cpu().clone()]
    if isinstance(value, (tuple, list)):
        return [tensor for child in value for tensor in flatten_tensors(child)]
    return []


def capture(model):
    records, handles = {}, []
    names = ["conv_in", "time_proj", "time_embedding", "add_time_proj", "add_embedding",
             "control_type_proj", "control_add_embedding", "controlnet_cond_embedding",
             "transformer_layes.0", "transformer_layes.0.attn", "spatial_ch_projs",
             "down_blocks.0", "mid_block", "controlnet_down_blocks.0",
             "controlnet_down_blocks.1", "controlnet_mid_block"]
    modules = dict(model.named_modules())
    for name in names:
        def hook(module, inputs, output, key=name):
            records.setdefault(key, []).append({"inputs": flatten_tensors(inputs), "outputs": flatten_tensors(output)})
        handles.append(modules[name].register_forward_hook(hook))
    return records, handles


def compare(expected, actual):
    assert expected.shape == actual.shape
    assert expected.dtype == actual.dtype == torch.float32
    assert torch.isfinite(expected).all() and torch.isfinite(actual).all()
    difference = (expected - actual).abs()
    return {"shape": list(expected.shape), "max_abs": float(difference.max()),
            "mean_abs": float(difference.mean()), "expected_abs_max": float(expected.abs().max()),
            "actual_abs_max": float(actual.abs().max())}


def compare_captures(original, native):
    assert set(original) == set(native)
    result = {}
    for name in original:
        assert len(original[name]) == len(native[name])
        comparisons = []
        for orig_call, native_call in zip(original[name], native[name]):
            for kind in ["inputs", "outputs"]:
                assert len(orig_call[kind]) == len(native_call[kind])
                comparisons.extend({"kind": kind, **compare(a, b)}
                                   for a, b in zip(orig_call[kind], native_call[kind]))
        result[name] = comparisons
    return result


def shared_models(author):
    config = dict(
        in_channels=4, conditioning_channels=3,
        down_block_types=("CrossAttnDownBlock2D",), block_out_channels=(320,),
        layers_per_block=1, norm_num_groups=32,
        cross_attention_dim=16, attention_head_dim=8,
        transformer_layers_per_block=1,
        conditioning_embedding_out_channels=(16, 32, 96, 256),
        addition_embed_type="text_time", addition_time_embed_dim=4,
        projection_class_embeddings_input_dim=8 + 6 * 4,
        num_control_type=8, controlnet_conditioning_channel_order="rgb",
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(712934)
        original = author.ControlNetModel_Union(**config).cpu().float().eval()
        native = ControlNetUnionModel(**config, num_trans_channel=320,
                num_trans_head=8, num_trans_layer=1, num_proj_channel=320).cpu().float().eval()
        # Author checkpoints train these zero-initialized projections. Set
        # distinguishable nonzero synthetic values before copying the state.
        with torch.no_grad():
            for module in [original.controlnet_cond_embedding.conv_out,
                           original.spatial_ch_projs, *original.controlnet_down_blocks,
                           original.controlnet_mid_block]:
                fan_in = module.weight.shape[1] * math.prod(module.weight.shape[2:])
                module.weight.normal_(std=0.25 / math.sqrt(fan_in))
                if module.bias is not None:
                    module.bias.normal_(std=0.025)
        state = original.state_dict()
        native_state = native.state_dict()
        assert set(state) == set(native_state)
        assert list(state) == list(native_state), "Parameter registration order changed"
        assert all(a.shape == native_state[key].shape and a.dtype == native_state[key].dtype
                   for key, a in state.items())
        result = native.load_state_dict(state, strict=True)
        assert not result.missing_keys and not result.unexpected_keys
        assert all(torch.equal(value, native.state_dict()[key]) for key, value in state.items())
    assert original.task_embedding.shape == native.task_embedding.shape == (8, 320)
    assert not original.transformer_layes[0].attn.batch_first
    assert not native.transformer_layes[0].attn.batch_first
    assert all(param.device.type == "cpu" and param.dtype == torch.float32
               for model in [original, native] for param in model.parameters())
    data_hash = hashlib.sha256()
    for key, value in state.items():
        data_hash.update(key.encode())
        data_hash.update(value.numpy().tobytes())
    metadata = {"shared_config": config, "original_parameter_count": sum(v.numel() for v in original.parameters()),
                "native_parameter_count": sum(v.numel() for v in native.parameters()),
                "state_tensor_count": len(state), "strict_keys_shapes_dtypes_and_order_match": True,
                "all_state_values_exact_after_strict_load": True, "state_key_value_sha256": data_hash.hexdigest(),
                "state_keys": list(state), "task_attention_batch_first": False,
                "task_attention_dim": 320, "task_attention_heads": 8,
                "nonzero_synthetic_cond_projection_fuser_and_residual_heads": True}
    return original, native, metadata


def case_inputs(batch, indices):
    generator = torch.Generator(device="cpu").manual_seed(2991)
    # Distinct CFG batch elements prevent a batch/order false pass.
    sample = torch.randn((2, 4, 4, 4), generator=generator)[:batch]
    prompt = torch.randn((2, 5, 16), generator=generator)[:batch]
    pooled = torch.randn((2, 8), generator=generator)[:batch]
    time_ids = torch.tensor([[1152, 512, 0, 0, 1152, 512], [1152, 512, 16, 8, 1152, 512]], dtype=torch.float32)[:batch]
    controls = []
    for i in range(8):
        yy, xx = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
        # Unequal RGB channels, spatial patterns, tasks and CFG branches.
        image = torch.stack([(xx.float() + i) / 40, (yy.float() + 2 * i) / 48,
                             ((xx + yy + i) % 13).float() / 13])[None]
        image = torch.cat([image, image * 0.65 + 0.2], dim=0)[:batch]
        image[:, :, :8] = 0  # Literal black regeneration pixels, not -1 normalized.
        controls.append(image)
    control_type = torch.zeros((batch, 8), dtype=torch.float32)
    control_type[:, indices] = 1
    return sample, prompt, pooled, time_ids, controls, control_type


def run_case(original, native, batch, indices, scale=1.0, guess=False):
    sample, prompt, pooled, time_ids, controls, control_type = case_inputs(batch, indices)
    assert indices == sorted(indices), "Original author iterates nonzero task indices in ascending order"
    added = {"text_embeds": pooled, "time_ids": time_ids}
    orig_added = {**added, "control_type": control_type}
    original_capture, orig_handles = capture(original)
    native_capture, native_handles = capture(native)
    try:
        with torch.inference_mode():
            expected = original(sample, torch.tensor(500.0), encoder_hidden_states=prompt,
                controlnet_cond_list=controls, conditioning_scale=scale,
                added_cond_kwargs=orig_added, guess_mode=guess, return_dict=False)
            actual = native(sample, torch.tensor(500.0), encoder_hidden_states=prompt,
                controlnet_cond=[controls[i] for i in indices], control_type=control_type,
                control_type_idx=indices, conditioning_scale=float(scale),
                added_cond_kwargs=added, guess_mode=guess, return_dict=False)
    finally:
        for handle in [*orig_handles, *native_handles]:
            handle.remove()
    comparisons = [compare(a, b) for a, b in zip(expected[0], actual[0])]
    assert len(expected[0]) == len(actual[0]) == 2
    mid = compare(expected[1], actual[1])
    assert all(float(value.abs().max()) > 0.01 for value in [*expected[0], expected[1]]), "Residuals must be meaningful and nonzero"
    stages = compare_captures(original_capture, native_capture)
    max_error = max([mid["max_abs"], *[c["max_abs"] for c in comparisons],
                     *[record["max_abs"] for stage in stages.values() for record in stage]])
    return {"batch": batch, "control_indices": indices, "conditioning_scale": scale,
            "guess_mode": guess, "control_type": control_type.tolist(),
            "task_attention_actual_input_shape": list(original_capture["transformer_layes.0.attn"][0]["inputs"][0].shape),
            "control_rgb_range": [min(float(controls[i].min()) for i in indices), max(float(controls[i].max()) for i in indices)],
            "residuals": comparisons, "mid_residual": mid, "stage_comparisons": stages,
            "max_abs_error": max_error}, actual, (sample, prompt, pooled, time_ids, controls, control_type)


def task_attention_axis_oracle(original):
    """Explicit QKV/softmax oracle preserves author's S=B, N=tasks axis usage."""
    module = original.transformer_layes[0].attn
    generator = torch.Generator(device="cpu").manual_seed(453)
    value = torch.randn((2, 3, 320), generator=generator)
    with torch.inference_mode():
        actual = module(value, value, value, need_weights=False)[0]
        q, k, v = torch.nn.functional.linear(value, module.in_proj_weight, module.in_proj_bias).chunk(3, dim=-1)
        # MHA expects sequence first. Here sequence length is CFG batch2;
        # logical task count3 is MHA's batch axis, exactly as author trained it.
        sequence, mha_batch, dim = value.shape
        head_dim = dim // module.num_heads
        def heads(tensor):
            return tensor.reshape(sequence, mha_batch, module.num_heads, head_dim).permute(1, 2, 0, 3)
        q, k, v = map(heads, (q, k, v))
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
        output = torch.matmul(torch.softmax(scores, dim=-1), v)
        output = output.permute(2, 0, 1, 3).reshape(sequence, mha_batch, dim)
        expected = torch.nn.functional.linear(output, module.out_proj.weight, module.out_proj.bias)
        logical_batch_first, _ = module(value.transpose(0, 1), value.transpose(0, 1), value.transpose(0, 1), need_weights=False)
        corrected_axis_delta = float((actual - logical_batch_first.transpose(0, 1)).abs().max())
    assert corrected_axis_delta > 0.01
    error = compare(expected, actual)
    assert error["max_abs"] < 2e-6
    return {"explicit_qkv_softmax": error, "original_axes_preserved": True,
            "actual_sequence_axis": "input dimension0 (CFG sample batch)",
            "actual_mha_batch_axis": "input dimension1 (control/task slots)",
            "batch_first_or_transpose_changes_output_max_abs": corrected_axis_delta,
            "author_semantics_not_reinterpreted": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "original_native_cpu_parity.json")
    args = parser.parse_args()
    author, author_path = import_author()
    original, native, models = shared_models(author)
    cases, retained = [], {}
    # Single repaint7 CFG batch2 is first and is the priority acceptance case.
    for batch, indices, scale, guess in [
        (2, [7], 1.0, False), (1, [7], 1.0, False),
        (2, [3], 1.0, False), (2, [7], 0.35, False),
        (2, [7], 0.7, True), (2, [3, 7], 1.0, False),
    ]:
        record, output, _ = run_case(original, native, batch, indices, scale, guess)
        cases.append(record)
        retained[(batch, tuple(indices), scale, guess)] = output
        print(json.dumps({"phase": "case", "batch": batch, "indices": indices,
                          "scale": scale, "guess": guess, "max_abs_error": record["max_abs_error"]}), flush=True)
    task_delta = float((retained[(2, (7,), 1.0, False)][1] - retained[(2, (3,), 1.0, False)][1]).abs().max())
    assert task_delta > 0.01, "Task index/control_type must affect final outputs"
    batch_delta = float((retained[(2, (7,), 1.0, False)][1][:1] - retained[(1, (7,), 1.0, False)][1]).abs().max())
    # Shared task embeddings dominate the task-7 tensor, so this full-model
    # batch effect can be small. The separate randomized axis oracle provides
    # a strong independent counterexample to batch_first reinterpretation.
    assert batch_delta > 1e-7, f"Actual cross-sample effect must exceed roundoff: {batch_delta}"
    baseline = retained[(2, (7,), 1.0, False)]
    scaled = retained[(2, (7,), 0.35, False)]
    scale_errors = [compare(a * 0.35, b) for a, b in zip([*baseline[0], baseline[1]], [*scaled[0], scaled[1]])]
    axis = task_attention_axis_oracle(original)
    maximum = max([case["max_abs_error"] for case in cases] + [axis["explicit_qkv_softmax"]["max_abs"]])
    native_path = Path(inspect.getsourcefile(ControlNetUnionModel))
    report = {"status": "passed" if maximum < 2e-5 else "failed", "device": "CPU only",
              "dtype": "float32", "torch_version": torch.__version__, "diffusers_version": diffusers.__version__,
              "checkpoint_weights_used": False, "original_module_modified": False,
              "original_source": {"url": AUTHOR_URL, "revision": REVISION, "path": str(author_path), "sha256": sha(author_path)},
              "native_source": {"path": str(native_path), "sha256": sha(native_path)},
              "test_sha256": sha(__file__), "models": models, "cases": cases,
              "task_attention_axis_oracle": axis, "task3_vs_task7_output_delta_max_abs": task_delta,
              "batch2_first_vs_batch1_output_delta_max_abs": batch_delta,
              "single_control_output_scaling": scale_errors, "max_abs_error": maximum,
              "threshold": 2e-5,
              "scope_limits": [
                  "Shallow one320-channel downblock+midblock, tiny4x4latent/32x32RGB; real full-depth checkpoint not loaded.",
                  "Original author module imports common UNet/attention/embedding primitives from current installed Diffusers0.39.0; historical dependency-version drift is not tested.",
                  "Float32 CPU only; real MPS/halfprecision kernel behavior and dark generated-image cause remain unverified.",
                  "Multicontrol parity only equal scale1 with original ascending task order; heterogeneous native multicontrol scaling differs from original API and is not claimed.",
              ]}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "max_abs_error": maximum,
                      "cases": len(cases), "output": str(args.output.resolve())}))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
