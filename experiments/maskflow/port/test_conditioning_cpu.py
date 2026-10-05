#!/usr/bin/env python3
"""Weight-free CPU tests of exact MaskFlow VLM and VAE affine plumbing.

Executes the installed, pinned Diffusers prompt method as the oracle with fake
hidden states, and actual local2511 tokenizer plus FastTorch image processing.
No model weights are loaded; no GPU operations are permitted by this script.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import inspect
import json
import subprocess
import time
from pathlib import Path
from types import MethodType, SimpleNamespace

import mlx.core as mx

mx.set_default_device(mx.cpu)

import numpy as np
import torch
from diffusers import QwenImageEditPlusPipeline
from PIL import Image
from transformers import AutoTokenizer

from maskflow_conditioning import (
    EDIT_TEMPLATE,
    MaskFlowEncoder,
    PROCESSOR_CONFIG_SHA256,
    PROCESSOR_CONFIG_URL,
    build_official_processor,
    encode_cfg_branches,
    format_prompt,
    prepare_vlm_conditions,
    tokenize_conditioning,
)
from maskflow_math import prepare_condition_images, normalize_vae_latents, denormalize_vae_latents
from maskflow_vae import MaskFlowVAE
from mflux.models.qwen.model.qwen_text_encoder.qwen_vision_language_encoder import QwenVisionLanguageEncoder
from mflux.models.qwen.model.qwen_text_encoder.qwen_encoder import QwenEncoder
from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
from mflux.models.qwen.tokenizer.qwen_image_processor import QwenImageProcessor
from mflux.models.qwen.tokenizer.qwen_vision_language_processor import QwenVisionLanguageProcessor
from mflux.models.qwen.tokenizer.qwen_vision_language_tokenizer import QwenVisionLanguageTokenizer

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RESEARCH = HERE.parent / "research" / "2026-10-05"
EXPECTED_RUNTIME = "99fb94dd3eaa9dd1931cd3cd8eae1ae3e20f2ef3"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float().numpy()
    return np.asarray(value.astype(mx.float32), dtype=np.float32)


def exact(actual, expected, label: str) -> dict:
    a, b = array(actual), array(expected)
    if a.shape != b.shape:
        raise AssertionError(f"{label}: shape{a.shape} != {b.shape}")
    diff = np.abs(a - b)
    result = {"shape": list(a.shape), "max_abs_error": float(diff.max(initial=0)), "exact": bool(np.array_equal(a, b))}
    if not result["exact"]:
        raise AssertionError(f"{label}: {result}")
    return result


def hidden_numpy(input_ids, pixel_values):
    ids = np.asarray(input_ids, dtype=np.float32)
    pixels = torch.from_numpy(np.asarray(pixel_values, dtype=np.float32)).to(torch.bfloat16).float().numpy()
    signal = np.float32(pixels.mean(dtype=np.float64))
    position = np.broadcast_to(np.arange(ids.shape[1], dtype=np.float32), ids.shape)
    features = np.stack((np.remainder(ids, 251) / 128, np.remainder(ids, 17) / 32, np.remainder(position, 31) / 16, np.full_like(ids, signal)), axis=-1)
    return torch.from_numpy(features).to(torch.bfloat16)


class FakeTorchVL:
    dtype = torch.bfloat16

    def __init__(self):
        self.captured = None

    def __call__(self, **kwargs):
        self.captured = kwargs
        hidden = hidden_numpy(kwargs["input_ids"].numpy(), kwargs["pixel_values"].float().numpy())
        return SimpleNamespace(hidden_states=[hidden])


class FakeMLXVL:
    def __init__(self):
        self.calls = []

    def __call__(self, input_ids, attention_mask, pixel_values, image_grid_thw):
        self.calls.append(np.asarray(input_ids))
        hidden = hidden_numpy(np.asarray(input_ids), array(pixel_values)).float().numpy()
        return mx.array(hidden, dtype=mx.bfloat16)


def official_prompt(prompt, conditions, raw_tokenizer):
    encoder = FakeTorchVL()
    pipe = SimpleNamespace(
        _execution_device=torch.device("cpu"),
        text_encoder=encoder,
        processor=build_official_processor(raw_tokenizer),
        prompt_template_encode=EDIT_TEMPLATE,
        prompt_template_encode_start_idx=64,
    )
    pipe._extract_masked_hidden = MethodType(QwenImageEditPlusPipeline._extract_masked_hidden, pipe)
    images = [torch.from_numpy(conditions[key]).to(torch.bfloat16) for key in ("source", "mask")]
    embeds, mask = QwenImageEditPlusPipeline._get_qwen_prompt_embeds(pipe, [prompt], images, torch.device("cpu"))
    return embeds, mask, encoder.captured


def vision_spy_test() -> dict:
    """Execute upstream Torch method and actual subclass with tiny CPU spies."""
    import textwrap
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel

    # Compile the unmodified method body without unrelated return decorators.
    tree = ast.parse(textwrap.dedent(inspect.getsource(Qwen2_5_VLModel.get_image_features)))
    method = tree.body[0]
    method.decorator_list = []
    method.returns = None
    for argument in (*method.args.posonlyargs, *method.args.args, *method.args.kwonlyargs):
        argument.annotation = None
    method.args.kwarg.annotation = None
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(tree), "official_qwen25vl_get_image_features", "exec"), namespace)

    class TorchVisionSpy:
        dtype = torch.bfloat16
        spatial_merge_size = 2

        def __call__(self, pixels, grid_thw=None, **kwargs):
            self.received_dtype = pixels.dtype
            self.received_values = pixels.clone()
            count = int((grid_thw.prod(-1) // 4).sum())
            return SimpleNamespace(pooler_output=torch.arange(count * 4).reshape(count, 4).to(pixels.dtype))

    class MLXVisionSpy:
        spatial_merge_size = 2

        def __call__(self, pixels, grid):
            self.received_dtype = pixels.dtype
            self.received_values = pixels
            count = int((grid.prod(axis=-1) // 4).sum().item())
            return mx.arange(count * 4).reshape(count, 4).astype(pixels.dtype)

    grid_np = np.array([[1, 4, 4], [1, 6, 8], [1, 2, 10]], dtype=np.int32)
    pixels_np = np.linspace(-1.9, 1.7, 84 * 12, dtype=np.float32).reshape(84, 12)
    torch_spy, port_spy, stock_spy = TorchVisionSpy(), MLXVisionSpy(), MLXVisionSpy()
    expected = namespace["get_image_features"](SimpleNamespace(visual=torch_spy), torch.from_numpy(pixels_np), torch.from_numpy(grid_np)).pooler_output
    # Prove the intended in-place class reassignment on an actual tiny encoder.
    existing_encoder = QwenEncoder(vocab_size=8, hidden_size=28, num_hidden_layers=0)
    identity_before = id(existing_encoder)
    weights_before = existing_encoder.embed_tokens.weight
    existing_encoder.__class__ = MaskFlowEncoder
    existing_encoder.visual = port_spy
    assert id(existing_encoder) == identity_before
    assert existing_encoder.embed_tokens.weight is weights_before
    port_result = existing_encoder.get_image_features(mx.array(pixels_np), mx.array(grid_np))
    stock_result = QwenEncoder.get_image_features(SimpleNamespace(visual=stock_spy), mx.array(pixels_np), mx.array(grid_np))
    assert port_spy.received_dtype == mx.bfloat16 and torch_spy.received_dtype == torch.bfloat16
    assert stock_spy.received_dtype == mx.float32
    checks = [exact(actual, reference, f"vision_split_{index}") for index, (actual, reference) in enumerate(zip(port_result, expected))]
    assert len(port_result) == len(expected) == len(stock_result) == 3
    exact(port_spy.received_values, torch_spy.received_values, "visual_pixel_boundary")
    return {
        "official_visual_input_dtype": str(torch_spy.received_dtype),
        "port_visual_input_dtype": str(port_spy.received_dtype),
        "stock_visual_input_dtype": str(stock_spy.received_dtype),
        "split_sizes": [len(item) for item in port_result],
        "split_checks": checks,
        "existing_encoder_identity_and_weights_preserved": True,
        "official_get_image_features_sha256": hashlib.sha256(inspect.getsource(Qwen2_5_VLModel.get_image_features).encode()).hexdigest(),
        "no_model_weights_or_large_allocation": True,
    }


def run() -> dict:
    started = time.monotonic()
    torch.set_num_threads(2)
    runtime = HERE.parent / "runtime" / "mlx-gen"
    runtime_sha = subprocess.check_output(["git", "-C", str(runtime), "rev-parse", "HEAD"], text=True).strip()
    assert runtime_sha == EXPECTED_RUNTIME
    assert importlib.metadata.version("diffusers") == "0.37.1"
    track = REPO / "experiments" / "evaluation" / "inputs" / "track3"
    canvas = Image.open(track / "canvas.png").convert("RGB")
    mask = Image.open(track / "mask.png").convert("RGB")
    canvas_array = np.array(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255
    mask_array = np.array(mask, dtype=np.float32).transpose(2, 0, 1)[None] / 255
    prepared = prepare_condition_images(canvas_array, mask_array, backend="numpy", dtype="bfloat16")
    conditions = prepared["vlm_conditions"]
    fallback = prepare_vlm_conditions(canvas, prepared["mask"])
    tests = {"fallback_source": exact(torch.from_numpy(fallback["source"]), torch.from_numpy(conditions["source"]), "sourcepreprocess"), "fallback_mask": exact(torch.from_numpy(fallback["mask"]), torch.from_numpy(conditions["mask"]), "maskpreprocess")}
    tok_path = HERE.parent / "models" / "qwen-image-edit-2511-4bit" / "tokenizer"
    raw = AutoTokenizer.from_pretrained(tok_path, local_files_only=True)
    tokenizer = QwenVisionLanguageTokenizer(processor=QwenVisionLanguageProcessor(raw), use_picture_prefix=True)
    assert tokenizer.edit_template_picture_prefix == EDIT_TEMPLATE
    prompt_file = REPO / "experiments" / "qwen" / "controlnet_runs" / "2026-10-05" / "track3-localized40-outpaint" / "localized_reference_metrics.json"
    prompt = json.loads(prompt_file.read_text())["prompt"]
    fake_encoder = FakeMLXVL()
    encoder = QwenVisionLanguageEncoder(encoder=fake_encoder)
    branches = encode_cfg_branches(prompt=prompt, negative_prompt="", vlm_conditions=conditions, qwen_vl_tokenizer=tokenizer, qwen_vl_encoder=encoder)
    oracle_metadata = {}
    for name, text in (("pm", prompt), ("nm", "")):
        expected, expected_mask, captured = official_prompt(text, conditions, raw)
        tests[f"{name}_hidden_and_trim"] = exact(branches[name]["prompt_embeds"], expected, name)
        tests[f"{name}_attention_mask"] = exact(branches[name]["prompt_embeds_mask"], expected_mask, f"{name}mask")
        tokenized = tokenize_conditioning(text, conditions, raw_tokenizer=raw)
        for key in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw"):
            assert np.array_equal(tokenized[key], captured[key].numpy()), key
        oracle_metadata[name] = tokenized["metadata"]
    assert len(fake_encoder.calls) == 2
    assert not np.array_equal(fake_encoder.calls[0], fake_encoder.calls[1])
    tests["empty_negative_independent"] = {"calls": len(fake_encoder.calls), "input_ids_differ": True}
    long_text = "Blue sky. " * 600
    long_tokens = tokenize_conditioning(long_text, conditions, raw_tokenizer=raw)
    assert long_tokens["metadata"]["embedding_token_count"] > 1024
    tests["no_forced_1024_truncation"] = {"embedding_token_count": long_tokens["metadata"]["embedding_token_count"], "oracle_parameter_unused": "max_sequence_length is unused in Diffusers0.37.1 encode_prompt"}
    # Quantify why the stock PIL/uint8 path must not replace actual tensors.
    png_images = [Image.fromarray(np.rint(conditions[key][0].transpose(1, 2, 0) * 255).astype(np.uint8)) for key in ("source", "mask")]
    png_values, png_grid = QwenImageProcessor().preprocess(png_images)
    native = tokenize_conditioning(prompt, conditions, raw_tokenizer=raw)
    assert np.array_equal(png_grid, native["image_grid_thw"])
    pixel_diff = np.abs(png_values - native["pixel_values"])
    assert pixel_diff.max() > 1
    tests["png_path_changes_vlm"] = {"max_abs_difference": float(pixel_diff.max()), "mean_abs_difference": float(pixel_diff.mean()), "same_grid": True, "png_path_used_by_port": False}
    quantized_conditions = {"source": conditions["source"], "mask": np.rint(conditions["mask"] * 255) / 255}
    quantized_mask_tokens = tokenize_conditioning(prompt, quantized_conditions, raw_tokenizer=raw)
    mask_quantization_diff = np.abs(quantized_mask_tokens["pixel_values"] - native["pixel_values"])
    assert mask_quantization_diff.max() > 0
    tests["softmask_png_quantization_alone"] = {
        "mask_value_max_abs_difference_before_bf16_cast": float(np.abs(quantized_conditions["mask"] - conditions["mask"]).max()),
        "processor_pixel_max_abs_difference_same_fast_backend": float(mask_quantization_diff.max()),
        "processor_pixel_mean_abs_difference_same_fast_backend": float(mask_quantization_diff.mean()),
        "exact_equivalence": False,
        "png_path_used_by_port": False,
    }
    original_source_rows = prepared["mask"][0, 0, 320:832, 0]
    affected_rows = np.flatnonzero(original_source_rows > 0)
    assert np.array_equal(affected_rows, np.r_[np.arange(24), np.arange(488, 512)])
    tests["native_softmask_inside_original_boundary"] = {
        "affected_source_rows_at_top": 24,
        "affected_source_rows_at_bottom": 24,
        "native_zero_mask_source_rows": 464,
        "hard_source_paste_is_separate": True,
    }
    # Execute actual stock VAE affine on synthetic posterior moments, with no weights.
    rng = np.random.default_rng(20261005)
    means = rng.normal(size=(1, 16, 1, 4, 6)).astype(np.float32)
    means_pt = torch.from_numpy(means).to(torch.bfloat16)
    config = json.loads((RESEARCH / "base" / "vae" / "config.json").read_text())
    mean_pt = torch.tensor(config["latents_mean"], dtype=torch.bfloat16).reshape(1, 16, 1, 1, 1)
    std_pt = torch.tensor(config["latents_std"], dtype=torch.bfloat16).reshape(1, 16, 1, 1, 1)
    official_norm = (means_pt - mean_pt) / std_pt
    means_mx = mx.array(means, dtype=mx.bfloat16)
    tests["vae_bf16_affine_normalize"] = exact(normalize_vae_latents(means_mx, backend="mlx"), official_norm, "vaenorm")
    tests["vae_bf16_affine_denormalize"] = exact(denormalize_vae_latents(mx.array(official_norm.float().numpy(), dtype=mx.bfloat16), backend="mlx"), official_norm * std_pt + mean_pt, "vaedenorm")
    moments = mx.concatenate([means_mx, mx.ones_like(means_mx) * 7], axis=1)
    fake_vae = SimpleNamespace(encoder=lambda _: moments, quant_conv=lambda value: value, decoder=lambda value: value, post_quant_conv=lambda value: value)
    tests["vae_argmax_first16"] = exact(MaskFlowVAE.encode(fake_vae, mx.zeros((1, 3, 1, 32, 48), dtype=mx.bfloat16)), official_norm, "vaeargmax")
    tests["vae_decode_raw_affine"] = exact(MaskFlowVAE.decode(fake_vae, normalize_vae_latents(means_mx, backend="mlx")), official_norm * std_pt + mean_pt, "vaedecode")
    stock = QwenVAE.encode(fake_vae, mx.zeros((1, 3, 1, 32, 48), dtype=mx.bfloat16))
    mx.eval(stock)
    stock_difference = np.abs(array(stock) - array(official_norm))
    assert stock.dtype == mx.float32 and stock_difference.max() > 0
    tests["stock_vae_affine_requires_override"] = {"stock_dtype": str(stock.dtype), "official_dtype": "torch.bfloat16", "max_abs_difference": float(stock_difference.max()), "port_override_exact": True}
    tests["visual_input_bf16_and_exact_split"] = vision_spy_test()
    source_paths = [Path(__file__), HERE / "maskflow_conditioning.py", HERE / "maskflow_math.py", HERE / "maskflow_vae.py", RESEARCH / "code" / "pipelines" / "qwenimage" / "qwenimage_maskflow.py", RESEARCH / "code" / "pipelines" / "qwenimage" / "qwenimage_edit_plus.py", RESEARCH / "code" / "data_module" / "utils.py", Path(inspect.getfile(QwenImageEditPlusPipeline)), Path(inspect.getfile(type(build_official_processor(raw).image_processor))), Path(inspect.getfile(QwenVisionLanguageEncoder)), Path(inspect.getfile(QwenEncoder)), Path(inspect.getfile(QwenVAE)), tok_path / "tokenizer.json", tok_path / "tokenizer_config.json"]
    return {
        "status": "passed", "device": "CPU", "full_model_weights_loaded": False,
        "elapsed_seconds": time.monotonic() - started,
        "runtime_revision": runtime_sha,
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "transformers", "diffusers", "mlx", "mlx-gen", "numpy", "Pillow")},
        "processor_config": {"url": PROCESSOR_CONFIG_URL, "sha256": PROCESSOR_CONFIG_SHA256, "pinned_payload_bytes": 826, "fast_tensor_rescale_preserved": True},
        "source_sha256": {str(path): sha(path) for path in source_paths},
        "native_conditioning": branches["metadata"],
        "tests": tests,
        "limitations": ["No pretrained VL/VAE forward or quantized-weight quality parity has been run.", "Tests cover actual tokenizer/processor, official prompt trimming, independent empty negative branch and BF16 affine only.", "The official fast processor divides already [0,1] BF16 images by255; this implementation preserves that upstream recipe rather than correcting it.", "Native processed mask permits24pixels inside the source boundary; hard source paste is a separate output operation."],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=HERE / "conditioning_cpu_validation.json")
    args = parser.parse_args()
    try:
        report = run()
    except Exception as error:
        report = {"status": "failed", "device": "CPU", "error": f"{type(error).__name__}: {error}"}
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        raise
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "report": str(args.report), "tests": len(report["tests"]), "elapsed_seconds": report["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
