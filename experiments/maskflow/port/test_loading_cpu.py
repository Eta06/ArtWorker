"""Tiny CPU gates for strict saved loading and the official 2511 VAE affine.

This test uses temporary synthetic safetensors only. It never reads full model
payloads, constructs the full VAE, or selects the GPU.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import inspect
import json
from pathlib import Path
import resource
import struct
import sys
import subprocess
import tempfile
import traceback
import unittest
from unittest.mock import patch

import mlx.core as mx

mx.set_default_device(mx.cpu)

from mlx import nn
from mlx.utils import tree_flatten
import numpy as np
import torch

from maskflow_loading import load_saved_component, normalize_saved_shapes
from maskflow_vae import MaskFlowVAE
from maskflow_adapter import read_safetensors_header
from maskflow_transformer import MaskFlowTransformer
from mflux.models.qwen.model.qwen_text_encoder.qwen_text_encoder import QwenTextEncoder
from mflux.models.qwen.model.qwen_text_encoder.qwen_vision_transformer import VisionTransformer
from mflux.models.qwen.model.qwen_vae.qwen_image_rms_norm import QwenImageRMSNorm
from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNTIME = ROOT / "runtime/mlx-gen"
PIN = "99fb94dd3eaa9dd1931cd3cd8eae1ae3e20f2ef3"
VAE_CONFIG = ROOT / "research/2026-10-05/base/vae/config.json"
OFFICIAL_AFFINE = ROOT / "research/2026-10-05/code/pipelines/qwenimage/qwenimage_edit_plus.py"
SOURCES = [HERE / "maskflow_loading.py", HERE / "maskflow_vae.py",
           HERE / "maskflow_math.py", HERE / "maskflow_transformer.py",
           HERE / "maskflow_adapter.py", Path(__file__).resolve(),
           RUNTIME / "src/mflux/models/qwen/model/qwen_vae/qwen_vae.py",
           VAE_CONFIG, OFFICIAL_AFFINE]
REPORT = {"schema_version": 1, "status": "running", "device": "cpu",
          "gpu_used": False, "full_checkpoint_loaded": False,
          "full_vae_tensor_payload_materialized": False, "seed": 261005,
          "created_at_utc": datetime.now(timezone.utc).isoformat(), "checks": []}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def as_float(value):
    mx.eval(value)
    return np.asarray(value.astype(mx.float32))


def raw_numpy(value):
    mx.eval(value)
    # BF16 has no native NumPy buffer dtype; FP32 carries its exact values.
    return as_float(value) if value.dtype == mx.bfloat16 else np.asarray(value)


class TinyMixed(nn.Module):
    def __init__(self):
        super().__init__()
        self.q4 = nn.Linear(64, 8, bias=True)
        self.q8 = nn.Linear(64, 8, bias=False)
        self.norm_out = nn.LayerNorm(8, affine=True, bias=True)
        self.dense = nn.Linear(8, 3, bias=True)

    def __call__(self, x):
        return self.dense(self.norm_out(self.q4(x) + self.q8(x)))


def mixed_weights():
    """Explicit BF16 seed weights, then saved module-specific Q4 and Q8."""
    rng = np.random.default_rng(REPORT["seed"])
    model = TinyMixed()
    values = {}
    for key, current in tree_flatten(model.parameters()):
        value = rng.normal(0.0, 0.4, size=current.shape).astype(np.float32)
        if key == "norm_out.weight":
            value = 1.0 + value * 0.05
        if key == "norm_out.bias":
            value = np.array([0.125, -0.25, 0.375, -0.5, 0.625, -0.75, 0.875, -1.0], np.float32)
        values[key] = mx.array(value, dtype=mx.bfloat16)
    model.load_weights(list(values.items()), strict=True)
    nn.quantize(model, class_predicate=lambda name, module:
                {"bits": 4 if name == "q4" else 8, "group_size": 64}
                if name in ("q4", "q8") else False)
    weights = dict(tree_flatten(model.parameters()))
    mx.eval(weights)
    return model, weights


def write_shards(root, weights, *, omit=(), additions=None, one_shard=False):
    component = Path(root) / "tiny"
    component.mkdir()
    selected = {key: value for key, value in weights.items() if key not in omit}
    selected.update(additions or {})
    keys = sorted(selected)
    if one_shard:
        mx.save_safetensors(str(component / "model.safetensors"), selected)
    else:
        mid = len(keys) // 2
        for index, group in enumerate((keys[:mid], keys[mid:]), 1):
            mx.save_safetensors(str(component / f"model-{index:05d}-of-00002.safetensors"),
                                {key: selected[key] for key in group})
    return component


class LoadingGate(unittest.TestCase):
    def setUp(self):
        self.original, self.weights = mixed_weights()

    def test_mixed_saved_quantization_and_exact_values(self):
        with tempfile.TemporaryDirectory(prefix="maskflow-tiny-loader-") as root:
            write_shards(root, self.weights)
            loaded, metadata = load_saved_component(root, "tiny", TinyMixed)
            self.assertIsInstance(loaded.q4, nn.QuantizedLinear)
            self.assertIsInstance(loaded.q8, nn.QuantizedLinear)
            self.assertEqual(loaded.q4.bits, 4)
            self.assertEqual(loaded.q8.bits, 8)
            self.assertEqual(loaded.q4.group_size, 64)
            self.assertEqual(loaded.q8.group_size, 64)
            self.assertIsInstance(loaded.dense, nn.Linear)
            self.assertEqual(loaded.dense.weight.dtype, mx.bfloat16)
            self.assertEqual(loaded.norm_out.bias.dtype, mx.bfloat16)
            actual = dict(tree_flatten(loaded.parameters()))
            self.assertEqual(set(actual), set(self.weights))
            for key, expected in self.weights.items():
                self.assertEqual(actual[key].dtype, expected.dtype, key)
                np.testing.assert_array_equal(raw_numpy(actual[key]), raw_numpy(expected), err_msg=key)
            x = mx.array(np.random.default_rng(1005).normal(size=(2, 64)), dtype=mx.bfloat16)
            expected, observed = self.original(x), loaded(x)
            np.testing.assert_array_equal(as_float(observed), as_float(expected))
            self.assertFalse(tree_flatten(loaded.trainable_parameters()))
            REPORT["mixed_loading"] = {**metadata, "quantization_bits": {"q4": 4, "q8": 8},
                "unquantized_dtype": "bfloat16", "all_saved_tensor_values_bit_exact": True,
                "forward_max_absolute_error": float(np.max(np.abs(as_float(observed) - as_float(expected)))),
                "all_parameters_frozen": True}

    def test_learned_norm_bias_sidecar_is_exact_and_used(self):
        learned = self.weights["norm_out.bias"]
        self.assertTrue(np.any(as_float(learned) != 0))
        with tempfile.TemporaryDirectory(prefix="maskflow-tiny-sidecar-") as root:
            write_shards(root, self.weights, omit=("norm_out.bias",))
            sidecar = Path(root) / "official_norm_bias.safetensors"
            mx.save_safetensors(str(sidecar), {"norm_out.bias": learned})
            loaded, metadata = load_saved_component(root, "tiny", TinyMixed, sidecar=sidecar)
            np.testing.assert_array_equal(as_float(loaded.norm_out.bias), as_float(learned))
            x = mx.array(np.random.default_rng(37).normal(size=(3, 64)), dtype=mx.bfloat16)
            np.testing.assert_array_equal(as_float(loaded(x)), as_float(self.original(x)))
            learned_output = as_float(loaded(x)).copy()
            loaded.norm_out.bias = mx.zeros_like(learned)
            zero_bias_output = as_float(loaded(x))
            self.assertGreater(float(np.max(np.abs(learned_output - zero_bias_output))), 0.01)
            REPORT["sidecar"] = {"tensor": "norm_out.bias", "dtype": "bfloat16",
                "nonzero_learned_values_preserved_exactly": True, "source_fixture_zero_filled": False,
                "forward_sensitive_to_learned_bias": True, "tensor_count": metadata["tensor_count"]}

    def test_missing_tensor_rejected(self):
        with tempfile.TemporaryDirectory(prefix="maskflow-tiny-missing-") as root:
            write_shards(root, self.weights, omit=("dense.bias",))
            with self.assertRaisesRegex(ValueError, "tensor coverage failed.*dense.bias"):
                load_saved_component(root, "tiny", TinyMixed)

    def test_missing_learned_bias_rejected_without_sidecar(self):
        with tempfile.TemporaryDirectory(prefix="maskflow-tiny-missing-bias-") as root:
            write_shards(root, self.weights, omit=("norm_out.bias",))
            with self.assertRaisesRegex(ValueError, "tensor coverage failed.*norm_out.bias"):
                load_saved_component(root, "tiny", TinyMixed)

    def test_extra_tensor_rejected(self):
        with tempfile.TemporaryDirectory(prefix="maskflow-tiny-extra-") as root:
            write_shards(root, self.weights, additions={"alien.weight": mx.zeros((1,))})
            with self.assertRaisesRegex(ValueError, "tensor coverage failed.*alien.weight"):
                load_saved_component(root, "tiny", TinyMixed)

    def test_shape_mismatch_rejected(self):
        with tempfile.TemporaryDirectory(prefix="maskflow-tiny-shape-") as root:
            write_shards(root, self.weights, additions={"dense.bias": mx.zeros((4,), dtype=mx.bfloat16)})
            with self.assertRaisesRegex(ValueError, "Shape mismatch tiny.dense.bias"):
                load_saved_component(root, "tiny", TinyMixed)

    def test_duplicate_shard_keys_rejected(self):
        with tempfile.TemporaryDirectory(prefix="maskflow-tiny-duplicate-") as root:
            component = write_shards(root, self.weights, one_shard=True)
            mx.save_safetensors(str(component / "other.safetensors"), {"dense.bias": self.weights["dense.bias"]})
            with self.assertRaisesRegex(ValueError, "Duplicate saved tensor keys"):
                load_saved_component(root, "tiny", TinyMixed)

    def test_sidecar_collision_rejected(self):
        with tempfile.TemporaryDirectory(prefix="maskflow-tiny-collision-") as root:
            write_shards(root, self.weights)
            sidecar = Path(root) / "collision.safetensors"
            mx.save_safetensors(str(sidecar), {"norm_out.bias": self.weights["norm_out.bias"]})
            with self.assertRaisesRegex(ValueError, "Sidecar must supply only omitted"):
                load_saved_component(root, "tiny", TinyMixed, sidecar=sidecar)

    def test_unsupported_quantization_rejected(self):
        with tempfile.TemporaryDirectory(prefix="maskflow-tiny-unsupported-") as root:
            # A 4-word packed width and one group scale imply Q2, not Q4/Q8.
            write_shards(root, self.weights, additions={"q4.weight": mx.zeros((8, 4), dtype=mx.uint32)})
            with self.assertRaisesRegex(ValueError, "Unsupported saved quantization.*2"):
                load_saved_component(root, "tiny", TinyMixed)

    def test_scale_without_packed_weight_rejected(self):
        with tempfile.TemporaryDirectory(prefix="maskflow-tiny-no-packed-") as root:
            write_shards(root, self.weights, omit=("q4.weight",))
            with self.assertRaisesRegex(ValueError, "Saved quantized module cannot be constructed: q4"):
                load_saved_component(root, "tiny", TinyMixed)

    def test_nonquantizable_module_scale_rejected(self):
        with tempfile.TemporaryDirectory(prefix="maskflow-tiny-bad-module-") as root:
            write_shards(root, self.weights, additions={"norm_out.scales": mx.ones((8, 1), dtype=mx.bfloat16)})
            with self.assertRaisesRegex(ValueError, "Saved quantized module cannot be constructed: norm_out"):
                load_saved_component(root, "tiny", TinyMixed)

    def test_no_shards_rejected(self):
        with tempfile.TemporaryDirectory(prefix="maskflow-tiny-empty-") as root:
            with self.assertRaisesRegex(FileNotFoundError, "No pinned local shards"):
                load_saved_component(root, "tiny", TinyMixed)


def local_component_headers(component):
    """Actual local prefix + JSON reads; no mx.load or tensor byte reads."""
    root = ROOT / "models/qwen-image-edit-2511-4bit" / component
    index_path = root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())["weight_map"]
    paths = sorted(root.glob("*.safetensors"))
    if {path.name for path in paths} != set(index.values()):
        raise ValueError(f"Actual {component} shard set differs from pinned index")
    headers, receipts = {}, []
    for path in paths:
        header = read_safetensors_header(path)
        with path.open("rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
        entries = {key: value for key, value in header.items() if key != "__metadata__"}
        if set(headers).intersection(entries):
            raise ValueError(f"Duplicate actual {component} header tensor names")
        for key in entries:
            if index.get(key) != path.name:
                raise ValueError(f"Actual {component} header/index shard mismatch: {key}")
        end = 0
        widths = {"BF16": 2, "F32": 4, "U32": 4}
        for key, value in sorted(entries.items(), key=lambda pair: pair[1]["data_offsets"][0]):
            start, stop = value["data_offsets"]
            if start != end or stop - start != int(np.prod(value["shape"])) * widths[value["dtype"]]:
                raise ValueError(f"Invalid header data extent {component}.{key}")
            end = stop
        if path.stat().st_size != 8 + header_size + end:
            raise ValueError(f"Actual {component} shard size disagrees with its header")
        headers.update(entries)
        receipts.append({"path": str(path), "header_size_bytes": header_size,
                         "header_semantics_sha256": hashlib.sha256(json.dumps(header, sort_keys=True).encode()).hexdigest(),
                         "file_size_bytes": path.stat().st_size, "tensor_count": len(entries),
                         "tensor_payload_read": False})
    if set(index) != set(headers):
        raise ValueError(f"Actual {component} header tensor coverage differs from its index")
    return headers, receipts, index_path


def header_expected_shapes(model, header):
    """Derive quantized shapes from lazy full module dimensions, no quantize."""
    original = {key: tuple(value.shape) for key, value in tree_flatten(model.parameters())}
    modules = dict(model.named_modules())
    expected = dict(original)
    bit_counts = Counter()
    for key, value in header.items():
        if not key.endswith(".scales"):
            continue
        module_path = key.removesuffix(".scales")
        weight_key = module_path + ".weight"
        module = modules.get(module_path)
        if not isinstance(module, (nn.Linear, nn.Embedding)) or weight_key not in original:
            raise ValueError(f"Saved quantized header module unsupported: {module_path}")
        shape = original[weight_key]
        if len(shape) != 2 or shape[-1] % 64:
            raise ValueError(f"Full module is incompatible with group64 quantization: {module_path}")
        packed = header[weight_key]
        bits = packed["shape"][-1] * 32 // shape[-1]
        if bits not in (4, 8) or packed["dtype"] != "U32":
            raise ValueError(f"Full module has unsupported packed quantization: {module_path}")
        expected[weight_key] = shape[:-1] + (shape[-1] * bits // 32,)
        expected[key] = shape[:-1] + (shape[-1] // 64,)
        expected[module_path + ".biases"] = expected[key]
        if value["dtype"] != "BF16" or header[module_path + ".biases"]["dtype"] != "BF16":
            raise ValueError(f"Affine quantization scale/offset dtype differs: {module_path}")
        bit_counts[bits] += 1
    return expected, modules, bit_counts


class FullArchitectureHeaderGate(unittest.TestCase):
    def test_actual_saved_headers_cover_full_runtime_architectures(self):
        components = []
        dynamic_sources = {}
        constructors = {"transformer": lambda: MaskFlowTransformer(zero_cond_t=True),
                        "text_encoder": QwenTextEncoder, "vae": MaskFlowVAE}
        for component, constructor in constructors.items():
            with self.subTest(component=component):
                header, shards, index_path = local_component_headers(component)
                dynamic_sources[str(index_path)] = sha(index_path)
                sidecar_header = {}
                if component == "transformer":
                    sidecar_path = ROOT / "models/original-sidecar/norm_out.linear.bias.safetensors"
                    sidecar_header = {key: value for key, value in read_safetensors_header(sidecar_path).items()
                                      if key != "__metadata__"}
                    self.assertEqual(set(sidecar_header), {"norm_out.linear.bias"})
                    self.assertEqual(sidecar_header["norm_out.linear.bias"]["dtype"], "BF16")
                    self.assertEqual(sidecar_header["norm_out.linear.bias"]["shape"], [6144])
                    self.assertFalse(set(sidecar_header).intersection(header))
                    provenance_path = sidecar_path.with_suffix(".provenance.json")
                    dynamic_sources[str(provenance_path)] = sha(provenance_path)
                combined = {**header, **sidecar_header}
                rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                # Constructors create only lazy shape graphs. Any explicit
                # materialization or quantization attempt fails this CPU gate.
                with patch.object(mx, "eval", side_effect=AssertionError("Full lazy parameters cannot be evaluated")), \
                     patch.object(mx, "load", side_effect=AssertionError("Full tensor payload cannot be loaded")), \
                     patch.object(nn, "quantize", side_effect=AssertionError("Full random parameters cannot be quantized")):
                    model = constructor()
                    if component == "text_encoder":
                        model.encoder.visual = VisionTransformer()
                    expected, modules, bits = header_expected_shapes(model, combined)
                    class ShapeOnly:
                        def __init__(self, shape):
                            self.shape = tuple(shape)

                        def reshape(self, shape):
                            if np.prod(shape) != np.prod(self.shape):
                                raise ValueError("Shape-only reshape changed element count")
                            return ShapeOnly(shape)
                    normalized, shape_receipt = normalize_saved_shapes(
                        model, {key: ShapeOnly(value["shape"]) for key, value in combined.items()})
                    self.assertEqual(set(expected), set(combined),
                                     f"{component}: missing={sorted(set(expected)-set(combined))}, "
                                     f"extra={sorted(set(combined)-set(expected))}")
                    flattened_norms = []
                    shape_errors = []
                    for key, shape in expected.items():
                        observed = tuple(combined[key]["shape"])
                        if observed == shape:
                            continue
                        module_path, leaf = key.rsplit(".", 1)
                        module = modules.get(module_path)
                        # Saved VAE RMS coefficients are flat; their only
                        # legal layout bridge adds singleton broadcast axes.
                        if (component == "vae" and leaf == "weight" and isinstance(module, QwenImageRMSNorm)
                            and observed == (shape[0],) and all(size == 1 for size in shape[1:])):
                            flattened_norms.append({"tensor": key, "saved_shape": list(observed),
                                                    "runtime_shape": list(shape), "value_preserving_reshape": True})
                        else:
                            shape_errors.append({"tensor": key, "expected": list(shape), "saved": list(observed)})
                    self.assertFalse(shape_errors, shape_errors)
                    self.assertEqual({key: tuple(value.shape) for key, value in normalized.items()}, expected,
                                     "Actual strict shape bridge differs from full architecture/header shapes")
                    self.assertEqual({entry["tensor"] for entry in shape_receipt["vae_rms_broadcast_reshapes"]},
                                     {entry["tensor"] for entry in flattened_norms})
                    self.assertTrue(all(entry["values_changed"] is False
                                        for entry in shape_receipt["vae_rms_broadcast_reshapes"]))
                    if component == "transformer":
                        self.assertTrue(model.zero_cond_t)
                        self.assertEqual(len(model.transformer_blocks), 60)
                        self.assertEqual(set(expected) - set(header), {"norm_out.linear.bias"})
                    elif component == "text_encoder":
                        self.assertEqual(len(model.encoder.layers), 28)
                        self.assertEqual(len(model.encoder.visual.blocks), 32)
                    for key, value in combined.items():
                        if key.endswith(".weight") and not key.removesuffix(".weight") + ".scales" in combined:
                            self.assertIn(value["dtype"], ("BF16", "F32"), key)
                    components.append({"component": component, "actual_saved_tensor_count": len(header),
                        "sidecar_tensor_count": len(sidecar_header), "full_expected_tensor_count": len(expected),
                        "missing_tensors_after_sidecar": [], "unexpected_tensors": [], "shape_errors": [],
                        "quantized_module_counts": {str(key): value for key, value in sorted(bits.items())},
                        "vae_norm_layout_bridges": flattened_norms,
                        "dtype_counts": dict(Counter(value["dtype"] for value in combined.values())),
                        "lazy_architecture_peak_rss_increase_bytes": max(0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss-rss_before),
                        "actual_shards": shards, "parameters_evaluated": False, "parameters_quantized": False})
                del model, expected, modules
                gc.collect()
        REPORT["full_architecture_headers"] = {"components": components,
            "actual_local_safetensors_headers_only": True, "tensor_payload_read": False,
            "full_architectures_constructed_lazily_on_cpu": True,
            "payload_checksums_verified_by_this_gate": False}
        REPORT["header_source_sha256"] = dynamic_sources


class TinyRMSNorms(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_norm = QwenImageRMSNorm(8, images=True)
        self.video_norm = QwenImageRMSNorm(8, images=False)
        self.dense = nn.Linear(8, 2, bias=False)


class RMSNormShapeGate(unittest.TestCase):
    def setUp(self):
        self.reference = TinyRMSNorms()
        self.reference.image_norm.weight = mx.array([0.125, -0.25, 0.375, -0.5, 0.625, -0.75, 0.875, -1],
                                                    dtype=mx.bfloat16).reshape(8, 1, 1)
        self.reference.video_norm.weight = mx.array([1, 0.75, 0.5, 0.25, -0.25, -0.5, -0.75, -1],
                                                    dtype=mx.bfloat16).reshape(8, 1, 1, 1)
        self.reference.dense.weight = mx.array(np.arange(16, dtype=np.float32).reshape(2, 8) / 16,
                                               dtype=mx.bfloat16)
        self.weights = dict(tree_flatten(self.reference.parameters()))
        mx.eval(self.weights)
        self.saved = {key: value.reshape(8) if key.endswith("_norm.weight") else value
                      for key, value in self.weights.items()}

    def test_flat_norm_coefficients_reshape_without_value_change(self):
        with tempfile.TemporaryDirectory(prefix="maskflow-tiny-rms-shape-") as root:
            write_shards(root, self.saved)
            loaded, receipt = load_saved_component(root, "tiny", TinyRMSNorms)
            parameters = dict(tree_flatten(loaded.parameters()))
            for key, value in parameters.items():
                self.assertEqual(value.shape, self.weights[key].shape)
                self.assertEqual(value.dtype, mx.bfloat16)
                np.testing.assert_array_equal(as_float(value).ravel(), as_float(self.saved[key]).ravel())
            self.assertEqual({entry["tensor"] for entry in receipt["vae_rms_broadcast_reshapes"]},
                             {"image_norm.weight", "video_norm.weight"})
            x4 = mx.array(np.random.default_rng(8).normal(size=(1, 8, 3, 4)), dtype=mx.bfloat16)
            x5 = x4[:, :, None]
            np.testing.assert_array_equal(as_float(loaded.image_norm(x4)), as_float(self.reference.image_norm(x4)))
            np.testing.assert_array_equal(as_float(loaded.video_norm(x5)), as_float(self.reference.video_norm(x5)))
            REPORT["rms_norm_reshape"] = {"passed": True, "coefficient_values_bit_exact": True,
                "image_forward_bit_exact": True, "video_forward_bit_exact": True,
                "receipt": receipt["vae_rms_broadcast_reshapes"]}

    def test_wrong_norm_channel_count_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="maskflow-tiny-rms-wrong-channels-") as root:
            write_shards(root, self.saved, additions={"image_norm.weight": mx.ones((9,), dtype=mx.bfloat16)})
            with self.assertRaisesRegex(ValueError, "Shape mismatch tiny.image_norm.weight"):
                load_saved_component(root, "tiny", TinyRMSNorms)

    def test_ordinary_weights_are_not_reshaped_even_when_element_count_matches(self):
        with tempfile.TemporaryDirectory(prefix="maskflow-tiny-rms-no-general-reshape-") as root:
            write_shards(root, self.saved, additions={"dense.weight": self.saved["dense.weight"].reshape(16)})
            with self.assertRaisesRegex(ValueError, "Shape mismatch tiny.dense.weight"):
                load_saved_component(root, "tiny", TinyRMSNorms)


class CapturingIdentity(nn.Module):
    def __init__(self):
        super().__init__()
        self.observed = None

    def __call__(self, value):
        self.observed = value
        return value


class MomentsEncoder(CapturingIdentity):
    def __init__(self, moments):
        super().__init__()
        self.moments = moments

    def __call__(self, value):
        self.observed = value
        return self.moments


def stub_vae(moments):
    # __new__ and nn.Module.__init__ allocate only these tiny stub arrays. The
    # parent constructor is deliberately forbidden to prevent a full VAE load.
    with patch.object(QwenVAE, "__init__", side_effect=AssertionError("Full VAE construction forbidden")):
        model = MaskFlowVAE.__new__(MaskFlowVAE)
        nn.Module.__init__(model)
        model.encoder = MomentsEncoder(moments)
        model.quant_conv = CapturingIdentity()
        model.post_quant_conv = CapturingIdentity()
        model.decoder = CapturingIdentity()
    return model


class VAEAffineGate(unittest.TestCase):
    def test_official_model_dtype_affine_and_posterior_mode(self):
        config = json.loads(VAE_CONFIG.read_text())
        self.assertEqual(config["z_dim"], 16)
        cases = []
        for dtype_name, mx_dtype, torch_dtype in (("bfloat16", mx.bfloat16, torch.bfloat16),
                                                 ("float32", mx.float32, torch.float32)):
            for ndim in (4, 5):
                with self.subTest(dtype=dtype_name, input_ndim=ndim):
                    rng = np.random.default_rng(261005 + ndim)
                    latent_values = rng.normal(0, 2, (1, 16, 1, 4, 6)).astype(np.float32)
                    logvar = rng.normal(20, 3, latent_values.shape).astype(np.float32)
                    moments = mx.array(np.concatenate((latent_values, logvar), axis=1), dtype=mx_dtype)
                    model = stub_vae(moments)
                    pixels = mx.zeros((1, 3, 4, 6) if ndim == 4 else (1, 3, 1, 4, 6), dtype=mx_dtype)
                    output = model.encode(pixels)
                    torch_latents = torch.tensor(latent_values, dtype=torch_dtype)
                    mean = torch.tensor(config["latents_mean"], dtype=torch_dtype).view(1, 16, 1, 1, 1)
                    std = torch.tensor(config["latents_std"], dtype=torch_dtype).view(1, 16, 1, 1, 1)
                    # Exact equations and dtype locations from pinned official
                    # encode_image, independently evaluated with eager Torch.
                    expected = (torch_latents - mean) / std
                    observed_np = as_float(output)
                    expected_np = expected.float().numpy()
                    if dtype_name == "bfloat16":
                        np.testing.assert_array_equal(observed_np, expected_np)
                    else:
                        np.testing.assert_allclose(observed_np, expected_np, rtol=0, atol=1e-6)
                    self.assertEqual(output.dtype, mx_dtype)
                    self.assertEqual(model.encoder.observed.shape, (1, 3, 1, 4, 6))
                    self.assertEqual(model.quant_conv.observed.shape, (1, 32, 1, 4, 6))
                    # Changing only posterior log variance cannot change mode.
                    model.encoder.moments = mx.concatenate((moments[:, :16], mx.full_like(moments[:, 16:], -30)), axis=1)
                    np.testing.assert_array_equal(as_float(model.encode(pixels)), observed_np)
                    encoded_input = output[:, :, 0] if ndim == 4 else output
                    decoded = model.decode(encoded_input)
                    decode_expected = expected * std + mean
                    decoded_np = as_float(decoded)
                    expected_decode_np = decode_expected.float().numpy()
                    if dtype_name == "bfloat16":
                        np.testing.assert_array_equal(decoded_np, expected_decode_np)
                    else:
                        np.testing.assert_allclose(decoded_np, expected_decode_np, rtol=0, atol=1e-6)
                    self.assertEqual(decoded.dtype, mx_dtype)
                    self.assertEqual(model.post_quant_conv.observed.shape, (1, 16, 1, 4, 6))
                    np.testing.assert_array_equal(as_float(model.decoder.observed), decoded_np)
                    cases.append({"dtype": dtype_name, "input_ndim": ndim,
                                  "encode_max_absolute_error": float(np.max(np.abs(observed_np - expected_np))),
                                  "decode_max_absolute_error": float(np.max(np.abs(decoded_np - expected_decode_np))),
                                  "posterior_mode_independent_of_log_variance": True,
                                  "single_frame_axis_inserted_or_preserved": True})
        REPORT["vae_affine"] = {"official_equations_source": str(OFFICIAL_AFFINE),
                                "official_constants_source": str(VAE_CONFIG),
                                "full_vae_constructed": False, "cases": cases}


class RecordingResult(unittest.TextTestResult):
    def addSuccess(self, test):
        super().addSuccess(test)
        REPORT["checks"].append({"name": test.id(), "status": "passed"})

    def addFailure(self, test, err):
        super().addFailure(test, err)
        REPORT["checks"].append({"name": test.id(), "status": "failed", "error": self._exc_info_to_string(err, test)})

    def addError(self, test, err):
        super().addError(test, err)
        REPORT["checks"].append({"name": test.id(), "status": "failed", "error": self._exc_info_to_string(err, test)})

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err is not None:
            REPORT["checks"].append({"name": subtest.id(), "status": "failed", "error": self._exc_info_to_string(err, test)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=HERE / "loading_cpu_validation.json")
    args = parser.parse_args()
    runtime_sources = [Path(module.__file__).resolve()
                       for name, module in list(sys.modules.items())
                       if (name.startswith("mflux.models.qwen.model.")
                           or name == "mflux.models.flux.model.flux_transformer.ada_layer_norm_continuous")
                       and getattr(module, "__file__", None)
                       and Path(module.__file__).suffix == ".py"]
    tested_sources = list(dict.fromkeys(SOURCES + runtime_sources))
    source_before = {str(path): sha(path) for path in tested_sources}
    success = False
    try:
        revision = subprocess.check_output(["git", "-C", str(RUNTIME), "rev-parse", "HEAD"], text=True).strip()
        if revision != PIN:
            raise ValueError(f"Runtime revision mismatch: {revision}")
        runtime_class_path = Path(inspect.getfile(QwenVAE)).resolve()
        if not runtime_class_path.is_relative_to(RUNTIME.resolve()):
            raise ValueError(f"A different runtime was imported: {runtime_class_path}")
        REPORT.update(runtime_revision=revision, runtime_vae_path=str(runtime_class_path),
                      mlx_default_device=str(mx.default_device()), torch_version=torch.__version__)
        suite = unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
        result = unittest.TextTestRunner(verbosity=2, resultclass=RecordingResult).run(suite)
        source_after = {str(path): sha(path) for path in tested_sources}
        if source_before != source_after:
            raise ValueError("Tested source changed during CPU validation; rerun after edits finish")
        REPORT.update(source_sha256={**source_after, **REPORT.get("header_source_sha256", {})}, tests_run=result.testsRun,
                      failure_count=len(result.failures) + len(result.errors))
        success = result.wasSuccessful()
        REPORT["status"] = "passed" if success else "failed"
    except Exception as error:
        REPORT.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        REPORT["source_sha256"] = {str(path): sha(path) for path in tested_sources if path.exists()}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(REPORT, indent=2) + "\n")
    print(json.dumps({"status": REPORT["status"], "tests_run": REPORT.get("tests_run"),
                      "report": str(args.report)}))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
