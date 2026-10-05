"""Header and tiny numerical CPU gates; does not load either full checkpoint."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from maskflow_adapter import (
    ADAPTER_REVISION, BASE_REVISION, MODULE_PATHS, adapter_spec, apply_maskflow_adapter,
    bridge_maskflow_weights, build_adapter_plan, prefix_maskflow_key,
    read_safetensors_header, target_module,
)

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime/mlx-gen"
RUNTIME_REVISION = "99fb94dd3eaa9dd1931cd3cd8eae1ae3e20f2ef3"
RESEARCH = ROOT / "research/2026-10-05"
REPORT = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
          "status": "not_run", "device": "cpu", "gpu_used": False,
          "full_checkpoint_loaded": False, "base_revision": BASE_REVISION,
          "adapter_revision": ADAPTER_REVISION, "runtime_revision": RUNTIME_REVISION}


def inputs():
    header = json.loads((RESEARCH / "headers/latest.safetensors.json").read_text())
    base = {}
    sources = sorted((ROOT / "models/headers").glob("transformer-*.safetensors.json"))
    if len(sources) != 7:
        raise ValueError("Need pinned headers from all seven transformer shards")
    for source in sources:
        for key, value in json.loads(source.read_text()).items():
            if key == "__metadata__":
                continue
            if key in base:
                raise ValueError(f"Duplicate base key: {key}")
            base[key] = value
    index = json.loads((RESEARCH / "quantized_metadata/AbstractFramework--qwen-image-edit-2511-4bit/transformer/model.safetensors.index.json").read_text())
    return header, base, index["weight_map"]


def runtime_modules():
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    import mlx.nn as nn
    import numpy as np
    loader_module = importlib.import_module("mflux.models.common.lora.mapping.lora_loader")
    mapping_module = importlib.import_module("mflux.models.qwen.weights.qwen_lora_mapping")
    loaded_path = Path(loader_module.__file__).resolve()
    if not loaded_path.is_relative_to(RUNTIME.resolve()):
        raise ValueError(f"CPU gate imported a different runtime: {loaded_path}")
    revision = subprocess.check_output(["git", "-C", str(RUNTIME), "rev-parse", "HEAD"], text=True).strip()
    if revision != RUNTIME_REVISION:
        raise ValueError(f"Runtime revision changed: {revision}")
    REPORT["runtime_loader_path"] = str(loaded_path)
    REPORT["mlx_default_device"] = str(mx.default_device())
    return mx, nn, np, loader_module.LoRALoader, mapping_module.QwenLoRAMapping


class HeaderGate(unittest.TestCase):
    def setUp(self):
        self.header, self.base, self.weight_map = inputs()

    def test_all_600_pairs_match_packed_q4_base(self):
        spec, plans = build_adapter_plan(self.header, self.base, self.weight_map)
        self.assertEqual(len(plans), 600)
        self.assertEqual(len({p.target_module for p in plans}), 600)
        self.assertEqual(spec.alpha_over_rank, 1.0)
        self.assertEqual({p.input_dims for p in plans}, {3072, 12288})
        self.assertEqual({p.output_dims for p in plans}, {3072})
        REPORT["header_gate"] = {"status": "passed", "tensor_count": 1200,
                                 "target_count": len(plans), "spec": asdict(spec),
                                 "base_headers_only": True,
                                 "targets": [asdict(p) for p in plans]}

    def test_bridge_preserves_tensor_objects_and_raw_names(self):
        class HeaderOnlyTensor:
            def __init__(self, shape):
                self.shape = tuple(shape)
        weights = {k: HeaderOnlyTensor(v["shape"]) for k, v in self.header.items() if k != "__metadata__"}
        before = set(weights)
        bridged = bridge_maskflow_weights(weights, self.header)
        self.assertEqual(set(weights), before)
        self.assertEqual(len(bridged), 1200)
        self.assertTrue(all(bridged[prefix_maskflow_key(k)] is v for k, v in weights.items()))
        weights.pop(next(iter(weights)))
        with self.assertRaises(ValueError):
            bridge_maskflow_weights(weights, self.header)

    def test_incomplete_pair_is_rejected(self):
        self.header.pop("transformer_blocks.0.attn.to_q.lora_B.weight")
        with self.assertRaisesRegex(ValueError, "Incomplete A/B"):
            build_adapter_plan(self.header, self.base, self.weight_map)

    def test_unknown_tensor_is_rejected(self):
        self.header["unrecognized.weight"] = {"shape": [1], "dtype": "BF16"}
        with self.assertRaisesRegex(ValueError, "Unrecognized"):
            build_adapter_plan(self.header)

    def test_bad_rank_is_rejected(self):
        entry = self.header["transformer_blocks.0.attn.to_q.lora_A.weight"]
        entry["shape"][0] = 128
        entry["data_offsets"][1] = entry["data_offsets"][0] + 128 * 3072 * 2
        with self.assertRaisesRegex(ValueError, "Rank mismatch"):
            build_adapter_plan(self.header)

    def test_unsupported_peft_variants_are_rejected(self):
        for key, value in [("use_dora", True), ("use_rslora", True), ("fan_in_fan_out", True),
                           ("alpha_pattern", {"attn.to_q": 128}), ("lora_alpha", 128)]:
            changed = copy.deepcopy(self.header)
            config = json.loads(changed["__metadata__"]["lora_adapter_metadata"])
            config[key] = value
            changed["__metadata__"]["lora_adapter_metadata"] = json.dumps(config)
            with self.subTest(key=key), self.assertRaises(ValueError):
                adapter_spec(changed)

    def test_incompatible_base_header_or_index_is_rejected(self):
        key = "transformer_blocks.0.attn.to_q.weight"
        self.base[key]["shape"][1] = 385
        with self.assertRaisesRegex(ValueError, "Packed Q4"):
            build_adapter_plan(self.header, self.base, self.weight_map)
        self.base[key]["shape"][1] = 384
        self.weight_map.pop("transformer_blocks.0.attn.to_q.scales")
        with self.assertRaisesRegex(ValueError, "Base index lacks"):
            build_adapter_plan(self.header, self.base, self.weight_map)

    def test_downloaded_adapter_header_when_available(self):
        checkpoint = ROOT / "models/MaskFlow/latest.safetensors"
        if not checkpoint.exists():
            self.skipTest("Full adapter download not yet complete; only remote pinned header gated")
        self.assertEqual(read_safetensors_header(checkpoint), self.header)
        REPORT["local_adapter_header_matches_pinned_range_header"] = True

    def test_all_local_base_headers_match_pinned_remote_headers(self):
        for shard in range(7):
            local = ROOT / f"models/qwen-image-edit-2511-4bit/transformer/{shard}.safetensors"
            if not local.exists():
                self.skipTest("Base download not complete; remote pinned headers only")
            remote = json.loads((ROOT / f"models/headers/transformer-{shard}.safetensors.json").read_text())
            self.assertEqual(read_safetensors_header(local), remote)
        REPORT["local_base_headers_match_pinned_range_headers"] = True


class RuntimeCPUGate(unittest.TestCase):
    def test_original_bias_sidecar_is_cpu_bf16_finite_nonzero(self):
        mx, _, _, _, _ = runtime_modules()
        directory = ROOT / "models/original-sidecar"
        provenance = json.loads((directory / "norm_out.linear.bias.provenance.json").read_text())
        path = directory / "norm_out.linear.bias.safetensors"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), provenance["safetensors_sidecar_sha256"])
        self.assertFalse(provenance["source_full_shard_payload_verified"])
        self.assertEqual(provenance["payload_range"], "bytes 4776-17063/982130472")
        weights = mx.load(str(path))
        self.assertEqual(set(weights), {"norm_out.linear.bias"})
        bias = weights["norm_out.linear.bias"]
        self.assertEqual((tuple(bias.shape), bias.dtype), ((6144,), mx.bfloat16))
        mx.eval(bias)
        self.assertTrue(bool(mx.all(mx.isfinite(bias))))
        self.assertTrue(bool(mx.any(bias != 0)))
        REPORT["norm_out_bias_sidecar_gate"] = {"status": "passed", "shape": [6144], "dtype": "BF16",
                  "finite": True, "nonzero": True, "device": str(mx.default_device()),
                  "sidecar_sha256": provenance["safetensors_sidecar_sha256"],
                  "original_full_shard_payload_verified": False}

    def test_full_public_bridge_with_shape_only_mock_preserves_base_and_rejects_before_mutation(self):
        mx, nn, _, loader, _ = runtime_modules()
        header, _, _ = inputs()
        _, plans = build_adapter_plan(header)

        class ShapeTensor:
            def __init__(self, shape, dtype):
                self.shape = tuple(shape)
                self.dtype = dtype
            @property
            def T(self):
                return ShapeTensor(tuple(reversed(self.shape)), self.dtype)

        class ShapeQ4:
            bits, group_size, mode = 4, 64, "affine"
            def __init__(self, input_dims, output_dims):
                self.weight = ShapeTensor((output_dims, input_dims // 8), mx.uint32)
                self.scales = ShapeTensor((output_dims, input_dims // 64), mx.bfloat16)
                self.biases = ShapeTensor((output_dims, input_dims // 64), mx.bfloat16)
                self.bias = ShapeTensor((output_dims,), mx.bfloat16)

        class ShapeLoRA:
            @staticmethod
            def from_linear(linear, r=16, scale=1.0):
                wrapper = ShapeLoRA()
                wrapper.linear, wrapper.scale = linear, scale
                return wrapper

        def fixture():
            root = SimpleNamespace(transformer_blocks=[SimpleNamespace(attn={}, img_ff={}, txt_ff={}) for _ in range(60)])
            bases = {}
            for plan in plans:
                block = root.transformer_blocks[int(plan.target_module.split(".")[1])]
                family, suffix = plan.target_module.split(".", 3)[2:]
                base = ShapeQ4(plan.input_dims, plan.output_dims)
                if suffix == "attn_to_out.0":
                    block.attn["attn_to_out"] = [base]
                else:
                    getattr(block, family)[suffix] = base
                bases[plan.target_module] = base
            weights = {key: ShapeTensor(value["shape"], mx.bfloat16) for key, value in header.items() if key != "__metadata__"}
            return root, bases, weights

        loader_module = importlib.import_module(loader.__module__)
        with patch.object(nn, "QuantizedLinear", ShapeQ4), patch.object(loader_module, "LoRALinear", ShapeLoRA):
            for mutation in ("missing_tensor", "wrong_dtype", "wrong_base_dims", "nan_scale"):
                root, bases, weights = fixture()
                scale = 1.0
                if mutation == "missing_tensor":
                    weights.pop(next(iter(weights)))
                elif mutation == "wrong_dtype":
                    weights[next(iter(weights))].dtype = mx.float32
                elif mutation == "wrong_base_dims":
                    bases[plans[-1].target_module].weight.shape = (1, 1)
                elif mutation == "nan_scale":
                    scale = float("nan")
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    apply_maskflow_adapter(root, weights, header, scale=scale)
                self.assertTrue(all(target_module(root, target) is base for target, base in bases.items()))
            root, bases, weights = fixture()
            snapshots = {target: {name: getattr(base, name) for name in ("weight", "scales", "biases", "bias")} for target, base in bases.items()}
            result = apply_maskflow_adapter(root, weights, header, scale=.75)
            self.assertEqual((result["targets"], result["matched_keys"]), (600, 1200))
            for target, base in bases.items():
                wrapper = target_module(root, target)
                self.assertIs(wrapper.linear, base)
                for name, before in snapshots[target].items():
                    self.assertIs(getattr(wrapper.linear, name), before)
            REPORT["full_public_apply_shape_only_gate"] = {"status": "passed", "targets": 600,
                  "matched_keys": 1200, "backend": "shape-only mocks with unchanged pinned mapping and loader APIs",
                  "negative_preflight_cases_leave_all_targets_unchanged": True,
                  "base_parameter_identity_preserved": True, "no_large_tensor_allocation": True}

    def test_every_bridged_key_matches_stock_mapper_once(self):
        _, _, _, loader, mapping = runtime_modules()
        header, _, _ = inputs()
        _, plans = build_adapter_plan(header)
        patterns = loader._build_pattern_mappings(mapping.get_mapping())
        expected = {key: p.target_module for p in plans for key in (p.down_key, p.up_key)}
        raw_matches = 0
        targets = set()
        for key, target in expected.items():
            raw_matches += sum(loader._match_pattern(key, p.source_pattern) is not None for p in patterns)
            bridged = prefix_maskflow_key(key)
            matches = [p for p in patterns if loader._match_pattern(bridged, p.source_pattern) is not None]
            self.assertEqual(len(matches), 1, key)
            match = matches[0]
            block = loader._match_pattern(bridged, match.source_pattern)
            self.assertEqual(match.target_path.format(block=block), target)
            self.assertEqual(match.matrix_name, "lora_A" if ".lora_A." in key else "lora_B")
            self.assertTrue(match.transpose)
            self.assertIsNone(match.transform)
            targets.add(target)
        self.assertEqual(raw_matches, 0)
        REPORT["stock_mapper_gate"] = {"status": "passed", "raw_key_matches": raw_matches,
                                       "bridged_key_matches": 1200, "targets": len(targets),
                                       "A_B_transposed_once": True}

    def test_tiny_q4_all_ten_paths_match_independent_numpy(self):
        mx, nn, np, loader, mapping = runtime_modules()
        rng = np.random.default_rng(260805)
        root = nn.Module()
        block = nn.Module()
        block.attn = {}
        block.img_ff = {}
        block.txt_ff = {}
        root.transformer_blocks = [block]
        weights, bases, snapshots, refs = {}, {}, {}, {}
        x_np = rng.normal(size=(2, 64)).astype(np.float32)
        x = mx.array(x_np)
        scale = 0.75
        for source_suffix, target_suffix in MODULE_PATHS.items():
            linear = nn.Linear(64, 8, bias=True)
            linear.weight = mx.array(rng.normal(scale=.1, size=(8, 64)).astype(np.float32))
            linear.bias = mx.array(rng.normal(scale=.1, size=(8,)).astype(np.float32))
            quantized = nn.QuantizedLinear.from_linear(linear, group_size=64, bits=4)
            if target_suffix == "attn.attn_to_out.0":
                block.attn["attn_to_out"] = [quantized]
            else:
                family, name = target_suffix.split(".")
                getattr(block, family)[name] = quantized
            target = "transformer_blocks.0." + target_suffix
            bases[target] = quantized
            mx.eval(quantized.parameters())
            snapshots[target] = {name: np.array(getattr(quantized, name)) for name in ("weight", "scales", "biases", "bias")}
            a_np = rng.normal(scale=.01, size=(256, 64)).astype(np.float32)
            b_np = rng.normal(scale=.01, size=(8, 256)).astype(np.float32)
            source = "transformer_blocks.0." + source_suffix
            weights[prefix_maskflow_key(source + ".lora_A.weight")] = mx.array(a_np)
            weights[prefix_maskflow_key(source + ".lora_B.weight")] = mx.array(b_np)
            dense = mx.dequantize(quantized.weight, quantized.scales, quantized.biases, group_size=64, bits=4)
            mx.eval(dense)
            refs[target] = x_np @ np.array(dense).T + np.array(quantized.bias) + scale * ((x_np @ a_np.T) @ b_np.T)
        patterns = loader._build_pattern_mappings(mapping.get_mapping())
        count, matched = loader._apply_lora_with_mapping(root, weights, scale, patterns, role="maskflow-cpu-gate")
        self.assertEqual((count, len(matched)), (10, 20))
        max_error = 0.0
        for target, expected in refs.items():
            adapted = target_module(root, target)
            self.assertIs(adapted.linear, bases[target])
            self.assertEqual((adapted.linear.bits, adapted.linear.group_size), (4, 64))
            actual = adapted(x)
            mx.eval(actual)
            actual_np = np.array(actual)
            np.testing.assert_allclose(actual_np, expected, rtol=2e-5, atol=2e-5)
            max_error = max(max_error, float(np.max(np.abs(actual_np - expected))))
            for name, before in snapshots[target].items():
                np.testing.assert_array_equal(np.array(getattr(adapted.linear, name)), before)
        REPORT["tiny_q4_numeric_gate"] = {"status": "passed", "seed": 260805, "device": str(mx.default_device()),
                                         "targets": count, "matched_keys": len(matched), "rank": 256,
                                         "input_dims": 64, "output_dims": 8, "scale": scale,
                                         "max_abs_error_vs_independent_numpy": max_error,
                                         "base_quantized_tensors_unchanged": True,
                                         "full_model_application_tested": False}

    def test_bf16_q4_cpu_at_zero_positive_negative_scale(self):
        mx, nn, np, loader, mapping = runtime_modules()
        rng = np.random.default_rng(260806)
        x = mx.array(rng.normal(size=(2, 64)).astype(np.float32)).astype(mx.bfloat16)
        a = mx.array(rng.normal(scale=.02, size=(256, 64)).astype(np.float32)).astype(mx.bfloat16)
        b = mx.array(rng.normal(scale=.02, size=(8, 256)).astype(np.float32)).astype(mx.bfloat16)
        errors = {}
        for scale in (0.0, .75, -.5):
            linear = nn.Linear(64, 8, bias=True)
            linear.weight = mx.array(rng.normal(scale=.1, size=(8, 64)).astype(np.float32)).astype(mx.bfloat16)
            linear.bias = mx.array(rng.normal(scale=.1, size=(8,)).astype(np.float32)).astype(mx.bfloat16)
            base = nn.QuantizedLinear.from_linear(linear, group_size=64, bits=4)
            root = nn.Module()
            root.transformer_blocks = [{"attn": {"to_q": base}}]
            source = "transformer_blocks.0.attn.to_q"
            weights = {prefix_maskflow_key(source + ".lora_A.weight"): a,
                       prefix_maskflow_key(source + ".lora_B.weight"): b}
            dense = mx.dequantize(base.weight, base.scales, base.biases, group_size=64, bits=4)
            mx.eval(dense, x, a, b, base.bias)
            as_np = lambda value: np.array(value.astype(mx.float32))
            expected = as_np(x) @ as_np(dense).T + as_np(base.bias) + scale * ((as_np(x) @ as_np(a).T) @ as_np(b).T)
            count, matched = loader._apply_lora_with_mapping(root, weights, scale,
                    loader._build_pattern_mappings(mapping.get_mapping()), role="maskflow-bf16-cpu-gate")
            self.assertEqual((count, len(matched)), (1, 2))
            wrapper = root.transformer_blocks[0]["attn"]["to_q"]
            self.assertIs(wrapper.linear, base)
            actual = wrapper(x)
            mx.eval(actual)
            error = float(np.max(np.abs(as_np(actual) - expected)))
            # BF16 rounds the separate base/delta matmuls and the final sum.
            np.testing.assert_allclose(as_np(actual), expected, rtol=.02, atol=.02)
            errors[str(scale)] = error
        REPORT["bf16_q4_cpu_gate"] = {"status": "passed", "device": str(mx.default_device()),
                                    "dtype": "BF16", "scales": [0.0, .75, -.5],
                                    "max_abs_errors_vs_float32_numpy": errors,
                                    "tolerance_rtol_atol": [.02, .02], "rank": 256}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=ROOT / "port/adapter_cpu_validation.json")
    parser.add_argument("--headers-only", action="store_true")
    args = parser.parse_args()
    suite = unittest.TestLoader().loadTestsFromTestCase(HeaderGate)
    if not args.headers_only:
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(RuntimeCPUGate))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    REPORT.update(status="passed_headers_only" if result.wasSuccessful() and args.headers_only else "passed" if result.wasSuccessful() else "failed",
                  tests_run=result.testsRun, errors=len(result.errors), failures=len(result.failures),
                  skipped=[{"test": str(test), "reason": reason} for test, reason in result.skipped],
                  notes=["Header compatibility and tiny CPU arithmetic only; no full model allocation, GPU inference, or output-quality claim.",
                         "Full payload integrity is recorded separately in download_manifest.json."])
    REPORT["code_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), Path(__file__).with_name("maskflow_adapter.py")]}
    args.report.write_text(json.dumps(REPORT, indent=2) + "\n")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
