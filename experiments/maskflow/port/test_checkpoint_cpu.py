"""CPU-only retained checkpoint integrity gates; never loads teacher weights."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import mlx.core as mx

# This must precede the creation or evaluation of any MLX array.
mx.set_default_device(mx.cpu)

from maskflow_checkpoint import (  # noqa: E402
    PROTOCOL_KEYS,
    REQUIRED,
    file_sha256,
    restore_checkpoint,
    save_checkpoint,
    validate_checkpoint,
)

HERE = Path(__file__).resolve().parent
PACKED_SHAPE = (1, 2304, 64)
CHECKS = {}


def make_arrays():
    mx.random.seed(260805)
    arrays = {
        "latents": mx.random.normal(PACKED_SHAPE).astype(mx.bfloat16),
        "initial_noise": mx.random.normal(PACKED_SHAPE).astype(mx.bfloat16),
        "source_latents": mx.full(PACKED_SHAPE, 0.375, dtype=mx.bfloat16),
        "mask_image_latents": mx.full(PACKED_SHAPE, 0.625, dtype=mx.bfloat16),
        "mask_latents": mx.concatenate(
            [mx.zeros((1, 1152, 64), dtype=mx.bfloat16),
             mx.ones((1, 1152, 64), dtype=mx.bfloat16)], axis=1
        ),
        "pm.prompt_embeds": mx.random.normal((1, 13, 10)).astype(mx.bfloat16),
        "pm.prompt_embeds_mask": mx.array([[True] * 11 + [False] * 2]),
        "nm.prompt_embeds": mx.random.normal((1, 7, 10)).astype(mx.bfloat16),
        "nm.prompt_embeds_mask": mx.array([[True] * 6 + [False]]),
    }
    mx.eval(*arrays.values())
    return arrays


def make_metadata():
    metadata = {
        "source_file_sha256": "1" * 64,
        "source_rgb_sha256": "2" * 64,
        "canvas_rgb_sha256": "3" * 64,
        "prompt_sha256": "4" * 64,
        "seed": 260805,
        "width": 512,
        "height": 1152,
        "known_box_xyxy": [0, 320, 512, 832],
        "unknown_fill": "black",
        "steps": 50,
        "text_cfg": 4.0,
        "mask_cfg": 1.0,
        "rescale_cfg": True,
        "runtime_revision": "99fb94dd3eaa9dd1931cd3cd8eae1ae3e20f2ef3",
        "official_revision": "3125f1ecd72f5a4e068c5a1954e9d6a8df38d443",
        "port_sha256": {"maskflow_checkpoint.py": file_sha256(HERE / "maskflow_checkpoint.py")},
        "initial_noise_sha256": "5" * 64,
        "vlm": {"retained_in_memory": False, "conditioning_only": True},
        "cpu_gates": {"checkpoint": "test-fixture"},
        "downloads": {"full_teacher_loaded": False},
    }
    assert set(PROTOCOL_KEYS) <= set(metadata)
    return metadata


class CheckpointCPU(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="maskflow-checkpoint-cpu-")
        self.directory = Path(self.temp.name)
        self.arrays = make_arrays()
        self.metadata = make_metadata()
        self.saved = save_checkpoint(self.directory, 2, self.arrays, self.metadata)
        self.receipt_path = Path(self.saved["receipt"])
        self.receipt = json.loads(self.receipt_path.read_text())
        self.protocol = {key: self.metadata[key] for key in PROTOCOL_KEYS}

    def tearDown(self):
        self.temp.cleanup()

    def receipt_variant(self, receipt):
        path = self.directory / "variant.json"
        path.write_text(json.dumps(receipt))
        return path

    def assert_rejected(self, receipt, protocol=None, restore=False):
        with self.assertRaises((ValueError, KeyError, FileNotFoundError)):
            checked = validate_checkpoint(self.receipt_variant(receipt), protocol or self.protocol)
            if restore:
                restore_checkpoint(checked)

    def test_bf16_roundtrip_real_packed_geometry_and_both_cfg_branches(self):
        checked = validate_checkpoint(self.receipt_path, self.protocol)
        restored = restore_checkpoint(checked)
        self.assertEqual(set(restored), set(REQUIRED))
        for key in REQUIRED:
            self.assertEqual(restored[key].shape, self.arrays[key].shape)
            self.assertEqual(restored[key].dtype, self.arrays[key].dtype)
            self.assertTrue(bool(mx.array_equal(restored[key], self.arrays[key]).item()), key)
        self.assertEqual(restored["pm.prompt_embeds"].shape, (1, 13, 10))
        self.assertEqual(restored["nm.prompt_embeds"].shape, (1, 7, 10))
        for key in REQUIRED[:5]:
            self.assertEqual(restored[key].shape, PACKED_SHAPE)
            self.assertEqual(restored[key].dtype, mx.bfloat16)
        self.assertFalse(bool(mx.array_equal(restored["source_latents"], restored["mask_image_latents"]).item()))
        self.assertEqual(float(mx.min(restored["mask_latents"]).item()), 0.0)
        self.assertEqual(float(mx.max(restored["mask_latents"]).item()), 1.0)
        CHECKS["bf16_roundtrip"] = {
            "packed_shape": list(PACKED_SHAPE), "packed_dtype": "BF16",
            "all_nine_tensors_equal": True, "distinct_pm_nm_lengths": [13, 7],
            "mask_inventory_retained": True,
            "state_bytes": Path(checked["state_path"]).stat().st_size,
        }

    def test_prior_states_and_receipt_retained_until_atomic_replacement(self):
        original_path = Path(self.receipt["state_path"])
        original_hash = file_sha256(original_path)
        saved2 = save_checkpoint(self.directory, 2, self.arrays, self.metadata)
        saved3 = save_checkpoint(self.directory, 3, self.arrays, self.metadata)
        self.assertEqual(len({self.saved["state_path"], saved2["state_path"], saved3["state_path"]}), 3)
        for saved in [self.saved, saved2, saved3]:
            self.assertTrue(Path(saved["state_path"]).is_file())
            self.assertEqual(file_sha256(saved["state_path"]), saved["state_sha256"])
        self.assertEqual(file_sha256(original_path), original_hash)
        checked = validate_checkpoint(self.receipt_path, self.protocol)
        self.assertEqual(checked["completed_steps"], 3)
        self.assertEqual(checked["state_path"], saved3["state_path"])
        self.assertFalse(any(p.name.startswith(".") or p.suffix == ".tmp" for p in self.directory.iterdir()))
        before_receipt = self.receipt_path.read_bytes()
        bad_metadata = dict(self.metadata)
        del bad_metadata["source_file_sha256"]
        with self.assertRaises(KeyError):
            save_checkpoint(self.directory, 4, self.arrays, bad_metadata)
        self.assertEqual(self.receipt_path.read_bytes(), before_receipt)
        self.assertEqual(file_sha256(original_path), original_hash)
        CHECKS["atomic_retention"] = {
            "three_unique_committed_states_retained": True,
            "failed_save_preserves_prior_receipt_and_state": True,
            "temporary_files_remaining_after_success": 0,
        }

    def test_byte_corruption_rejected_before_restore(self):
        path = Path(self.receipt["state_path"])
        payload = bytearray(path.read_bytes())
        payload[-1] ^= 1
        path.write_bytes(payload)
        self.assert_rejected(self.receipt, restore=True)
        CHECKS["state_byte_corruption_rejected"] = True

    def test_protocol_source_seed_port_hash_and_all_other_fields_rejected(self):
        tested = []
        for key in PROTOCOL_KEYS:
            with self.subTest(key=key):
                changed = copy.deepcopy(self.protocol)
                if isinstance(changed[key], dict):
                    changed[key] = {**changed[key], "changed": "different"}
                elif isinstance(changed[key], list):
                    changed[key] = [1, 320, 512, 832]
                elif isinstance(changed[key], bool):
                    changed[key] = not changed[key]
                elif isinstance(changed[key], (int, float)):
                    changed[key] += 1
                else:
                    changed[key] += "-changed"
                self.assert_rejected(self.receipt, changed)
                tested.append(key)
        CHECKS["protocol_changes_rejected"] = tested

    def test_invalid_schema_step_and_missing_receipt_inventory_rejected(self):
        for key, value in [("schema", "unknown"), ("completed_steps", -1), ("completed_steps", 51)]:
            changed = copy.deepcopy(self.receipt)
            changed[key] = value
            self.assert_rejected(changed)
        for key in REQUIRED:
            changed = copy.deepcopy(self.receipt)
            del changed["saved_tensor_shapes"][key]
            self.assert_rejected(changed)
        changed = copy.deepcopy(self.receipt)
        changed["saved_tensor_shapes"]["extra"] = [1]
        self.assert_rejected(changed)
        CHECKS["schema_step_inventory_rejected"] = True

    def test_packed_receipt_shape_rejected(self):
        for key in REQUIRED[:5]:
            changed = copy.deepcopy(self.receipt)
            changed["saved_tensor_shapes"][key] = [1, 2303, 64]
            self.assert_rejected(changed)
        CHECKS["all_five_packed_receipt_shapes_checked"] = True

    def test_actual_missing_or_extra_tensors_rejected_even_with_updated_hash(self):
        for key in REQUIRED:
            with self.subTest(missing=key):
                modified = dict(self.arrays)
                del modified[key]
                changed = self.write_modified_payload(modified)
                self.assert_rejected(changed, restore=True)
        modified = {**self.arrays, "extra": mx.array([1])}
        self.assert_rejected(self.write_modified_payload(modified), restore=True)
        CHECKS["actual_inventory_checked_against_receipt"] = True

    def test_actual_shape_and_dtype_mismatch_rejected_even_with_updated_hash(self):
        for key in REQUIRED:
            with self.subTest(shape=key):
                modified = dict(self.arrays)
                modified[key] = modified[key][:, :-1]
                self.assert_rejected(self.write_modified_payload(modified), restore=True)
        modified = dict(self.arrays)
        modified["latents"] = modified["latents"].astype(mx.float32)
        self.assert_rejected(self.write_modified_payload(modified), restore=True)
        CHECKS["actual_shapes_and_dtype_checked_against_receipt"] = True

    def test_save_rejects_incomplete_inventory(self):
        before_receipt = self.receipt_path.read_bytes()
        for key in REQUIRED:
            modified = dict(self.arrays)
            del modified[key]
            with self.assertRaises(ValueError):
                save_checkpoint(self.directory, 4, modified, self.metadata)
        with self.assertRaises(ValueError):
            save_checkpoint(self.directory, 4, {**self.arrays, "extra": mx.array([1])}, self.metadata)
        self.assertEqual(self.receipt_path.read_bytes(), before_receipt)
        CHECKS["incomplete_save_rejected_without_receipt_change"] = True

    def write_modified_payload(self, arrays):
        mx.eval(*arrays.values())
        path = self.directory / "modified.safetensors"
        mx.save_safetensors(str(path), arrays)
        changed = copy.deepcopy(self.receipt)
        changed["state_path"] = str(path)
        changed["state_sha256"] = file_sha256(path)
        return changed


def main():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CheckpointCPU)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if result.wasSuccessful() else "failed",
        "device": str(mx.default_device()),
        "gpu_used": False,
        "full_model_loaded": False,
        "source_sha256": {
            "maskflow_checkpoint.py": file_sha256(HERE / "maskflow_checkpoint.py"),
            "test_checkpoint_cpu.py": file_sha256(Path(__file__)),
        },
        "checks": CHECKS,
        "tests_run": result.testsRun,
        "errors": len(result.errors),
        "failures": len(result.failures),
        "notes": [
            "Real native packed geometry with CPU BF16 tensors; tiny CFG embeddings only.",
            "Temporary test states are removed after each case; full teacher and GPU are not used.",
            "Tests check committed receipts and failure retention; they do not simulate OS power loss.",
        ],
    }
    (HERE / "checkpoint_cpu_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "tests_run": result.testsRun,
                      "report": str(HERE / "checkpoint_cpu_validation.json")}, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
