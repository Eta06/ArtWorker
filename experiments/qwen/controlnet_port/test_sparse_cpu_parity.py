#!/usr/bin/env python3
"""Independent CPU support geometry, nearest packing and zero/keep parity.

Euclidean support uses direct integer squared distances instead of OpenCV's
distance transform. Packing uses Torch nearest interpolation instead of the
helper's sampled indexing. Tagged nonzero 129-channel float32/BF16 tensors
exercise gating and exact preservation of the last 65 conditioning channels.
No model is imported and no checkpoint/decoder/GPU computation is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import mlx.core as mx

mx.set_default_device(mx.cpu)
torch.set_num_threads(2)

from conditioning import gather_control_context
from sparse_conditioning import prepare_sparse_support, apply_sparse_structural_context


def support_oracle(width, height, rect, guide, radius):
    """Retain the ROI and the Euclidean radius around UNKNOWN guide pixels."""
    x1, y1, x2, y2 = rect
    yy, xx = np.indices((height, width), dtype=np.int64)
    known = (xx >= x1) & (xx < x2) & (yy >= y1) & (yy < y2)
    guide_pixels = np.any(guide > 0, axis=-1) & ~known
    points = np.argwhere(guide_pixels)
    dilation = np.zeros((height, width), dtype=bool)
    for gy, gx in points:
        dilation |= (yy - gy) ** 2 + (xx - gx) ** 2 <= radius ** 2
    return known | dilation, int(len(points))


def nearest_oracle(support):
    height, width = support.shape
    value = torch.tensor(support.astype(np.float32))[None, None]
    sampled = F.interpolate(value, size=(height // 16, width // 16), mode="nearest")
    return sampled.numpy().astype(bool).reshape(1, -1, 1)


def geometry_checks():
    width, height, rect = 128, 160, [19, 37, 86, 109]
    guide = np.zeros((height, width, 3), dtype=np.uint8)
    guide[0, 0, 0] = 1
    guide[15, 10, 1] = 17
    guide[130, 100, 2] = 255
    guide[50, 40] = 255  # In the ROI: excluded before any unknown dilation.
    before = guide.copy()
    results, packed_cases = [], []
    for radius in [0, 1, 3, 16, 64]:
        pixels, packed, metadata = prepare_sparse_support(np, Image, width, height, rect, guide, radius)
        expected_pixels, unknown_count = support_oracle(width, height, rect, guide, radius)
        expected_packed = nearest_oracle(expected_pixels)
        assert pixels.dtype == np.bool_ and packed.dtype == np.bool_
        assert np.array_equal(pixels, expected_pixels)
        assert np.array_equal(packed, expected_packed)
        assert metadata["unknown_guide_pixels"] == unknown_count == 3
        assert metadata["known_pixels"] == (rect[2] - rect[0]) * (rect[3] - rect[1])
        assert metadata["supported_pixels"] == int(expected_pixels.sum())
        assert metadata["supported_target_tokens"] == int(expected_packed.sum())
        assert np.array_equal(guide, before)
        results.append({"radius_pixels": radius, "pixel_support_exact": True,
                        "torch_nearest_packed_exact": True, "unknown_guide_pixels": unknown_count,
                        "supported_pixels": metadata["supported_pixels"],
                        "supported_tokens": metadata["supported_target_tokens"]})
        packed_cases.append(packed)
    known_pixels, known_packed, _ = prepare_sparse_support(np, Image, width, height, rect)
    only_known_guide = np.zeros_like(guide)
    only_known_guide[50, 40] = 255
    for special_guide in [np.zeros_like(guide), only_known_guide]:
        actual_pixels, actual_packed, info = prepare_sparse_support(np, Image, width, height, rect, special_guide, 64)
        assert np.array_equal(actual_pixels, known_pixels)
        assert np.array_equal(actual_packed, known_packed)
        assert info["unknown_guide_pixels"] == 0
    expected_roi_samples = np.zeros((10, 8), dtype=bool)
    expected_roi_samples[3:7, 2:6] = True
    assert np.array_equal(known_packed.reshape(10, 8), expected_roi_samples)
    return results, [known_packed, *packed_cases], {
        "off_grid_roi_floor_samples_exact": True,
        "empty_and_known_only_guides_do_not_expand_support": True,
        "guide_input_unchanged": True,
        "known_support_polarity": "True retains encoded first64 control channels",
        "guide_inside_known_excluded_before_dilation": True,
    }


def gate_checks(support_cases):
    tokens = 80
    # Every row/channel differs; last65 include signed zero and negative values.
    data = (np.arange(tokens * 129, dtype=np.float32).reshape(1, tokens, 129) + 1) / 128
    data[..., 64:] *= -1
    data[0, 13, 98] = -0.0
    original_bytes = data.tobytes()
    masks = [np.ones((1, tokens, 1), dtype=bool), np.zeros((1, tokens, 1), dtype=bool), *support_cases]
    records = []
    for dtype_name, mlx_dtype, torch_dtype in [
        ("float32", mx.float32, torch.float32),
        ("bfloat16", mx.bfloat16, torch.bfloat16),
    ]:
        context = mx.array(data).astype(mlx_dtype)
        expected_input = torch.tensor(data).to(torch_dtype)
        saved_input = np.asarray(context.astype(mx.float32)).copy()
        for index, support in enumerate(masks):
            expected = torch.cat([
                torch.where(torch.tensor(support), expected_input[..., :64], torch.zeros_like(expected_input[..., :64])),
                expected_input[..., 64:],
            ], dim=-1).to(torch.float32).numpy()
            output = apply_sparse_structural_context(mx, np, context, support)
            assert output.dtype == mlx_dtype
            actual = np.asarray(output.astype(mx.float32))
            assert np.array_equal(actual, expected)
            assert actual[..., 64:].tobytes() == saved_input[..., 64:].tobytes()
            keep = support.reshape(-1)
            assert actual[:, keep, :64].tobytes() == saved_input[:, keep, :64].tobytes()
            assert np.all(actual[:, ~keep, :64] == 0)
            # Gather includes nonmonotonic and duplicated final-canvas raster IDs.
            ids = np.array([79, 0, 13, 40, 13, 2], dtype=np.int32)
            gathered = gather_control_context(mx, output, ids)
            expected_gather = torch.tensor(expected).index_select(1, torch.tensor(ids.astype(np.int64))).numpy()
            assert np.array_equal(np.asarray(gathered.astype(mx.float32)), expected_gather)
            assert np.array_equal(np.asarray(context.astype(mx.float32)), saved_input)
            records.append({"dtype": dtype_name, "support_case": index,
                            "torch_gate_max_abs_error": float(np.abs(actual - expected).max()),
                            "supported_control64_exact": True,
                            "unsupported_control64_literal_zero": True,
                            "source65_bytes_exact_after_float32_view": True,
                            "dtype_preserved": True, "absolute_nonmonotonic_duplicate_gather_exact": True})
    assert data.tobytes() == original_bytes
    # Unsupported structural inputs must not silently accept wrong polarity/dtype/shape.
    invalid = [np.ones((1, 80, 1), dtype=np.uint8), np.ones((1, 79, 1), dtype=bool), np.ones((80,), dtype=bool)]
    for mask in invalid:
        try:
            apply_sparse_structural_context(mx, np, mx.array(data), mask)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid support was accepted")
    return records


def existing_white_artifacts():
    root = Path(__file__).resolve().parents[3]
    paths = [
        root / "experiments/qwen/controlnet_runs/2026-10-05/track3-20/mask-only",
        root / "experiments/qwen/controlnet_runs/2026-10-05/track3-20/tangent-canny",
        root / "experiments/qwen/controlnet_runs/2026-10-05/track3-40-knownbridge/tangent-canny",
    ]
    records = []
    for directory in paths:
        if not (directory / "raw.png").is_file():
            continue
        images = {}
        for name in ["raw", "composite"]:
            path = directory / f"{name}.png"
            image = Image.open(path)
            rgb = np.asarray(image.convert("RGB"))
            white = rgb.min(axis=-1) > 240
            rows = np.flatnonzero(white.mean(axis=1) > 0.95)
            ranges = []
            for row in rows:
                if ranges and ranges[-1][1] == int(row) - 1:
                    ranges[-1][1] = int(row)
                else:
                    ranges.append([int(row), int(row)])
            images[name] = {"path": str(path), "mode": image.mode, "size": list(image.size),
                            "rgb_min_gt240_fraction": float(white.mean()),
                            "rows_with_gt95pct_white_width_inclusive": ranges,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        records.append(images)
    return {"artifacts": records,
            "interpretation": "Existing saved images are RGB. White regions appear in raw and composite at identical target rows; runner discards decoded alpha and never composites it onto white. Decoder alpha itself was not saved/inspected, and no decoder rerun occurred."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("sparse_cpu_parity.json"))
    args = parser.parse_args()
    geometry, supports, extra = geometry_checks()
    gate = gate_checks(supports)
    report = {
        "status": "passed", "device": "CPU only", "checkpoint_or_model_weights_used": False,
        "decoder_rerun": False, "geometry_oracle": "integer squared Euclidean distances to input-only unknown guide pixels",
        "packing_oracle": "Torch interpolate(mode=nearest) then y-major reshape",
        "geometry_cases": geometry, "geometry_invariants": extra, "gate_cases": gate,
        "max_abs_gate_error": max(case["torch_gate_max_abs_error"] for case in gate),
        "scope": "Experimental input context gating only. Literal-zero input control64 does not prove zero learned ControlNet hints. This test does not assess generated image quality or remove VAE receptive-field effects inside retained support.",
        "source_hashes": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ["test_sparse_cpu_parity.py", "sparse_conditioning.py", "conditioning.py", "run_sparse_control_trial.py"]
        },
        "existing_white_artifacts": existing_white_artifacts(),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "geometry_cases": len(geometry), "gate_cases": len(gate),
                      "max_abs_gate_error": report["max_abs_gate_error"], "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
