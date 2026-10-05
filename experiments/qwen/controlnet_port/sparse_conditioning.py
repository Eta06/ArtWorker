"""Experimental spatially absent structural control; source channels stay exact.

This is not the published VideoX inference recipe. The baseline VAE-encodes
an entire RGB control map. Here only its first64 control channels are retained
where an input-derived support is present; other positions become literal zero.
The final known-mask1 and masked-source64 channels are never changed.
"""

from __future__ import annotations

import hashlib


def prepare_sparse_support(np, Image, width, height, source_rect, guide_rgb=None,
                           dilation_pixels=64):
    """Known ROI plus an exact Euclidean dilation of UNKNOWN guide pixels.

    White/True means retain encoded structural control. The packed grid uses
    the same floor/nearest samples as the official one-channel known mask.
    Geometry comes only from the supplied guide; no output is inspected.
    """
    from conditioning import _dimensions, _rgb_array

    x1, y1, x2, y2 = _dimensions(width, height, source_rect)
    if not isinstance(dilation_pixels, int) or not 0 <= dilation_pixels <= max(width, height):
        raise ValueError("dilation_pixels must be a bounded nonnegative integer")
    known = np.zeros((height, width), dtype=bool)
    known[y1:y2, x1:x2] = True
    guide = np.zeros_like(known)
    guide_sha = None
    if guide_rgb is not None:
        rgb = _rgb_array(np, Image, guide_rgb, width, height, "guide_rgb")
        guide_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
        guide = np.any(rgb != 0, axis=-1) & ~known
    dilated = np.zeros_like(known)
    if guide.any():
        if dilation_pixels == 0:
            dilated = guide.copy()
        else:
            import cv2
            # Exact L2 distance to a nonzero guide pixel. No resized detector,
            # approximate chamfer kernel or generated output geometry is used.
            distances = cv2.distanceTransform((~guide).astype(np.uint8), cv2.DIST_L2,
                                              cv2.DIST_MASK_PRECISE)
            dilated = distances <= float(dilation_pixels)
    support = known | (dilated & ~known)
    ys = np.arange(height // 16, dtype=np.int64) * height // (height // 16)
    xs = np.arange(width // 16, dtype=np.int64) * width // (width // 16)
    packed = np.ascontiguousarray(support[ys[:, None], xs[None, :]].reshape(1, -1, 1))
    metadata = {
        "experimental_sparse_control": True,
        "support_polarity": "white/True=retain encoded control64; black/False=literal zero64",
        "guide_geometry_source": "input guide outside known ROI only; never model output",
        "source_rect_xyxy": list(source_rect),
        "dilation_pixels": dilation_pixels,
        "dilation_metric": "exact Euclidean L2 pixel distance",
        "guide_rgb_uint8_sha256": guide_sha,
        "unknown_guide_pixels": int(guide.sum()),
        "known_pixels": int(known.sum()),
        "supported_pixels": int(support.sum()),
        "supported_target_tokens": int(packed.sum()),
        "total_target_tokens": packed.shape[1],
        "packed_shape": list(packed.shape),
        "nearest_coordinates": "floor(output_index*input_size/output_size), raster y-major",
        "support_pixels_bool_sha256": hashlib.sha256(support.tobytes()).hexdigest(),
        "support_tokens_bool_sha256": hashlib.sha256(packed.tobytes()).hexdigest(),
        "source_conditioning_channels_changed": False,
    }
    return support, packed, metadata


def apply_sparse_structural_context(mx, np, context, packed_support):
    """Drop exactly first64 channels outside support, preserving source65 bytes."""
    support = np.asarray(packed_support)
    if context.ndim != 3 or context.shape[0] != 1 or context.shape[-1] != 129:
        raise ValueError("Expected one packed129-channel context")
    if support.dtype != np.bool_ or support.shape != (1, context.shape[1], 1):
        raise ValueError("Expected bool packed support shaped [1,target_tokens,1]")
    structural = mx.where(mx.array(support), context[:, :, :64], mx.zeros_like(context[:, :, :64]))
    return mx.concatenate([structural, context[:, :, 64:]], axis=-1)


def cpu_preflight():
    """Weights-free zero/keep/parity, polarity, Euclidean and absolute ID tests."""
    import mlx.core as mx
    import numpy as np
    from PIL import Image
    from conditioning import gather_control_context

    mx.set_default_device(mx.cpu)
    width, height, rect = 96, 128, [16, 32, 80, 96]
    support, tokens, metadata = prepare_sparse_support(np, Image, width, height, rect)
    expected = np.zeros((8, 6), dtype=bool)
    expected[2:6, 1:5] = True
    assert np.array_equal(tokens.reshape(8, 6), expected)
    assert support.sum() == 64 * 64 and tokens.sum() == 16
    # Distinct nonzero channel values prevent an all-zero false pass.
    data = (np.arange(48 * 129, dtype=np.float32).reshape(1, 48, 129) + 1) / 128
    context = mx.array(data)
    keep = np.ones((1, 48, 1), dtype=bool)
    drop = np.zeros_like(keep)
    kept = np.asarray(apply_sparse_structural_context(mx, np, context, keep))
    dropped = np.asarray(apply_sparse_structural_context(mx, np, context, drop))
    sparse = np.asarray(apply_sparse_structural_context(mx, np, context, tokens))
    assert np.array_equal(kept, data)
    assert np.array_equal(dropped[:, :, 64:], data[:, :, 64:])
    assert np.all(dropped[:, :, :64] == 0)
    assert np.array_equal(sparse[:, :, 64:], data[:, :, 64:])
    known_ids = np.flatnonzero(tokens.reshape(-1))
    unknown_ids = np.flatnonzero(~tokens.reshape(-1))
    assert np.array_equal(sparse[:, known_ids, :64], data[:, known_ids, :64])
    assert np.all(sparse[:, unknown_ids, :64] == 0)
    ids = [47, 13, 0, 26]
    assert np.array_equal(np.asarray(gather_control_context(mx, mx.array(sparse), ids)), sparse[:, ids])
    # Single unknown guide checks exact Euclidean radius3, not square radius3.
    guide = np.zeros((height, width, 3), dtype=np.uint8)
    guide[10, 10] = 255
    guide[40, 20] = 255  # known guide pixels do not enlarge unknown support.
    dilated, _, dilation_info = prepare_sparse_support(np, Image, width, height, rect, guide, 3)
    expected_circle = np.zeros((height, width), dtype=bool)
    yy, xx = np.ogrid[:height, :width]
    expected_circle[(xx - 10) ** 2 + (yy - 10) ** 2 <= 9] = True
    expected_circle[32:96, 16:80] = True
    assert np.array_equal(dilated, expected_circle)
    assert dilation_info["unknown_guide_pixels"] == 1
    empty, empty_tokens, _ = prepare_sparse_support(np, Image, width, height, rect,
                                                   np.zeros_like(guide), 64)
    assert np.array_equal(empty, support) and np.array_equal(empty_tokens, tokens)
    assert not dilated[13, 13] and dilated[13, 10]
    return {
        "status": "passed", "device": "CPU", "model_weights_loaded": False,
        "all_keep_context_exact": True, "all_drop_control64_zero": True,
        "source65_exact_under_all_cases": True, "known_support_polarity_exact": True,
        "packed_nearest_matches_independent_expected_grid": True,
        "absolute_nonmonotonic_target_gather_exact": True,
        "euclidean_dilation_matches_independent_circle": True,
        "empty_guide_matches_known_only": True, "nonzero_context_test": True,
        "known_only_test_metadata": metadata,
    }
