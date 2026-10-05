"""Absolute target conditioning gathers for the experimental Turbo/control path."""
from __future__ import annotations

import mlx.core as mx
from conditioning import control_prefix_padding, gather_control_context


def gather_active_conditioning(np, target_context, full_target_hint_weights,
                               absolute_ids, reference_tokens=1024):
    ids = np.asarray(absolute_ids)
    if (ids.ndim != 1 or ids.dtype.kind not in "iu" or len(ids) == 0
            or np.any(ids < 0) or np.any(ids >= target_context.shape[1])
            or np.any(np.diff(ids.astype(np.int64)) <= 0)):
        raise ValueError("Active conditioning requires increasing unique final raster IDs")
    if full_target_hint_weights.shape != (target_context.shape[0], target_context.shape[1], 1):
        raise ValueError("Hint weights must align with the final target context")
    active_context = gather_control_context(mx, target_context, ids.astype(np.int32))
    active_hint_weights = full_target_hint_weights[:, mx.array(ids, dtype=mx.int32)]
    padded = control_prefix_padding(mx, active_context, reference_tokens)
    if padded.shape != (1, reference_tokens + len(ids), 129):
        raise AssertionError("Future/reference/text rows leaked into active control context")
    return active_context, padded, active_hint_weights


def edge_known_flow_bridge(latents, known_mask, known_clean, active_noise, sigma):
    """Old geometry clean target codec; distinct from masked-source129 channels."""
    if (known_mask.shape != (*latents.shape[:2], 1)
            or known_clean.shape != latents.shape or active_noise.shape != latents.shape):
        raise ValueError("Edge-known bridge must align with active target latents only")
    known = ((1 - sigma) * known_clean.astype(mx.float32)
             + sigma * active_noise.astype(mx.float32)).astype(latents.dtype)
    return mx.where(known_mask, known, latents)


def predicted_clean_with_known(before, prediction, known_mask, known_clean, sigma):
    if prediction.shape != before.shape or known_clean.shape != before.shape:
        raise ValueError("Predicted clean must cover the current active target only")
    clean = (before.astype(mx.float32) - sigma * prediction.astype(mx.float32)).astype(before.dtype)
    return mx.where(known_mask, known_clean, clean)
