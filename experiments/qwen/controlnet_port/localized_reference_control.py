"""Experimental post-chain hint localization; full denoiser compute remains.

All16 control hints are computed on the unmodified full joint before scalar
gating. Prefix and known-source target hints are zero. Unknown target hints
use only distance to a supplied input guide; base/source/control states are
never masked and geometry never comes from generated output.
"""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

import mlx.core as mx
from mlx import nn
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer_block import Qwen21TransformerBlock


def runtime_dependency_hashes():
    """Hash the actual imported modules without changing device selection."""
    names = ("qwen21_attention", "qwen21_transformer_block", "qwen21_layout",
             "qwen21_transformer", "qwen21_norm_out", "qwen21_text_projection",
             "qwen21_time_text_embed", "qwen21_rope")
    records = {}
    for name in names:
        path = Path(importlib.import_module("mflux.models.qwen21.model.qwen21_transformer." + name).__file__).resolve()
        records[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return records


def prepare_local_hint_weights(np, Image, width, height, source_rect, guide_rgb,
                               radius_pixels=64, shape="cosine"):
    from conditioning import _dimensions, _rgb_array
    x1, y1, x2, y2 = _dimensions(width, height, source_rect)
    if not isinstance(radius_pixels, int) or not 0 <= radius_pixels <= max(width, height):
        raise ValueError("Hint radius must be a bounded nonnegative integer")
    if shape not in ("cosine", "hard"):
        raise ValueError("Hint shape must be cosine or hard")
    rgb = _rgb_array(np, Image, guide_rgb, width, height, "hint_guide_rgb")
    known = np.zeros((height, width), dtype=bool)
    known[y1:y2, x1:x2] = True
    guide = np.any(rgb != 0, axis=-1) & ~known
    weights = np.zeros((height, width), dtype=np.float32)
    distances = np.full((height, width), np.inf, dtype=np.float32)
    if guide.any():
        import cv2
        distances = cv2.distanceTransform((~guide).astype(np.uint8), cv2.DIST_L2,
                                          cv2.DIST_MASK_PRECISE)
        if radius_pixels == 0:
            weights[guide] = 1
        elif shape == "hard":
            weights[distances <= radius_pixels] = 1
        else:
            inside = distances < radius_pixels
            weights[inside] = 0.5 * (1 + np.cos(np.pi * distances[inside] / float(radius_pixels)))
            weights[guide] = 1
    weights[known] = 0
    ys = np.arange(height // 16, dtype=np.int64) * height // (height // 16)
    xs = np.arange(width // 16, dtype=np.int64) * width // (width // 16)
    packed = np.ascontiguousarray(weights[ys[:, None], xs[None, :]].reshape(1, -1, 1))
    known_packed = known[ys[:, None], xs[None, :]].reshape(1, -1, 1)
    if np.any(packed[known_packed] != 0) or not np.isfinite(packed).all():
        raise AssertionError("Known-source hint weights must be zero and all weights finite")
    metadata = {"experimental_localized_hints": True, "shape": shape,
                "radius_pixels": radius_pixels, "distance_metric": "exact Euclidean L2 pixel distance",
                "geometry_source": "nonblack input guide outside source ROI only; no output-derived geometry",
                "cosine_formula": "0.5*(1+cos(pi*distance/radius)) for distance<radius; otherwise0" if shape == "cosine" else None,
                "source_roi_always_zero": True, "source_rect_xyxy": list(source_rect),
                "unknown_guide_pixels": int(guide.sum()), "positive_hint_pixels": int((weights > 0).sum()),
                "positive_hint_target_tokens": int((packed > 0).sum()), "total_target_tokens": packed.shape[1],
                "target_hint_weight_sum": float(packed.sum()), "packed_shape": list(packed.shape),
                "nearest_coordinates": "floor(output_index*input_size/output_size), raster y-major",
                "guide_rgb_uint8_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                "pixel_weights_float32_sha256": hashlib.sha256(weights.tobytes()).hexdigest(),
                "target_weights_float32_sha256": hashlib.sha256(packed.tobytes()).hexdigest(),
                "full_joint_computation": True, "sparse_compute": False}
    return weights, packed, metadata


def localized_reference_controlled_forward(base, branch, hidden_states, prompt_embeds, timestep,
                                          layout, control_context, target_hint_weights,
                                          control_scale=1.0, cache=None,
                                          encoder_hidden_states_mask=None, trace=None):
    """Validated reference forward mechanics plus a completed-hint scalar field.

    Only hints are multiplied. Known target states can still change indirectly
    through target attention; zero direct hints do not freeze those states.
    Prefix gate is always zero. Target field is
    caller-supplied [1,target_tokens,1] in [0,1]; production construction sets
    known source ROI to zero. An all-one target field reproduces the validated
    prefix-hints-disabled forward; all-zero reproduces uncached base output.
    """
    if cache is not None:
        raise ValueError("Prefix cache is disabled for localized reference control")
    if control_context.shape[:2] != hidden_states.shape[:2] or control_context.shape[-1] != branch.in_channels:
        raise ValueError("Control context must align with image tokens without text padding")
    if target_hint_weights.shape != (hidden_states.shape[0], layout.target_tokens, 1):
        raise ValueError("Hint weights must index target raster only")
    if len(base.transformer_blocks) <= max(branch.control_layers):
        raise ValueError("Control/base layer configuration mismatch")
    if not bool(mx.all(mx.isfinite(target_hint_weights) & (target_hint_weights >= 0) & (target_hint_weights <= 1))):
        raise ValueError("Hint weights must be finite and in [0,1]")
    images = base.img_in(hidden_states)
    text = base.txt_in(prompt_embeds)
    joint = mx.concatenate([text, mx.zeros((text.shape[0], layout.target_tokens // 4,
                                           text.shape[-1]), dtype=text.dtype)], axis=1)
    joint = joint[:, layout.repeat_indices]
    joint[:, layout.image_indices] = images
    control_joint = mx.zeros_like(joint)
    control_joint[:, layout.image_indices] = branch.control_img_in(control_context.astype(joint.dtype))
    key_valid = None
    if encoder_hidden_states_mask is not None:
        mask = mx.concatenate([encoder_hidden_states_mask.astype(mx.bool_),
                               mx.ones((joint.shape[0], layout.target_tokens // 4), dtype=mx.bool_)], axis=1)
        key_valid = mask[:, layout.repeat_indices]
        key_valid[:, layout.image_indices] = True
    time_rows = mx.concatenate([timestep.astype(joint.dtype).reshape(-1),
                                mx.zeros((1,), dtype=joint.dtype)])
    time = base.time_text_embed(time_rows, joint.dtype)
    modulation = base.modulation(time)
    hints = branch.forward_hints(control_joint, joint, modulation,
                                 layout.target_mask, layout, layout.rope, key_valid)
    if trace is not None:
        import numpy as np
        mx.eval(joint, *hints)
        trace["base_prefix_initial"] = np.asarray(joint[:, :layout.prefix_length]).copy()
        trace["prefix_hints_before"] = [np.asarray(h[:, :layout.prefix_length]).copy() for h in hints]
        trace["target_hints_before"] = [np.asarray(h[:, layout.prefix_length:]).copy() for h in hints]
        trace["base_prefix_after_block"] = []
    gate = mx.concatenate([mx.zeros((joint.shape[0], layout.prefix_length, 1), dtype=joint.dtype),
                           target_hint_weights.astype(joint.dtype)], axis=1)
    hints = [hint * gate for hint in hints]
    if trace is not None:
        mx.eval(*hints)
        trace["joint_gate"] = np.asarray(gate).copy()
        trace["prefix_hints_after"] = [np.asarray(h[:, :layout.prefix_length]).copy() for h in hints]
        trace["target_hints_after"] = [np.asarray(h[:, layout.prefix_length:]).copy() for h in hints]
    for index, block in enumerate(base.transformer_blocks):
        joint, stored = block.forward_reference(
            joint, modulation, layout.target_mask, layout, layout.rope,
            cached=None, extract=False, key_valid=key_valid)
        if stored is not None:
            raise AssertionError("Localized control path must not extract a prefix cache")
        hint_id = branch.control_layer_mapping.get(index)
        if hint_id is not None:
            joint = joint + hints[hint_id] * control_scale
        mx.eval(joint)
        if trace is not None:
            trace["base_prefix_after_block"].append(np.asarray(joint[:, :layout.prefix_length]).copy())
    scale = Qwen21TransformerBlock.select_rows(base.norm_out.linear(nn.silu(time)), layout.target_mask)
    return base.proj_out(base.norm_out(joint, scale))[:, -layout.target_tokens:]
