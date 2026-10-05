"""Isolated known-top-collar extension of frozen post-chain hint localization.

The copied unknown-guide field stays bitwise unchanged. A declared scalar may
enable hints only on the first 32 (or optionally64) pixels of the known source.
All16 hints are computed before gating. Prefix hints remain zero, and no bottom
or deeper source hints are enabled. Base/source/control states are not masked.
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


def prepare_known_collar_hint_weights(np, Image, width, height, source_rect, guide_rgb,
                                      radius_pixels=64, shape="cosine",
                                      known_collar_pixels=32, known_collar_weight=1.0):
    """Add an explicit known-top field to the copied original unknown field.

    Zero pixels or zero scalar reproduce the original field exactly. This is
    scalar gating of completed hints only; it does not edit context129 or the
    per-step target known-latent bridge.
    """
    if (not isinstance(known_collar_pixels,(int,np.integer)) or isinstance(known_collar_pixels,(bool,np.bool_))
            or known_collar_pixels not in (0,32,64)):
        raise ValueError("Known collar must be 0,32, or64 pixels")
    known_collar_pixels=int(known_collar_pixels)
    if not np.isfinite(known_collar_weight) or not 0<=known_collar_weight<=1:
        raise ValueError("Known collar scalar must be finite in [0,1]")
    from conditioning import _dimensions
    x1,y1,x2,y2=_dimensions(width,height,source_rect)
    if known_collar_pixels>y2-y1 or y1%16 or known_collar_pixels%16:
        raise ValueError("Known top collar must fit the source and align to packed rows")
    pixels,old_packed,info=prepare_local_hint_weights(np,Image,width,height,source_rect,
                                                    guide_rgb,radius_pixels,shape)
    old_pixels=pixels.copy()
    collar=np.zeros((height,width),dtype=bool)
    collar[y1:y1+known_collar_pixels,x1:x2]=True
    if known_collar_pixels and known_collar_weight:
        pixels[collar]=np.float32(known_collar_weight)
    ys=np.arange(height//16,dtype=np.int64)*height//(height//16)
    xs=np.arange(width//16,dtype=np.int64)*width//(width//16)
    packed=np.ascontiguousarray(pixels[ys[:,None],xs[None,:]].reshape(1,-1,1))
    known=np.zeros((height,width),dtype=bool);known[y1:y2,x1:x2]=True
    if not np.array_equal(pixels[~known],old_pixels[~known]):
        raise AssertionError("Known collar changed unknown-guide field")
    if np.any(pixels[known & ~collar]!=0) or not np.isfinite(packed).all():
        raise AssertionError("Hints escaped the intended known top collar")
    if not known_collar_pixels or not known_collar_weight:
        if not np.array_equal(pixels,old_pixels) or not np.array_equal(packed,old_packed):
            raise AssertionError("Zero collar must reproduce original localization exactly")
    packed_collar=collar[ys[:,None],xs[None,:]].reshape(1,-1,1)
    packed_known=known[ys[:,None],xs[None,:]].reshape(1,-1,1)
    info.update({"experimental_known_top_collar_hints":True,
                 "known_collar_pixels":known_collar_pixels,"known_collar_scalar_weight":float(known_collar_weight),
                 "source_roi_always_zero":not bool(known_collar_pixels and known_collar_weight),
                 "known_collar_pixel_rows_half_open":[y1,y1+known_collar_pixels],
                 "known_collar_packed_rows_half_open":[y1//16,(y1+known_collar_pixels)//16],
                 "known_collar_positive_tokens":int(((packed>0)&packed_collar).sum()),
                 "known_outside_collar_positive_tokens":int(((packed>0)&packed_known&~packed_collar).sum()),
                 "unknown_field_exact_against_old_localized":True,
                 "bottom_and_source_interior_hints_zero":True,
                 "reference_prefix_hints_zero":True,"source_conditioning_channels_changed":False,
                 "positive_hint_pixels":int((pixels>0).sum()),"positive_hint_target_tokens":int((packed>0).sum()),
                 "target_hint_weight_sum":float(packed.sum()),
                 "pixel_weights_float32_sha256":hashlib.sha256(pixels.tobytes()).hexdigest(),
                 "target_weights_float32_sha256":hashlib.sha256(packed.tobytes()).hexdigest(),
                 "old_target_weights_float32_sha256":hashlib.sha256(old_packed.tobytes()).hexdigest()})
    return pixels,packed,info


def known_collar_reference_controlled_forward(base, branch, hidden_states, prompt_embeds, timestep,
                                          layout, control_context, target_hint_weights,
                                          control_scale=1.0, cache=None,
                                          encoder_hidden_states_mask=None, trace=None):
    """Validated reference forward mechanics plus a completed-hint scalar field.

    Only hints are multiplied. Known target states can still change indirectly
    through target attention; zero direct hints do not freeze those states.
    Prefix gate is always zero. Target field is
    caller-supplied [1,target_tokens,1] in [0,1]; production construction permits
    only the declared known top collar in addition to the original unknown-guide
    support. An all-one target field reproduces the validated
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
