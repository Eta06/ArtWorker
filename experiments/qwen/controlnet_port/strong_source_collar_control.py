"""Isolated strong source-top32 completed-hint gate; no runtime edits."""
from __future__ import annotations

import mlx.core as mx
from mlx import nn
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer_block import Qwen21TransformerBlock
from localized_reference_control import runtime_dependency_hashes
from known_collar_saved_baseline import array_sha, bf16_values


def prepare_strong_source_collar_weights(np, requested_scalar1, effective_scalar1, collar_scalar):
    """Copy saved scalar1 fields and alter only known top32 IDs640:704.

    Production arms use 2/4. Scalar0/1 are CPU control cases. Requested FP32
    weights and effective FP32 representations of BF16 stay distinct.
    """
    if (isinstance(collar_scalar, (bool, np.bool_))
            or not isinstance(collar_scalar, (int, float, np.integer, np.floating))
            or not np.isfinite(collar_scalar) or collar_scalar not in (0, 1, 2, 4)):
        raise ValueError("Collar scalar must be one of 0,1,2,4; production uses 2/4")
    fields = (requested_scalar1, effective_scalar1)
    for value in fields:
        if (not isinstance(value, np.ndarray) or value.shape != (1, 2304, 1)
                or str(value.dtype) != "float32" or not np.isfinite(value).all()
                or np.any(value < 0) or np.any(value > 1)):
            raise ValueError("Saved scalar1 field must be finite FP32 [1,2304,1] in [0,1]")
        if not np.all(value[:, 640:704] == 1) or np.any(value[:, 704:1664] != 0):
            raise ValueError("Saved scalar1 known hints must cover top32 only")
    if not np.array_equal(bf16_values(np, requested_scalar1), effective_scalar1):
        raise ValueError("Saved requested/effective scalar1 BF16 values disagree")
    requested, effective = (value.copy() for value in fields)
    requested[:, 640:704] = np.float32(collar_scalar)
    effective[:, 640:704] = np.float32(collar_scalar)
    unchanged = np.ones(2304, dtype=bool); unchanged[640:704] = False
    if (not np.array_equal(requested[:, unchanged], requested_scalar1[:, unchanged])
            or not np.array_equal(effective[:, unchanged], effective_scalar1[:, unchanged])
            or not np.array_equal(effective, bf16_values(np, requested))):
        raise AssertionError("Strong collar escaped top32 or changed saved BF16 values")
    changed = np.flatnonzero((requested != requested_scalar1).reshape(-1))
    metadata = {"collar_scalar": float(collar_scalar), "root_control_scale": 1.0,
        "production_arm": bool(collar_scalar in (2, 4)), "top32_absolute_ids_half_open": [640, 704],
        "changed_token_ids": changed.tolist(), "changed_token_count": len(changed),
        "changed_packed_rows": sorted(set((changed // 32).tolist())),
        "unknown_requested_and_effective_fields_exact": True,
        "known_outside_top32_hints_zero": True, "reference_prefix_hints_zero": True,
        "requested_weight_sha256": array_sha(requested), "effective_weight_sha256": array_sha(effective),
        "saved_requested_scalar1_sha256": array_sha(requested_scalar1),
        "saved_effective_scalar1_sha256": array_sha(effective_scalar1),
        "completed_hints_gated_after_all16_control_blocks": True,
        "raw_control_chain_unchanged_by_scalar": True, "output_linear_scaling_claim": False}
    return requested, effective, metadata


def strong_source_collar_controlled_forward(base, branch, hidden_states, prompt_embeds, timestep,
                                          layout, control_context, target_hint_weights,
                                          control_scale=1.0, cache=None,
                                          encoder_hidden_states_mask=None, trace=None, *, allowed_collar_mask):
    """Frozen completed-hint forward with a declared strong known collar.

    All sixteen raw hints compute before gating. Only caller-declared known
    target collar rows may exceed one, up to four. Prefix gates remain zero.
    This does not rescale the raw control chain or promise linear outputs.
    """
    if cache is not None:
        raise ValueError("Prefix cache is disabled for localized reference control")
    if control_context.shape[:2] != hidden_states.shape[:2] or control_context.shape[-1] != branch.in_channels:
        raise ValueError("Control context must align with image tokens without text padding")
    if target_hint_weights.shape != (hidden_states.shape[0], layout.target_tokens, 1):
        raise ValueError("Hint weights must index target raster only")
    if len(base.transformer_blocks) <= max(branch.control_layers):
        raise ValueError("Control/base layer configuration mismatch")
    if allowed_collar_mask.shape != target_hint_weights.shape or allowed_collar_mask.dtype != mx.bool_:
        raise ValueError("Allowed collar mask must be boolean and align with target hint weights")
    known_mask = control_context[:, -layout.target_tokens:, 64:65] > 0.5
    if bool(mx.any(allowed_collar_mask & ~known_mask)):
        raise ValueError("Allowed strong collar must cover known target rows only")
    upper_bound = mx.where(allowed_collar_mask, 4.0, 1.0)
    if not bool(mx.all(mx.isfinite(target_hint_weights) & (target_hint_weights >= 0) & (target_hint_weights <= upper_bound))):
        raise ValueError("Hints must be finite, nonnegative, <=4 inside the allowed collar and <=1 elsewhere")
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
