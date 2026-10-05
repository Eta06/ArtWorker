"""Experimental reference-prefix ControlNet extension, separate from the port.

Literal zero129 reference rows are an untrained conditioning extension. They
do not disable hints: biased projections and the first control block's base
joint merge can produce nonzero hints on the full text/image prefix. The
optional gate below changes only completed hints; base representations and
the control chain itself remain intact. No prefix cache is supported.
"""
from __future__ import annotations

import mlx.core as mx
from mlx import nn

from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer_block import Qwen21TransformerBlock


def validate_reference_layout(np, layout, reference_tokens, target_tokens):
    """Context rows index image tokens only; the joint prefix also has text."""
    image_ids = np.asarray(layout.image_indices)
    target_mask = np.asarray(layout.target_mask)
    if layout.target_tokens != target_tokens:
        raise ValueError("Target layout must contain exactly the final target raster")
    if len(image_ids) != reference_tokens + target_tokens:
        raise ValueError("Reference/target image token count mismatch")
    if not (np.all(image_ids[:reference_tokens] < layout.prefix_length)
            and np.array_equal(image_ids[reference_tokens:],
                               np.arange(layout.prefix_length, layout.prefix_length + target_tokens))):
        raise ValueError("Reference image tokens must precede the target raster")
    expected_mask = np.arange(len(target_mask)) >= layout.prefix_length
    if not np.array_equal(target_mask, expected_mask):
        raise ValueError("Hint gate requires the full text+reference prefix mask")
    return {"reference_image_tokens": reference_tokens, "target_tokens": target_tokens,
            "joint_prefix_tokens": layout.prefix_length,
            "prefix_text_tokens": layout.prefix_length - reference_tokens,
            "context_rows": reference_tokens + target_tokens,
            "control_text_padding_rows": 0,
            "full_prefix_hint_gate_uses_target_mask": True}


def assemble_reference_input(source_latents, target_latents):
    if (source_latents.ndim != 3 or target_latents.ndim != 3
            or source_latents.shape[0] != target_latents.shape[0]
            or source_latents.shape[-1] != target_latents.shape[-1]):
        raise ValueError("Reference and target must be compatible packed image latents")
    return mx.concatenate([source_latents, target_latents], axis=1)


def target_known_bridge(target_latents, target_noise, target_context, sigma):
    """Optional flow bridge affects target rows only, never reference input."""
    if (target_context.shape != (*target_latents.shape[:2], 129)
            or target_noise.shape != target_latents.shape
            or target_latents.shape[-1] != 64):
        raise ValueError("Known bridge requires target-only64 latents and target-only129 context")
    known = target_context[:, :, 64:65] > 0.5
    clean = target_context[:, :, 65:129].astype(mx.float32)
    bridge = ((1 - sigma) * clean + sigma * target_noise.astype(mx.float32)).astype(target_latents.dtype)
    return mx.where(known, bridge, target_latents)


def reference_controlled_forward(base, branch, hidden_states, prompt_embeds, timestep,
                                layout, control_context, control_scale=1.0,
                                prefix_hints_disabled=False, cache=None,
                                encoder_hidden_states_mask=None, trace=None):
    """Copied full-joint control forward with an optional post-chain hint gate.

    This is not the official text-only control recipe. ``control_context`` has
    literal zero129 rows for reference IMAGE latents, followed by target rows;
    no text rows are added. All hints are computed before optional gating of
    every text/reference prefix hint. The target hint rows remain exact.
    ``trace`` is a tiny-CPU diagnostic only and records prefix representations.
    """
    if cache is not None:
        raise ValueError("Prefix cache is disabled for experimental reference control")
    if control_context.shape[:2] != hidden_states.shape[:2] or control_context.shape[-1] != branch.in_channels:
        raise ValueError("Control context must align with image tokens without text padding")
    if len(base.transformer_blocks) <= max(branch.control_layers):
        raise ValueError("Control/base layer configuration mismatch")

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
    if prefix_hints_disabled:
        # Gate the completed hints, never control_joint/base_joint/reference
        # representations. This includes text AND reference image positions.
        gate = layout.target_mask[None, :, None].astype(joint.dtype)
        hints = [hint * gate for hint in hints]
    if trace is not None:
        mx.eval(*hints)
        trace["prefix_hints_after"] = [np.asarray(h[:, :layout.prefix_length]).copy() for h in hints]
        trace["target_hints_after"] = [np.asarray(h[:, layout.prefix_length:]).copy() for h in hints]

    for index, block in enumerate(base.transformer_blocks):
        joint, stored = block.forward_reference(
            joint, modulation, layout.target_mask, layout, layout.rope,
            cached=None, extract=False, key_valid=key_valid)
        if stored is not None:
            raise AssertionError("Experimental control path must not extract a prefix cache")
        hint_id = branch.control_layer_mapping.get(index)
        if hint_id is not None:
            joint = joint + hints[hint_id] * control_scale
        mx.eval(joint)
        if trace is not None:
            trace["base_prefix_after_block"].append(np.asarray(joint[:, :layout.prefix_length]).copy())
    scale = Qwen21TransformerBlock.select_rows(base.norm_out.linear(nn.silu(time)), layout.target_mask)
    return base.proj_out(base.norm_out(joint, scale))[:, -layout.target_tokens:]
