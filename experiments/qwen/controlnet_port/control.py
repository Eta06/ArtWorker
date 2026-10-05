"""Isolated MLX Qwen-Image-2.1 Fun ControlNet-Union reference port.

Architecture adapted from Alibaba PAI / VideoX-Fun, Apache-2.0 code,
revision 4b7b6402a1e0f0406bd6801fb66c0a00bd922621. Checkpoint licence is
Qwen Research, independently of runtime code. Existing MFLUX modules are
reused but never modified. Full joint sequence is recomputed every step;
prefix caching with control is deliberately unsupported, matching the
current official pipeline. See README.md for conditioning and parity scope.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten

from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer_block import Qwen21TransformerBlock


UPSTREAM_REVISION = "4b7b6402a1e0f0406bd6801fb66c0a00bd922621"
CHECKPOINT_REVISION = "8a4702014d4dabb5f896fcba917e2ee0a961465f"
CHECKPOINT_FILENAME = "Qwen-Image-2.1-Fun-Controlnet-Union.safetensors"
CHECKPOINT_BYTES = 7_550_979_904
CHECKPOINT_SHA256 = "65d6b66d734da9e7ff5ef04e7db3a133553a52a3f29a7fcb3e9cce8fa21dcfcd"


class ControlBlock(Qwen21TransformerBlock):
    def __init__(self, dim, num_heads, head_dim, mlp_ratio=3, eps=1e-6, first=False):
        super().__init__(dim=dim, num_attention_heads=num_heads,
                         attention_head_dim=head_dim, mlp_ratio=mlp_ratio, eps=eps)
        if first:
            self.before_proj = nn.Linear(dim, dim, bias=True)
        self.after_proj = nn.Linear(dim, dim, bias=True)
        self.first = first

    def forward_control(self, control, base_joint, modulation, target_mask,
                        layout, rope, key_valid=None):
        if self.first:
            control = self.before_proj(control)+base_joint
        control, stored = super().forward_reference(
            control,modulation,target_mask,layout,rope,cached=None,
            extract=False,key_valid=key_valid)
        if stored is not None:
            raise AssertionError("Control branch must not extract a prefix cache")
        return self.after_proj(control), control


class ControlBranch(nn.Module):
    def __init__(self, dim=4096, num_heads=32, head_dim=128, mlp_ratio=3,
                 num_layers=32, control_layers=None, in_channels=129, eps=1e-6):
        super().__init__()
        if dim != num_heads*head_dim:
            raise ValueError("Control attention dimension mismatch")
        self.control_layers = list(range(0,num_layers,2)) if control_layers is None else list(control_layers)
        if not self.control_layers or self.control_layers[0] != 0 or len(set(self.control_layers)) != len(self.control_layers):
            raise ValueError("Control injection layers must be unique and begin at base block zero")
        if any(i<0 or i>=num_layers for i in self.control_layers):
            raise ValueError("Invalid base control injection layer")
        self.control_layer_mapping = {layer:index for index,layer in enumerate(self.control_layers)}
        self.control_img_in = nn.Linear(in_channels,dim,bias=True)
        self.control_blocks = [ControlBlock(dim,num_heads,head_dim,mlp_ratio,eps,first=i==0)
                               for i in self.control_layers]
        self.in_channels = in_channels

    def forward_hints(self, control_joint, base_joint, modulation, target_mask,
                      layout, rope, key_valid=None):
        control = control_joint
        hints = []
        for block in self.control_blocks:
            hint, control = block.forward_control(
                control,base_joint,modulation,target_mask,layout,rope,key_valid)
            # The upstream stacks all earlier hints and running state. Keeping
            # separate tensors preserves identical math without stack copies.
            mx.eval(hint,control)
            hints.append(hint)
        return hints


def controlled_forward(base, branch, hidden_states, prompt_embeds, timestep,
                       layout, control_context, control_scale=1.0, cache=None,
                       encoder_hidden_states_mask=None):
    """Full-joint reference forward with VACE hints injected AFTER base blocks.

    hidden_states contains any image-reference tokens followed by current
    target tokens. control_context must align with every image token one-to-one.
    Official control inpaint conditioning has no image-reference prefix. A
    caller adding references must explicitly supply corresponding control rows
    and label that conditioning as a separate experimental extension.
    Absolute target RoPE is supplied by layout; no resizing/recentering occurs.
    """
    if cache is not None:
        raise ValueError("Prefix cache is disabled whenever the control branch is active")
    if control_context.shape[:2] != hidden_states.shape[:2] or control_context.shape[-1] != branch.in_channels:
        raise ValueError("Control context must align with every current image token")
    if len(base.transformer_blocks) <= max(branch.control_layers):
        raise ValueError("Control/base layer configuration mismatch")

    images = base.img_in(hidden_states)
    text = base.txt_in(prompt_embeds)
    joint = mx.concatenate([text,mx.zeros((text.shape[0],layout.target_tokens//4,text.shape[-1]),dtype=text.dtype)],axis=1)
    joint = joint[:,layout.repeat_indices]
    joint[:,layout.image_indices] = images

    control_joint = mx.zeros_like(joint)
    control_joint[:,layout.image_indices] = branch.control_img_in(control_context.astype(joint.dtype))

    key_valid = None
    if encoder_hidden_states_mask is not None:
        mask = mx.concatenate([encoder_hidden_states_mask.astype(mx.bool_),
                               mx.ones((joint.shape[0],layout.target_tokens//4),dtype=mx.bool_)],axis=1)
        key_valid = mask[:,layout.repeat_indices]
        key_valid[:,layout.image_indices] = True

    time_rows = mx.concatenate([timestep.astype(joint.dtype).reshape(-1),mx.zeros((1,),dtype=joint.dtype)])
    time = base.time_text_embed(time_rows,joint.dtype)
    modulation = base.modulation(time)
    hints = branch.forward_hints(control_joint,joint,modulation,
                                 layout.target_mask,layout,layout.rope,key_valid)
    for index,block in enumerate(base.transformer_blocks):
        joint,stored = block.forward_reference(
            joint,modulation,layout.target_mask,layout,layout.rope,
            cached=None,extract=False,key_valid=key_valid)
        if stored is not None:
            raise AssertionError("Base control path must not extract a prefix cache")
        hint_id = branch.control_layer_mapping.get(index)
        if hint_id is not None:
            joint = joint+hints[hint_id]*control_scale
        mx.eval(joint)
    scale = Qwen21TransformerBlock.select_rows(base.norm_out.linear(nn.silu(time)),layout.target_mask)
    output = base.proj_out(base.norm_out(joint,scale))[:,-layout.target_tokens:]
    return output


def checkpoint_header(path):
    with Path(path).open("rb") as stream:
        size = struct.unpack("<Q",stream.read(8))[0]
        header_bytes = stream.read(size)
    header = json.loads(header_bytes)
    return header,hashlib.sha256(header_bytes).hexdigest()


def load_control_branch(path,quantize=None):
    """Strictly load all 180 trained branch tensors; no partial-key fallback.

    Call after deleting the text encoder, to avoid retaining the encoder while
    this ~7.55 GB BF16 branch loads. Quantization is optional and changes the
    checkpoint numerics; record it as an independent inference approximation.
    """
    path = Path(path)
    if path.stat().st_size != CHECKPOINT_BYTES:
        raise ValueError("Wrong/incomplete pinned Union checkpoint size")
    header,header_sha = checkpoint_header(path)
    tensor_names = [name for name in header if name!="__metadata__"]
    if len(tensor_names) != 180 or any(not name.startswith(("control_blocks.","control_img_in.")) for name in tensor_names):
        raise ValueError("Unexpected Union tensor names/count")
    if any(header[name]["dtype"]!="BF16" for name in tensor_names):
        raise ValueError("Pinned branch must contain original BF16 weights")
    branch = ControlBranch()
    expected = dict(tree_flatten(branch.parameters()))
    if set(expected)!=set(tensor_names):
        raise ValueError(f"Strict Union key mismatch: missing={set(expected)-set(tensor_names)}, extra={set(tensor_names)-set(expected)}")
    for name,param in expected.items():
        if list(param.shape)!=header[name]["shape"]:
            raise ValueError(f"Checkpoint shape mismatch: {name}")
    weights = mx.load(str(path))
    branch.load_weights(list(weights.items()),strict=True)
    del weights,expected
    if quantize is not None:
        if quantize not in (4,8):
            raise ValueError("Only experimental Q4/Q8 control quantization supported")
        nn.quantize(branch,group_size=64,bits=quantize,
                    class_predicate=lambda _,module:isinstance(module,nn.Linear) and module.weight.shape[-1]%64==0)
    mx.eval(branch.parameters())
    return branch,{"checkpoint_revision":CHECKPOINT_REVISION,"checkpoint_filename":CHECKPOINT_FILENAME,
                   "checkpoint_bytes":CHECKPOINT_BYTES,"expected_checkpoint_sha256":CHECKPOINT_SHA256,
                   "header_sha256":header_sha,"tensor_count":len(tensor_names),
                   "strict_keys_and_shapes":True,"control_layers":branch.control_layers,
                   "control_in_channels":129,"control_quantization":quantize,
                   "loaded_parameter_bytes":sum(value.nbytes for _,value in tree_flatten(branch.parameters())),
                   "upstream_revision":UPSTREAM_REVISION,"prefix_cache_enabled":False}
