"""Strict, in-memory bridge for the pinned public MaskFlow ``latest`` LoRA.

The downloaded checkpoint is never rewritten.  Header inspection and planning
need only the standard library; actual application imports the isolated MLX-Gen
runtime lazily and keeps its quantized base linears unbaked.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
import json
import math
from pathlib import Path
import re
import struct
from typing import Any, Mapping

BASE_REPOSITORY = "AbstractFramework/qwen-image-edit-2511-4bit"
BASE_REVISION = "dbc6d597c1316fc34c0f7726904121a284a69c0b"
ADAPTER_REPOSITORY = "ReyChiaro/MaskFlow"
ADAPTER_REVISION = "f18f0828ab6a40a7f3456feace7a33bebe8c511e"
ADAPTER_SHA256 = "773e815a209ebf948dbe0cf76492dd658cc16030f8119471cc553edb208de7b0"
ADAPTER_SIZE_BYTES = 2453820032
MODULE_PATHS = {
    "attn.to_q": "attn.to_q",
    "attn.to_k": "attn.to_k",
    "attn.to_v": "attn.to_v",
    "attn.to_out.0": "attn.attn_to_out.0",
    "attn.add_q_proj": "attn.add_q_proj",
    "attn.add_k_proj": "attn.add_k_proj",
    "attn.add_v_proj": "attn.add_v_proj",
    "attn.to_add_out": "attn.to_add_out",
    "img_mlp.net.2": "img_ff.mlp_out",
    "txt_mlp.net.2": "txt_ff.mlp_out",
}
_KEY = re.compile(r"^transformer_blocks\.(\d+)\.(.+)\.lora_([AB])\.weight$")


@dataclass(frozen=True)
class AdapterSpec:
    rank: int
    alpha: float
    alpha_over_rank: float
    use_rslora: bool = False
    use_dora: bool = False


@dataclass(frozen=True)
class AdapterTarget:
    source_module: str
    target_module: str
    down_key: str
    up_key: str
    rank: int
    input_dims: int
    output_dims: int
    base_shard: str | None = None


def read_safetensors_header(path: str | Path) -> dict[str, Any]:
    """Read only the length prefix and JSON header, never tensor payloads."""
    with Path(path).open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("Truncated safetensors length prefix")
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= 5 * 1024 * 1024:
            raise ValueError(f"Invalid safetensors header length: {length}")
        payload = stream.read(length)
        if len(payload) != length:
            raise ValueError("Truncated safetensors header")
    header = json.loads(payload)
    if not isinstance(header, dict):
        raise ValueError("Safetensors header must be a JSON object")
    return header


def adapter_spec(header: Mapping[str, Any]) -> AdapterSpec:
    """Reject unsupported PEFT variants rather than silently changing scaling."""
    metadata = header.get("__metadata__", {})
    config = json.loads(metadata["lora_adapter_metadata"])
    for flag in ("use_rslora", "use_dora", "use_qalora"):
        if config.get(flag):
            raise ValueError(f"Unsupported adapter mode: {flag}")
    for field in ("rank_pattern", "alpha_pattern", "modules_to_save", "target_parameters"):
        if config.get(field):
            raise ValueError(f"Unsupported adapter configuration: {field}")
    if config.get("fan_in_fan_out") or config.get("lora_bias") or config.get("bias") != "none":
        raise ValueError("Only plain, bias-free LoRA is supported")
    if config.get("peft_type") != "LORA":
        raise ValueError("Expected plain PEFT LORA metadata")
    rank, alpha = config.get("r"), config.get("lora_alpha")
    if rank != 256 or alpha != 256:
        raise ValueError("Pinned latest adapter requires rank=256 and alpha=256")
    if set(config.get("target_modules", [])) != set(MODULE_PATHS):
        raise ValueError("Pinned latest adapter target modules differ")
    return AdapterSpec(rank=rank, alpha=float(alpha), alpha_over_rank=float(alpha) / rank)


def build_adapter_plan(
    header: Mapping[str, Any],
    base_header: Mapping[str, Any] | None = None,
    base_weight_map: Mapping[str, str] | None = None,
) -> tuple[AdapterSpec, list[AdapterTarget]]:
    """Validate every A/B pair and optionally every packed-Q4 base dimension."""
    spec = adapter_spec(header)
    modules: dict[str, dict[str, tuple[str, Mapping[str, Any]]]] = {}
    for key, tensor in header.items():
        if key == "__metadata__":
            continue
        match = _KEY.fullmatch(key)
        if match is None:
            raise ValueError(f"Unrecognized adapter tensor: {key}")
        block, suffix, matrix = match.groups()
        if not 0 <= int(block) < 60 or str(int(block)) != block or suffix not in MODULE_PATHS:
            raise ValueError(f"Unexpected adapter target: {key}")
        if tensor.get("dtype") != "BF16" or len(tensor.get("shape", [])) != 2:
            raise ValueError(f"Expected rank-two BF16 tensor: {key}")
        shape = tensor["shape"]
        if any(type(d) is not int or d <= 0 for d in shape):
            raise ValueError(f"Invalid adapter dimensions: {key}")
        offsets = tensor.get("data_offsets", [])
        if len(offsets) != 2 or offsets[1] - offsets[0] != math.prod(shape) * 2:
            raise ValueError(f"Invalid adapter tensor byte length: {key}")
        source = f"transformer_blocks.{block}.{suffix}"
        modules.setdefault(source, {})[matrix] = (key, tensor)
    expected = {f"transformer_blocks.{block}.{suffix}" for block in range(60) for suffix in MODULE_PATHS}
    if set(modules) != expected:
        raise ValueError(f"Expected all 600 adapter targets; found {len(modules)}")
    plans = []
    for source in sorted(modules):
        matrices = modules[source]
        if set(matrices) != {"A", "B"}:
            raise ValueError(f"Incomplete A/B pair: {source}")
        down_key, down = matrices["A"]
        up_key, up = matrices["B"]
        rank, input_dims = down["shape"]
        output_dims, up_rank = up["shape"]
        if rank != spec.rank or up_rank != rank:
            raise ValueError(f"Rank mismatch: {source}")
        parts = source.split(".")
        suffix = ".".join(parts[2:])
        target = f"transformer_blocks.{parts[1]}.{MODULE_PATHS[suffix]}"
        weight_key = target + ".weight"
        if base_weight_map is not None:
            for ending in ("weight", "scales", "biases"):
                if target + "." + ending not in base_weight_map:
                    raise ValueError(f"Base index lacks target: {target}.{ending}")
        if base_header is not None:
            weight = base_header.get(weight_key, {})
            scales = base_header.get(target + ".scales", {})
            biases = base_header.get(target + ".biases", {})
            if weight.get("dtype") != "U32" or weight.get("shape") != [output_dims, input_dims // 8]:
                raise ValueError(f"Packed Q4 base weight does not match LoRA: {target}")
            expected_groups = [output_dims, input_dims // 64]
            if input_dims % 64 or scales.get("shape") != expected_groups or biases.get("shape") != expected_groups:
                raise ValueError(f"Group64 scales/biases do not match LoRA: {target}")
            if scales.get("dtype") != "BF16" or biases.get("dtype") != "BF16":
                raise ValueError(f"Expected BF16 affine Q4 scales/biases: {target}")
        plans.append(AdapterTarget(source, target, down_key, up_key, rank, input_dims, output_dims,
                                   base_weight_map.get(weight_key) if base_weight_map is not None else None))
    return spec, plans


def bridge_maskflow_weights(weights: Mapping[str, Any], header: Mapping[str, Any]) -> dict[str, Any]:
    """Prefix existing tensors by reference; the runtime transposes A/B once."""
    _, plans = build_adapter_plan(header)
    expected = {key for p in plans for key in (p.down_key, p.up_key)}
    if set(weights) != expected:
        raise ValueError("Loaded tensor keys differ from validated adapter header")
    for key in expected:
        if list(weights[key].shape) != header[key]["shape"]:
            raise ValueError(f"Loaded adapter shape differs from header: {key}")
    return {prefix_maskflow_key(key): value for key, value in weights.items()}


def prefix_maskflow_key(key: str) -> str:
    """Bridge one recognized raw key without transforming its tensor."""
    match = _KEY.fullmatch(key)
    if match is None or not 0 <= int(match[1]) < 60 or match[2] not in MODULE_PATHS:
        raise ValueError(f"Unrecognized MaskFlow key: {key}")
    return "diffusion_model." + key


def target_module(transformer: Any, path: str) -> Any:
    """Resolve the model's object/list/dict path before replacing any layer."""
    module = transformer
    for part in path.split("."):
        if part.isdigit():
            module = module[int(part)]
        elif isinstance(module, dict) and part in module:
            module = module[part]
        else:
            module = getattr(module, part)
    return module


def apply_maskflow_adapter(
    transformer: Any,
    weights: Mapping[str, Any],
    header: Mapping[str, Any],
    *,
    scale: float = 1.0,
    role: str = "maskflow",
    runtime_package: str = "mflux",
) -> dict[str, Any]:
    """Apply only after full preflight; never bake/dequantize the base weights.

    The caller controls device selection and file loading.  The default package
    is the isolated pinned MLX-Gen runtime, not the pre-existing mflux checkout.
    """
    if not math.isfinite(scale):
        raise ValueError("Adapter scale must be finite")
    spec, plans = build_adapter_plan(header)
    bridged = bridge_maskflow_weights(weights, header)
    nn = importlib.import_module("mlx.nn")
    mx = importlib.import_module("mlx.core")
    loader = importlib.import_module(runtime_package + ".models.common.lora.mapping.lora_loader").LoRALoader
    mapping = importlib.import_module(runtime_package + ".models.qwen.weights.qwen_lora_mapping").QwenLoRAMapping
    for plan in plans:
        module = target_module(transformer, plan.target_module)
        if (not isinstance(module, nn.QuantizedLinear) or module.bits != 4 or module.group_size != 64
                or module.mode != "affine"):
            raise ValueError(f"Expected untouched group64 Q4 target: {plan.target_module}")
        if (tuple(module.weight.shape) != (plan.output_dims, plan.input_dims // 8)
                or module.weight.dtype != mx.uint32):
            raise ValueError(f"Runtime base dimensions differ: {plan.target_module}")
        for name in ("scales", "biases"):
            parameter = getattr(module, name)
            if (tuple(parameter.shape) != (plan.output_dims, plan.input_dims // 64)
                    or parameter.dtype != mx.bfloat16):
                raise ValueError(f"Runtime base affine parameters differ: {plan.target_module}.{name}")
        for key in (plan.down_key, plan.up_key):
            if weights[key].dtype != mx.bfloat16:
                raise ValueError(f"Runtime adapter dtype differs from BF16 header: {key}")
    patterns = loader._build_pattern_mappings(mapping.get_mapping())
    # Validate the runtime mapping before replacing any module.  Zero/partial
    # matches must not produce a run described as MaskFlow.
    matched_targets = set()
    for key in bridged:
        matches = [p for p in patterns if loader._match_pattern(key, p.source_pattern) is not None]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one runtime mapping for {key}; found {len(matches)}")
        match = matches[0]
        block = loader._match_pattern(key, match.source_pattern)
        target = match.target_path.format(block=block)
        raw_key = key.removeprefix("diffusion_model.")
        expected_target = next(p.target_module for p in plans if raw_key in (p.down_key, p.up_key))
        matrix_name = "lora_A" if ".lora_A." in raw_key else "lora_B"
        if (target != expected_target or not match.transpose or match.matrix_name != matrix_name
                or match.transform is not None):
            raise ValueError(f"Runtime target/orientation differs for {key}")
        matched_targets.add(target)
    if len(matched_targets) != 600:
        raise ValueError("Runtime mapping did not cover all 600 targets")
    count, matched = loader._apply_lora_with_mapping(
        transformer, bridged, scale * spec.alpha_over_rank, patterns, role=role)
    if count != 600 or len(matched) != 1200:
        raise RuntimeError(f"MaskFlow application incomplete: {count} targets, {len(matched)} keys")
    return {"status": "applied_unbaked", "targets": count, "matched_keys": len(matched),
            "scale": scale, "effective_scale": scale * spec.alpha_over_rank,
            "spec": asdict(spec), "checkpoint_rewritten": False}
