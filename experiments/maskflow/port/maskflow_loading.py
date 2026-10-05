"""Strict local-only staged loading for the isolated pinned saved MLX model.

Saved mixed quantization is inferred per module from packed weights/scales;
no dequantization, requantization of downloaded values or loose key updates.
"""
from __future__ import annotations

import gc
from pathlib import Path
import time

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten, tree_unflatten


def normalize_saved_shapes(model, weights):
    """Only restore Qwen VAE RMS channel-vector broadcast dimensions.

    The saved converter flattens these learned weights. Restoring singleton
    axes changes no values; all other shape mismatches remain strict failures.
    """
    from mflux.models.qwen.model.qwen_vae.qwen_image_rms_norm import QwenImageRMSNorm
    expected = dict(tree_flatten(model.parameters()))
    result = dict(weights)
    reshaped = []
    for key, value in weights.items():
        if key not in expected or tuple(value.shape) == tuple(expected[key].shape):
            continue
        if not key.endswith(".weight"):
            continue
        module = model
        for part in key.rsplit(".", 1)[0].split("."):
            module = module[int(part)] if isinstance(module, list) else getattr(module, part)
        shape = tuple(expected[key].shape)
        if (isinstance(module, QwenImageRMSNorm) and len(value.shape) == 1
                and len(shape) in (3, 4) and shape[0] == value.shape[0]
                and all(size == 1 for size in shape[1:])):
            result[key] = value.reshape(shape)
            reshaped.append({"tensor": key, "from": list(value.shape), "to": list(shape),
                             "values_changed": False})
    return result, {"vae_rms_broadcast_reshapes": reshaped}


def load_saved_component(base_path, name, constructor, *, sidecar=None):
    started = time.perf_counter()
    path = Path(base_path) / name
    shards = sorted(path.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No pinned local shards for {name}")
    weights = {}
    for shard in shards:
        payload = mx.load(str(shard))
        if set(weights).intersection(payload):
            raise ValueError(f"Duplicate saved tensor keys in {name}")
        weights.update(payload)
    if sidecar is not None:
        extra = mx.load(str(sidecar))
        if set(weights).intersection(extra):
            raise ValueError("Sidecar must supply only omitted official tensors")
        weights.update(extra)
    model = constructor()
    def predicate(module_path, module):
        scale = weights.get(module_path + ".scales")
        packed = weights.get(module_path + ".weight")
        if scale is None:
            return False
        if not hasattr(module, "to_quantized") or packed is None:
            raise ValueError(f"Saved quantized module cannot be constructed: {module_path}")
        bits = packed.shape[-1] * 32 // (scale.shape[-1] * 64)
        if bits not in (4, 8):
            raise ValueError(f"Unsupported saved quantization for {module_path}: {bits}")
        return {"bits": int(bits), "group_size": 64}
    nn.quantize(model, class_predicate=predicate)
    weights, shape_receipt = normalize_saved_shapes(model, weights)
    expected = dict(tree_flatten(model.parameters()))
    missing, extra = set(expected) - set(weights), set(weights) - set(expected)
    if missing or extra:
        raise ValueError(f"Strict {name} tensor coverage failed; missing={sorted(missing)[:12]}, extra={sorted(extra)[:12]}")
    for key, value in weights.items():
        if tuple(value.shape) != tuple(expected[key].shape):
            raise ValueError(f"Shape mismatch {name}.{key}: {value.shape} != {expected[key].shape}")
    model.update(tree_unflatten(list(weights.items())), strict=True)
    model.freeze()
    mx.eval(model.parameters())
    count = len(weights)
    del weights, expected
    gc.collect()
    mx.clear_cache()
    return model, {"component": name, "shards": len(shards), "tensor_count": count,
                   "strict_tensor_coverage": True, "base_values_requantized": False,
                   "load_seconds": time.perf_counter()-started, **shape_receipt}


def release_model(model):
    del model
    gc.collect()
    mx.clear_cache()
