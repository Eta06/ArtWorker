"""Load a native affine export without quantizing random dense initial weights.

Components are instantiated on demand by the caller. Every checkpoint key and
shape is validated; incomplete exports fail instead of silently using defaults.
"""
from pathlib import Path


def load_component(module, directory: Path, materialize=True, prefix=None, sequential=False):
    if sequential and prefix is not None:
        raise ValueError("Sequential loading requires a complete component")
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_unflatten, tree_map_with_path
    weights = {}
    if prefix is not None:
        import json
        manifest = json.loads((directory / "repack_manifest.json").read_text())
        files = [directory / entry["file"] for entry in manifest["files"] if entry["group"] == prefix or entry["group"].startswith(prefix + ".")]
        if not files:
            raise ValueError(f"Missing layerwise component: {prefix}")
    else:
        files = sorted(directory.glob("*.safetensors"))
    for file in files:
        if sequential:
            from run_trial import header
            entries = header(file)
            metadata = entries.pop("__metadata__")
            types = {"F16": mx.float16, "BF16": mx.bfloat16, "F32": mx.float32, "U32": mx.uint32, "I32": mx.int32}
            loaded = {key: mx.zeros(tuple(entry["shape"]), dtype=types[entry["dtype"]]) for key, entry in entries.items()}
        else:
            loaded, metadata = mx.load(str(file), return_metadata=True)
        if prefix is not None:
            if any(not key.startswith(prefix + ".") for key in loaded):
                raise ValueError(f"Unexpected layerwise key in {file}")
            loaded = {key[len(prefix) + 1:]: value for key, value in loaded.items()}
        if metadata.get("quantization_level") != "4":
            raise ValueError(f"Not a native q4 shard: {file}")
        overlap = set(weights) & set(loaded)
        if overlap:
            raise ValueError(f"Duplicate checkpoint keys: {overlap}")
        weights.update(loaded)
        del loaded

    def replace(path, layer):
        base = path + "." if path else ""
        if f"{base}scales" not in weights:
            return layer
        if not isinstance(layer, (nn.Linear, nn.Embedding)):
            raise ValueError(f"Packed weights target unsupported layer {path}: {type(layer)}")
        packed = weights[f"{base}weight"]
        scales = weights[f"{base}scales"]
        input_dims = layer.weight.shape[-1]
        bits = packed.shape[-1] * 32 // input_dims
        group = input_dims // scales.shape[-1]
        if bits != 4 or group != 64:
            raise ValueError(f"Unexpected quantization at {path}: {bits}/{group}")
        if isinstance(layer, nn.Embedding):
            new = nn.QuantizedEmbedding(1, 64, bits=4, group_size=64)
            new.num_embeddings = layer.weight.shape[0]
            new.dims = input_dims
        else:
            # Tiny placeholders avoid quantizing a randomly initialized full model.
            new = nn.QuantizedLinear(64, 1, bias=f"{base}bias" in weights, bits=4, group_size=64)
        new.weight = packed
        new.scales = scales
        new.biases = weights[f"{base}biases"]
        if f"{base}bias" in weights:
            new.bias = weights[f"{base}bias"]
        return new

    if "scales" in weights:
        module = replace("", module)
    replacements = tree_map_with_path(replace, module.leaf_modules(), is_leaf=nn.Module.is_module)
    module.update_modules(replacements)
    del replacements
    expected = dict(tree_flatten(module.parameters()))
    missing = {k for k in set(expected) - set(weights) if not k.endswith(".inv_freq")}
    unexpected = set(weights) - set(expected)
    mismatched = [k for k in set(expected) & set(weights) if expected[k].shape != weights[k].shape]
    if missing or unexpected or mismatched:
        raise ValueError(f"{directory.name}: missing={sorted(missing)}, unexpected={sorted(unexpected)}, shapes={mismatched}")
    module.update(tree_unflatten(list(weights.items())), strict=False)
    del expected, weights
    if sequential:
        for file in files:
            loaded = mx.load(str(file))
            module.update(tree_unflatten(list(loaded.items())), strict=False)
            mx.eval(loaded)
            del loaded
            mx.clear_cache()
    elif materialize:
        mx.eval(module.parameters())
    mx.clear_cache()
    return module
