"""Tensor-wise BFL -> Swift MLX native int4 export, without a dense model load.

Matches the pinned runtime's BFL mapping, including QKV splitting and final
scale/shift row order. The runtime must still validate every key/shape before
using this export. Source files and their mtimes remain untouched.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def mapped(key, value, mx):
    block = re.fullmatch(r"double_blocks\.(\d+)\.(.*)", key)
    if block:
        index, suffix = block.groups()
        root = f"transformerBlocks.{index}."
        if suffix in ("img_attn.qkv.weight", "txt_attn.qkv.weight"):
            width = value.shape[0] // 3
            paths = ["toQ", "toK", "toV"] if suffix.startswith("img") else ["addQProj", "addKProj", "addVProj"]
            return [(root + f"attn.{name}.weight", value[i * width:(i + 1) * width]) for i, name in enumerate(paths)]
        fields = {"img_attn.proj.weight": "attn.toOut.weight", "txt_attn.proj.weight": "attn.toAddOut.weight",
            "img_attn.norm.query_norm.scale": "attn.normQ.weight", "img_attn.norm.key_norm.scale": "attn.normK.weight",
            "txt_attn.norm.query_norm.scale": "attn.normAddedQ.weight", "txt_attn.norm.key_norm.scale": "attn.normAddedK.weight",
            "img_mlp.0.weight": "ff.activation.proj.weight", "img_mlp.2.weight": "ff.linearOut.weight",
            "txt_mlp.0.weight": "ffContext.activation.proj.weight", "txt_mlp.2.weight": "ffContext.linearOut.weight"}
        return [(root + fields[suffix], value)]
    block = re.fullmatch(r"single_blocks\.(\d+)\.(.*)", key)
    if block:
        index, suffix = block.groups()
        fields = {"linear1.weight": "attn.toQkvMlp.weight", "linear2.weight": "attn.toOut.weight",
            "norm.query_norm.scale": "attn.normQ.weight", "norm.key_norm.scale": "attn.normK.weight"}
        return [(f"singleTransformerBlocks.{index}." + fields[suffix], value)]
    if key == "final_layer.adaLN_modulation.1.weight":
        half = value.shape[0] // 2
        return [("normOut.linear.weight", mx.concatenate([value[half:], value[:half]], axis=0))]
    fields = {"img_in.weight": "xEmbedder.weight", "txt_in.weight": "contextEmbedder.weight",
        "time_in.in_layer.weight": "timeGuidanceEmbed.timestepEmbedder.linear1.weight",
        "time_in.out_layer.weight": "timeGuidanceEmbed.timestepEmbedder.linear2.weight",
        "double_stream_modulation_img.lin.weight": "doubleStreamModulationImg.linear.weight",
        "double_stream_modulation_txt.lin.weight": "doubleStreamModulationTxt.linear.weight",
        "single_stream_modulation.lin.weight": "singleStreamModulation.linear.weight",
        "final_layer.linear.weight": "projOut.weight"}
    return [(fields[key], value)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["base", "distilled"], default="base")
    parser.add_argument("--bake-outpaint", action="store_true")
    args = parser.parse_args()
    import mlx.core as mx
    import numpy as np
    mx.set_cache_limit(0)
    mx.set_memory_limit(1024**3)
    name = "FLUX.2-klein-base-4B-klein4b-base-bf16" if args.model == "base" else "FLUX.2-klein-4B-klein4b-bf16"
    source_dir = ROOT / ".build/models/flux2/black-forest-labs" / name
    source = next(source_dir.glob("*.safetensors"))
    original_source_dir = source_dir
    pairs = {}
    applied = set()
    if args.bake_outpaint:
        adapter = ROOT / ".build/models/flux2-outpaint-lora/flux-outpaint-lora.safetensors"
        arrays = mx.load(str(adapter))
        for key in arrays:
            if not key.endswith(".lora_A.weight"):
                continue
            prefix = key.removesuffix(".lora_A.weight")
            b = arrays[prefix + ".lora_B.weight"].astype(mx.float16)
            a = arrays[key].astype(mx.float16)
            raw = prefix.removeprefix("base_model.model.")
            for target, split_b in mapped(raw + ".weight", b, mx):
                pairs[target] = (a, split_b)
        if len(pairs) != 88:
            raise ValueError(f"Expected 88 mapped adapter pairs, got {len(pairs)}")
        # Separate model root prevents overwriting or disguising the unbaked cache.
        baked_root = ROOT / ".build/models/flux2-outpaint-baked"
        source_dir = baked_root / "black-forest-labs" / original_source_dir.name
        source_dir.mkdir(parents=True, exist_ok=True)
        for item in original_source_dir.iterdir():
            if item.is_file() and not (source_dir / item.name).exists():
                os.link(item, source_dir / item.name)
        original_root = ROOT / ".build/models/flux2"
        for child in original_root.iterdir():
            if child.name == "black-forest-labs":
                continue
            link = baked_root / child.name
            if not link.exists():
                link.symlink_to(child.resolve(), target_is_directory=child.is_dir())
        for child in (original_root / "black-forest-labs").iterdir():
            if child.name == source_dir.name:
                continue
            link = baked_root / "black-forest-labs" / child.name
            if not link.exists():
                link.symlink_to(child.resolve(), target_is_directory=child.is_dir())
    destination = source_dir / "mlx-prequantized/int4/transformer.safetensors"
    if destination.exists():
        raise FileExistsError("An export already exists; it must be validated rather than overwritten")
    destination.parent.mkdir(parents=True, exist_ok=True)
    scratch = destination.with_suffix(".partial-data")
    temporary = destination.with_suffix(".partial")
    stat = source.stat()
    metadata = {"format": "flux2-mlx-prequantized-v1", "quantization": "int4", "bits": "4", "group_size": "64",
        "mode": "affine", "component": "transformer", "source": source_dir.name,
        "source_fingerprint": f"{source.name}:{stat.st_size}:{int(stat.st_mtime)}", "created_by": "ArtWorker tensor-wise native exporter"}
    if args.bake_outpaint:
        metadata.update(lora_baked="true", lora_source="fal/flux-2-klein-4B-outpaint-lora", lora_scale="1.1")
    entries = {"__metadata__": metadata}
    offset = 0
    try:
        with source.open("rb") as stream, scratch.open("wb") as body:
            length = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(length))
            for key, spec in header.items():
                if key == "__metadata__":
                    continue
                a, b = spec["data_offsets"]
                stream.seek(8 + length + a)
                storage = stream.read(b - a)
                dtype = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32}[spec["dtype"]]
                value = mx.array(np.frombuffer(storage, dtype=dtype)).reshape(spec["shape"])
                if spec["dtype"] == "BF16":
                    value = value.view(mx.bfloat16)
                value = value.astype(mx.float16)
                targets = mapped(key, value, mx)
                for name, tensor in targets:
                    if name in pairs:
                        a, b = pairs[name]
                        if b.shape[0] != tensor.shape[0] or a.shape[1] != tensor.shape[1]:
                            raise ValueError(f"Adapter shape mismatch: {name}")
                        tensor = tensor + 1.1 * mx.matmul(b, a)
                        applied.add(name)
                    if tensor.ndim == 2:
                        packed, scales, biases = mx.quantize(tensor, group_size=64, bits=4)
                        tensors = {name: packed, name.removesuffix(".weight") + ".scales": scales,
                            name.removesuffix(".weight") + ".biases": biases}
                    else:
                        tensors = {name: tensor}
                    mx.eval(*tensors.values())
                    for target, array in tensors.items():
                        if target in entries:
                            raise ValueError(f"Duplicate native key: {target}")
                        cpu = np.asarray(array)
                        payload = cpu.tobytes()
                        label = {np.dtype("float16"): "F16", np.dtype("uint32"): "U32"}[cpu.dtype]
                        entries[target] = {"dtype": label, "shape": list(array.shape), "data_offsets": [offset, offset + len(payload)]}
                        body.write(payload)
                        offset += len(payload)
                    if tensor.ndim == 2:
                        del packed, scales, biases
                    del tensors, array, cpu, payload
                del targets, value, tensor, storage
                mx.clear_cache()
        if args.bake_outpaint and applied != set(pairs):
            raise ValueError(f"Unused adapter keys: {set(pairs) - applied}")
        encoded = json.dumps(entries, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 8)
        with temporary.open("wb") as output, scratch.open("rb") as body:
            output.write(struct.pack("<Q", len(encoded)))
            output.write(encoded)
            shutil.copyfileobj(body, output, length=1024**2)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(destination)
    finally:
        scratch.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
    digest = hashlib.sha256()
    with destination.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    record = {"model": args.model, "source_file": source.name, "source_bytes": stat.st_size,
        "file": str(destination.relative_to(ROOT)), "bytes": destination.stat().st_size, "sha256": digest.hexdigest(),
        "tensor_count": len(entries) - 1, "method": "tensor-wise native int4, all Linear matrices, QKV split, swapped final scale/shift",
        "license": "Apache-2.0 upstream", "lora_baked": args.bake_outpaint,
        "adapter_pairs_applied": len(applied), "merge_method": "FP16 source + FP16 adapter, then one INT4 quantization" if args.bake_outpaint else None,
        "runtime_validation": "pending"}
    suffix = "_outpaint_baked" if args.bake_outpaint else ""
    (ROOT / f"experiments/klein_outpaint/native_q4_{args.model}{suffix}_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
