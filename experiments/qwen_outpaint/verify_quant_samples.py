"""CPU-only checks of exported affine packing against sampled original weights."""
from pathlib import Path
import json
import struct
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / ".build/models/qwen/base"
EXPORT = ROOT / ".build/models/qwen/native-edit-q4"


def locate(directory, key):
    for file in sorted(directory.glob("*.safetensors")):
        with file.open("rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(length))
        if key in header:
            return file, 8 + length, header[key]
    raise KeyError(key)


def row(directory, key, index):
    file, base, spec = locate(directory, key)
    shape = spec["shape"]
    count = shape[-1]
    dtype = {"BF16": "uint16", "F32": "float32", "F16": "float16", "U32": "uint32"}[spec["dtype"]]
    itemsize = np.dtype(dtype).itemsize
    with file.open("rb") as stream:
        stream.seek(base + spec["data_offsets"][0] + index * count * itemsize)
        values = np.frombuffer(stream.read(count * itemsize), dtype=dtype)
    if spec["dtype"] == "BF16":
        values = (values.astype(np.uint32) << 16).view(np.float32)
    return values, shape


def main():
    report = {"scope": "three matrices / three rows each; packing and affine error only, not image acceptance", "samples": []}
    pairs = [
        ("transformer", "img_in.weight", "img_in.weight"),
        ("text_encoder", "model.language_model.embed_tokens.weight", "language_model.embed_tokens.weight"),
        ("text_encoder", "model.visual.blocks.0.attn.qkv.weight", "visual.blocks.0.attn.qkv.weight"),
    ]
    for component, source_key, target_key in pairs:
        spec = locate(ORIGINAL / component, source_key)[2]
        rows = sorted({0, spec["shape"][0] // 2, spec["shape"][0] - 1})
        before, after = [], []
        for index in rows:
            original, shape = row(ORIGINAL / component, source_key, index)
            packed, qshape = row(EXPORT / component, target_key, index)
            scales, _ = row(EXPORT / component, target_key.removesuffix(".weight") + ".scales", index)
            biases, _ = row(EXPORT / component, target_key.removesuffix(".weight") + ".biases", index)
            if qshape != [shape[0], shape[1] // 8] or len(scales) != shape[1] // 64:
                raise ValueError("Invalid packed geometry")
            codes = ((packed[:, None] >> (np.arange(8, dtype=np.uint32) * 4)) & 15).reshape(-1)
            reconstructed = codes * np.repeat(scales, 64) + np.repeat(biases, 64)
            before.append(original.astype(np.float32))
            after.append(reconstructed.astype(np.float32))
        before, after = np.concatenate(before), np.concatenate(after)
        relative = float(np.linalg.norm(after - before) / np.linalg.norm(before))
        if not np.isfinite(after).all() or relative >= 0.20:
            raise ValueError(f"Unexpected reconstruction error: {component}/{target_key}: {relative}")
        report["samples"].append({"component": component, "key": target_key, "rows": rows, "relative_l2_error": relative, "finite": True})
    report["status"] = "passed"
    (ROOT / "experiments/qwen_outpaint/quant_sample_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
