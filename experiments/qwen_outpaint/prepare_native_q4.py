"""Export the cached full Qwen editing checkpoint one tensor at a time.

This is affine quantization, not distillation. No model is instantiated. Original
weights are never modified; output has MFLUX's native reference-edit key layout.
Use scripts/run_bounded_model.py to supervise Metal and physical memory.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import struct
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / ".build/models/qwen/base"
REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / ".build/models/qwen/native-edit-q4")
    args = parser.parse_args()
    import mlx.core as mx
    import numpy as np
    mx.set_cache_limit(0)
    mx.set_memory_limit(4 * 1024**3)
    if args.output.exists():
        raise FileExistsError("Use a fresh export directory; partial exports are not ready checkpoints")
    args.output.mkdir(parents=True)
    receipt = {"source_repo": "Qwen/Qwen-Image-2.1", "source_revision": REVISION,
        "license": "qwen-research", "method": "tensor-wise affine q4, group64; full vision encoder included",
        "distilled": False, "status": "running", "files": [], "mlx_version": mx.__version__}
    try:
        for name in ("processor", "scheduler"):
            shutil.copytree(SOURCE / name, args.output / name)
        shutil.copyfile(SOURCE / "model_index.json", args.output / "model_index.json")
        shutil.copyfile(SOURCE / "LICENSE", args.output / "LICENSE")
        for component in ("vae", "transformer", "text_encoder"):
            target = args.output / component
            target.mkdir()
            shutil.copyfile(SOURCE / component / "config.json", target / "config.json")
            for index, original in enumerate(sorted((SOURCE / component).glob("*.safetensors"))):
                output = target / f"{index}.safetensors"
                scratch = target / f"{index}.partial-data"
                entries = {"__metadata__": {"mflux_version": "0.20.0", "quantization_level": "4"}}
                offset = 0
                with original.open("rb") as stream, scratch.open("wb") as body:
                    hlen = struct.unpack("<Q", stream.read(8))[0]
                    header = json.loads(stream.read(hlen))
                    data_start = 8 + hlen
                    for key, info in header.items():
                        if key == "__metadata__":
                            continue
                        newkey = key
                        if component == "text_encoder":
                            newkey = key.removeprefix("model.")
                            if newkey == "language_model.norm.weight" or newkey.startswith("lm_head."):
                                continue
                        if component == "transformer" and (key == "time_text_embed.time_proj.freqs" or key.startswith("pos_embed.cos_tables.") or key.startswith("pos_embed.sin_tables.")):
                            continue
                        if component == "transformer" and key.startswith("modulation.1."):
                            newkey = key.replace("modulation.1.", "modulation.layers.1.", 1)
                        start, end = info["data_offsets"]
                        stream.seek(data_start + start)
                        storage = stream.read(end - start)
                        dtype = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32}[info["dtype"]]
                        value = mx.array(np.frombuffer(storage, dtype=dtype)).reshape(info["shape"])
                        if info["dtype"] == "BF16":
                            value = value.view(mx.bfloat16)
                        value = value.astype(mx.float32 if component == "vae" else mx.bfloat16)
                        if component == "vae":
                            if newkey.endswith(".gamma"):
                                value = value.reshape(-1)
                            if newkey.endswith(".weight") and value.ndim == 4:
                                value = value.transpose(0, 2, 3, 1)
                        elif component == "text_encoder" and newkey.endswith("patch_embed.proj.weight"):
                            value = value.transpose(0, 2, 3, 4, 1)
                        quantized = (component != "vae" and newkey.endswith(".weight") and value.ndim == 2
                            and value.shape[-1] % 64 == 0
                            and not any(part in newkey for part in ("modulation", "time_text_embed", "norm_out")))
                        if quantized:
                            packed, scales, biases = mx.quantize(value, group_size=64, bits=4)
                            tensors = {newkey: packed, newkey.removesuffix(".weight") + ".scales": scales,
                                newkey.removesuffix(".weight") + ".biases": biases}
                        else:
                            tensors = {newkey: value}
                        mx.eval(*tensors.values())
                        for name, tensor in tensors.items():
                            if name in entries:
                                raise ValueError(f"Duplicate output key: {name}")
                            if tensor.dtype == mx.bfloat16:
                                array = np.asarray(tensor.view(mx.uint16))
                                label = "BF16"
                            else:
                                array = np.asarray(tensor)
                                label = {np.dtype("uint32"): "U32", np.dtype("float32"): "F32", np.dtype("float16"): "F16"}[array.dtype]
                            payload = array.tobytes()
                            body.write(payload)
                            entries[name] = {"dtype": label, "shape": list(tensor.shape), "data_offsets": [offset, offset + len(payload)]}
                            offset += len(payload)
                        # Discard every lazy graph and buffer before the next source tensor.
                        del value, tensors, tensor, array, payload, storage
                        if quantized:
                            del packed, scales, biases
                        mx.clear_cache()
                encoded = json.dumps(entries, separators=(",", ":")).encode()
                encoded += b" " * (-len(encoded) % 8)
                with output.open("wb") as dest, scratch.open("rb") as body:
                    dest.write(struct.pack("<Q", len(encoded)))
                    dest.write(encoded)
                    shutil.copyfileobj(body, dest, length=1024**2)
                scratch.unlink()
                digest = hashlib.sha256()
                with output.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024**2), b""):
                        digest.update(chunk)
                receipt["files"].append({"file": str(output.relative_to(args.output)), "bytes": output.stat().st_size,
                    "sha256": digest.hexdigest(), "source_shard": original.name, "tensor_count": len(entries) - 1})
                print(json.dumps(receipt["files"][-1]), flush=True)
        receipt.update(status="completed", payload_bytes=sum(f["bytes"] for f in receipt["files"]))
    except BaseException as exc:
        receipt.update(status="failed", error=str(exc))
        raise
    finally:
        (args.output / "export_manifest.json").write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
