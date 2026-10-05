"""Atomic retained iteration checkpoints for the bounded MaskFlow teacher."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import uuid

REQUIRED = ("latents", "initial_noise", "source_latents", "mask_image_latents", "mask_latents",
            "pm.prompt_embeds", "pm.prompt_embeds_mask", "nm.prompt_embeds", "nm.prompt_embeds_mask")
PROTOCOL_KEYS = ("source_file_sha256", "source_rgb_sha256", "canvas_rgb_sha256", "prompt_sha256",
                 "seed", "width", "height", "known_box_xyxy", "unknown_fill", "steps", "text_cfg",
                 "mask_cfg", "rescale_cfg", "runtime_revision", "official_revision", "port_sha256")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def save_checkpoint(output_dir, completed_steps, arrays, metadata):
    import mlx.core as mx
    output_dir = Path(output_dir)
    if set(arrays) != set(REQUIRED) or not 0 <= completed_steps <= 50:
        raise ValueError("Incomplete native MaskFlow sampling state")
    mx.eval(*arrays.values())
    # Retain old states; the receipt points to a fully committed unique file.
    name = f"state-step{completed_steps:02d}-{uuid.uuid4().hex[:8]}.safetensors"
    temporary = output_dir / ("." + name)
    destination = output_dir / name
    mx.save_safetensors(str(temporary), arrays)
    temporary.replace(destination)
    receipt = {"schema": "maskflow-native-latest-v1", "completed_steps": completed_steps,
               "state_path": str(destination.resolve()), "state_sha256": file_sha256(destination),
               "protocol": {key: metadata[key] for key in PROTOCOL_KEYS},
               "initial_noise_sha256": metadata["initial_noise_sha256"],
               "vlm": metadata["vlm"], "cpu_gates": metadata["cpu_gates"],
               "downloads": metadata["downloads"],
               "saved_tensor_shapes": {key: list(value.shape) for key, value in arrays.items()},
               "saved_tensor_dtypes": {key: str(value.dtype) for key, value in arrays.items()}}
    atomic_json(output_dir / "checkpoint.json", receipt)
    return {"receipt": str((output_dir / "checkpoint.json").resolve()),
            "state_path": str(destination.resolve()), "completed_steps": completed_steps,
            "state_sha256": receipt["state_sha256"]}


def validate_checkpoint(receipt_path, current_protocol):
    receipt = json.loads(Path(receipt_path).read_text())
    if receipt.get("schema") != "maskflow-native-latest-v1" or not 0 <= receipt.get("completed_steps", -1) <= 50:
        raise ValueError("Unknown/incomplete checkpoint protocol")
    for key in PROTOCOL_KEYS:
        if receipt["protocol"][key] != current_protocol[key]:
            raise ValueError(f"Checkpoint protocol changed: {key}")
    if file_sha256(receipt["state_path"]) != receipt["state_sha256"]:
        raise ValueError("Checkpoint state bytes changed")
    if set(receipt["saved_tensor_shapes"]) != set(REQUIRED):
        raise ValueError("Checkpoint tensor inventory differs")
    for key in REQUIRED[:5]:
        if receipt["saved_tensor_shapes"][key] != [1, 2304, 64]:
            raise ValueError(f"Checkpoint packed geometry changed: {key}")
    return receipt


def restore_checkpoint(receipt):
    import mlx.core as mx
    arrays = mx.load(receipt["state_path"])
    if set(arrays) != set(REQUIRED):
        raise ValueError("Checkpoint actual tensor inventory differs")
    for key, value in arrays.items():
        if list(value.shape) != receipt["saved_tensor_shapes"][key] or str(value.dtype) != receipt["saved_tensor_dtypes"][key]:
            raise ValueError(f"Checkpoint tensor changed: {key}")
    mx.eval(*arrays.values())
    return arrays
