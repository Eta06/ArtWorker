#!/usr/bin/env python3
"""Local FLUX.2 masked-outpainting experiments; no player source modifications."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CLI = ROOT / ".build/flux2-native/Build/Products/Release/Flux2CLI"
MODELS = ROOT / ".build/models/flux2"
INPUTS = ROOT / "experiments/evaluation/inputs"
PROMPTS = json.loads((HERE / "prompts.json").read_text())


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def profile_metadata(text: str) -> dict:
    phases = {}
    for label, amount, unit in re.findall(r"^\s+(\d+[a-z]?\. [A-Za-z :\-]+?)\s+(\d+\.\d+)(ms|s)\s+", text, re.MULTILINE):
        phases.setdefault(label.strip(), float(amount) / (1000 if unit == "ms" else 1))
    footprint = re.search(r"(\d+)\s+peak memory footprint", text)
    return {
        "profile_phase_seconds": phases,
        "process_peak_memory_footprint_bytes": int(footprint.group(1)) if footprint else None,
        "mlx_profile_unit": "MiB (upstream prints MB but divides by 1_048_576)",
        "elapsed_seconds_scope": "Inpainting chain including lazy encoder/transformer loading; initialization before chain is not full model load.",
        "memory_caveat": "MLX cumulative allocation peak, macOS peak footprint and RSS are separate, overlapping quantities. None is an iPhone measurement.",
    }


def prepare(track: str, seed: int) -> tuple[Path, Path]:
    """Match the upstream outpainting recipe: neutral noise + inner mask ramps."""
    folder = HERE / "prepared" / f"{track}-seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    canvas_path, mask_path = folder / "canvas.png", folder / "mask.png"
    if canvas_path.exists() and mask_path.exists():
        return canvas_path, mask_path
    rng = random.Random(seed)
    data = bytes(max(0, min(255, round(rng.gauss(128, 32)))) for _ in range(512 * 1152))
    canvas = Image.frombytes("L", (512, 1152), data).convert("RGB")
    canvas.paste(Image.open(INPUTS / track / "source_512.png").convert("RGB"), (0, 320))
    canvas.save(canvas_path)
    mask = Image.new("L", (512, 1152), 255)
    mask.paste(0, (0, 320, 512, 832))
    for offset in range(32):
        value = (255 * (32 - offset) + 16) // 32
        mask.paste(value, (0, 320 + offset, 512, 321 + offset))
        mask.paste(value, (0, 831 - offset, 512, 832 - offset))
    mask.save(mask_path)
    return canvas_path, mask_path


def run(track: str, args: argparse.Namespace) -> list[dict]:
    name = f"{track}-{args.model}-{args.quant}-{args.steps}step-seed{args.seed}"
    folder = HERE / "results"
    folder.mkdir(exist_ok=True)
    metadata_path = folder / f"{name}.json"
    if metadata_path.exists() and not args.force:
        print(f"Already recorded: {metadata_path}", flush=True)
        return json.loads(metadata_path.read_text())["runs"]
    canvas, mask = prepare(track, args.seed)
    raw = folder / f"{name}-raw.png"
    composite = folder / f"{name}-composite.png"
    log = folder / f"{name}.log"
    command = [str(CLI), "inpaint", "--models-dir", str(MODELS),
               "--flux-model", args.model, "--image", str(canvas), "--mask", str(mask),
               "--reference", str(INPUTS / track / "source_512.png"),
               "--prompt", PROMPTS[track], "--output", str(raw),
               "--steps", str(args.steps), "--guidance", str(args.guidance),
               "--seed", str(args.seed), "--text-quant", "4bit", "--transformer-quant", args.quant,
               "--max-pixels", str(512 * 1152), "--profile", "--repeat-count", str(args.repeat_count)]
    print(f"Starting {name}", flush=True)
    start = time.monotonic()
    with log.open("w") as stream:
        result = subprocess.run(["/usr/bin/time", "-l", *command], stdout=stream, stderr=subprocess.STDOUT)
    wall = time.monotonic() - start
    text = log.read_text(errors="replace")
    times = [float(value) for value in re.findall(r"Inpainting run \d+/\d+ done in ([\d.]+)s", text)]
    peaks = [int(value) for value in re.findall(r"peak=(\d+) MB", text)]
    rss = re.search(r"(\d+)\s+maximum resident set size", text)
    load = re.search(r"pipeline ready in ([\d.]+)s", text)
    common = {
        "model": f"FLUX.2 {args.model} / {args.quant}", "track_id": track, "seed": args.seed,
        "steps": args.steps, "guidance": args.guidance, "transformer_quantization": args.quant,
        "text_encoder_quantization": "4bit", "vae": "FLUX.2-small-decoder",
        "status": "completed" if result.returncode == 0 and raw.exists() else "failed",
        "exit_code": result.returncode, "elapsed_seconds": times[-1] if times else None,
        "all_inference_seconds": times, "process_wall_seconds_including_load": wall,
        "pipeline_initialization_seconds": float(load.group(1)) if load else None,
        "mlx_peak_allocated_mb": max(peaks) if peaks else None,
        "process_maximum_rss_bytes": int(rss.group(1)) if rss else None,
        "memory_caveat": "MLX allocation peak and process RSS are different quantities; neither is an iPhone measurement.",
        "method": "Source reference + RePaint latent blending; 32px soft ramp inside preserved square; neutral noise canvas.",
        "log": str(log), "command": command,
        "prepared_canvas_sha256": hashlib.sha256(canvas.read_bytes()).hexdigest(),
        "prepared_mask_sha256": hashlib.sha256(mask.read_bytes()).hexdigest(),
        **profile_metadata(text),
    }
    runs = [{**common, "variant": "raw", "image": str(raw) if raw.exists() else None}]
    if common["status"] == "completed":
        image = Image.open(raw).convert("RGB")
        if image.size != (512, 1152):
            raise RuntimeError(f"Unexpected output geometry {image.size}; retaining raw result")
        image.paste(Image.open(INPUTS / track / "source_512.png").convert("RGB"), (0, 320))
        image.save(composite)
        runs.append({**common, "variant": "exact-source-composite", "image": str(composite),
                     "method": common["method"] + " Original 512px square pasted back exactly after decoding."})
    save_json(metadata_path, {"runs": runs})
    print(json.dumps({key: value for key, value in common.items() if key in
          ("status", "elapsed_seconds", "mlx_peak_allocated_mb", "process_wall_seconds_including_load", "exit_code")}), flush=True)
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", nargs="+", default=["track1", "track2", "track3"], choices=list(PROMPTS))
    parser.add_argument("--model", default="klein-4b")
    parser.add_argument("--quant", default="int4", choices=["int4", "bf16", "qint8"])
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeat-count", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    runs = []
    for track in args.tracks:
        runs.extend(run(track, args))
    save_json(HERE / "runs.json", {"runs": runs})


if __name__ == "__main__":
    main()
