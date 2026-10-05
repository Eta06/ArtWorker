"""Task-specific Klein adapter trial with bounded execution and raw/composite arms."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from run_bounded_model import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=["track1", "track2", "track3"], default="track3")
    parser.add_argument("--model", choices=["klein-4b", "klein-4b-base"], default="klein-4b")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--vae", choices=["small-decoder", "standard"], default="small-decoder")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-gib", type=float, default=6.0)
    parser.add_argument("--require-prequantized", action="store_true")
    parser.add_argument("--baked-outpaint", action="store_true")
    parser.add_argument("--release-before-decode", action="store_true")
    args = parser.parse_args()
    from PIL import Image
    import numpy as np
    args.output.mkdir(parents=True, exist_ok=True)
    shared = json.loads((ROOT / "experiments/evaluation/inputs/inputs.json").read_text())
    track = next(t for t in shared["tracks"] if t["id"] == args.track)
    size = shared["canvas_size"]
    rect = shared["source_rect_xyxy"]
    source = Image.open(track["source_512"]).convert("RGB")
    canvas = Image.new("RGB", tuple(size), (0, 255, 0))
    canvas.paste(source, (rect[0], rect[1]))
    canvas.save(args.output / "input.png")
    prompt = "Fill the green spaces according to the image"
    if args.describe:
        prompt += ". " + json.loads((ROOT / "experiments/flux2/prompts.json").read_text())[args.track]
    cli = ROOT / ".build/flux2-native/Build/Products/Release/Flux2CLI"
    command = [str(cli), "i2i", prompt, "--images", str(args.output / "input.png"),
        "--model", args.model, "--models-dir", str(ROOT / (".build/models/flux2-outpaint-baked" if args.baked_outpaint else ".build/models/flux2")),
        "--text-quant", "4bit", "--transformer-quant", "int4", "--steps", str(args.steps),
        "--guidance", str(args.guidance), "--seed", str(args.seed), "--width", str(size[0]), "--height", str(size[1]),
        "--vae-variant", args.vae, "--memory-profile", "conservative", "--profile", "--verbose",
        "--output", str(args.output / "raw.png")]
    if not args.baked_outpaint:
        command += ["--lora", str(ROOT / ".build/models/flux2-outpaint-lora/flux-outpaint-lora.safetensors"), "--lora-scale", "1.1"]
    if args.release_before_decode:
        if not args.baked_outpaint:
            parser.error("Release-before-decode is limited to the baked-adapter experiment")
        command = ["env", "ARTWORKER_RELEASE_TRANSFORMER_BEFORE_DECODE=1", *command]
    record = {"track": args.track, "model": args.model, "steps": args.steps, "guidance": args.guidance,
        "seed": args.seed, "prompt": prompt, "adapter_scale": 1.1, "vae": args.vae, "input": "green #00ff00, sole i2i reference",
        "source_rect": rect, "width": size[0], "height": size[1], "runtime_patch": "bfl-timestep-mapping.patch",
        "cli_sha256": hashlib.sha256(cli.read_bytes()).hexdigest(),
        "visual_quality_accepted": False, "status": "running"}
    record["adapter_baked"] = args.baked_outpaint
    record["release_before_decode"] = args.release_before_decode
    if args.release_before_decode:
        record["runtime_memory_patch"] = "sequential-decode-memory.patch"
    try:
        receipt = run(command, args.output, args.max_gib, 600)
        record["execution"] = receipt
        if receipt["status"] != "completed":
            raise RuntimeError(f"Model stopped: {receipt['stop_reason']}")
        text = (args.output / "execution.log").read_text()
        if args.require_prequantized and "Loaded pre-quantized" not in text:
            raise RuntimeError("Native pre-quantized load was not confirmed")
        matches = re.findall(r"\[LoRA\] Merged (\d+) layers \((\d+) not found\)", text)
        if not args.baked_outpaint and matches != [("88", "0")]:
            raise RuntimeError(f"Incomplete adapter application: {matches}")
        if args.baked_outpaint and "LoRA weights BAKED IN" not in text:
            raise RuntimeError("Baked adapter metadata was not confirmed")
        raw = Image.open(args.output / "raw.png").convert("RGB")
        if raw.size != tuple(size):
            raise ValueError(f"Wrong output dimensions: {raw.size}")
        record["raw_source_mae_255"] = float(np.abs(np.asarray(raw.crop(rect), dtype=np.float32) - np.asarray(source, dtype=np.float32)).mean())
        composite = raw.copy()
        composite.paste(source, (rect[0], rect[1]))
        composite.save(args.output / "composite.png")
        record["source_exact_in_composite"] = bool(np.array_equal(np.asarray(composite.crop(rect)), np.asarray(source)))
        record["status"] = "completed"
    except BaseException as exc:
        record.update(status="failed", error=str(exc))
        raise
    finally:
        (args.output / "metrics.json").write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
