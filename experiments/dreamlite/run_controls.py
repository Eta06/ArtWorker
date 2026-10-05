"""1024-square controls for DreamLite; tests general edit vs outpaint configuration."""

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers import DreamLiteMobilePipeline, DreamLitePipeline

from run_mobile import ROOT, MemoryMonitor, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["mobile", "base"], default="mobile")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--tests", nargs="+", choices=["recolor", "square_outpaint", "portrait_outpaint"], default=["recolor", "square_outpaint"])
    args = parser.parse_args()
    output = ROOT / (args.output_dir or f"experiments/dreamlite/results/{args.variant}-1024-controls-seed42")
    output.mkdir(parents=True, exist_ok=True)
    inputs = json.loads((ROOT / "experiments/evaluation/inputs/inputs.json").read_text())
    track = next(t for t in inputs["tracks"] if t["id"] == "track3")
    square = Image.open(track["source_square"]).convert("RGB")
    recolor_input = square.resize((1024, 1024), Image.Resampling.LANCZOS)
    square_outpaint_input = Image.new("RGB", (1024, 1024), "black")
    square_outpaint_input.paste(Image.open(track["source_512"]).convert("RGB"), (256, 256))
    tests = {
        "recolor": (recolor_input, "the red car changed to a blue car. Keep the man, road, sea, camera angle and all other details unchanged.", None),
        "square_outpaint": (square_outpaint_input, "the empty black area surrounding the central square filled with a seamless continuation of the same coastal scene. Extend the sea, road and vegetation naturally in all four directions. Preserve the car and man in the central square. Do not add other people, cars, letters, borders or panels.", (256, 256, 768, 768)),
        "portrait_outpaint": (Image.open(track["canvas"]).convert("RGB"), "the empty black bands above and below the central square filled with a seamless continuation of the same coastal scene. Extend the sea, road and vegetation naturally upward and downward. Preserve the car and man in the central square. Do not add other people, cars, letters, borders or panels.", tuple(inputs["source_rect_xyxy"])),
    }
    monitor = MemoryMonitor()
    monitor.thread.start()
    steps = 4 if args.variant == "mobile" else 28
    cls = DreamLiteMobilePipeline if args.variant == "mobile" else DreamLitePipeline
    report = {
        "model": f"carlofkl/DreamLite-{args.variant}",
        "revision": "6695c3f4be230f0493fa5dbf78be3bc4d3bb2ab4" if args.variant == "mobile" else "751cb8dbb9072a8c8ffd8684e0f254b50f20531b",
        "steps": steps, "dtype": "bfloat16", "device": "mps", "seed": 42,
        "guidance_scale": None if args.variant == "mobile" else 3.5,
        "image_guidance_scale": None if args.variant == "mobile" else 1.5,
        "purpose": "Diagnostic controls; 1024-square is the official example resolution. Source track3. Native instruction edit, no hard mask.",
        "runs": [],
    }
    try:
        started = time.perf_counter()
        print(f"Loading {args.variant}", flush=True)
        pipe = cls.from_pretrained(ROOT / f".build/models/dreamlite/{args.variant}", torch_dtype=torch.bfloat16, local_files_only=True).to("mps")
        torch.mps.synchronize()
        report["load_seconds"] = time.perf_counter() - started
        for name in args.tests:
            image, prompt, rect = tests[name]
            image.save(output / f"{name}_input.png")
            print(f"Running {name}, {image.size}, {steps} steps", flush=True)
            started = time.perf_counter()
            result = pipe(prompt=prompt, image=image, width=image.width, height=image.height, num_inference_steps=steps, generator=torch.Generator("cpu").manual_seed(42)).images[0].convert("RGB")
            torch.mps.synchronize()
            seconds = time.perf_counter() - started
            path = output / f"{name}_raw.png"
            result.save(path)
            row = {"test": name, "prompt": prompt, "generation_seconds": seconds, "input_size": list(image.size), "actual_output_size": list(result.size), "raw_path": str(path), "raw_sha256": sha256(path), **monitor.result()}
            if rect:
                source = image.crop(rect)
                composite = result.copy()
                composite.paste(source, rect[:2])
                cp = output / f"{name}_composite.png"
                composite.save(cp)
                row.update({"source_rect_xyxy": list(rect), "composite_path": str(cp), "composite_center_byte_exact": bool(np.array_equal(np.asarray(composite.crop(rect)), np.asarray(source))), "raw_center_mae_rgb_0_255": float(np.abs(np.asarray(result.crop(rect)).astype(np.int16) - np.asarray(source).astype(np.int16)).mean())})
            report["runs"].append(row)
            (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(row), flush=True)
        report["status"] = "completed"
    except Exception as exc:
        report.update({"status": "failed", "error": repr(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        monitor.stop.set()
        monitor.thread.join()
        report.update(monitor.result())
        (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
