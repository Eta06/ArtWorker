"""Generate in 1024-square model space; crop a phone frame, restore source exactly.

This changes model context/work resolution and is not the same inference setting
as the direct 512x1152 comparison. Both intermediate raw and phone outputs survive.
"""

import argparse
import json
import time
import traceback

import numpy as np
import torch
from PIL import Image
from diffusers import DreamLiteMobilePipeline, DreamLitePipeline

from run_mobile import ROOT, MemoryMonitor, sha256


SCENES = {
    "track1": "the blue textured moon, dark night atmosphere and subtle lunar lighting",
    "track2": "the cyan blue studio wall and photography lighting",
    "track3": "the sea, coastal road and roadside vegetation",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["mobile", "base"], default="mobile")
    parser.add_argument("--tracks", nargs="+", default=["track1", "track2", "track3"])
    args = parser.parse_args()
    inputs = json.loads((ROOT / "experiments/evaluation/inputs/inputs.json").read_text())
    output = ROOT / f"experiments/dreamlite/results/{args.variant}-square-context-phone-seed42"
    output.mkdir(parents=True, exist_ok=True)
    steps = 4 if args.variant == "mobile" else 28
    cls = DreamLiteMobilePipeline if args.variant == "mobile" else DreamLitePipeline
    monitor = MemoryMonitor()
    monitor.thread.start()
    report = {
        "model": f"carlofkl/DreamLite-{args.variant}",
        "revision": "6695c3f4be230f0493fa5dbf78be3bc4d3bb2ab4" if args.variant == "mobile" else "751cb8dbb9072a8c8ffd8684e0f254b50f20531b",
        "steps": steps, "dtype": "bfloat16", "device": "mps", "seed": 42,
        "canvas_size": inputs["canvas_size"], "source_rect_xyxy": inputs["source_rect_xyxy"],
        "model_work_resolution": [1024, 1024], "model_source_rect_xyxy": [288, 288, 736, 736],
        "phone_crop_xyxy_in_model_space": [288, 8, 736, 1016],
        "method": "native instruction edit in larger square context, crop448x1008 then LANCZOS resize512x1152; final byte-exact original512 composite; no native hard mask",
        "comparison_caveat": "Work resolution, visible context, source size in model space and scene-specific prompts differ from the direct portrait baseline.",
        "runs": [],
    }
    try:
        started = time.perf_counter()
        print(f"Loading {args.variant}", flush=True)
        pipe = cls.from_pretrained(ROOT / f".build/models/dreamlite/{args.variant}", torch_dtype=torch.bfloat16, local_files_only=True).to("mps")
        torch.mps.synchronize()
        report["load_seconds"] = time.perf_counter() - started
        for track in inputs["tracks"]:
            if track["id"] not in args.tracks:
                continue
            source = Image.open(track["source_512"]).convert("RGB")
            square_input = Image.new("RGB", (1024, 1024), "black")
            square_input.paste(source.resize((448, 448), Image.Resampling.LANCZOS), (288, 288))
            square_input.save(output / f"{track['id']}_square_input.png")
            prompt = (
                "the empty black area surrounding the central square filled with a seamless "
                f"continuation of {SCENES[track['id']]}. Extend the existing scene and texture "
                "naturally in all four directions. Preserve the central square album artwork, "
                "all existing people, objects and text in their positions. Do not add other "
                "people, cars, letters, borders, frames or panels."
            )
            print(f"Running {track['id']}, 1024-square, {steps} steps", flush=True)
            started = time.perf_counter()
            square_raw = pipe(prompt=prompt, image=square_input, width=1024, height=1024, num_inference_steps=steps, generator=torch.Generator("cpu").manual_seed(42)).images[0].convert("RGB")
            torch.mps.synchronize()
            seconds = time.perf_counter() - started
            square_path = output / f"{track['id']}_square_raw.png"
            square_raw.save(square_path)
            phone = square_raw.crop((288, 8, 736, 1016)).resize((512, 1152), Image.Resampling.LANCZOS)
            phone_path = output / f"{track['id']}_raw.png"
            phone.save(phone_path)
            composite = phone.copy()
            rect = tuple(inputs["source_rect_xyxy"])
            composite.paste(source, rect[:2])
            cp = output / f"{track['id']}_composite.png"
            composite.save(cp)
            row = {
                "track_id": track["id"], "prompt": prompt, "generation_seconds": seconds,
                "actual_output_size": list(phone.size), "raw_square_path": str(square_path),
                "raw_path": str(phone_path), "composite_path": str(cp),
                "raw_sha256": sha256(phone_path), "composite_sha256": sha256(cp),
                "composite_center_byte_exact": bool(np.array_equal(np.asarray(composite.crop(rect)), np.asarray(source))),
                "raw_center_mae_rgb_0_255": float(np.abs(np.asarray(phone.crop(rect)).astype(np.int16) - np.asarray(source).astype(np.int16)).mean()),
                **monitor.result(),
            }
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
