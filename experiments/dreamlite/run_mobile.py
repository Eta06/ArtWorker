"""Measured DreamLite Mobile instruction-edit trial; no native spatial hard mask.

Run with .build/dreamlite-venv/bin/python. Model weights stay in ignored .build.
The original resized cover is composited back exactly only in *_composite.png.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import psutil
import torch
from PIL import Image
from diffusers import DreamLiteMobilePipeline


ROOT = Path(__file__).resolve().parents[2]
MODEL_SHA = "6695c3f4be230f0493fa5dbf78be3bc4d3bb2ab4"
REPO_SHA = "a6e20c8cc94027f37dd7c5a81b0b3b472aa18409"
PROMPT = (
    "the empty black bands above and below the central square album cover filled "
    "with a seamless continuation of the album artwork. Extend the existing scene, "
    "colors, lighting and texture naturally upward and downward into a tall portrait "
    "composition. Preserve the central square artwork, all people and existing text "
    "exactly in their current positions. Do not add new text or duplicate subjects."
)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class MemoryMonitor:
    def __init__(self):
        self.stop = threading.Event()
        self.peak_rss = 0
        self.peak_mps_allocated = 0
        self.peak_mps_driver = 0
        self.thread = threading.Thread(target=self.sample_loop, daemon=True)

    def sample_loop(self):
        process = psutil.Process()
        while not self.stop.is_set():
            self.peak_rss = max(self.peak_rss, process.memory_info().rss)
            if torch.backends.mps.is_available():
                self.peak_mps_allocated = max(self.peak_mps_allocated, torch.mps.current_allocated_memory())
                self.peak_mps_driver = max(self.peak_mps_driver, torch.mps.driver_allocated_memory())
            self.stop.wait(0.1)

    def result(self):
        return {
            "peak_process_rss_bytes": self.peak_rss,
            "peak_mps_allocated_bytes": self.peak_mps_allocated,
            "peak_mps_driver_bytes": self.peak_mps_driver,
            "memory_note": "RSS and MPS driver memory overlap on unified memory; do not add them.",
            "sampling_interval_seconds": 0.1,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", nargs="+", default=["track1", "track2", "track3"])
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--latent-preservation", action="store_true", help="Experimental restoration of source latents at each FlowMatch sigma")
    parser.add_argument("--output-dir", default="experiments/dreamlite/results/mobile-4step-seed42")
    args = parser.parse_args()
    inputs = json.loads((ROOT / "experiments/evaluation/inputs/inputs.json").read_text())
    model = ROOT / ".build/models/dreamlite/mobile"
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    monitor = MemoryMonitor()
    monitor.thread.start()
    report = {
        "model": "carlofkl/DreamLite-mobile",
        "model_revision": MODEL_SHA,
        "official_repository_revision": REPO_SHA,
        "weight_license": "CC BY-NC 4.0; research trial only",
        "device": "mps",
        "host": platform.platform(),
        "dtype": args.dtype,
        "steps": args.steps,
        "experimental_latent_preservation": args.latent_preservation,
        "seed": args.seed,
        "prompt": PROMPT,
        "canvas_size": inputs["canvas_size"],
        "source_rect_xyxy": inputs["source_rect_xyxy"],
        "method": "native instruction image edit of padded canvas; no native mask argument; final original-source composite",
        "versions": {name: importlib.metadata.version(name) for name in ["torch", "diffusers", "transformers", "huggingface-hub", "Pillow", "safetensors", "accelerate"]},
        "runs": [],
    }
    try:
        print("Loading DreamLite Mobile onto MPS", flush=True)
        load_start = time.perf_counter()
        pipe = DreamLiteMobilePipeline.from_pretrained(
            model, torch_dtype=getattr(torch, args.dtype), local_files_only=True,
        ).to("mps")
        if args.latent_preservation:
            latent_cache = {}
            original_prepare_latents = pipe.prepare_latents
            original_prepare_image_latents = pipe.prepare_image_latents
            original_scheduler_step = pipe.scheduler.step

            def capture_noise(*call_args, **call_kwargs):
                result = original_prepare_latents(*call_args, **call_kwargs)
                latent_cache["noise"] = result.clone()
                return result

            def capture_image(*call_args, **call_kwargs):
                result = original_prepare_image_latents(*call_args, **call_kwargs)
                latent_cache["image"] = result.clone()
                mask = Image.open(inputs["tracks"][0]["mask"]).convert("L")
                mask = mask.resize((result.shape[-1], result.shape[-2]), Image.Resampling.NEAREST)
                latent_cache["generate_mask"] = torch.tensor(np.array(mask) / 255, device=result.device, dtype=result.dtype)[None, None]
                return result

            def preserve_source_step(*call_args, **call_kwargs):
                result = original_scheduler_step(*call_args, **call_kwargs)
                sigma = pipe.scheduler.sigmas[pipe.scheduler.step_index].to(latent_cache["image"].device)
                protected = (1 - sigma) * latent_cache["image"] + sigma * latent_cache["noise"]
                mask = latent_cache["generate_mask"]
                if isinstance(result, tuple):
                    return (mask * result[0] + (1 - mask) * protected,) + result[1:]
                result.prev_sample = mask * result.prev_sample + (1 - mask) * protected
                return result

            pipe.prepare_latents = capture_noise
            pipe.prepare_image_latents = capture_image
            pipe.scheduler.step = preserve_source_step
            report["method"] += "; experimental per-step FlowMatch source latent restoration (sampler adaptation, no model retraining)"
        torch.mps.synchronize()
        report["load_seconds"] = time.perf_counter() - load_start
        print(f"Loaded in {report['load_seconds']:.2f}s", flush=True)
        for track in inputs["tracks"]:
            if track["id"] not in args.tracks:
                continue
            track_start = time.perf_counter()
            print(f"Generating {track['id']} at {inputs['canvas_size']}, {args.steps} steps", flush=True)
            raw = pipe(
                prompt=PROMPT,
                image=Image.open(track["canvas"]).convert("RGB"),
                width=inputs["canvas_size"][0], height=inputs["canvas_size"][1],
                num_inference_steps=args.steps,
                generator=torch.Generator("cpu").manual_seed(args.seed),
            ).images[0].convert("RGB")
            torch.mps.synchronize()
            generation_seconds = time.perf_counter() - track_start
            if list(raw.size) != inputs["canvas_size"]:
                raise ValueError(f"Actual output {raw.size} differs from requested canvas {inputs['canvas_size']}")
            source = Image.open(track["source_512"]).convert("RGB")
            rect = tuple(inputs["source_rect_xyxy"])
            raw_array = np.asarray(raw.crop(rect)).astype(np.int16)
            source_array = np.asarray(source).astype(np.int16)
            composite = raw.copy()
            composite.paste(source, rect[:2])
            raw_path = output / f"{track['id']}_raw.png"
            composite_path = output / f"{track['id']}_composite.png"
            raw.save(raw_path)
            composite.save(composite_path)
            row = {
                "track_id": track["id"], "generation_seconds": generation_seconds,
                "actual_output_size": list(raw.size),
                "raw_center_mae_rgb_0_255": float(np.abs(raw_array - source_array).mean()),
                "composite_center_byte_exact": np.array_equal(np.asarray(composite.crop(rect)), np.asarray(source)),
                "raw_path": str(raw_path), "composite_path": str(composite_path),
                "raw_sha256": sha256(raw_path), "composite_sha256": sha256(composite_path),
                **monitor.result(),
            }
            report["runs"].append(row)
            (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(row), flush=True)
        report["status"] = "completed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = repr(exc)
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        monitor.stop.set()
        monitor.thread.join()
        report.update(monitor.result())
        (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
