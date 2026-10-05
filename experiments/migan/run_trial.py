"""CPU-only complete MI-GAN ONNX pipeline on the common outpaint canvas."""
import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=["track1", "track2", "track3"], default="track3")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=["full", "strips"], default="full")
    parser.add_argument("--band", type=int, default=64)
    args = parser.parse_args()
    import numpy as np
    from PIL import Image
    import onnxruntime as ort
    manifest = json.loads((ROOT / "experiments/migan/download_manifest.json").read_text())
    if not manifest["verified"]:
        raise ValueError("Unverified model")
    args.output.mkdir(parents=True, exist_ok=True)
    shared = json.loads((ROOT / "experiments/evaluation/inputs/inputs.json").read_text())
    track = next(t for t in shared["tracks"] if t["id"] == args.track)
    source = Image.open(track["source_512"]).convert("RGB")
    rect = shared["source_rect_xyxy"]
    width, height = shared["canvas_size"]
    canvas = Image.new("RGB", (width, height), (128, 128, 128))
    canvas.paste(source, (rect[0], rect[1]))
    mask = np.zeros((1, 1, height, width), dtype=np.uint8)
    mask[:, :, rect[1]:rect[3], rect[0]:rect[2]] = 255
    image = np.asarray(canvas, dtype=np.uint8).transpose(2, 0, 1)[None]
    record = {"track": args.track, "repo": manifest["repo"], "revision": manifest["revision"],
        "model_bytes": manifest["bytes"], "provider": "CPUExecutionProvider", "threads": 2,
        "width": width, "height": height, "source_rect": rect, "mask": "255 known; 0 generate",
        "runtime_version": ort.__version__, "visual_quality_accepted": False, "method": args.method,
        "true_single_pass_spatial_streaming": False,
        "raw_note": "Upstream pipeline already blends known pixels; this is pipeline output, not an uncomposited generator prediction",
        "status": "running"}
    (args.output / "metrics.json").write_text(json.dumps(record, indent=2))
    start = time.perf_counter()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(ROOT / ".build/models/migan/migan_pipeline_v2.onnx"), options,
        providers=["CPUExecutionProvider"])
    record["load_seconds"] = time.perf_counter() - start
    start = time.perf_counter()
    if args.method == "full":
        prediction = session.run(None, {"image": image, "mask": mask})[0]
    else:
        if not 1 <= args.band < width or rect[0] != 0 or rect[2] != width:
            raise ValueError("Strip trial requires full-width source and a band smaller than the working window")
        prediction = image.copy()
        top, bottom = rect[1], rect[3]
        calls = []
        while top > 0 or bottom < height:
            for edge in ("top", "bottom"):
                if (edge == "top" and top == 0) or (edge == "bottom" and bottom == height):
                    continue
                a, b = (max(0, top - args.band), top) if edge == "top" else (bottom, min(height, bottom + args.band))
                y = a if edge == "top" else b - width
                tile = prediction[:, :, y:y + width, :].copy()
                known = np.ones((1, 1, width, width), dtype=np.uint8) * 255
                known[:, :, a-y:b-y] = 0
                call_start = time.perf_counter()
                result = session.run(None, {"image": tile, "mask": known})[0]
                prediction[:, :, a:b] = result[:, :, a-y:b-y]
                calls.append({"edge": edge, "rows": [a, b], "working_window": [y, y + width], "seconds": time.perf_counter() - call_start})
                if edge == "top":
                    top = a
                else:
                    bottom = b
        record["calls"] = calls
        record["iteration_note"] = "Ten small-window sequential predictions for band64; not one-pass generative streaming, not a matched-cost baseline"
    record["pipeline_inference_seconds"] = time.perf_counter() - start
    if prediction.shape != image.shape or prediction.dtype != np.uint8:
        raise ValueError(f"Unexpected output: {prediction.shape}/{prediction.dtype}")
    raw = Image.fromarray(prediction[0].transpose(1, 2, 0))
    raw.save(args.output / "pipeline.png")
    record["source_exact_in_pipeline"] = bool(np.array_equal(np.asarray(raw.crop(rect)), np.asarray(source)))
    record["pipeline_source_mae_255"] = float(np.abs(np.asarray(raw.crop(rect), dtype=np.float32) - np.asarray(source, dtype=np.float32)).mean())
    composite = raw.copy()
    composite.paste(source, (rect[0], rect[1]))
    composite.save(args.output / "composite.png")
    record["source_exact_in_composite"] = bool(np.array_equal(np.asarray(composite.crop(rect)), np.asarray(source)))
    record["status"] = "completed"
    (args.output / "metrics.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
