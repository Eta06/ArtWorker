#!/usr/bin/env python3
"""Prepare shared cover inputs and inspect model outputs; Pillow is the only dependency."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageStat


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CANVAS = (512, 1152)
SOURCE_RECT = (0, 320, 512, 832)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    tracks = json.loads((ROOT / "Player/Resources/tracks.json").read_text())
    metadata = {
        "canvas_size": list(CANVAS),
        "source_rect_xyxy": list(SOURCE_RECT),
        "source_rect_xywh": [0, 320, 512, 512],
        "mask_convention": "white 255 = generate; black 0 = preserve",
        "resize_filter": "Pillow LANCZOS",
        "generate_fraction": 640 / 1152,
        "reference_phone_size": [402, 874],
        "aspect_note": "512x1152 uses 64-pixel multiples; all candidates share this canvas. It is slightly taller than 402x874.",
        "tracks": [],
    }
    for track in tracks:
        source = ROOT / "Player/Resources" / f"{track['id']}.jpg"
        with Image.open(source) as im:
            image = im.convert("RGB")
        width, height = image.size
        # Match Track.artwork: crop only the square centered in a 16:9 export.
        crop = ((width - height) // 2, 0, (width + height) // 2, height) if width * 9 == height * 16 else (0, 0, width, height)
        square = image.crop(crop)
        item_dir = directory / track["id"]
        item_dir.mkdir(exist_ok=True)
        square_path = item_dir / "source_square.png"
        square.save(square_path)
        working = square.resize((512, 512), Image.Resampling.LANCZOS)
        working_path = item_dir / "source_512.png"
        working.save(working_path)
        canvas = Image.new("RGB", CANVAS, (0, 0, 0))
        canvas.paste(working, SOURCE_RECT[:2])
        canvas_path = item_dir / "canvas.png"
        canvas.save(canvas_path)
        mask = Image.new("L", CANVAS, 255)
        mask.paste(0, SOURCE_RECT)
        mask_path = item_dir / "mask.png"
        mask.save(mask_path)
        rgba = Image.new("RGBA", CANVAS, (0, 0, 0, 0))
        rgba.paste(working.convert("RGBA"), SOURCE_RECT[:2])
        rgba_path = item_dir / "canvas_rgba.png"
        rgba.save(rgba_path)
        metadata["tracks"].append({
            **track,
            "original_jpg": str(source),
            "original_jpg_sha256": digest(source),
            "original_size": [width, height],
            "original_crop_xyxy": list(crop),
            "original_square_size": list(square.size),
            "source_square": str(square_path),
            "source_512": str(working_path),
            "canvas": str(canvas_path),
            "mask": str(mask_path),
            "canvas_rgba": str(rgba_path),
        })
    write_json(directory / "inputs.json", metadata)
    print(directory / "inputs.json")


def difference_stats(a: Image.Image, b: Image.Image) -> dict:
    diff = ImageChops.difference(a.convert("RGB"), b.convert("RGB"))
    extrema = diff.getextrema()
    changed = sum(1 for pixel in diff.getdata() if any(pixel))
    pixels = a.width * a.height
    return {
        "identical_pixels_fraction": (pixels - changed) / pixels,
        "mean_absolute_rgb_error_0_to_255": sum(ImageStat.Stat(diff).mean) / 3,
        "maximum_channel_error_0_to_255": max(x[1] for x in extrema),
    }


def row_delta(image: Image.Image, row_a: int, row_b: int) -> float:
    a = image.crop((0, row_a, image.width, row_a + 1))
    b = image.crop((0, row_b, image.width, row_b + 1))
    return sum(ImageStat.Stat(ImageChops.difference(a, b)).mean) / 3


def inspect_output(run: dict, inputs: dict, manifest_dir: Path) -> dict:
    result = dict(run)
    if run.get("status", "completed") != "completed" or not run.get("image"):
        result["metrics_status"] = "no completed image"
        return result
    path = Path(run["image"])
    if not path.is_absolute():
        path = manifest_dir / path
    result["image"] = str(path.resolve())
    if not path.exists():
        result["metrics_status"] = "image missing"
        return result
    try:
        image = Image.open(path).convert("RGB")
        result["output_size"] = list(image.size)
        if run.get("diagnostic_control") or run.get("shared_canvas_comparable") is False:
            result["metrics_status"] = "diagnostic control; excluded from shared-canvas metrics"
            return result
        if image.size != tuple(inputs["canvas_size"]):
            result["metrics_status"] = "dimension mismatch; no resizing for metrics"
            return result
        track = next(t for t in inputs["tracks"] if t["id"] == run["track_id"])
        original = Image.open(track["source_512"]).convert("RGB")
        result["source_preservation_reference_image"] = track["source_512"]
        result["source_preservation_reference_size"] = list(original.size)
        result["source_preservation_caveat"] = "Equality is against the Lanczos-resized 512×512 reference, not original decoded 720×720 pixels."
        rect = tuple(inputs["source_rect_xyxy"])
        result["source_preservation"] = difference_stats(image.crop(rect), original)
        top, bottom = rect[1], rect[3]
        boundaries = {}
        for name, row in (("top", top), ("bottom", bottom)):
            boundary_delta = row_delta(image, row - 1, row)
            adjacent = [row_delta(image, r, r + 1) for r in (row - 4, row - 3, row - 2, row, row + 1, row + 2)]
            nearby_delta = sum(adjacent) / len(adjacent)
            boundaries[name] = {
                "boundary_adjacent_row_mean_rgb_difference": boundary_delta,
                "nearby_adjacent_rows_mean_rgb_difference": nearby_delta,
                "boundary_to_nearby_ratio": boundary_delta / nearby_delta if nearby_delta else None,
            }
        result["boundary_heuristic"] = boundaries
        result["metrics_status"] = "measured"
    except (OSError, StopIteration, ValueError) as error:
        result["metrics_status"] = f"could not measure: {error}"
    return result


def font(size: int) -> ImageFont.ImageFont:
    for path in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def contact_sheet(runs: list[dict], destination: Path, columns: int) -> None:
    tile_w, tile_h = 256, 650
    rows = (len(runs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_w, rows * tile_h), (22, 23, 27))
    draw = ImageDraw.Draw(sheet)
    for index, run in enumerate(runs):
        x, y = index % columns * tile_w, index // columns * tile_h
        label = f"{run.get('model', '?')}\n{run.get('track_id', '?')} / seed {run.get('seed', '?')}"
        draw.multiline_text((x + 12, y + 10), label, fill="white", font=font(13), spacing=4)
        path = Path(run.get("image") or "")
        if path.is_file():
            try:
                image = Image.open(path).convert("RGB")
                image.thumbnail((232, 530), Image.Resampling.LANCZOS)
                sheet.paste(image, (x + (tile_w - image.width) // 2, y + 68))
            except OSError:
                pass
        details = run.get("status", "completed")
        if run.get("elapsed_seconds") is not None:
            details += f" / {run['elapsed_seconds']:.1f}s"
        draw.text((x + 12, y + 608), details, fill=(185, 187, 197), font=font(12))
    sheet.save(destination)


def compare(manifest: Path, input_manifest: Path, destination: Path, columns: int) -> None:
    inputs = json.loads(input_manifest.read_text())
    raw = json.loads(manifest.read_text())
    runs = raw["runs"] if isinstance(raw, dict) else raw
    results = [inspect_output(run, inputs, manifest.parent) for run in runs]
    destination.mkdir(parents=True, exist_ok=True)
    report = {
        "input_manifest": str(input_manifest.resolve()),
        "caveat": "Boundary deltas are a crude seam flag, not perceptual quality scores. Exact source-pixel preservation also does not prove coherent outpainting; it compares the resized source_512.png reference, not original 720x720 pixels. Compare raw model outputs; label final compositing separately. Timings and memory measurements are supplied by the inference runner, not this tool.",
        "runs": results,
    }
    write_json(destination / "report.json", report)
    if results:
        contact_sheet(results, destination / "contact_sheet.png", columns)
    cards = []
    for run in results:
        image = run.get("image")
        img = f'<img src="{html.escape(Path(image).as_uri())}" loading="lazy">' if image and Path(image).is_file() else ""
        cards.append(f"<article><h2>{html.escape(str(run.get('model', '?')))}</h2><p>{html.escape(str(run.get('track_id', '?')))} / seed {html.escape(str(run.get('seed', '?')))}</p>{img}<pre>{html.escape(json.dumps(run, ensure_ascii=False, indent=2))}</pre></article>")
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>ArtWorker outpainting comparison</title>
<style>body{{background:#15161a;color:#eee;font:14px system-ui;margin:24px}}main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px}}article{{background:#22242a;padding:16px;border-radius:12px}}img{{width:100%;max-height:780px;object-fit:contain}}pre{{font-size:11px;white-space:pre-wrap;overflow-wrap:anywhere}}h2{{font-size:18px}}</style>
<h1>Outpainting comparison</h1><p>{html.escape(report['caveat'])}</p><main>{''.join(cards)}</main></html>"""
    (destination / "comparison.html").write_text(page)
    print(destination / "report.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--destination", type=Path, default=HERE / "inputs")
    c = sub.add_parser("compare")
    c.add_argument("manifest", type=Path)
    c.add_argument("--inputs", type=Path, default=HERE / "inputs/inputs.json")
    c.add_argument("--destination", type=Path, default=HERE / "comparison")
    c.add_argument("--columns", type=int, default=3)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.destination.resolve())
    elif args.columns < 1:
        parser.error("--columns must be positive")
    else:
        compare(args.manifest.resolve(), args.inputs.resolve(), args.destination.resolve(), args.columns)


if __name__ == "__main__":
    main()
