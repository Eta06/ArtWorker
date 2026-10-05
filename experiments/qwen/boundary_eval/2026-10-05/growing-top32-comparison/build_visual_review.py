#!/usr/bin/env python3
"""Pixel-derived review sheets; read the frozen completed images only."""
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
OLD = ROOT / "experiments/qwen/controlnet_runs/2026-10-05/track3-growing-localized6"
NEW = ROOT / "experiments/qwen/controlnet_runs/2026-10-05/track3-growing-top32-6step"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def font(size=20):
    for path in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Helvetica.ttf"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def main():
    images = []
    for scale in ("0.5", "1"):
        for label, base in (("Saved zero-source-hint", OLD), ("Source top 32 hint", NEW)):
            images.append((f"{label}; scale {scale}", base / f"tangent-canny/scale{scale}/composite.png"))
    previews = [NEW / f"tangent-canny/scale{scale}/previews/step{step:02d}_predicted_clean_composite.png"
                for scale in ("0.5", "1") for step in (2, 4, 6)]
    inputs = {str(path): sha(path) for _, path in images}
    inputs.update({str(path): sha(path) for path in previews})
    outputs = {}
    for name, rect, factor in (("rail-join-6x.png", (256, 286, 432, 350), 6),
                               ("asphalt-join-2x.png", (0, 768, 512, 896), 2)):
        w, h = (rect[2] - rect[0]) * factor, (rect[3] - rect[1]) * factor
        sheet = Image.new("RGB", (2 * (w + 20) + 12, 2 * (h + 52) + 12), "#11181c")
        draw = ImageDraw.Draw(sheet)
        for i, (label, path) in enumerate(images):
            x, y = 12 + (i % 2) * (w + 20), 12 + (i // 2) * (h + 52)
            pixels = Image.open(path).convert("RGB").crop(rect).resize((w, h), Image.Resampling.NEAREST)
            sheet.paste(pixels, (x, y + 32))
            draw.text((x, y), label, font=font(), fill="#f1f5f7")
            boundary = 320 if name.startswith("rail") else 832
            by = y + 32 + (boundary - rect[1]) * factor
            draw.line((x - 5, by, x - 1, by), fill="#ffcd6b", width=2)
            draw.line((x + w + 1, by, x + w + 5, by), fill="#ffcd6b", width=2)
        path = OUT / name
        sheet.save(path)
        outputs[name] = {"sha256": sha(path), "source_crop_xyxy": list(rect), "nearest_scale": factor}
    # Step2 has only the true active896 rows at final y128:1024. Grey bands
    # explicitly show absent target support, rather than invented output pixels.
    scale_factor = 0.75
    w, h = 384, 864
    sheet = Image.new("RGB", (3 * (w + 20) + 12, 2 * (h + 60) + 12), "#11181c")
    draw = ImageDraw.Draw(sheet)
    for i, path in enumerate(previews):
        arm = "0.5" if i < 3 else "1"
        step = (2, 4, 6)[i % 3]
        actual = Image.open(path).convert("RGB")
        canvas = Image.new("RGB", (512, 1152), "#3d454b")
        offset = 128 if step == 2 else 0
        expected = (512, 896 if step == 2 else 1152)
        if actual.size != expected:
            raise ValueError(f"Unexpected actual preview dimensions: {path}")
        canvas.paste(actual, (0, offset))
        x, y = 12 + (i % 3) * (w + 20), 12 + (i // 3) * (h + 60)
        sheet.paste(canvas.resize((w, h), Image.Resampling.NEAREST), (x, y + 42))
        label = f"Scale {arm}; step {step}; active y{offset}:{offset + actual.height}"
        draw.text((x, y), label, font=font(16), fill="#f1f5f7")
        if step == 2:
            draw.text((x + 8, y + 54), "Future region outside current support", font=font(15), fill="#d9e0e5")
            draw.text((x + 8, y + 42 + int(1038 * scale_factor)), "Future region outside current support", font=font(15), fill="#d9e0e5")
    path = OUT / "step-progression.png"
    sheet.save(path)
    outputs[path.name] = {"sha256": sha(path), "step2_actual_window_xyxy": [0, 128, 512, 1024],
                          "grey_bands_definition": "future target rows absent from this preview", "nearest_scale": scale_factor}
    for path, expected in inputs.items():
        if sha(path) != expected:
            raise AssertionError(f"Review changed a frozen input: {path}")
    report = {"status": "passed", "script_sha256": sha(__file__), "inputs_sha256": inputs,
              "outputs": outputs, "original_outputs_unchanged": True, "generated_pixels": False}
    (OUT / "review_artifact_provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "passed", "artifacts": list(outputs)}))


if __name__ == "__main__":
    main()
