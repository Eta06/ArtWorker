"""CPU diagnostic: global source registration, without a bending flow collar.

All transforms are estimated from the decoded original square, never from
hand-picked rail coordinates. This postprocessor cannot invent missing content.
Reflected border pixels and exact-source compositing are reported explicitly.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/qwen"))
from run_boundary_compositor import compose, harmonic_transfer

RECT = (0, 320, 512, 832)


def global_matrix(local, rect=RECT):
    """Lift an output->input transform from source-crop to canvas coordinates."""
    x, y = rect[:2]
    origin = np.array([[1, 0, x], [0, 1, y], [0, 0, 1]], dtype=np.float32)
    matrix = np.eye(3, dtype=np.float32)
    matrix[:2] = local
    return (origin @ matrix @ np.linalg.inv(origin))[:2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if (args.output / "metrics.json").exists():
        raise FileExistsError("Choose a fresh diagnostic directory")
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(2)
    start = time.monotonic()
    raw = np.asarray(Image.open(args.raw).convert("RGB"))
    source_path = ROOT / "experiments/evaluation/inputs/track3/source_512.png"
    source = np.asarray(Image.open(source_path).convert("RGB"))
    if raw.shape != (1152, 512, 3):
        raise ValueError("This diagnostic is frozen to the shared track3 geometry")
    template = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    decoded = cv2.cvtColor(raw[320:832], cv2.COLOR_RGB2GRAY)
    # Suppress fine stochastic water texture; retain scene structure.
    template = cv2.GaussianBlur(template, (5, 5), 1)
    decoded = cv2.GaussianBlur(decoded, (5, 5), 1)
    arms = {"baseline": compose(raw, source)}
    records = {}
    for name, mode in [("translation", cv2.MOTION_TRANSLATION),
                       ("rigid", cv2.MOTION_EUCLIDEAN), ("affine", cv2.MOTION_AFFINE)]:
        try:
            correlation, local = cv2.findTransformECC(template, decoded, np.eye(2, 3, dtype=np.float32), mode,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5), None, 5)
            matrix = global_matrix(local)
            corners = np.array([[0, 0], [511, 0], [0, 1151], [511, 1151]], dtype=float)
            moved = corners @ matrix[:, :2].T + matrix[:, 2]
            max_shift = float(np.linalg.norm(moved - corners, axis=1).max())
            determinant = float(np.linalg.det(matrix[:, :2]))
            if not .8 <= determinant <= 1.2 or max_shift > 64:
                raise ValueError(f"Unstable global fit: det={determinant}, max shift={max_shift}")
            aligned = cv2.warpAffine(raw, matrix, (512, 1152), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                     borderMode=cv2.BORDER_REFLECT_101)
            valid = cv2.warpAffine(np.full((1152, 512), 255, np.uint8), matrix, (512, 1152),
                flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT)
            before = float(np.abs(raw[320:832].astype(float) - source).mean())
            after = float(np.abs(aligned[320:832].astype(float) - source).mean())
            arms[name] = compose(aligned, source)
            corrected, color = harmonic_transfer(aligned, source, strip=128, sigma=4, residual_cap=40)
            arms[name + "-color"] = compose(corrected, source)
            records[name] = {"status": "completed", "ecc": correlation, "matrix_output_to_raw": matrix.tolist(),
                "source_before_mae_255": before, "source_after_mae_255": after,
                "reflected_border_fraction": float((valid == 0).mean()), "max_corner_shift_px": max_shift,
                "color_transfer": color, "visual_quality_accepted": False}
        except (cv2.error, ValueError) as exc:
            records[name] = {"status": "failed", "error": str(exc), "visual_quality_accepted": False}
    sheet = Image.new("RGB", (len(arms) * 256, 608), "#16171a")
    draw = ImageDraw.Draw(sheet)
    for index, (name, output) in enumerate(arms.items()):
        Image.fromarray(output).save(args.output / (name + ".png"))
        sheet.paste(Image.fromarray(output).resize((256, 576)), (index * 256, 32))
        draw.text((index * 256 + 8, 8), name, fill="white")
        if not np.array_equal(output[320:832], source):
            raise AssertionError("Source changed in composite")
    sheet.save(args.output / "contact-sheet.png")
    result = {"raw_sha256": hashlib.sha256(args.raw.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(), "seconds": time.monotonic() - start,
        "method": "global ECC registration from known square; reflected border; optional harmonic color collar",
        "source_exact_all_composites": True, "visual_quality_accepted": False, "arms": records}
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
