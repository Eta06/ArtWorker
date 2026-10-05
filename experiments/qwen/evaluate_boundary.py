#!/usr/bin/env python3
"""CPU-only track3 rail/source-boundary diagnostic, not a universal quality score.

The rail identity and search corridors are frozen from the original source and
the first saved square-known/edge-context probe. Selected image features are
shown on a separate annotated diagnostic so false edges/posts can be audited.
The untouched image montage is also saved. No output pixels are changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments/evaluation/inputs/track3/source_512.png"
RECT = (0, 320, 512, 832)
DEFAULT_OUT = ROOT / "experiments/qwen/boundary_eval/2026-10-05"
PROFILE = {
    "id": "track3-upper-metal-edge-v1",
    "frozen_from": "original 512 source and geometry probe, before later boundary trials",
    "source_y_half_open": [320, 371],
    "source_nominal_x_at_y320": 311.0,
    "source_nominal_dx_dy": -1.94,
    "generated_y_half_open": [270, 320],
    "generated_nominal_x_at_y270": 372.0,
    "generated_nominal_dx_dy": -1.43,
    "search_half_width_pixels": 11,
    "far_fit_generated_y_half_open": [270, 311],
    "near_join_generated_y_half_open": [312, 320],
    "rail_identity": "left/top bright metal ridge, not parallel lower ridge or vertical post",
    "selection": "largest positive luminance x-gradient within frozen corridor; metal color gate",
    "warning": "a rail outside the corridor or a post can invalidate feature identity; inspect annotated points",
}


def rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def robust_line(points: list[dict], y_min: int, y_max: int) -> dict:
    pts = [(p["y"], p["x"]) for p in points if y_min <= p["y"] < y_max and p["valid"]]
    if len(pts) < 10:
        return {"valid": False, "point_count": len(pts), "reason": "fewer than ten detected rail-edge rows"}
    y, x = np.asarray(pts, dtype=float).T
    # Theil-Sen: robust to single posts/texture edges; local rail curvature is
    # retained as fit residual and window sensitivity, never treated as noise.
    slopes = [(x[j] - x[i]) / (y[j] - y[i]) for i in range(len(y)) for j in range(i + 1, len(y)) if y[j] - y[i] >= 8]
    m = float(np.median(slopes))
    b320 = float(np.median(x - m * (y - 320)))
    residual = x - (b320 + m * (y - 320))
    return {
        "valid": True, "point_count": len(pts), "y_range_half_open": [y_min, y_max],
        "dx_dy": m, "x_extrapolated_at_y320": b320,
        "angle_from_downward_y_degrees": float(np.degrees(np.arctan(m))),
        "median_absolute_fit_residual_px": float(np.median(np.abs(residual))),
        "p90_absolute_fit_residual_px": float(np.percentile(np.abs(residual), 90)),
    }


def rail_points(a: np.ndarray, source: bool) -> list[dict]:
    smooth = np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(.55)), dtype=float)
    lum = smooth @ np.array([.2126, .7152, .0722])
    grad = np.zeros_like(lum)
    grad[:, 1:-1] = (lum[:, 2:] - lum[:, :-2]) / 2
    rows = range(*(PROFILE["source_y_half_open"] if source else PROFILE["generated_y_half_open"]))
    points = []
    for y in rows:
        center = (311 - 1.94 * (y - 320)) if source else (372 - 1.43 * (y - 270))
        lo, hi = max(2, int(round(center)) - 11), min(509, int(round(center)) + 11)
        xs = np.arange(lo, hi + 1)
        # Look just beyond a dark->metal edge: bright + blue rather than reeds.
        ahead = smooth[y, xs + 2]
        gate = (ahead[:, 2] - ahead[:, 0] >= 8) & (ahead[:, 2] - ahead[:, 1] >= 2) & (lum[y, xs + 2] >= 72)
        scores = np.where(gate, grad[y, xs], -1)
        j = int(np.argmax(scores))
        x = int(xs[j])
        strength = float(scores[j])
        valid = strength >= 7
        # Subpixel parabolic interpolation around the chosen gradient maximum.
        offset = 0.0
        if valid and lo < x < hi:
            prev, curr, nxt = grad[y, x - 1:x + 2]
            denom = prev - 2 * curr + nxt
            if abs(denom) > 1e-8:
                offset = float(np.clip(.5 * (prev - nxt) / denom, -.5, .5))
        sorted_scores = sorted((float(s), int(xx)) for s, xx in zip(scores, xs) if s >= 7)
        competitors = [xx for s, xx in sorted_scores if abs(xx - x) >= 5 and s >= .75 * strength]
        points.append({"y": y, "x": x + offset, "valid": bool(valid), "gradient_strength": strength,
                       "corridor_edge_hit": bool(x in (lo, hi)), "competing_peak_x": competitors,
                       "nominal_x": center, "corridor": [lo, hi]})
    return points


def boundary_stats(a: np.ndarray, source: np.ndarray) -> dict:
    delta = np.abs(a[320:832].astype(float) - source.astype(float))
    l = a.astype(float) @ np.array([.2126, .7152, .0722])
    bands = {}
    for name, y in [("top", 320), ("bottom", 832)]:
        # Adjacent-row jumps are normalized by actual nearby generated/source
        # row-to-row changes. This is a seam alarm, not semantic correctness.
        jump = np.abs(l[y] - l[y - 1])
        nearby = np.abs(l[y - 12:y + 12][1:] - l[y - 12:y + 12][:-1])
        nearby = np.delete(nearby, 11, axis=0)
        band = l[y - 16:y + 16]
        # X gradients measure texture separately on each side of the seam.
        gx = np.abs(np.diff(band, axis=1))
        bands[name] = {
            "luminance_adjacent_row_mae_255": float(jump.mean()),
            "luminance_adjacent_row_p90_255": float(np.percentile(jump, 90)),
            "nearby_row_change_mae_255": float(nearby.mean()),
            "join_to_nearby_change_ratio": float(jump.mean() / max(nearby.mean(), 1e-8)),
            "x_texture_gradient_above_mean_255": float(gx[:16].mean()),
            "x_texture_gradient_below_mean_255": float(gx[16:].mean()),
            "above_luminance_mean_255": float(band[:16].mean()),
            "below_luminance_mean_255": float(band[16:].mean()),
        }
    return {"source_pixels_exact": bool(np.array_equal(a[320:832], source)),
            "source_mae_255": float(delta.mean()), "source_top16_mae_255": float(delta[:16].mean()),
            "source_bottom16_mae_255": float(delta[-16:].mean()),
            "source_inner_mae_255": float(delta[16:-16].mean()), "boundaries": bands}


def summarize(points: list[dict], source_fit: dict, source_points: list[dict]) -> dict:
    far = robust_line(points, 270, 311)
    near = robust_line(points, 305, 320)
    windows = [robust_line(points, 270, 311), robust_line(points, 280, 311), robust_line(points, 290, 311)]
    source_windows = [source_fit, robust_line(source_points, 320, 351), robust_line(source_points, 320, 341)]
    out = {"far_rail_fit": far, "near_join_fit": near,
           "invalid_feature_rows": [p["y"] for p in points if not p["valid"]],
           "corridor_edge_rows": [p["y"] for p in points if p["corridor_edge_hit"]],
           "ambiguous_peak_rows": [p["y"] for p in points if p["competing_peak_x"]],
           "fit_window_sensitivity": {"generated": windows, "source": source_windows}}
    if far["valid"] and source_fit["valid"]:
        out.update({
            "far_tangent_angle_difference_degrees": far["angle_from_downward_y_degrees"] - source_fit["angle_from_downward_y_degrees"],
            "far_tangent_endpoint_signed_dx_px": far["x_extrapolated_at_y320"] - source_fit["x_extrapolated_at_y320"],
            "far_tangent_endpoint_abs_dx_px": abs(far["x_extrapolated_at_y320"] - source_fit["x_extrapolated_at_y320"]),
        })
        if near["valid"]:
            out["near_to_far_tangent_angle_change_degrees"] = near["angle_from_downward_y_degrees"] - far["angle_from_downward_y_degrees"]
            out["near_source_tangent_angle_difference_degrees"] = near["angle_from_downward_y_degrees"] - source_fit["angle_from_downward_y_degrees"]
        last = [p for p in points if p["valid"] and 312 <= p["y"] < 320]
        if last:
            dx = [p["x"] - (source_fit["x_extrapolated_at_y320"] + source_fit["dx_dy"] * (p["y"] - 320)) for p in last]
            out["near_join_source_tangent_offset"] = {"point_count": len(dx), "median_signed_dx_px": float(np.median(dx)),
                                                      "p90_absolute_dx_px": float(np.percentile(np.abs(dx), 90)),
                                                      "per_row_signed_dx_px": [{"y": p["y"], "dx": float(v)} for p, v in zip(last, dx)]}
    return out


def font(size: int):
    for p in ["/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def montage(items: list[tuple[str, Path]], output: Path, crop=None, scale=1, points=None, source_points=None):
    w, h = (512, 1152) if crop is None else (crop[2] - crop[0], crop[3] - crop[1])
    sw, sh = int(w * scale), int(h * scale)
    panel_w, panel_h = sw + 24, sh + 62
    columns = min(4, len(items))
    rows = (len(items) + columns - 1) // columns
    result = Image.new("RGB", (columns * panel_w + 12, rows * panel_h + 12), "#101619")
    draw = ImageDraw.Draw(result)
    for i, (label, path) in enumerate(items):
        x, y = 12 + (i % columns) * panel_w, 12 + (i // columns) * panel_h
        im = Image.open(path).convert("RGB")
        if crop is not None:
            im = im.crop(crop)
        if points is not None and crop is not None:
            # Annotation is confined to this derived diagnostic. Raw outputs
            # and the separate untouched crop montage retain exact pixels.
            overlay = ImageDraw.Draw(im)
            for p in (points[label] + (source_points or [])):
                if p["valid"] and crop[1] <= p["y"] < crop[3]:
                    px, py = p["x"] - crop[0], p["y"] - crop[1]
                    col = "#ffcc55" if p["y"] < 320 else "#4dffc8"
                    overlay.point((round(px), py), fill=col)
            overlay.line((0, 320 - crop[1], im.width - 1, 320 - crop[1]), fill="#ef5571")
        im = im.resize((sw, sh), Image.Resampling.NEAREST if crop else Image.Resampling.LANCZOS)
        result.paste(im, (x, y + 38))
        draw.text((x, y), label, font=font(16), fill="#edf3f5")
    result.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", action="append", default=[], metavar="LABEL=PATH", help="saved 512x1152 raw/composite; repeat for new candidates")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.image:
        items = [(s.split("=", 1)[0], Path(s.split("=", 1)[1]).resolve()) for s in args.image]
    else:
        base = ROOT / "experiments/qwen/geometry_runs/2026-10-05/track3"
        items = [(f"{arm} {kind}", base / arm / f"{kind}.png") for arm in ("square-known", "edge-context") for kind in ("raw", "composite")]
    if len({label for label, _ in items}) != len(items):
        raise ValueError("Labels must be unique")
    args.output.mkdir(parents=True, exist_ok=True)
    source = rgb(SOURCE)
    source_canvas = np.zeros((1152, 512, 3), dtype=np.uint8)
    source_canvas[320:832] = source
    source_points = rail_points(source_canvas, source=True)
    source_fit = robust_line(source_points, 320, 371)
    report = {"status": "diagnostic", "profile": PROFILE, "source": str(SOURCE), "source_rect_xyxy": RECT,
              "source_upper_edge_fit": source_fit, "source_edge_points": source_points,
              "limitations": ["Track3 only; frozen ROI may miss a relocated/invented rail and then cannot judge geometry.",
                              "Angle/endpoint measurements compare a local visible metal ridge, not a ground-truth extension.",
                              "A rail may curve legitimately farther outside the source. Far angle difference alone is not a failure; the local join must be viewed.",
                              "Parallel rails, posts, bushes, blur or vanished ridges can fool feature extraction; inspect annotated selection.",
                              "Source is gently curved; alternate fit windows quantify model-free sensitivity, not statistical confidence.",
                              "Hard source paste guarantees source equality but does not make the join correct.",
                              "Texture-gradient/adjacent-row jumps flag seams but can flag valid object edges."], "images": {}}
    point_map = {}
    for label, path in items:
        a = rgb(path)
        if a.shape != (1152, 512, 3):
            raise ValueError(f"{path}: expected 512x1152, got {a.shape}")
        pts = rail_points(a, source=False)
        point_map[label] = pts
        report["images"][label] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                   "rail": summarize(pts, source_fit, source_points), "edge_points": pts,
                                   "pixels": boundary_stats(a, source)}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    montage(items, args.output / "full.png", scale=.5)
    crop = (230, 265, 422, 375)
    montage(items, args.output / "rail-crop-untouched.png", crop=crop, scale=3)
    montage(items, args.output / "rail-feature-selection.png", crop=crop, scale=3, points=point_map, source_points=source_points)
    montage(items, args.output / "bottom-crop-untouched.png", crop=(0, 800, 512, 864), scale=1)
    table = ["# Track3 source-boundary diagnostic", "", "This is a local rail feature diagnostic, not an overall quality score. The profile/search corridor is fixed; inspect selected points to reject false edges.", "",
             "| Image | far angle difference deg | far endpoint dx px | near offset median px | top seam ratio | bottom seam ratio | source exact |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for label, rec in report["images"].items():
        rail, pix = rec["rail"], rec["pixels"]
        def f(key):
            return f"{rail[key]:.2f}" if key in rail else "invalid"
        near = rail.get("near_join_source_tangent_offset", {}).get("median_signed_dx_px")
        table.append(f"| {label} | {f('far_tangent_angle_difference_degrees')} | {f('far_tangent_endpoint_signed_dx_px')} | {near:.2f} | {pix['boundaries']['top']['join_to_nearby_change_ratio']:.2f} | {pix['boundaries']['bottom']['join_to_nearby_change_ratio']:.2f} | {pix['source_pixels_exact']} |" if near is not None else f"| {label} | invalid feature |")
    table += ["", "Far fit: generated rows270–310; source rows320–370. Near offset: detected generated rows312–319 against the source tangent. Positive angle difference means the generated rail is less steep. Signed endpoint dx is generated minus source at y320. Raw source pixels are decoded approximations; composites paste the resized source exactly.", "",
              "A smaller near offset can be an elbow that connects incompatible far tangents. Report far angle, endpoint, near residual and full images together; do not collapse them into a winning scalar.", "",
              "Source curvature and rail-edge selection create uncertainty. Per-window sensitivity, rejected rows and competing peaks are in metrics.json. Source fit should remain identical for all later candidates.", "",
              "![Actual unmodified full outputs](full.png)", "", "![Actual untouched rail crops](rail-crop-untouched.png)", "", "![Selected features, yellow generated and green original, pink source boundary](rail-feature-selection.png)", "", "![Actual untouched lower source-boundary crops](bottom-crop-untouched.png)"]
    (args.output / "REPORT.md").write_text("\n".join(table) + "\n")
    print(json.dumps({"output": str(args.output), "source_fit": source_fit,
                      "images": {k: {"angle_diff_deg": v["rail"].get("far_tangent_angle_difference_degrees"),
                                      "endpoint_dx_px": v["rail"].get("far_tangent_endpoint_signed_dx_px"),
                                      "near_offset_px": v["rail"].get("near_join_source_tangent_offset", {}).get("median_signed_dx_px"),
                                      "source_pixels_exact": v["pixels"]["source_pixels_exact"]}
                                 for k, v in report["images"].items()}}, indent=2))


if __name__ == "__main__":
    main()
