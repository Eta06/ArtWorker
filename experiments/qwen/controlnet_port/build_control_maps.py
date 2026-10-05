"""CPU-only structural inputs for the Qwen Fun Union ControlNet experiment.

Maps are conditioning inputs, never painted into an outpainted result. The
known rectangle comes from the supplied mask. Long, locally straight edges
that reach an outward-facing source border may be continued into the unknown
region when another nearby parallel edge supports the detection. This rejects
the short isolated text/logo fragments in the current test set. It is a
geometric heuristic, not a semantic guardrail detector or a production claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def known_rectangle(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return the complete rectangular known area; reject unexpected masks."""
    if mask.ndim != 2 or set(np.unique(mask).tolist()) != {0, 255}:
        raise ValueError("Expected binary mask with known=0, regenerate=255")
    ys, xs = np.nonzero(mask == 0)
    rect = int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1
    x1, y1, x2, y2 = rect
    expected = np.full(mask.shape, 255, dtype=np.uint8)
    expected[y1:y2, x1:x2] = 0
    if not np.array_equal(expected, mask):
        raise ValueError("Known pixels must form one filled rectangle")
    return rect


def angle_difference(a: float, b: float) -> float:
    return abs((a-b+math.pi/2) % math.pi-math.pi/2)


def line_candidates(gray: np.ndarray, canny: np.ndarray, active_borders: list[str],
                    min_length: float, max_border_gap: float) -> list[dict]:
    h, w = gray.shape
    detected = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV).detect(gray)[0]
    if detected is None:
        return []
    candidates = []
    supported = cv2.dilate(canny, np.ones((5, 5), np.uint8))
    for row in detected[:, 0]:
        p1 = row[:2].astype(np.float64)
        p2 = row[2:].astype(np.float64)
        delta = p2-p1
        length = float(np.linalg.norm(delta))
        if length < min_length:
            continue
        direction = delta/length
        theta = math.atan2(float(direction[1]), float(direction[0])) % math.pi
        sample = p1[None] + np.linspace(0, 1, max(16, int(length)))[:, None]*delta[None]
        sample = np.round(sample).astype(int)
        sample[:, 0] = np.clip(sample[:, 0], 0, w-1)
        sample[:, 1] = np.clip(sample[:, 1], 0, h-1)
        # LSD may fit the dark side of an edge: allow a two-pixel Canny band.
        support = float(np.mean(supported[sample[:, 1], sample[:, 0]] > 0))
        if support < 0.65:
            continue
        for side in active_borders:
            axis = 1 if side in {"top", "bottom"} else 0
            bound = 0.0 if side in {"top", "left"} else float((h if axis else w)-1)
            gap = float(min(abs(p1[axis]-bound), abs(p2[axis]-bound)))
            if gap > max_border_gap or abs(direction[axis]) < 0.25:
                continue
            # Physical boundary is the pixel center immediately outside source.
            crossing_bound = -0.5 if side in {"top", "left"} else bound+0.5
            crossing = p1 + (crossing_bound-p1[axis])/direction[axis]*direction
            other = 1-axis
            if not -0.5 <= crossing[other] <= (w if axis else h)-0.5:
                continue
            outward_sign = -1.0 if side in {"top", "left"} else 1.0
            outward = direction*np.sign(outward_sign*direction[axis])
            candidates.append({"side": side, "points": [p1.tolist(), p2.tolist()],
                               "length": length, "border_gap": gap, "canny_support": support,
                               "angle_radians": theta, "crossing": crossing.tolist(),
                               "outward_unit": outward.tolist(), "accepted": False,
                               "parallel_support": []})
    return candidates


def supported_parallel_edges(candidates: list[dict], max_angle_degrees: float,
                             min_separation: float, max_separation: float) -> list[dict]:
    """Keep a nearby parallel family, merging duplicate fits to the same edge."""
    for i, a in enumerate(candidates):
        normal = np.array([-math.sin(a["angle_radians"]), math.cos(a["angle_radians"])])
        crossing = np.array(a["crossing"])
        for j, b in enumerate(candidates):
            if i == j or a["side"] != b["side"]:
                continue
            if math.degrees(angle_difference(a["angle_radians"], b["angle_radians"])) > max_angle_degrees:
                continue
            separation = abs(float(np.dot(np.array(b["crossing"])-crossing, normal)))
            if min_separation <= separation <= max_separation:
                a["parallel_support"].append(j)
        a["accepted"] = bool(a["parallel_support"])
    retained = []
    for candidate in sorted(candidates, key=lambda c: c["length"]*c["canny_support"], reverse=True):
        if not candidate["accepted"]:
            continue
        duplicate = False
        for prior in retained:
            if candidate["side"] != prior["side"]:
                continue
            n = np.array([-math.sin(prior["angle_radians"]), math.cos(prior["angle_radians"])])
            distance = abs(float(np.dot(np.array(candidate["crossing"])-np.array(prior["crossing"]), n)))
            if distance < 2.0 and math.degrees(angle_difference(candidate["angle_radians"], prior["angle_radians"])) < 4.0:
                duplicate = True
                break
        if not duplicate:
            retained.append(candidate)
    return retained


def make_maps(source: np.ndarray, mask: np.ndarray, min_length: float = 64,
              max_border_gap: float = 12, max_extrapolation: float = 256) -> dict:
    rect = known_rectangle(mask)
    x1, y1, x2, y2 = rect
    if source.shape != (y2-y1, x2-x1, 3):
        raise ValueError("Source dimensions do not match the mask's known area")
    h, w = mask.shape
    black_canvas = np.zeros((h, w, 3), dtype=np.uint8)
    black_canvas[y1:y2, x1:x2] = source
    gray_canvas = np.full((h, w, 3), 128, dtype=np.uint8)
    gray_canvas[y1:y2, x1:x2] = source
    gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    canny = cv2.Canny(gray, 80, 160, L2gradient=True)
    edges = np.zeros((h, w), dtype=np.uint8)
    edges[y1:y2, x1:x2] = canny
    sides = [name for name, active in [("top", y1>0), ("bottom", y2<h),
                                      ("left", x1>0), ("right", x2<w)] if active]
    candidates = line_candidates(gray, canny, sides, min_length, max_border_gap)
    retained = supported_parallel_edges(candidates, 6, 2.0, 24.0)
    guide_only = np.zeros_like(edges)
    for line in retained:
        start = np.array(line["crossing"])+np.array([x1, y1])
        end = start+max_extrapolation*np.array(line["outward_unit"])
        # cv2 clip handles image edges. Unknown-only application never changes
        # known-source Canny pixels, including the physical boundary pixel.
        cv2.line(guide_only, tuple(np.round(start).astype(int)),
                 tuple(np.round(end).astype(int)), 255, 1, cv2.LINE_AA)
        line["canvas_start"] = start.tolist()
        line["canvas_end_before_clip"] = end.tolist()
    guide_only[mask == 0] = 0
    extended = np.maximum(edges, guide_only)
    if not np.array_equal(extended[mask == 0], edges[mask == 0]):
        raise AssertionError("Structural extension changed known source edges")
    return {"source_black": black_canvas, "source_midgray": gray_canvas,
            "canny_known": edges, "canny_extrapolated": extended,
            "extrapolated_only": guide_only, "mask": mask,
            "mask_known_white": 255-mask, "rect": rect,
            "candidates": candidates, "retained": retained,
            "settings": {"min_length_px": min_length, "max_border_gap_px": max_border_gap,
                         "max_extrapolation_euclidean_px": max_extrapolation,
                         "parallel_angle_degrees": 6, "parallel_normal_separation_px": [2, 24],
                         "canny_thresholds": [80, 160], "line_width": 1,
                         "known_source_edges_preserved_exact": True}}


def save_maps(track: dict, inputs: dict, output: Path, args) -> dict:
    source_path, mask_path = Path(track["source_512"]), Path(track["mask"])
    source = np.asarray(Image.open(source_path).convert("RGB"))
    mask = np.asarray(Image.open(mask_path).convert("L"))
    maps = make_maps(source, mask, args.min_length, args.max_border_gap, args.max_extrapolation)
    if tuple(maps["rect"]) != tuple(inputs["source_rect_xyxy"]):
        raise AssertionError("Source ROI differs from shared evaluator geometry")
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in ["source_black", "source_midgray", "canny_known", "canny_extrapolated", "extrapolated_only"]:
        array = maps[name]
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=2)
        path = output/f"{name}.png"
        Image.fromarray(array, "RGB").save(path)
        files[name] = str(path)
    for name in ["mask", "mask_known_white"]:
        path = output/f"{name}.png"
        Image.fromarray(maps[name], "L").save(path)
        files[name] = str(path)
    # Diagnostic overlay only; this RGB image is not a generated result.
    overlay = maps["source_midgray"].copy()
    overlay[maps["extrapolated_only"] > 0] = [255, 70, 55]
    for line in maps["retained"]:
        p1, p2 = np.array(line["points"])+np.array(maps["rect"][:2])
        cv2.line(overlay, tuple(np.round(p1).astype(int)), tuple(np.round(p2).astype(int)),
                 (255, 70, 55), 2, cv2.LINE_AA)
    overlay_path = output/"diagnostic_overlay.png"
    Image.fromarray(overlay, "RGB").save(overlay_path)
    files["diagnostic_overlay"] = str(overlay_path)
    record = {"track": track["id"], "canvas_size": inputs["canvas_size"],
              "source_rect_from_mask": list(maps["rect"]),
              "source_sha256": sha256(source_path), "mask_sha256": sha256(mask_path),
              "original_jpg_sha256": sha256(Path(track["original_jpg"])),
              "mask_convention": "mask.png: 255=regenerate, 0=known; mask_known_white.png is inverted",
              "source_midgray_note": "PNG rounds midgray to128. Official codec must mask normalized RGB to exact0 before opaque-alpha VAE encoding.",
              "mask_only_note": "Missing edge control must be literal zero64 latent, not black RGB VAE encoding.",
              "purpose": "conditioning input; no outpainted result is modified",
              "heuristic_not_semantic_detector": True, "settings": maps["settings"],
              "candidate_lines": maps["candidates"], "retained_lines": maps["retained"],
              "generated_edge_pixels": int(np.count_nonzero(maps["extrapolated_only"])),
              "files": files, "file_sha256": {k: sha256(Path(v)) for k, v in files.items()}}
    if record["original_jpg_sha256"] != track["original_jpg_sha256"]:
        raise AssertionError("Original artwork hash changed")
    (output/"maps.json").write_text(json.dumps(record, indent=2)+"\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=ROOT/"experiments/evaluation/inputs/inputs.json")
    parser.add_argument("--output", type=Path, default=ROOT/"experiments/qwen/controlnet_inputs/2026-10-05")
    parser.add_argument("--tracks", nargs="+", default=["track1", "track2", "track3"])
    parser.add_argument("--min-length", type=float, default=64)
    parser.add_argument("--max-border-gap", type=float, default=12)
    parser.add_argument("--max-extrapolation", type=float, default=256)
    args = parser.parse_args()
    inputs = json.loads(args.inputs.read_text())
    records = [save_maps(t, inputs, args.output/t["id"], args) for t in inputs["tracks"] if t["id"] in args.tracks]
    if len(records) != len(args.tracks):
        raise ValueError("Unknown or duplicate track requested")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "prepared_not_model_evaluated", "device": "CPU", "model_weights_loaded": False,
                "script_sha256": sha256(Path(__file__)), "inputs_sha256": sha256(args.inputs),
                "tracks": records, "all_original_artworks_unchanged": True}
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"status": manifest["status"], "output": str(args.output),
                      "tracks": [{"track": r["track"], "retained_lines": len(r["retained_lines"]),
                                  "generated_edge_pixels": r["generated_edge_pixels"]} for r in records]}, indent=2))


if __name__ == "__main__":
    main()
