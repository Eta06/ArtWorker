#!/usr/bin/env python3
"""CPU-only, exterior-only boundary repair diagnostic.

This is a compositor experiment, not another denoising/model trial. It registers
the decoded known source against the original and extends that displacement
into the generated collar. A separate DCT harmonic color residual can reduce
the hard-paste color seam. Neither procedure can invent missing scene geometry.
The original source is copied exactly after every variant.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "experiments/qwen/boundary_compositor_runs/2026-10-05"
RECT = (0, 320, 512, 832)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, array: np.ndarray) -> None:
    Image.fromarray(np.clip(np.rint(array), 0, 255).astype(np.uint8)).save(path)


def compose(raw: np.ndarray, source: np.ndarray, rect=RECT) -> np.ndarray:
    x0, y0, x1, y1 = rect
    output = np.clip(np.rint(raw), 0, 255).astype(np.uint8)
    output[y0:y1, x0:x1] = source
    if not np.array_equal(output[y0:y1, x0:x1], source):
        raise AssertionError("Compositor changed original source pixels")
    return output


def smooth_field(field: np.ndarray, sigma: float = 2.5) -> np.ndarray:
    smoothed = np.empty_like(field)
    for channel in range(field.shape[-1]):
        smoothed[..., channel] = cv2.GaussianBlur(
            field[..., channel], (0, 0), sigma, borderType=cv2.BORDER_REFLECT101)
    return smoothed


def displacement(raw: np.ndarray, source: np.ndarray, method: str, cap: float):
    """Return output->raw remap displacement, derived from known pixels only.

    Flow is original -> decoded source. Its direction is therefore already the
    inverse image-warp direction needed by cv2.remap. We cannot observe true
    correspondence beyond the source; collar continuation is a heuristic.
    """
    x0, y0, x1, y1 = RECT
    decoded = raw[y0:y1, x0:x1].astype(np.uint8)
    ref_gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    raw_gray = cv2.cvtColor(decoded, cv2.COLOR_RGB2GRAY)
    ref_gray = cv2.GaussianBlur(ref_gray, (5, 5), 0.7)
    raw_gray = cv2.GaussianBlur(raw_gray, (5, 5), 0.7)
    if method == "dis":
        estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        estimator.setUseSpatialPropagation(True)
        flow = estimator.calc(ref_gray, raw_gray, None)
    elif method == "farneback":
        flow = cv2.calcOpticalFlowFarneback(
            ref_gray, raw_gray, None, 0.5, 4, 21, 5, 7, 1.5, 0)
    else:
        raise ValueError(method)
    flow = smooth_field(flow)
    mag = np.linalg.norm(flow, axis=-1)
    flow *= np.minimum(1.0, cap / np.maximum(mag, 1e-6))[..., None]
    rows, cols = np.indices(source.shape[:2], dtype=np.float32)
    aligned = cv2.remap(decoded, cols + flow[..., 0], rows + flow[..., 1],
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    stats = {"algorithm": method, "cap_pixels": cap,
             "source_before_mae_255": float(np.abs(decoded.astype(float)-source).mean()),
             "source_after_mae_255": float(np.abs(aligned.astype(float)-source).mean()),
             "source_top16_after_mae_255": float(np.abs(aligned[:16].astype(float)-source[:16]).mean()),
             "source_bottom16_after_mae_255": float(np.abs(aligned[-16:].astype(float)-source[-16:]).mean()),
             "flow_magnitude_p50": float(np.quantile(mag, .5)),
             "flow_magnitude_p95": float(np.quantile(mag, .95)),
             "flow_magnitude_max_before_cap": float(mag.max())}
    return flow, stats


def extend_flow(flow: np.ndarray, raw_shape, strip: int, tangent: bool):
    """Extend only observed boundary displacement; taper to zero by strip end.

    No manual rail coordinates or synthesized line masks appear here. The
    optional normal derivative comes from the first 16 known source rows and
    is capped to avoid a runaway extrapolation. It does not guarantee a scene
    line tangent because flow estimates do not know the scene's geometry.
    """
    height, width = raw_shape[:2]
    _, y0, _, y1 = RECT
    field = np.zeros((height, width, 2), dtype=np.float32)
    field[y0:y1] = flow
    details = {}
    for name, known, start, direction in (
        ("top", flow[:16], y0, -1),
        ("bottom", flow[-16:][::-1], y1-1, 1),
    ):
        # Rows nearest the border dominate, while a small average suppresses
        # codec texture/noise in a single pixel row.
        weights = np.exp(-np.arange(6, dtype=np.float32)/2.0)
        weights /= weights.sum()
        boundary = (known[:6] * weights[:, None, None]).sum(axis=0)
        normal_derivative = (known[8:16].mean(axis=0)-known[:8].mean(axis=0))/8.0
        normal_derivative = np.clip(normal_derivative, -.06, .06)
        for distance in range(1, strip+1):
            row = start + direction*distance
            if row < 0 or row >= height:
                break
            u = distance/strip
            taper = 1 - 3*u*u + 2*u*u*u
            extrapolation = boundary.copy()
            if tangent:
                extrapolation -= min(distance, 24) * normal_derivative
            field[row] = extrapolation * taper
        details[name] = {"boundary_mean_dx": float(boundary[:, 0].mean()),
                         "boundary_mean_dy": float(boundary[:, 1].mean()),
                         "boundary_max_displacement": float(np.linalg.norm(boundary, axis=-1).max())}
    return field, details


def warp(raw: np.ndarray, field: np.ndarray, interpolation: str = "linear") -> np.ndarray:
    rows, cols = np.indices(raw.shape[:2], dtype=np.float32)
    mode={"linear":cv2.INTER_LINEAR,"cubic":cv2.INTER_CUBIC}[interpolation]
    return cv2.remap(raw.astype(np.float32), cols+field[..., 0], rows+field[..., 1],
                     mode, borderMode=cv2.BORDER_REFLECT101)


def harmonic_transfer(raw: np.ndarray, source: np.ndarray, strip: int,
                      sigma: float, residual_cap: float):
    """Add a smooth border residual only outside the known source.

    Dirichlet residual is averaged from the first four known rows, then blurred
    across x. Its cosine modes solve Laplace's equation in a finite strip with
    a zero residual at the far edge and reflecting side boundaries. Color is
    corrected additively, so generated textures remain except for clipping.
    This can create a color halo or ghost on misregistered geometry.
    """
    _, y0, _, y1 = RECT
    output = raw.copy().astype(np.float32)
    details = {}
    width = raw.shape[1]
    modes = np.pi*np.arange(width, dtype=np.float64)/width
    for name, reference, decoded, edge, direction in (
        ("top", source[:4], raw[y0:y0+4], y0, -1),
        ("bottom", source[-4:][::-1], raw[y1-4:y1][::-1], y1-1, 1),
    ):
        weights = np.array([.4,.3,.2,.1], dtype=np.float32)
        residual = ((reference.astype(np.float32)-decoded.astype(np.float32))*weights[:, None, None]).sum(axis=0)
        residual = cv2.GaussianBlur(residual[None], (0,0), sigmaX=sigma, sigmaY=0)[0]
        residual = np.clip(residual, -residual_cap, residual_cap)
        spectra = np.stack([cv2.dct(residual[:, channel:channel+1])[:,0] for channel in range(3)], axis=-1)
        for distance in range(1, strip+1):
            row = edge + direction*distance
            if row < 0 or row >= raw.shape[0]:
                break
            # Stable equivalent of sinh(k*(L-d))/sinh(k*L).
            damping = np.zeros(width, dtype=np.float64)
            damping[0] = 1-distance/strip
            k = modes[1:]
            damping[1:] = (np.exp(-k*distance)-np.exp(-k*(2*strip-distance))) / (-np.expm1(-2*k*strip))
            correction = np.stack([
                cv2.idct((spectra[:, channel]*damping.astype(np.float32))[:,None])[:,0]
                for channel in range(3)], axis=-1)
            output[row] += correction
        details[name] = {"residual_mean_rgb": residual.mean(axis=0).tolist(),
                         "residual_abs_p95_255": float(np.quantile(np.abs(residual), .95)),
                         "residual_max_255": float(np.abs(residual).max())}
    return output, details


def metrics(output: np.ndarray, original_composite: np.ndarray, source: np.ndarray,
            raw: np.ndarray, strip: int):
    _, y0, _, y1 = RECT
    outside = np.ones(output.shape[:2], bool)
    outside[y0:y1] = False
    untouched = outside.copy()
    untouched[y0-strip:y1+strip] = False
    differences = np.abs(output.astype(float)-original_composite.astype(float))
    gray = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY).astype(float)
    baseline_gray = cv2.cvtColor(original_composite, cv2.COLOR_RGB2GRAY).astype(float)
    stripe = np.concatenate([gray[y0-strip:y0], gray[y1:y1+strip]])
    baseline_stripe = np.concatenate([baseline_gray[y0-strip:y0], baseline_gray[y1:y1+strip]])
    gradient = np.gradient(stripe, axis=1)
    baseline_gradient = np.gradient(baseline_stripe, axis=1)
    return {"source_pixels_exact": bool(np.array_equal(output[y0:y1], source)),
            "untouched_outer_pixels_exact": bool(np.array_equal(output[untouched], original_composite[untouched])),
            "generated_changed_pixel_fraction": float(np.any(output!=original_composite, axis=-1)[outside].mean()),
            "generated_mean_abs_change_255": float(differences[outside].mean()),
            "top_join_mean_abs_adjacent_row_jump_255": float(np.abs(output[y0].astype(float)-output[y0-1].astype(float)).mean()),
            "bottom_join_mean_abs_adjacent_row_jump_255": float(np.abs(output[y1].astype(float)-output[y1-1].astype(float)).mean()),
            "collar_horizontal_gradient_energy_ratio": float(np.mean(gradient**2)/max(1e-6,np.mean(baseline_gradient**2))),
            "rgb_clipped_low_fraction_in_generated": float(np.mean(output[outside]==0)),
            "rgb_clipped_high_fraction_in_generated": float(np.mean(output[outside]==255)),
            "metric_limit": "Row jump is photometric only; gradient energy is not rail continuity or scene quality."}


def montage(run_dir: Path, variants: dict, raw: np.ndarray, source: np.ndarray):
    font = ImageFont.load_default(size=16)
    thumbnail_width=256
    full = Image.new("RGB", (len(variants)*thumbnail_width, 624), "#15181c")
    # The identical, broad crop applies to every variant; no rail-specific
    # alignment or coordinates enter the compositor algorithm itself.
    crop_rect = (0, 292, 512, 352)
    bottom_rect = (0, 806, 512, 858)
    closeup = Image.new("RGB", (1024+180, len(variants)*(120+30)), "#15181c")
    bottom = Image.new("RGB", (1024+180, len(variants)*(104+30)), "#15181c")
    draw_full, draw_close, draw_bottom = ImageDraw.Draw(full), ImageDraw.Draw(closeup), ImageDraw.Draw(bottom)
    for index, (name, array) in enumerate(variants.items()):
        image = Image.fromarray(array)
        draw_full.text((index*thumbnail_width+8,8), name, fill="white", font=font)
        full.paste(image.resize((256,576), Image.Resampling.LANCZOS), (index*thumbnail_width, 36))
        draw_close.text((8,index*150+50), name, fill="white", font=font)
        closeup.paste(image.crop(crop_rect).resize((1024,120),Image.Resampling.NEAREST), (180,index*150+24))
        draw_bottom.text((8,index*134+40), name, fill="white", font=font)
        bottom.paste(image.crop(bottom_rect).resize((1024,104),Image.Resampling.NEAREST), (180,index*134+24))
        image.crop(crop_rect).save(run_dir/name/"top-boundary.png")
        image.crop(bottom_rect).save(run_dir/name/"bottom-boundary.png")
    full.save(run_dir/"final-comparison.png")
    closeup.save(run_dir/"top-boundary-comparison-2x.png")
    bottom.save(run_dir/"bottom-boundary-comparison-2x.png")


def preflight():
    # Verify inverse-warp sign using a known synthetic translation, independent
    # of the approximate optical-flow estimator's accuracy.
    rng=np.random.default_rng(314)
    synthetic=rng.uniform(10,240,(60,80,3)).astype(np.float32)
    rows,cols=np.indices(synthetic.shape[:2],dtype=np.float32)
    shifted=cv2.remap(synthetic,cols-3,rows+2,cv2.INTER_NEAREST,borderMode=cv2.BORDER_REFLECT101)
    field=np.zeros((60,80,2),np.float32)
    field[...,0]=3
    field[...,1]=-2
    restored=warp(shifted,field)
    if not np.array_equal(restored[4:-4,4:-4],synthetic[4:-4,4:-4]):
        raise AssertionError("Output->raw remap translation sign is incorrect")
    # A uniform residual has the analytic linear solution in the finite strip.
    reference=np.full((512,512,3),100,dtype=np.uint8)
    test_raw=np.full((1152,512,3),90,dtype=np.float32)
    corrected,_=harmonic_transfer(test_raw,reference,96,6,24)
    expected=90+10*(1-1/96)
    if not np.allclose(corrected[319],expected,atol=1e-3):
        raise AssertionError("Harmonic constant mode is not the expected linear solution")
    output=compose(corrected,reference)
    if not np.array_equal(output[320:832],reference):
        raise AssertionError("Preflight changed known source pixels")
    if not np.array_equal(output[:224],test_raw[:224].astype(np.uint8)):
        raise AssertionError("Color correction escaped its finite strip")
    return {"inverse_warp_direction_passed": True,"harmonic_constant_mode_passed": True,
            "original_source_exact_passed": True,"far_exterior_exact_passed": True,
            "cpu_only": True,"no_model_weights_loaded": True,"opencv_version": cv2.__version__}


def run(args):
    cv2.setNumThreads(1)
    start=time.perf_counter()
    shared=json.loads((ROOT/"experiments/evaluation/inputs/inputs.json").read_text())
    track=next(row for row in shared["tracks"] if row["id"]==args.track)
    raw_path=args.raw or ROOT/f"experiments/qwen/geometry_runs/2026-10-05/{args.track}/edge-context/raw.png"
    source_path=Path(track["source_512"])
    raw=np.asarray(Image.open(raw_path).convert("RGB"),dtype=np.float32)
    source=np.asarray(Image.open(source_path).convert("RGB"),dtype=np.uint8)
    if raw.shape!=(1152,512,3) or source.shape!=(512,512,3):
        raise ValueError("Expected shared 512x1152 raw decode and original 512 square")
    out=args.output or DEFAULT_RUN/args.track
    if (out/"compositor_metrics.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed experiment {out}")
    out.mkdir(parents=True,exist_ok=True)
    before={str(path):digest(path) for path in (raw_path,source_path,Path(track["original_jpg"]))}
    suite={"status":"running","kind":"CPU post-generation compositor diagnostic",
           "track":args.track,"input_raw":str(raw_path),"input_source":str(source_path),
           "source_rect_xyxy":list(RECT),"strip_pixels":args.strip,
           "input_sha256_before":before,"preflight":preflight(),"variants":{}}
    variants={"baseline":compose(raw,source)}
    configurations=[("color",None,False,True),
                    ("dis-flow","dis",False,False),
                    ("dis-color","dis",False,True),
                    ("dis-tangent-color","dis",True,True),
                    ("farneback-color","farneback",False,True)]
    flows={}
    for name,method,tangent,color in configurations:
        elapsed=time.perf_counter()
        work=raw.copy()
        record={"optical_flow":method,"flow_tangent_extrapolation":tangent,
                "harmonic_color":color,"strip_pixels":args.strip,
                "warp_interpolation":args.warp_interpolation,
                "color_sigma_x":args.color_sigma,"residual_cap_255":args.residual_cap}
        if method:
            if method not in flows:
                flows[method]=displacement(raw,source,method,args.flow_cap)
            flow,stats=flows[method]
            field,details=extend_flow(flow,raw.shape,args.strip,tangent)
            work=warp(work,field,args.warp_interpolation)
            record.update({"registration":stats,"exterior_flow":details})
        if color:
            work,details=harmonic_transfer(work,source,args.strip,args.color_sigma,args.residual_cap)
            record["color_residual"]=details
        variants[name]=compose(work,source)
        record["processing_seconds"]=time.perf_counter()-elapsed
        suite["variants"][name]=record
    for name,array in variants.items():
        directory=out/name
        directory.mkdir(exist_ok=True)
        path=directory/"composite.png"
        save(path,array)
        record=suite["variants"].setdefault(name,{"processing_seconds":0.0})
        record.update(metrics(array,variants["baseline"],source,raw,args.strip))
        record.update({"composite_path":str(path),"composite_sha256":digest(path)})
        if not record["source_pixels_exact"] or not record["untouched_outer_pixels_exact"]:
            raise AssertionError(f"Source/far exterior preservation failed: {name}")
        (directory/"metrics.json").write_text(json.dumps(record,indent=2)+"\n")
    montage(out,variants,raw,source)
    after={path:digest(Path(path)) for path in before}
    suite.update({"input_sha256_after":after,"inputs_unchanged":before==after,
                  "status":"success","elapsed_seconds":time.perf_counter()-start,
                  "runner_sha256":digest(Path(__file__)),
                  "all_source_pixels_exact":all(row["source_pixels_exact"] for row in suite["variants"].values()),
                  "all_far_exterior_exact":all(row["untouched_outer_pixels_exact"] for row in suite["variants"].values()),
                  "quality_verdict":"pending direct image inspection; metrics do not establish geometry quality"})
    (out/"compositor_metrics.json").write_text(json.dumps(suite,indent=2)+"\n")
    print(json.dumps({"status":suite["status"],"output":str(out),"elapsed_seconds":suite["elapsed_seconds"],
                      "source_exact":suite["all_source_pixels_exact"],"inputs_unchanged":suite["inputs_unchanged"]}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track",choices=("track1","track2","track3"),default="track3")
    parser.add_argument("--raw",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--strip",type=int,default=96)
    parser.add_argument("--flow-cap",type=float,default=6)
    parser.add_argument("--color-sigma",type=float,default=6)
    parser.add_argument("--residual-cap",type=float,default=24)
    parser.add_argument("--warp-interpolation",choices=("linear","cubic"),default="linear")
    parser.add_argument("--preflight",action="store_true")
    args=parser.parse_args()
    if args.preflight:
        cv2.setNumThreads(1)
        print(json.dumps(preflight()))
    else:
        if not 1<=args.strip<=320:
            raise ValueError("Strip must stay inside the 320-pixel generated exterior")
        run(args)
