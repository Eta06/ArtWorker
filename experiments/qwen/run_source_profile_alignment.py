#!/usr/bin/env python3
"""CPU-only, source-derived multi-edge collar resampling diagnostic.

This isolated prototype changes sampling coordinates, never paints, copies,
blurs, recolors, or adds grain to generated pixels. It cannot remove invented
rail topology. Source pixels and the distant exterior retain exact uint8 bytes.
Analytical Gaussian filtering is used only to measure edges, not on the output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from evaluate_boundary import (RECT, SOURCE, boundary_stats, font, rail_points,
                               rgb, robust_line)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "experiments/qwen/controlnet_runs/2026-10-05/track3-localized40-outpaint/tangent-canny/localized-hints"
DEFAULT_OUTPUT = ROOT / "experiments/qwen/profile_alignment_runs/2026-10-05/track3-localized40-prototype"
EDGE_NAMES = ("outer_positive", "highlight_negative", "inner_positive", "closing_negative")
POLARITIES = (1, -1, 1, -1)
ANCHORS = (0, 2, 3)
HELD_OUT = 1
LUMINANCE = np.array([.2126, .7152, .0722])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def profiles(array):
    # Keep the measurement convention of the frozen evaluator. This array is
    # used exclusively for diagnostics; resampling uses original uint8 pixels.
    from PIL import ImageFilter
    measured = np.asarray(Image.fromarray(array).filter(ImageFilter.GaussianBlur(.55)), float)
    luminance = measured @ LUMINANCE
    gradient = np.zeros_like(luminance)
    gradient[:, 1:-1] = (luminance[:, 2:] - luminance[:, :-2]) / 2
    return luminance, gradient


def peaks(gradient, y, polarity, lo, hi, threshold=2.5):
    values = polarity * gradient[y]
    result = []
    for x in range(max(2, int(np.ceil(lo))), min(len(values)-2, int(np.floor(hi)))+1):
        if values[x] < threshold or values[x] < values[x-1] or values[x] <= values[x+1]:
            continue
        before, center, after = values[x-1:x+2]
        denominator = before - 2*center + after
        offset = 0.0 if abs(denominator) < 1e-8 else float(np.clip(.5*(before-after)/denominator, -.5, .5))
        result.append({"y": int(y), "x": float(x+offset), "strength": float(values[x]),
                       "polarity": int(polarity), "valid": True})
    return result


def fit(points, edge):
    value = robust_line(points, 320, 341)
    value["edge"] = edge
    value["polarity"] = POLARITIES[EDGE_NAMES.index(edge)]
    if not value["valid"]:
        raise ValueError(f"Insufficient original-source evidence for {edge}")
    return value


def x_at(line, y):
    return line["x_extrapolated_at_y320"] + line["dx_dy"] * (y-320)


def source_features(source):
    canvas = np.zeros((1152, 512, 3), np.uint8)
    canvas[320:832] = source
    source_outer = rail_points(canvas, source=True)
    outer_fit = robust_line(source_outer, 320, 341)
    # All four source identities come from the original's first profile. The
    # strong positive edge after this signed sequence belongs to the post and
    # is deliberately not used as a lower-rail correspondence.
    _, gradient = profiles(source)
    outer = max(peaks(gradient, 0, 1, x_at(outer_fit, 320)-3, x_at(outer_fit, 320)+3), key=lambda p: p["strength"])
    initial = [outer]
    for polarity in POLARITIES[1:]:
        choices = peaks(gradient, 0, polarity, initial[-1]["x"]+1.5, outer["x"]+16, 3)
        if not choices:
            raise ValueError("Original source does not contain the required ordered rail profile")
        initial.append(choices[0])
    points = [[] for _ in EDGE_NAMES]
    for y in range(21):
        for index, initial_point in enumerate(initial):
            nominal = initial_point["x"] + outer_fit["dx_dy"]*y
            choices = peaks(gradient, y, POLARITIES[index], nominal-2.5, nominal+2.5, 3)
            if choices:
                point = max(choices, key=lambda p: p["strength"])
                point = dict(point, y=y+320)
                points[index].append(point)
    fits = [fit(p, n) for p, n in zip(points, EDGE_NAMES)]
    return fits, points


def generated_features(array, source_fits):
    _, gradient = profiles(array)
    outer_map = {p["y"]: p for p in rail_points(array, source=False)}
    source_gaps = [x_at(source_fits[i+1], 320)-x_at(source_fits[i], 320) for i in range(3)]
    span = x_at(source_fits[2], 320)-x_at(source_fits[0], 320)
    rows = []
    previous_outer=None
    for y in range(304, 320):
        original_outer = outer_map[y]
        features = [None]*4
        record = {"y": y, "features": features, "ambiguities": [], "extra_ridges": []}
        if not original_outer["valid"]:
            # Preserve the continuing outer identity through a failed color
            # gate. The frozen evaluator itself remains invalid at this row.
            predicted=previous_outer["x"]+source_fits[0]["dx_dy"] if previous_outer else None
            choices=[] if predicted is None else peaks(gradient,y,1,predicted-2.5,predicted+2.5,7)
            if not choices:
                record["invalid_reason"] = "frozen outer metal-edge extraction is invalid and no narrow continuation exists"
                rows.append(record)
                continue
            outer=max(choices,key=lambda p:p["strength"])
            record["ambiguities"].append({"edge":EDGE_NAMES[0],"reason":"frozen color gate failed; narrow signed-gradient continuation used", "selected":outer})
        else:
            outer = dict(original_outer, strength=original_outer["gradient_strength"], polarity=1)
        previous_outer=outer
        features[0] = outer
        # The first falling highlight stays immediately after the fixed outer
        # identity. Its position is measured but held out from the fitted warp.
        negatives = peaks(gradient, y, -1, outer["x"]+1.5, outer["x"]+4*source_gaps[0])
        if negatives:
            features[1] = max(negatives, key=lambda p: np.log1p(p["strength"])-.035*(p["x"]-outer["x"]-source_gaps[0])**2)
            if features[1]["x"]-outer["x"]>2*source_gaps[0]:
                record["ambiguities"].append({"edge":EDGE_NAMES[1],"reason":"first highlight fall is no longer narrow; identity lost", "selected":features[1]})
                features[1]["identity_valid"]=False
        # The inner positive stripe may be much weaker, wider, or missing. Do
        # not substitute the post: its search span is bounded by source width.
        start = features[1]["x"]+1.5 if features[1] else outer["x"]+3
        positives = peaks(gradient, y, 1, start, outer["x"]+4*span)
        if positives:
            ranked = sorted(positives, key=lambda p: p["strength"], reverse=True)
            features[2] = ranked[0]
            if len(ranked)>1 and ranked[1]["strength"] >= .65*ranked[0]["strength"] and abs(ranked[1]["x"]-ranked[0]["x"])>=3:
                record["ambiguities"].append({"edge": EDGE_NAMES[2], "selected": ranked[0], "alternative": ranked[1]})
            # A closing edge must precede the next positive metal ridge. A
            # negative after that ridge would relabel a different rail/post.
            next_positive = peaks(gradient, y, 1, features[2]["x"]+2, features[2]["x"]+4*source_gaps[2])
            strong_next = [p for p in next_positive if p["strength"]>=max(5, .75*features[2]["strength"])]
            end = min(p["x"] for p in strong_next)-1 if strong_next else features[2]["x"]+4*source_gaps[2]
            closings = peaks(gradient, y, -1, features[2]["x"]+1.5, end)
            if closings:
                features[3] = max(closings, key=lambda p: p["strength"])
            record["extra_ridges"] = strong_next
        rows.append(record)
    return rows


def measured_correspondences(array, original_rows, field):
    """Measure fixed original identities in the actual resampled pixel array.

    The inverse monotone map supplies only a search prior. Actual signed peaks
    are detected again in the pixels; predicted anchor positions are never
    reported as observed success. Missing/ambiguous original identities persist.
    """
    _,gradient=profiles(array)
    xs=np.arange(array.shape[1],dtype=float)
    result=[]
    for original in original_rows:
        y=original["y"]
        row={"y":y,"features":[None]*4,"ambiguities":original["ambiguities"],
             "extra_ridges":original["extra_ridges"],"measurement_prior":"inverse map of fixed original feature identity"}
        for index,p in enumerate(original["features"]):
            if p is None:continue
            expected=float(np.interp(p["x"],xs+field[y],xs))
            candidates=peaks(gradient,y,POLARITIES[index],expected-2,expected+2)
            if candidates:
                selected=min(candidates,key=lambda q:abs(q["x"]-expected))
                selected["predicted_x"]=expected
                selected["original_identity_valid"]=p.get("identity_valid",True)
                row["features"][index]=selected
        result.append(row)
    return result


def boundary_offsets(rows, source_fits):
    offsets = []
    evidence = []
    for index in ANCHORS:
        values = [(r["y"], r["features"][index]) for r in rows if r["y"]>=316 and r["features"][index] is not None]
        if len(values)<3:
            raise ValueError(f"Fewer than three near-join anchor observations for {EDGE_NAMES[index]}")
        differences = np.array([p["x"]-x_at(source_fits[index], y) for y, p in values])
        delta = float(np.median(differences))
        if abs(delta)>24:
            raise ValueError("Required displacement exceeds the preregistered 24px collar cap")
        offsets.append(delta)
        evidence.append({"edge": EDGE_NAMES[index], "polarity": POLARITIES[index], "observed_rows": [y for y,p in values],
                         "offsets": differences.tolist(), "median_offset": delta,
                         "minimum_gradient_strength": float(min(p["strength"] for y,p in values)),
                         "offset_mad": float(np.median(np.abs(differences-delta))),
                         "confidence": "weak or ambiguous" if any(p["strength"]<7 for y,p in values) else "strong gradient only; identity still requires visual review"})
    return offsets, evidence


def sampling_field(shape, source_fits, offsets, collar, strength=1):
    field = np.zeros(shape[:2], np.float64)
    xs = np.arange(shape[1], dtype=float)
    for y in range(320-collar, 320):
        distance = 320-y
        t = distance/collar
        taper = 1-3*t*t+2*t*t*t
        anchors = np.array([x_at(source_fits[i], y) for i in ANCHORS])
        knots = np.r_[anchors[0]-40, anchors, anchors[-1]+40]
        shifts = np.r_[0, np.array(offsets)*strength*taper, 0]
        mapped = knots+shifts
        jacobians = np.diff(mapped)/np.diff(knots)
        if np.any(jacobians<.25) or np.any(jacobians>4):
            raise ValueError("Monotonic bounded-strain correspondence fails; no output will be accepted")
        field[y] = np.interp(xs, knots, shifts, left=0, right=0)
    return field


def resample(composite, field, interpolation="linear"):
    output = composite.copy()
    xs = np.arange(composite.shape[1], dtype=float)
    for y in np.flatnonzero(np.any(field!=0, axis=1)):
        coords = xs+field[y]
        if coords.min()<0 or coords.max()>composite.shape[1]-1:
            raise ValueError("Resampling would require an exterior padding policy")
        if interpolation=="nearest":
            sampled=composite[y,np.rint(coords).astype(int)]
        elif interpolation=="linear":
            lower = np.floor(coords).astype(int)
            upper = np.minimum(lower+1, composite.shape[1]-1)
            fraction = coords-lower
            sampled = composite[y,lower].astype(float)*(1-fraction[:,None])+composite[y,upper].astype(float)*fraction[:,None]
        else:raise ValueError(interpolation)
        active = field[y]!=0
        output[y,active] = np.clip(np.rint(sampled[active]),0,255).astype(np.uint8)
    return output


def feature_metrics(rows, source_fits):
    result = {}
    for i,name in enumerate(EDGE_NAMES):
        observations = [(r["y"],r["features"][i]) for r in rows if r["y"]>=316 and r["features"][i] is not None]
        differences = [p["x"]-x_at(source_fits[i],y) for y,p in observations]
        valid_differences=[d for (y,p),d in zip(observations,differences) if p.get("original_identity_valid",p.get("identity_valid",True))]
        result[name] = {"role": "held out" if i==HELD_OUT else "fitted anchor",
                        "observations": len(differences), "median_offset_px": float(np.median(differences)) if differences else None,
                        "p90_absolute_offset_px": float(np.percentile(np.abs(differences),90)) if differences else None,
                        "minimum_strength": float(min(p["strength"] for y,p in observations)) if observations else None,
                        "identity_invalid_rows":[y for y,p in observations if not p.get("original_identity_valid",p.get("identity_valid",True))],
                        "valid_identity_observations":len(valid_differences),
                        "valid_identity_median_offset_px":float(np.median(valid_differences)) if valid_differences else None,
                        "identity_status":"insufficient near rows" if len(valid_differences)<3 else "observed identity; no quality acceptance",
                        "points": [{"y":y, "x":p["x"], "offset_px":d} for (y,p),d in zip(observations,differences)]}
    widths = []
    for r in rows:
        p=r["features"]
        if r["y"]>=316 and p[0] is not None and p[3] is not None:
            widths.append({"y":r["y"], "actual_px":p[3]["x"]-p[0]["x"],
                           "source_tangent_px":x_at(source_fits[3],r["y"])-x_at(source_fits[0],r["y"])})
    return {"edges":result, "upper_band_widths":widths,
            "ambiguous_rows": [r["y"] for r in rows if r["ambiguities"]],
            "missing_closing_rows": [r["y"] for r in rows if r["features"][3] is None],
            "extra_positive_ridges": {str(r["y"]):r["extra_ridges"] for r in rows if r["extra_ridges"]}}


def diagnostics(output, source, baseline, field, collar):
    active=field!=0
    jacobian=1+np.diff(field,axis=1)
    outside_far=np.zeros(field.shape,bool);outside_far[:320-collar]=True;outside_far[832:]=True
    return {"source_pixels_exact":bool(np.array_equal(output[320:832],source)),
            "far_exterior_pixels_exact":bool(np.array_equal(output[outside_far],baseline[outside_far])),
            "all_pixels_outside_sampling_support_exact":bool(np.array_equal(output[~active],baseline[~active])),
            "bottom_exterior_exact":bool(np.array_equal(output[832:],baseline[832:])),
            "sampling_displacement_max_px":float(np.abs(field).max()),
            "horizontal_sampling_jacobian_min":float(jacobian.min()),
            "horizontal_sampling_jacobian_max":float(jacobian.max()),
            "changed_pixels":int(np.any(output!=baseline,axis=-1).sum()),
            "top_join_rgb_mae":float(np.abs(output[319].astype(float)-output[320]).mean()),
            "collar_far_row_rgb_jump_mae":float(np.abs(output[320-collar].astype(float)-output[319-collar]).mean()),
            "original_far_row_rgb_jump_mae":float(np.abs(baseline[320-collar].astype(float)-baseline[319-collar]).mean()),
            "pixels":boundary_stats(output,source)}


def sampling_self_tests():
    rows,cols=np.indices((20,64))
    synthetic=np.stack((cols,cols+40,cols+80),axis=-1).astype(np.uint8)
    zero=np.zeros((20,64),float)
    assert np.array_equal(resample(synthetic,zero),synthetic)
    field=zero.copy();field[8:12,10:40]=3
    for mode in ("linear","nearest"):
        result=resample(synthetic,field,mode)
        assert np.array_equal(result[8:12,10:40],synthetic[8:12,13:43])
        assert np.array_equal(result[field==0],synthetic[field==0])
    fractional=zero.copy();fractional[8:12,10:40]=.7
    nearest=resample(synthetic,fractional,"nearest")
    assert np.array_equal(nearest[8:12,10:40],synthetic[8:12,11:41])
    return {"identity_uint8_exact":True,"output_to_input_translation_sign_verified":True,
            "nearest_returns_exact_original_rgb_tuples":True,"zero_field_support_exact":True}


def montage(output_dir, images, records, source_points):
    crop=(270,296,380,342)
    w,h=crop[2]-crop[0],crop[3]-crop[1]
    for scale in (1,4):
        canvas=Image.new("RGB",(len(images)*(w*scale+24),h*scale+62),"#11191d")
        draw=ImageDraw.Draw(canvas)
        for i,(name,array) in enumerate(images.items()):
            x=i*(w*scale+24)+12
            draw.text((x,10),name,font=font(14),fill="white")
            im=Image.fromarray(array).crop(crop).resize((w*scale,h*scale),Image.Resampling.NEAREST)
            canvas.paste(im,(x,40))
            sy=40+(320-crop[1])*scale
            draw.line((x-5,sy,x-2,sy),fill="#edc267")
        canvas.save(output_dir/f"join-untouched-{scale}x.png")
    full=Image.new("RGB",(len(images)*256,612),"#11191d");d=ImageDraw.Draw(full)
    for i,(name,array) in enumerate(images.items()):
        d.text((i*256+8,8),name,font=font(14),fill="white")
        full.paste(Image.fromarray(array).resize((256,576),Image.Resampling.LANCZOS),(i*256,32))
    full.save(output_dir/"full-comparison.png")
    annotated=Image.new("RGB",(len(images)*(w*4+24),h*4+62),"#11191d");d=ImageDraw.Draw(annotated)
    colors=("#ffcb5e","#ee90fc","#54ddff","#ff716e")
    for i,(name,array) in enumerate(images.items()):
        im=Image.fromarray(array).crop(crop);paint=ImageDraw.Draw(im)
        for row in records[name]:
            for index,p in enumerate(row["features"]):
                if p and crop[0]<=p["x"]<crop[2]:paint.point((round(p["x"])-crop[0],p["y"]-crop[1]),fill=colors[index])
        for points in source_points:
            for p in points:
                if crop[1]<=p["y"]<crop[3]:paint.point((round(p["x"])-crop[0],p["y"]-crop[1]),fill="#55ffae")
        x=i*(w*4+24)+12;d.text((x,10),name,font=font(14),fill="white")
        annotated.paste(im.resize((w*4,h*4),Image.Resampling.NEAREST),(x,40))
    annotated.save(output_dir/"join-feature-diagnostic-4x.png")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=DEFAULT_INPUT)
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    if args.output.exists():raise ValueError("Use a fresh output directory; existing artifacts remain frozen")
    start=time.monotonic()
    source=rgb(SOURCE);raw=rgb(args.input/"raw.png");baseline=rgb(args.input/"composite.png")
    if raw.shape!=(1152,512,3) or baseline.shape!=raw.shape or source.shape!=(512,512,3):raise ValueError("Unexpected image geometry")
    if not np.array_equal(baseline[320:832],source):raise ValueError("Input composite does not preserve the original source")
    exterior=np.ones((1152,512),bool);exterior[320:832]=False
    if not np.array_equal(raw[exterior],baseline[exterior]):raise ValueError("Input raw and composite exteriors differ")
    inputs={str(p.resolve()):digest(p) for p in (SOURCE,args.input/"raw.png",args.input/"composite.png")}
    frozen={str(p.resolve()):digest(p) for p in (ROOT/"experiments/qwen/run_boundary_compositor.py",ROOT/"experiments/qwen/evaluate_boundary.py",ROOT/"experiments/qwen/evaluate_source_tangent.py")}
    source_fits,source_points=source_features(source)
    # Use raw for correspondence measurement to avoid hard-pasted source pixels
    # affecting the analytical vertical blur at the last generated row.
    raw_rows=generated_features(raw,source_fits)
    offsets,evidence=boundary_offsets(raw_rows,source_fits)
    identity_field=np.zeros(baseline.shape[:2],float)
    images={"original":baseline};records={"original":measured_correspondences(baseline,raw_rows,identity_field)}
    report={"status":"CPU compositor diagnostic; not quality accepted", "device":"CPU; NumPy and Pillow only",
            "sampling_self_tests":sampling_self_tests(),
            "source_rect_xyxy":RECT,"input_hashes":inputs,"frozen_script_hashes":frozen,
            "method":"Three ordered rail-edge anchor shifts from near exterior rows; monotonic horizontal linear resampling with smoothstep vertical taper",
            "fitted_edges":[EDGE_NAMES[i] for i in ANCHORS],"held_out_edge":EDGE_NAMES[HELD_OUT],
            "source_only_edge_fits":source_fits,"raw_correspondence_rows":raw_rows,"anchor_evidence":evidence,
            "limits":["Only track3, upper exterior collar; no new generation or model improvement.",
                      "Weak/missing closing edges and competing inner ridges reduce correspondence confidence.",
                      "A monotonic warp preserves ordering but cannot repair invented topology or recover absent texture.",
                      "Linear resampling can compress grain and introduce aliasing; no filtering, color correction, or grain synthesis is applied.",
                      "Good anchor locations alone do not imply scene correctness; inspect held-out edge, widths, untouched crops, and frozen evaluators."],
            "variants":{"original":{"edge_diagnostic":feature_metrics(records["original"],source_fits),
                                      "pixels":boundary_stats(baseline,source)}}}
    args.output.mkdir(parents=True)
    for name,collar,strength,mode in (("profile64",64,1,"linear"), ("profile96",96,1,"linear"), ("partial64",64,.75,"linear"), ("nearest64",64,1,"nearest")):
        field=sampling_field(baseline.shape,source_fits,offsets,collar,strength)
        output=resample(baseline,field,mode)
        checks=diagnostics(output,source,baseline,field,collar)
        for key in ("source_pixels_exact","far_exterior_pixels_exact","all_pixels_outside_sampling_support_exact","bottom_exterior_exact"):
            if not checks[key]:raise AssertionError(key)
        folder=args.output/name;folder.mkdir()
        Image.fromarray(output).save(folder/"composite.png")
        np.save(folder/"horizontal_sampling_displacement.npy",field.astype(np.float32))
        images[name]=output;records[name]=measured_correspondences(output,raw_rows,field)
        report["variants"][name]={"collar_pixels":collar,"displacement_strength":strength,"checks":checks,
                                 "interpolation":mode,"no_new_rgb_sampling":mode=="nearest",
                                 "edge_diagnostic":feature_metrics(records[name],source_fits),
                                 "output_sha256":digest(folder/"composite.png")}
    montage(args.output,images,records,source_points)
    report["input_and_frozen_files_unchanged"]=all(digest(Path(p))==value for p,value in {**inputs,**frozen}.items())
    report["seconds"]=time.monotonic()-start
    (args.output/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    lines=["# Source profile alignment diagnostic", "", "CPU resampling of the saved localized40 exterior. Source and distant pixels remain exact. This is not a new model run or accepted repair.", "",
           "Three fitted anchors: outer positive, inner positive, closing negative. The first negative highlight is held out. The strong positive post at x328 is excluded from source identities.", "",
           "| variant | outer dx | held-out highlight dx | inner dx | closing dx | source exact | distant exterior exact |", "|---|---:|---:|---:|---:|---|---|"]
    for name,r in report["variants"].items():
        edges=r["edge_diagnostic"]["edges"]
        vals=[edges[e]["valid_identity_median_offset_px"] for e in EDGE_NAMES]
        textvals=["unknown" if v is None else f"{v:.2f}" for v in vals]
        checks=r.get("checks",{})
        lines.append(f"| {name} | {' | '.join(textvals)} | {r.get('pixels',{}).get('source_pixels_exact',checks.get('source_pixels_exact'))} | {checks.get('far_exterior_pixels_exact','original')} |")
    lines += ["", "The held-out highlight has only one near-join observation with intact identity; two widened falls are invalid and its last-row identity is absent. Its numeric position therefore cannot establish a successful four-edge repair. Missing/ambiguous edges remain flagged in JSON.", "",
              "Nearest64 selects exact original RGB tuples; it adds no interpolated RGB values. It can duplicate/remove grain samples and make stair steps. Linear variants interpolate RGB and may soften/alias grain. All variants geometrically stretch/compress texture, bounded by recorded sampling Jacobians.", "",
              "An extra ridge or post can fool profile matching; anchor residuals are not an acceptance score.", "",
              "![Native untouched joins](join-untouched-1x.png)", "", "![Enlarged untouched joins](join-untouched-4x.png)", "", "![Separate feature diagnostic](join-feature-diagnostic-4x.png)", "", "![Full comparison](full-comparison.png)", ""]
    (args.output/"REPORT.md").write_text("\n".join(lines))
    print(json.dumps({"output":str(args.output),"seconds":report["seconds"],"anchor_evidence":evidence,
                      "variants":{n:{"edges":r["edge_diagnostic"]["edges"],"checks":r.get("checks")} for n,r in report["variants"].items()}},indent=2))


if __name__=="__main__":main()
