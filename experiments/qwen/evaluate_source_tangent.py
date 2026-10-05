#!/usr/bin/env python3
"""Separate source-derived track3 rail diagnostic; the older evaluator is unchanged.

Search coordinates and thresholds come from the original source, never a guide
map or candidate output. A successful feature extraction is not a quality pass.
Endpoints, tangent, competing paths and actual pixels must be reviewed together.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from evaluate_boundary import ROOT, SOURCE, boundary_stats, font, rail_points, rgb, robust_line

Y_RANGE = (296, 320)
NEAR_RANGE = (304, 320)
HALF_WIDTH = 20


def candidate_rows(a, source_fit, minimum_gradient):
    blur = np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(.55)), dtype=float)
    lum = blur @ np.array([.2126, .7152, .0722])
    gx = np.zeros_like(lum)
    gx[:, 1:-1] = (lum[:, 2:] - lum[:, :-2]) / 2
    rows = []
    for y in range(*Y_RANGE):
        nominal = source_fit["x_extrapolated_at_y320"] + source_fit["dx_dy"] * (y - 320)
        lo, hi = max(2, int(np.floor(nominal-HALF_WIDTH))), min(509, int(np.ceil(nominal+HALF_WIDTH)))
        candidates = []
        for x in range(lo, hi+1):
            # Color gate permits blue/neutral metal; it rejects brown/green
            # reeds but may reject a valid newly colored rail. That yields
            # unknown extraction, never a declaration of correct geometry.
            ahead = blur[y, x+2]
            metal = ahead[2] >= ahead[0]-5 and ahead[2] >= ahead[1]-4 and lum[y, x+2] >= 65
            peak = gx[y, x] >= gx[y, x-1] and gx[y, x] >= gx[y, x+1]
            if metal and peak and gx[y, x] >= minimum_gradient:
                prev, curr, nxt = gx[y, x-1:x+2]
                denom = prev-2*curr+nxt
                offset = 0.0 if abs(denom)<1e-8 else float(np.clip(.5*(prev-nxt)/denom,-.5,.5))
                candidates.append({"y":y,"x":x+offset,"gradient_strength":float(gx[y,x]),
                                   "valid":True,"corridor_edge_hit":x-lo<=1 or hi-x<=1})
        rows.append({"y":y,"nominal_x":float(nominal),"corridor":[lo,hi],"candidates":candidates})
    return rows


def best_path(rows, source_fit, forbid=None):
    """Short Viterbi track with explicit gaps and source-only correspondence prior.

    Strength and continuity identify a ridge; a source endpoint prior separates
    the upper metal ridge from the parallel lower ridge/post. This prior can
    prefer a false nearby edge, which is why candidates and competitors are
    saved and extraction validity never means visual success.
    """
    nodes = []
    previous = None
    for i,row in enumerate(rows):
        choices = [dict(c) for c in row["candidates"] if forbid is None or abs(c["x"]-forbid[i])>=5]
        # A missing-edge state stays in the source-derived corridor. It is
        # excluded from all fits and reduces extraction coverage.
        choices.append({"y":row["y"],"x":row["nominal_x"],"gradient_strength":0.0,"valid":False,"corridor_edge_hit":False})
        layer = []
        for c in choices:
            reward = np.log1p(c["gradient_strength"]) if c["valid"] else -4.0
            if previous is None:
                score = reward-.025*abs(c["x"]-row["nominal_x"])
                parent = None
            else:
                options = []
                for j,p in enumerate(previous):
                    dx = c["x"]-p["point"]["x"]
                    transition = .10*(dx-source_fit["dx_dy"])**2
                    if abs(dx)>6 and c["valid"] and p["point"]["valid"]:
                        transition += 8+abs(dx)
                    options.append((p["score"]-transition,j))
                best,parent = max(options)
                score = reward+best
            layer.append({"point":c,"score":float(score),"parent":parent})
        nodes.append(layer)
        previous = layer
    last = len(nodes)-1
    nominal = rows[-1]["nominal_x"]
    ranked = [(n["score"]-.8*abs(n["point"]["x"]-nominal),j) for j,n in enumerate(nodes[-1])]
    score,index = max(ranked)
    path=[]
    for i in range(last,-1,-1):
        n=nodes[i][index]
        path.append(n["point"])
        index=n["parent"]
    path.reverse()
    return {"score":float(score),"points":path,"coverage":sum(p["valid"] for p in path)/len(path)}


def evaluate(a, source_fit, source_points, minimum_gradient):
    rows = candidate_rows(a,source_fit,minimum_gradient)
    primary=best_path(rows,source_fit)
    alternative=best_path(rows,source_fit,[p["x"] for p in primary["points"]])
    points=primary["points"]
    near=[p for p in points if NEAR_RANGE[0]<=p["y"]<NEAR_RANGE[1]]
    fit=robust_line(points,*NEAR_RANGE)
    valid=[p for p in near if p["valid"]]
    edge_hits=[p["y"] for p in near if p["valid"] and p["corridor_edge_hit"]]
    gap=primary["score"]-alternative["score"]
    competing=alternative["coverage"]>=.8 and gap<.15*len(rows)
    reasons=[]
    if len(valid)/len(near)<.8: reasons.append("fewer than80 percent of near rows have a strong metal edge")
    if not fit["valid"]: reasons.append("insufficient near rows for tangent fit")
    elif fit["p90_absolute_fit_residual_px"]>2.5: reasons.append("near ridge is discontinuous or strongly curved/ambiguous in the local fit")
    if len(edge_hits)>max(2,len(near)*.2): reasons.append("near ridge repeatedly hits search corridor edge")
    if competing: reasons.append("a separated ridge/path has comparable source-correspondence likelihood")
    last=[p for p in valid if 316<=p["y"]<320]
    if len(last)<3: reasons.append("fewer than three strong edge rows immediately before source")
    metrics={"near_angle_difference_degrees":None,"near_endpoint_signed_dx_px":None,
             "last4_source_tangent_median_signed_dx_px":None}
    if not reasons:
        metrics={"near_angle_difference_degrees":fit["angle_from_downward_y_degrees"]-source_fit["angle_from_downward_y_degrees"],
                 "near_endpoint_signed_dx_px":fit["x_extrapolated_at_y320"]-source_fit["x_extrapolated_at_y320"],
                 "last4_source_tangent_median_signed_dx_px":float(np.median([p["x"]-(source_fit["x_extrapolated_at_y320"]+source_fit["dx_dy"]*(p["y"]-320)) for p in last]))}
    return {"status":"feature-extracted" if not reasons else "invalid-feature",
            "quality_pass":False,"invalid_reasons":reasons,"metrics":metrics,
            "near_fit_unpromoted":fit,"source_fit":source_fit,"near_edge_hit_rows":edge_hits,
            "near_strong_edge_rows":len(valid),"near_total_rows":len(near),
            "alternative_path_comparable":competing,"primary_minus_alternative_score":float(gap),
            "primary_path":primary,"alternative_path":alternative,"rows":rows,
            "source_points":source_points}


def crops(items,records,out,source_fit,source_points):
    rect=(256,286,432,350)
    scale=3
    w,h=rect[2]-rect[0],rect[3]-rect[1]
    pw,ph=w*scale+28,h*scale+50
    cols=min(3,len(items)); rows=(len(items)+cols-1)//cols
    for annotated in (False,True):
        montage=Image.new("RGB",(cols*pw+12,rows*ph+12),"#101619")
        draw=ImageDraw.Draw(montage)
        for i,(label,path) in enumerate(items):
            im=Image.open(path).convert("RGB").crop(rect)
            if annotated:
                d=ImageDraw.Draw(im)
                for row in records[label]["rows"]:
                    y=row["y"]-rect[1]
                    for x in row["corridor"]:
                        d.point((x-rect[0],y),fill="#747474")
                for p in records[label]["alternative_path"]["points"]:
                    if p["valid"]:d.point((round(p["x"])-rect[0],p["y"]-rect[1]),fill="#b883ee")
                for p in records[label]["primary_path"]["points"]:
                    if p["valid"]:d.point((round(p["x"])-rect[0],p["y"]-rect[1]),fill="#ffd565")
                for p in source_points:
                    if p["valid"] and rect[1]<=p["y"]<rect[3]:d.point((round(p["x"])-rect[0],p["y"]-rect[1]),fill="#54ffc5")
            im=im.resize((w*scale,h*scale),Image.Resampling.NEAREST)
            x,y=12+(i%cols)*pw,12+(i//cols)*ph
            montage.paste(im,(x+6,y+36))
            draw.text((x+6,y),label,font=font(17),fill="#eef4f6")
            sy=y+36+(320-rect[1])*scale
            draw.line((x,sy,x+4,sy),fill="#f4ba4e")
            draw.line((x+6+w*scale+2,sy,x+6+w*scale+6,sy),fill="#f4ba4e")
        montage.save(out/("source-tangent-selected-features.png" if annotated else "source-tangent-untouched-crops.png"))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image",action="append",required=True,metavar="LABEL=PATH")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    items=[(v.split("=",1)[0],Path(v.split("=",1)[1]).resolve()) for v in args.image]
    if len(set(k for k,_ in items))!=len(items):raise ValueError("Labels must be unique")
    args.output.mkdir(parents=True,exist_ok=True)
    source=rgb(SOURCE)
    canvas=np.zeros((1152,512,3),dtype=np.uint8);canvas[320:832]=source
    points=rail_points(canvas,source=True)
    # Local source tangent uses ONLY original rows0..20, with no candidate
    # tuning. The source's farther curvature is a separate sensitivity check.
    fit=robust_line(points,320,341)
    source_strength=[p["gradient_strength"] for p in points if p["valid"] and p["y"]<341]
    threshold=max(7.0,.25*float(np.median(source_strength)))
    profile={"id":"track3-source-derived-local-tangent-v1","source_fit_rows_half_open":[320,341],
             "generated_search_rows_half_open":Y_RANGE,"near_fit_rows_half_open":NEAR_RANGE,
             "corridor_half_width_px":HALF_WIDTH,"minimum_gradient_source_derived":threshold,
             "source_file_sha256":hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
             "evaluator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "old_evaluator_sha256":hashlib.sha256((ROOT/"experiments/qwen/evaluate_boundary.py").read_bytes()).hexdigest(),
             "selection_prior":"source-only local tangent and endpoint; output does not retune corridor",
             "validity":"feature-extracted means sufficient confident correspondence, never successful outpaint"}
    report={"profile":profile,"source_fit":fit,"source_window_sensitivity":[robust_line(points,320,351),robust_line(points,320,371)],
            "limitations":["Source-guided correspondence may choose a nearby false edge; inspect untouched and selected-feature crops.",
                           "Neutral/blue strong-ridge color gate can reject valid colored or blurred metal; that produces invalid-feature, not success.",
                           "A correctly curving rail may leave even this source-only corridor. Invalid is unknown geometry, not a quality verdict.",
                           "Far curvature is not judged; only a short source-adjacent segment is fitted.",
                           "Internal path score is feature-selection likelihood, not an image-quality ranking."],"images":{}}
    records={}
    for label,path in items:
        a=rgb(path)
        if a.shape!=(1152,512,3):raise ValueError("Expected512x1152 saved output")
        rec=evaluate(a,fit,points,threshold);records[label]=rec
        report["images"][label]={"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"rail":rec,"pixels":boundary_stats(a,source)}
    (args.output/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    crops(items,records,args.output,fit,points)
    lines=["# Source-derived near-boundary rail diagnostic","","Coordinates and source tangent use only the original source. The older generated-corridor evaluator is unchanged. Feature extraction is not a quality pass.","",
           "| Image | extraction | local tangent difference deg | endpoint dx px | last4 median dx px | invalid reason |",
           "|---|---|---:|---:|---:|---|"]
    for label,rec in records.items():
        values=[rec["metrics"][k] for k in ["near_angle_difference_degrees","near_endpoint_signed_dx_px","last4_source_tangent_median_signed_dx_px"]]
        vals=["invalid" if v is None else f"{v:.2f}" for v in values]
        lines.append(f"| {label} | {rec['status']} | {' | '.join(vals)} | {'; '.join(rec['invalid_reasons'])} |")
    lines += ["","A local edge/post can fool this source-correspondence prior. Yellow selected generated ridge, purple separated alternative, green original ridge, grey source-derived search corridor. Markers outside each crop identify y320. The untouched image remains decisive; metrics do not imply success.","",
              "![Actual untouched crops](source-tangent-untouched-crops.png)","","![Derived feature-selection diagnostic](source-tangent-selected-features.png)"]
    (args.output/"REPORT.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"output":str(args.output),"source_fit":fit,"images":{k:{"status":r["status"],"metrics":r["metrics"],"invalid_reasons":r["invalid_reasons"]} for k,r in records.items()}},indent=2))


if __name__=="__main__":main()
