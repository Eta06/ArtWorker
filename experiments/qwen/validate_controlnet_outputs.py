#!/usr/bin/env python3
"""Independent CPU artifact validation and true before/after crops; no GPU.

Passing artifact validation establishes recorded computation/source integrity,
not generation quality. Refuses unfinished suites. Original images stay intact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from evaluate_boundary import ROOT, font, montage, rgb


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def white_diagnostic(a):
    white=np.all(a>=230,axis=-1)
    ids=np.flatnonzero(white.mean(axis=1)>.9)
    bands=[]
    for y in ids:
        if not bands or y!=bands[-1][1]+1:bands.append([int(y),int(y)])
        else:bands[-1][1]=int(y)
    return {"definition":"all RGB channels>=230; row bands require90 percent pixels; brightness alone is not quality",
            "row_bands_inclusive":bands,"top_generated_fraction":float(white[:320].mean()),
            "bottom_generated_fraction":float(white[832:].mean())}


def crop_grid(items,out,name,rect,scale,seam):
    w,h=rect[2]-rect[0],rect[3]-rect[1]
    pw,ph=w*scale+28,h*scale+50
    canvas=Image.new("RGB",(len(items)*pw+12,2*ph+12),"#101619")
    draw=ImageDraw.Draw(canvas)
    for row,kind in enumerate(["raw","composite"]):
        for col,(label,directory) in enumerate(items):
            im=Image.open(directory/f"{kind}.png").convert("RGB").crop(rect)
            im=im.resize((w*scale,h*scale),Image.Resampling.NEAREST)
            x,y=12+col*pw,12+row*ph
            canvas.paste(im,(x+6,y+36))
            draw.text((x+6,y),f"{label} · {kind}",font=font(17),fill="#eef4f6")
            sy=y+36+(seam-rect[1])*scale
            draw.line((x,sy,x+4,sy),fill="#f4ba4e")
            draw.line((x+6+w*scale+2,sy,x+6+w*scale+6,sy),fill="#f4ba4e")
    canvas.save(out/name)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    suite=args.suite.resolve();output=args.output or suite/"independent-review"
    s=json.loads((suite/"controlnet_metrics.json").read_text())
    if s.get("status")!="success" or s.get("phase")!="complete":raise ValueError("Teacher suite is unfinished; do not infer quality from partial files")
    source=rgb(s["source_512_path"])
    inputs=json.loads((ROOT/"experiments/evaluation/inputs/inputs.json").read_text())
    steps=s["steps"]
    checks={};white={};records=s["runs"]
    for mode,record in records.items():
        if record.get("status")!="success":raise ValueError("Arm is unfinished")
        directory=suite/mode
        raw=rgb(directory/"raw.png");composite=rgb(directory/"composite.png")
        latent=np.load(directory/"final_latents.npz")["latents"]
        rows=record["step_records"]
        checks[mode]={"complete_record_count":len(rows)==steps==record["transformer_calls"],
                      "base_block_count":sum(r["base_blocks"] for r in rows)==record["base_block_calls"]==steps*32,
                      "control_block_count":sum(r["control_blocks"] for r in rows)==record["control_block_calls"]==steps*16,
                      "token_sum":sum(r["target_tokens_forwarded"] for r in rows)==record["target_token_forward_sum"]==steps*2304,
                      "full_target_in_every_step":all(r["target_tokens_forwarded"]==2304 for r in rows),
                      "raw_shape":raw.shape==(1152,512,3),"composite_shape":composite.shape==(1152,512,3),
                      "source_pixels_exact":bool(np.array_equal(composite[320:832],source)),
                      "latents_shape":latent.shape==(1,2304,64),"latents_finite":bool(np.isfinite(latent).all()),
                      "latent_hash":hashlib.sha256(latent.tobytes()).hexdigest()==record["final_latents_sha256"],
                      "composite_hash":sha(directory/"composite.png")==record["output_sha256"],
                      "outer_noise_matches_suite":record["noise_sha256"]==s["noise_sha256"],
                      "sigmas_match_suite":record["sigmas"]==s["sigmas"]}
        white[mode]=white_diagnostic(raw)
    flags={"source_file_hash_matches":sha(s["source_512_path"])==s["source_512_sha256"],
           "original_jpeg_hashes_unchanged":all(sha(t["original_jpg"])==t["original_jpg_sha256"] for t in inputs["tracks"]),
           "known_context_matched_record":s["known_mask_and_source_context_matched"],
           "full_compute_declared":s["growing_compute"] is False and s["prefix_cache_enabled"] is False,
           "no_turbo_adapter":s["adapter"] is None}
    if not all(all(c.values()) for c in checks.values()) or not all(flags.values()):raise AssertionError("Artifact validation failed")
    report={"status":"passed","quality_status":"not assessed by integrity validation; inspect actual finals",
            "suite":str(suite),"steps":steps,"arms":checks,"shared":flags,
            "raw_near_white_diagnostic":white,"known_bridge_extension":s["known_bridge_extension"],
            "suite_metrics_sha256":sha(suite/"controlnet_metrics.json"),
            "current_validator_sha256":sha(__file__),"baseline_comparison_limit":"Older edge-context uses Turbo, image prefix and real target growth. Comparison is qualitative, not a matched ControlNet-only ablation."}
    (suite/"independent_validation.json").write_text(json.dumps(report,indent=2)+"\n")
    output.mkdir(parents=True,exist_ok=True)
    baseline=ROOT/"experiments/qwen/geometry_runs/2026-10-05/track3/edge-context"
    items=[("old edge-context",baseline)]+[(mode,suite/mode) for mode in records]
    montage([(label,path/"composite.png") for label,path in items],output/"full-comparison.png",scale=.5)
    crop_grid(items,output,"rail-join-raw-composite.png",(256,296,384,344),4,320)
    crop_grid(items,output,"upper-source-border-raw-composite.png",(0,288,512,352),1,320)
    crop_grid(items,output,"lower-source-border-raw-composite.png",(0,800,512,896),1,832)
    print(json.dumps({"validation":str(suite/"independent_validation.json"),"review_images":str(output),
                      "status":report["status"],"arms":list(records),"white_diagnostic":white},indent=2))


if __name__=="__main__":main()
