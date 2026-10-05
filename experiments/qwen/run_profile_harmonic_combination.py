#!/usr/bin/env python3
"""CPU-only derived compositor: saved profile warp plus frozen harmonic RGB.

The original model raw stays untouched. Each labeled derived-raw intermediate
retains the original generator's decoded known region solely as residual input.
Only its exterior comes from the saved profile warp. The frozen harmonic helper
then applies a 96px RGB residual and the shared original source is hard pasted.
No optical-flow function, model, or GPU is used by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from evaluate_boundary import SOURCE, boundary_stats, font, rgb
from run_boundary_compositor import compose, harmonic_transfer
from run_source_profile_alignment import (DEFAULT_INPUT, DEFAULT_OUTPUT,
                                           feature_metrics, generated_features,
                                           measured_correspondences, source_features)

ROOT=Path(__file__).resolve().parents[2]
PROFILE_INPUT=DEFAULT_OUTPUT/"sampling-comparison"
DEFAULT_OUT=DEFAULT_OUTPUT/"combined-profile-harmonic96"
STRIP=96
SIGMA=6
CAP=24


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def array_sha(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def spectral(array,roi):
    x0,y0,x1,y1=roi
    l=array[y0:y1,x0:x1].astype(float)@np.array([.2126,.7152,.0722])
    l-=l.mean(axis=1,keepdims=True)
    window=np.hanning(l.shape[1])
    spectrum=np.fft.rfft(l*window[None],axis=1)
    power=np.mean(np.abs(spectrum)**2,axis=0)/max(float((window**2).sum()),1e-12)
    frequencies=np.fft.rfftfreq(l.shape[1])
    bands={}
    for name,lo,hi in (("low",0,.05),("middle",.05,.2),("high",.2,.500001)):
        bands[name]=float(power[(frequencies>=lo)&(frequencies<hi)].sum())
    return {"roi_xyxy":roi,"row_mean_removed":True,"window":"Hann along x only; analysis, no pixel filtering",
            "frequency_units":"cycles/pixel","band_power":bands,
            "x_difference_squared_mean":float(np.mean(np.diff(l,axis=1)**2)),
            "limitation":"This measures spectral texture including object edges, not a semantic grain estimator."}


def all_spectra(array):
    return {name:spectral(array,rect) for name,rect in {
        "top96_full_width":(0,224,512,320),
        "top_rail_patch_includes_edges":(280,288,380,320),
        "top_vegetation_no_geometry_warp":(200,288,260,320),
        "bottom96_full_width":(0,832,512,928)}.items()}


def ratios(a,b):
    return {roi:{band:a[roi]["band_power"][band]/max(b[roi]["band_power"][band],1e-12)
                 for band in ("low","middle","high")} for roi in a}


def preservation(output,source,baseline):
    far=np.r_[output[:224].ravel(),output[928:].ravel()]
    original_far=np.r_[baseline[:224].ravel(),baseline[928:].ravel()]
    return {"source_pixels_exact":bool(np.array_equal(output[320:832],source)),
            "far_exterior_pixels_exact":bool(np.array_equal(far,original_far)),
            "source_pixels_sha256":array_sha(output[320:832]),
            "original_source_pixels_sha256":array_sha(source),
            "far_exterior_pixels_sha256":array_sha(far),
            "original_far_exterior_pixels_sha256":array_sha(original_far),
            "far_region":"top rows0..223 and bottom rows928..1151, full width",
            "top_join_rgb_mae":float(np.abs(output[319].astype(float)-output[320]).mean()),
            "bottom_join_rgb_mae":float(np.abs(output[831].astype(float)-output[832]).mean()),
            "source_boundary_metrics":boundary_stats(output,source)}


def preview(out,images):
    for rect,label,scale in (((0,286,512,350),"top-fullwidth-join",2),((270,296,380,342),"rail-join",4),((0,806,512,858),"bottom-fullwidth-join",2)):
        w,h=rect[2]-rect[0],rect[3]-rect[1]
        canvas=Image.new("RGB",(w*scale+220,len(images)*(h*scale+44)),"#10191d")
        d=ImageDraw.Draw(canvas)
        for i,(name,a) in enumerate(images.items()):
            y=i*(h*scale+44)
            d.text((12,y+20),name,font=font(15),fill="white")
            crop=Image.fromarray(a).crop(rect)
            crop.save(out/f"{name}-{label}-native.png")
            canvas.paste(crop.resize((w*scale,h*scale),Image.Resampling.NEAREST),(210,y+24))
        canvas.save(out/f"{label}-comparison.png")
    canvas=Image.new("RGB",(len(images)*256,612),"#10191d");d=ImageDraw.Draw(canvas)
    for i,(name,a) in enumerate(images.items()):
        d.text((i*256+8,8),name,font=font(14),fill="white")
        canvas.paste(Image.fromarray(a).resize((256,576),Image.Resampling.LANCZOS),(i*256,32))
    canvas.save(out/"full-comparison.png")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=DEFAULT_INPUT)
    parser.add_argument("--profiles",type=Path,default=PROFILE_INPUT)
    parser.add_argument("--output",type=Path,default=DEFAULT_OUT)
    args=parser.parse_args()
    if args.output.exists():raise ValueError("Use a fresh unique output folder")
    cv2.setNumThreads(1);start=time.monotonic()
    raw=rgb(args.input/"raw.png");baseline=rgb(args.input/"composite.png");source=rgb(SOURCE)
    if raw.shape!=(1152,512,3) or source.shape!=(512,512,3):raise ValueError("Unexpected geometry")
    if not np.array_equal(baseline[320:832],source):raise ValueError("Input source crop differs")
    paths=(args.input/"raw.png",args.input/"composite.png",SOURCE,ROOT/"Player/Resources/track3.jpg",
           ROOT/"experiments/qwen/run_boundary_compositor.py",ROOT/"experiments/qwen/evaluate_boundary.py",ROOT/"experiments/qwen/evaluate_source_tangent.py")
    before={str(p.resolve()):sha(p) for p in paths}
    # Verify the imported frozen helper's constant mode, without invoking the
    # old runner's optical-flow-related preflight or any registration function.
    uniform=np.full(raw.shape,90,np.float32);reference=np.full(source.shape,100,np.uint8)
    test,_=harmonic_transfer(uniform,reference,STRIP,SIGMA,CAP)
    assert np.allclose(test[319],90+10*(1-1/96),atol=1e-3)
    assert np.array_equal(test[:224],uniform[:224])
    fits,points=source_features(source);raw_rows=generated_features(raw,fits)
    zero=np.zeros(raw.shape[:2],float)
    spectra_original=all_spectra(baseline)
    images={"original":baseline}
    report={"status":"CPU derived compositor; quality inconclusive", "source_rect_xyxy":[0,320,512,832],
            "device":"CPU only","no_optical_flow_or_model_calls":True,
            "original_generator_raw_unchanged":True,"input_and_frozen_hashes_before":before,
            "harmonic_helper":"imported unchanged run_boundary_compositor.harmonic_transfer",
            "harmonic_settings":{"strip_pixels":STRIP,"residual_source":"original generator decoded known pixels only","sigma_x":SIGMA,"residual_cap_255":CAP},
            "constant_mode_cpu_preflight_passed":True,
            "spectral_original":spectra_original,
            "limits":["A labeled derived_raw is a compositor intermediate, not a model-generated raw output.",
                      "The first highlight identity remains missing/invalid near the join; RGB residual cannot restore it.",
                      "Harmonic RGB adds a smooth correction without filtering model pixels, but quantization/clipping can change high frequencies.",
                      "Nearest profile sampling adds no interpolated RGB before RGB correction; harmonic correction does create new RGB values.",
                      "Profile sampling stretches/compresses texture. Spectral power also includes scene edges and cannot prove grain preservation."],"variants":{}}
    args.output.mkdir(parents=True)
    # Harmonic-only control isolates the color contribution from profile warp.
    color_work,_=harmonic_transfer(raw.astype(np.float32),source,STRIP,SIGMA,CAP)
    harmonic_only=compose(color_work,source);images["harmonic-only"]=harmonic_only
    control=preservation(harmonic_only,source,baseline)
    assert control["source_pixels_exact"] and control["far_exterior_pixels_exact"]
    Image.fromarray(harmonic_only).save(args.output/"harmonic-only-composite.png")
    report["variants"]["harmonic-only"]={"preservation":control,"spectral":all_spectra(harmonic_only)}
    for name in ("profile64","nearest64"):
        profile_path=args.profiles/name/"composite.png"
        warped=rgb(profile_path)
        field=np.load(args.profiles/name/"horizontal_sampling_displacement.npy").astype(float)
        derived=raw.copy();derived[:320]=warped[:320];derived[832:]=warped[832:]
        assert np.array_equal(derived[320:832],raw[320:832])
        assert np.array_equal(derived[:320],warped[:320]) and np.array_equal(derived[832:],warped[832:])
        folder=args.output/f"{name}-harmonic96";folder.mkdir()
        Image.fromarray(derived).save(folder/"derived_raw_original_known.png")
        (folder/"DERIVED_RAW.md").write_text("This is a derived compositor input, not model raw. Its exterior comes from the saved profile warp. Its known square retains the original generator decoder pixels solely for the original-source RGB residual calculation. The original model raw.png is untouched.\n")
        corrected,residual_details=harmonic_transfer(derived.astype(np.float32),source,STRIP,SIGMA,CAP)
        assert np.array_equal(corrected[320:832],raw[320:832].astype(np.float32))
        # The additive term must be exactly the harmonic-only term; registration
        # never enters the residual calculation because known pixels are raw.
        original_correction=color_work-raw.astype(np.float32)
        applied_correction=corrected-derived.astype(np.float32)
        max_error=float(np.abs(original_correction-applied_correction).max())
        assert max_error<3.1e-5
        output=compose(corrected,source)
        Image.fromarray(output).save(folder/"composite.png")
        images[name]=warped;images[f"{name}+RGB96"]=output
        checks=preservation(output,source,baseline)
        assert checks["source_pixels_exact"] and checks["far_exterior_pixels_exact"]
        spectra_warp=all_spectra(warped);spectra_output=all_spectra(output)
        measurement=measured_correspondences(output,raw_rows,field)
        report["variants"][name]={"input_profile_sha256":sha(profile_path),"derived_raw_sha256":sha(folder/"derived_raw_original_known.png"),
            "derived_known_crop_exact_original_generator_decode":True,
            "derived_exterior_exact_saved_profile":True,
            "original_decoder_known_crop_sha256":array_sha(raw[320:832]),"derived_known_crop_sha256":array_sha(derived[320:832]),
            "identical_harmonic_additive_term_max_float32_error":max_error,
            "harmonic_residual_details":residual_details,"preservation":checks,
            "spectral_before_warp":spectra_original,"spectral_warp":spectra_warp,"spectral_combined":spectra_output,
            "spectral_ratio_warp_to_original":ratios(spectra_warp,spectra_original),
            "spectral_ratio_combined_to_warp":ratios(spectra_output,spectra_warp),
            "clip_fraction_before_rounding_low":float((corrected<0).mean()),"clip_fraction_before_rounding_high":float((corrected>255).mean()),
            "edge_diagnostic":feature_metrics(measurement,fits),"output_sha256":sha(folder/"composite.png")}
    preview(args.output,images)
    after={p:sha(Path(p)) for p in before}
    report["input_and_frozen_hashes_after"]=after
    report["input_and_frozen_files_unchanged"]=before==after
    assert before==after
    report["seconds"]=time.monotonic()-start
    (args.output/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    lines=["# Profile geometry plus original-decoder harmonic RGB", "",
           "CPU compositor diagnostic only. Saved model raw and all frozen helpers remain unchanged. The original decoded known square is restored inside each labeled derived_raw only for the 96px residual calculation; exact shared source pixels are pasted at the end.", "",
           "| variant | top RGB adjacent-row MAE | source exact | distant exterior exact | rail-patch high-frequency ratio, combined / warp |", "|---|---:|---|---|---:|"]
    for name,row in report["variants"].items():
        checks=row["preservation"]
        high=row.get("spectral_ratio_combined_to_warp",{}).get("top_rail_patch_includes_edges",{}).get("high")
        value="control" if high is None else f"{high:.4f}"
        lines.append(f"| {name} | {checks['top_join_rgb_mae']:.3f} | {checks['source_pixels_exact']} | {checks['far_exterior_pixels_exact']} | {value} |")
    lines += ["", "The held-out highlight remains invalid/missing near the join. Harmonic RGB can reduce a photometric seam but cannot restore missing rail geometry. Nearest sampling preserves original RGB tuples before correction, while the additive RGB stage creates new values. Spectral texture includes object edges; it is not proof of retained grain.", "",
              "![Untouched rail comparisons](rail-join-comparison.png)", "", "![Full-width top join](top-fullwidth-join-comparison.png)", "", "![Full outputs](full-comparison.png)", ""]
    (args.output/"REPORT.md").write_text("\n".join(lines))
    print(json.dumps({"output":str(args.output.resolve()),"seconds":report["seconds"],"inputs_unchanged":before==after,
                      "variants":{n:{"preservation":v["preservation"],"high_frequency_ratios":v.get("spectral_ratio_combined_to_warp")} for n,v in report["variants"].items()}},indent=2))


if __name__=="__main__":main()
