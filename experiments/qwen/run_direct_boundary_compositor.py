#!/usr/bin/env python3
"""CPU diagnostics: direct generated/source boundary color matching.

Only generated exterior RGB values are corrected. There is no geometric warp,
line mask or line painting. Saved intermediates are postprocessed images, never
model raw. The frozen earlier decoder-residual harmonic method is a control.
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

import run_boundary_compositor as frozen


ROOT=Path(__file__).resolve().parents[2]
RECT=(0,320,512,832)
DEFAULT_RAW=ROOT/"experiments/qwen/controlnet_runs/2026-10-05/track3-saved-top32-outpaint/tangent-canny/top32/raw.png"
DEFAULT_OUTPUT=ROOT/"experiments/qwen/boundary_compositor_runs/2026-10-05/track3-top32-direct"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def smooth_x(row, sigma):
    return cv2.GaussianBlur(row[None].astype(np.float32),(0,0),sigmaX=sigma,sigmaY=0)[0]


def slope(rows):
    """Least-squares derivative for sample coordinate 0,1,...,n-1."""
    coordinates=np.arange(rows.shape[0],dtype=np.float64)
    centered=coordinates-coordinates.mean()
    return np.einsum("n,nwc->wc",centered,rows.astype(np.float64))/np.dot(centered,centered)


def residuals(raw,source,sigma,value_cap,gradient_cap):
    """Both edges use distance increasing outward, generated d=1 first.

    Original source edge is d=0. Desired d=1 value is the original edge plus
    its outward least-squares gradient. We compare that directly with the
    first generated row, not with the decoder's known-source reconstruction.
    """
    _,y0,_,y1=RECT
    residual={}
    for name,known,generated in (
        ("top",source[:8],raw[y0-8:y0][::-1]),
        ("bottom",source[-8:][::-1],raw[y1:y1+8]),
    ):
        original_edge=smooth_x(known[0],sigma)
        original_outward=-smooth_x(slope(known),sigma)
        generated_edge=smooth_x(generated[0],sigma)
        generated_outward=smooth_x(slope(generated),sigma)
        wanted_first=original_edge+original_outward
        original_value_residual=wanted_first-generated_edge
        original_gradient_residual=original_outward-generated_outward
        value=np.clip(original_value_residual,-value_cap,value_cap).astype(np.float32)
        gradient=np.clip(original_gradient_residual,-gradient_cap,gradient_cap).astype(np.float32)
        residual[name]={"value":value,"gradient":gradient,
                        "wanted_first":wanted_first,"original_outward":original_outward,
                        "source_edge":original_edge,"generated_edge":generated_edge,
                        "unclipped_value":original_value_residual,
                        "unclipped_gradient":original_gradient_residual}
    return residual


def dct_row(row):
    return np.stack([cv2.dct(row[:,c:c+1].astype(np.float32))[:,0] for c in range(3)],axis=-1)


def idct_row(spectra):
    return np.stack([cv2.idct(spectra[:,c:c+1].astype(np.float32))[:,0] for c in range(3)],axis=-1)


def correction_field(value,gradient,radius,method):
    """Correct first generated row, taper to zero by generated row radius.

    Direct harmonic is C0 only. Hermite preserves the discrete first-row
    derivative by a cubic clamped finite strip. Spectral biharmonic preserves
    value and continuous derivative, damps high spatial frequencies and clamps
    value/derivative at the far edge. All return radius rows, d=1..radius.
    """
    width=value.shape[0]
    length=radius-1
    coordinates=np.arange(radius,dtype=np.float64)
    u=coordinates/length
    if method=="hermite":
        h00=2*u**3-3*u**2+1
        h10=u**3-2*u**2+u
        one=1/length
        one_h00=2*one**3-3*one**2+1
        one_h10=one**3-2*one**2+one
        # Enforce r(2)-r(1)=gradient, rather than assuming the finite
        # difference is the continuous derivative of a sampled cubic.
        effective=(gradient+value*(1-one_h00))/(length*one_h10)
        return (h00[:,None,None]*value[None]+length*h10[:,None,None]*effective[None]).astype(np.float32)
    a=dct_row(value).astype(np.float64)
    b=dct_row(gradient).astype(np.float64)
    modes=np.pi*np.arange(width,dtype=np.float64)/width
    spectra=np.empty((radius,width,3),dtype=np.float64)
    if method=="harmonic":
        damping=np.empty((radius,width),dtype=np.float64)
        damping[:,0]=1-u
        k=modes[1:]
        d=coordinates[:,None]
        damping[:,1:]=(np.exp(-d*k)-np.exp(-(2*length-d)*k))/(-np.expm1(-2*length*k))
        spectra=a[None]*damping[...,None]
    elif method=="biharmonic":
        h00=2*u**3-3*u**2+1
        h10=u**3-2*u**2+u
        spectra[:,0]=h00[:,None]*a[None,0]+length*h10[:,None]*b[None,0]
        for index,k in enumerate(modes[1:],start=1):
            K=k*length
            q=np.exp(-K)
            # Stable exponentially decaying bases from both strip boundaries.
            matrix=np.array([[1,0,q,q],[-K,1,K*q,(-1+K)*q],
                             [q,q,1,0],[-K*q,(1-K)*q,K,-1]],dtype=np.float64)
            rhs=np.stack([a[index],length*b[index],np.zeros(3),np.zeros(3)])
            coefficients=np.linalg.solve(matrix,rhs)
            left=np.exp(-K*u)
            right=np.exp(-K*(1-u))
            basis=np.stack([left,u*left,right,(1-u)*right],axis=1)
            spectra[:,index]=basis@coefficients
    else:
        raise ValueError(method)
    result=np.stack([idct_row(row) for row in spectra]).astype(np.float32)
    # Clamped values are mathematical constraints; assign explicitly to remove
    # inverse-DCT floating-point endpoint noise, without touching source pixels.
    result[0]=value
    result[-1]=0
    return result


def apply_direct(raw,source,radius,method,sigma,value_cap=48,gradient_cap=.35):
    start=time.perf_counter()
    residual=residuals(raw,source,sigma,value_cap,gradient_cap)
    corrected=raw.copy().astype(np.float32)
    statistics={}
    _,y0,_,y1=RECT
    for name in ("top","bottom"):
        row=residual[name]
        profile_stats={}
        if method=="profile":
            # Match the actual generated low-frequency color profile row by
            # row, rather than propagating a one-row codec jump as a long
            # derivative. No pixels/geometry/texture are copied from source.
            generated=raw[y0-radius:y0][::-1] if name=="top" else raw[y1:y1+radius]
            source_normal=np.clip(row["original_outward"],-gradient_cap,gradient_cap)
            distances=np.arange(1,radius+1,dtype=np.float32)
            # First two rows continue the source's regularized affine color;
            # farther out that slope saturates smoothly instead of drifting.
            normal_distance=np.where(distances<=2,distances,2+8*(-np.expm1(-(distances-2)/8)))
            progress=np.maximum(0,(distances-2)/(radius-2))
            taper=1-3*progress**2+2*progress**3
            profile_residual=np.stack([row["source_edge"]+normal_distance[i]*source_normal-smooth_x(generated[i],sigma)
                                      for i in range(radius)])
            field=np.clip(profile_residual,-value_cap,value_cap)*taper[:,None,None]
            field[-1]=0
            profile_stats={"profile_source_normal_capped_fraction":float(np.mean(np.abs(row["original_outward"])>gradient_cap)),
                           "profile_per_row_value_capped_fraction":float(np.mean(np.abs(profile_residual)>value_cap)),
                           "profile_source_normal_max_after_cap":float(np.abs(source_normal).max()),
                           "profile_first_two_rows_full_weight":True,
                           "profile_normal_distance_saturation_pixels":10}
        else:
            field=correction_field(row["value"],row["gradient"],radius,method)
        if name=="top":
            corrected[y0-radius:y0]+=field[::-1]
        else:
            corrected[y1:y1+radius]+=field
        statistics[name]={**profile_stats,"value_abs_p95_255":float(np.quantile(np.abs(row["value"]),.95)),
                          "value_max_255":float(np.abs(row["value"]).max()),
                          "gradient_abs_p95_255_per_pixel":float(np.quantile(np.abs(row["gradient"]),.95)),
                          "value_residual_capped_fraction":float(np.mean(np.abs(row["unclipped_value"])>value_cap)),
                          "gradient_residual_capped_fraction":float(np.mean(np.abs(row["unclipped_gradient"])>gradient_cap)),
                          "correction_max_abs_255":float(np.abs(field).max()),
                          "correction_first_row_mean_rgb":field[0].mean(axis=0).tolist(),
                          "correction_far_row_max_abs":float(np.abs(field[-1]).max()),
                          "clipped_pixel_channels_before_uint8":int(np.sum((corrected[y0-radius:y0] if name=="top" else corrected[y1:y1+radius])<0)+np.sum((corrected[y0-radius:y0] if name=="top" else corrected[y1:y1+radius])>255))}
    output=frozen.compose(corrected,source)
    return output,{"method":method,"radius_pixels":radius,"smoothing_sigma_x":sigma,
                   "normal_slope_estimate_rows":8,"value_cap_255":value_cap,
                   "gradient_cap_255_per_pixel":gradient_cap,"edges":statistics,
                   "processing_seconds":time.perf_counter()-start}


def signed_fixture(method):
    """Independent affine-intensity oracle detects top/bottom sign mistakes."""
    width=512
    source=np.empty((512,width,3),dtype=np.float32)
    # Original y slope is +0.1. Thus top outward slope must be -0.1 and
    # bottom outward +0.1. Channel offsets are distinct to expose axis errors.
    for channel,offset in enumerate((0,7,14)):
        source[...,channel]=100+offset+.1*np.arange(512)[:,None]
    raw=np.empty((1152,width,3),dtype=np.float32)
    for channel,offset in enumerate((0,7,14)):
        raw[:320,:,channel]=75+offset+.03*np.arange(320)[:,None]
        raw[320:832,:,channel]=30+offset # Deliberately wrong decoded source.
        raw[832:,:,channel]=175+offset-.05*np.arange(320)[:,None]
    reference=source.astype(np.float32)
    residual=residuals(raw,reference,4,100,10)
    expected_top=np.array([99.9,106.9,113.9],dtype=np.float32)
    expected_bottom=np.array([151.2,158.2,165.2],dtype=np.float32)
    for name,wanted in (("top",expected_top),("bottom",expected_bottom)):
        if not np.allclose(residual[name]["wanted_first"],wanted[None],atol=2e-4):
            raise AssertionError(f"Outward gradient/source-value sign failed: {name}")
        field=correction_field(residual[name]["value"],residual[name]["gradient"],64,method)
        generated=(raw[319] if name=="top" else raw[832])
        if not np.allclose(generated+field[0],wanted[None],atol=2e-4):
            raise AssertionError(f"First generated row did not match affine source extrapolation: {name}")
        if method=="hermite":
            next_generated=raw[318] if name=="top" else raw[833]
            oracle=wanted+(-.1 if name=="top" else .1)
            if not np.allclose(next_generated+field[1],oracle[None],atol=2e-4):
                raise AssertionError(f"Second generated row derivative sign/value failed: {name}")
        if not np.array_equal(field[-1],np.zeros_like(field[-1])):
            raise AssertionError("Correction value is nonzero at far boundary")
        # A clamped continuous gradient has a far-row finite difference of
        # order 1/L^2. Harmonic deliberately promises only C0 continuity.
        if method!="harmonic" and float(np.abs(field[-2]).max())>.1:
            raise AssertionError("Clamped far-boundary gradient/taper is too abrupt")
    # Raw known source contents have no role in direct residuals. Change them
    # drastically and require identical boundary fields.
    changed=raw.copy()
    changed[320:832]=245
    changed_residual=residuals(changed,reference,4,100,10)
    for name in ("top","bottom"):
        if not np.array_equal(residual[name]["value"],changed_residual[name]["value"]) or not np.array_equal(residual[name]["gradient"],changed_residual[name]["gradient"]):
            raise AssertionError("Direct compositor incorrectly depends on decoded known source")
    return {"method":method,"affine_original_outward_sign":True,"first_generated_row_value":True,
            "hermite_second_generated_row_affine_oracle":method=="hermite",
            "far_value_zero":True,"clamped_far_slope_if_claimed":method!="harmonic",
            "decoded_known_source_independence":True}


def preflight():
    fixtures=[signed_fixture(method) for method in ("harmonic","hermite","biharmonic")]
    # A source already continuing with a constant color is a fixed point:
    # every method must change no pixels anywhere, without needing cap logic.
    source=np.full((512,512,3),83,dtype=np.uint8)
    raw=np.full((1152,512,3),83,dtype=np.float32)
    for method in ("harmonic","hermite","biharmonic","profile"):
        output,_=apply_direct(raw,source,32,method,4)
        if not np.array_equal(output,raw.astype(np.uint8)):
            raise AssertionError("Constant-color scene was not a fixed point")
    # A nonzero affine scene already continues through both seams. Direct
    # boundary residuals must be zero even though the edge row and first
    # exterior row naturally differ. Changing decoded source cannot affect it.
    source=np.empty((512,512,3),np.float32)
    raw=np.empty((1152,512,3),np.float32)
    for c,offset in enumerate((0,13,27)):
        source[...,c]=100+offset+.05*np.arange(512)[:,None]
        raw[...,c]=100+offset+.05*(np.arange(1152)-320)[:,None]
    zero=residuals(raw,source,4,100,10)
    for edge in zero.values():
        if np.abs(edge["value"]).max()>3e-5 or np.abs(edge["gradient"]).max()>3e-5:
            raise AssertionError("Already seamless nonzero affine scene is not a fixed point")
    # A half-sample cosine is a Neumann DCT mode. Reflection must commute with
    # extension, and its mean must remain zero at every depth; this exposes
    # axis/normalization errors independent of the implementation equations.
    x=np.arange(512,dtype=np.float32)
    mode=np.cos(np.pi*3*(x+.5)/512).astype(np.float32)
    value=np.stack([mode*5,mode*(-3),mode*2],axis=-1)
    gradient=np.zeros_like(value)
    for method in ("harmonic","hermite","biharmonic"):
        field=correction_field(value,gradient,64,method)
        mirrored=correction_field(value[::-1].copy(),gradient,64,method)
        if not np.allclose(field[:,::-1],mirrored,atol=2e-5):
            raise AssertionError("Horizontal mirror symmetry failed")
        if float(np.abs(field.mean(axis=1)).max())>2e-5:
            raise AssertionError("Zero-mean cosine acquired a color offset")
        if not np.allclose(field[0],value,atol=2e-5) or np.abs(field[-1]).max()!=0:
            raise AssertionError("Cosine boundary/far taper values failed")
    # Independent two-row source-ramp oracle for the per-row profile method.
    # Top and bottom source slopes have different signs and magnitudes; this
    # detects either an outward-sign error or accidental edge-row copying.
    source=np.full((512,512,3),100,dtype=np.uint8)
    raw=np.full((1152,512,3),75,dtype=np.float32)
    for c,offset in enumerate((0,7,14)):
        source[:8,:,c]=(100+offset-2*np.arange(8))[:,None]
        source[-8:,:,c]=(140+offset+3*np.arange(8)[::-1])[:,None]
        raw[256:320,:,c]=(80+offset-np.arange(64)[::-1])[:,None]
        raw[832:896,:,c]=(160+offset+np.arange(64))[:,None]
    output,_=apply_direct(raw,source,64,"profile",16,100,10)
    expected={319:[102,109,116],318:[104,111,118],832:[137,144,151],833:[134,141,148]}
    for row,wanted in expected.items():
        if not np.array_equal(output[row],np.broadcast_to(np.array(wanted,dtype=np.uint8),(512,3))):
            raise AssertionError("Profile first/second-row affine continuation oracle failed")
    baseline=frozen.compose(raw,source)
    if not np.array_equal(output[320:832],source) or not np.array_equal(output[:256],baseline[:256]) or not np.array_equal(output[896:],baseline[896:]):
        raise AssertionError("Profile changed source or far exterior")
    # Mirrored photographic-like color error must produce mirrored correction.
    # Verify on quantized images with at most one LSB tolerance for rounding.
    source=np.full((512,512,3),100,dtype=np.uint8)
    raw=np.full((1152,512,3),100,dtype=np.float32)
    wave=5*np.cos(np.pi*3*(np.arange(512,dtype=np.float32)+.5)/512)
    raw[:320]+=wave[None,:,None]
    raw[832:]-=wave[None,:,None]
    normal,_=apply_direct(raw,source,64,"profile",16)
    mirrored,_=apply_direct(raw[:,::-1].copy(),source[:,::-1].copy(),64,"profile",16)
    if int(np.abs(normal[:,::-1].astype(np.int16)-mirrored.astype(np.int16)).max())>1:
        raise AssertionError("Profile horizontal mirror symmetry failed")
    if float(np.abs(normal[[319,318,832,833]].astype(float).mean(axis=1)-100).max())>.05:
        raise AssertionError("Profile zero-mean cosine acquired a color offset")
    return {"status":"passed","fixtures":fixtures,"constant_scene_fixed_point":True,
            "seamless_nonzero_affine_fixed_point":True,"cosine_mean_and_mirror_symmetry":True,
            "seamless_nonzero_affine_fixed_point_scope":"harmonic/Hermite/biharmonic residuals only; profile intentionally saturates a source-normal color prior farther than two pixels",
            "profile_opposite_slope_first_two_rows_oracle":True,"profile_source_and_far_exact":True,
            "profile_constant_fixed_point":True,"profile_mirror_and_cosine_mean":True,
            "no_model_weights_loaded":True,"cpu_only":True}


def boundary_metrics(array,baseline,source,radius,sigma):
    _,y0,_,y1=RECT
    outside=np.ones(array.shape[:2],bool)
    outside[y0:y1]=False
    far=np.ones(array.shape[:2],bool)
    far[y0-radius:y1+radius]=False
    result={"source_pixels_exact":bool(np.array_equal(array[y0:y1],source)),
            "far_exterior_pixels_exact":bool(np.array_equal(array[far],baseline[far])),
            "no_geometric_warp":True,"no_line_mask_or_painting":True,
            "generated_mean_abs_change_255":float(np.abs(array.astype(float)-baseline.astype(float))[outside].mean())}
    for name,known,external,baseline_external in (
        ("top",source[:8],array[y0-8:y0][::-1],baseline[y0-8:y0][::-1]),
        ("bottom",source[-8:][::-1],array[y1:y1+8],baseline[y1:y1+8]),
    ):
        edge=smooth_x(known[0],sigma)
        normal=-smooth_x(slope(known),sigma)
        desired=edge+normal
        actual=smooth_x(external[0],sigma)
        result[name]={"adjacent_row_rgb_jump_255":float(np.abs(external[0].astype(float)-known[0].astype(float)).mean()),
                      "smoothed_first_generated_row_target_mae_255":float(np.abs(actual-desired).mean()),
                      "first_two_generated_rows_gradient_target_mae":float(np.abs((smooth_x(external[1],sigma)-actual)-normal).mean())}
    # Pure color correction can alter edge detection/gradient contrast; this
    # statistic alone never establishes preserved line topology or scene fit.
    collar=np.concatenate([array[y0-radius:y0],array[y1:y1+radius]])
    before=np.concatenate([baseline[y0-radius:y0],baseline[y1:y1+radius]])
    a=cv2.cvtColor(collar,cv2.COLOR_RGB2GRAY).astype(float)
    b=cv2.cvtColor(before,cv2.COLOR_RGB2GRAY).astype(float)
    result["collar_horizontal_gradient_energy_ratio"]=float(np.mean(np.gradient(a,axis=1)**2)/max(1e-8,np.mean(np.gradient(b,axis=1)**2)))
    return result


def previews(out,variants):
    font=ImageFont.load_default(size=16)
    width=256
    full=Image.new("RGB",(width*len(variants),624),"#16191d")
    draw=ImageDraw.Draw(full)
    # Fixed source-boundary regions used for viewing only. These coordinates
    # never condition the correction function.
    crops={"water-top":(0,292,512,352),"asphalt-bottom":(0,805,512,861),
           "rail-top":(295,278,395,360)}
    for index,(name,array) in enumerate(variants.items()):
        image=Image.fromarray(array)
        full.paste(image.resize((256,576),Image.Resampling.LANCZOS),(index*width,36))
        draw.text((index*width+5,8),name,fill="white",font=font)
        directory=out/name
        for label,rect in crops.items():
            crop=image.crop(rect)
            crop.save(directory/f"{label}-1x.png")
            crop.resize((crop.width*4,crop.height*4),Image.Resampling.NEAREST).save(directory/f"{label}-4x.png")
    full.save(out/"full-comparison.png")
    for label,rect in crops.items():
        cw,ch=(rect[2]-rect[0])*2,(rect[3]-rect[1])*2
        montage=Image.new("RGB",(cw+190,(ch+34)*len(variants)),"#16191d")
        painter=ImageDraw.Draw(montage)
        for i,(name,array) in enumerate(variants.items()):
            crop=Image.fromarray(array).crop(rect)
            montage.paste(crop.resize((cw,ch),Image.Resampling.NEAREST),(190,i*(ch+34)+26))
            painter.text((6,i*(ch+34)+35),name,fill="white",font=font)
        montage.save(out/f"{label}-comparison-2x.png")


def run(args):
    cv2.setNumThreads(1)
    start=time.perf_counter()
    raw_path=args.raw.resolve()
    out=args.output.resolve()
    if (out/"direct_compositor_metrics.json").exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    shared=json.loads((ROOT/"experiments/evaluation/inputs/inputs.json").read_text())
    track=next(row for row in shared["tracks"] if row["id"]==args.track)
    source_path=Path(track["source_512"])
    source=np.asarray(Image.open(source_path).convert("RGB"),dtype=np.uint8)
    raw=np.asarray(Image.open(raw_path).convert("RGB"),dtype=np.float32)
    if raw.shape!=(1152,512,3) or source.shape!=(512,512,3):
        raise ValueError("Expected shared 512x1152 raw and original512 square")
    before={str(path):sha(path) for path in (raw_path,source_path,Path(track["original_jpg"]),Path(frozen.__file__))}
    out.mkdir(parents=True,exist_ok=True)
    baseline=frozen.compose(raw,source)
    variants={"baseline":baseline}
    records={"baseline":{"method":"exact-source hard paste","radius_pixels":96,"smoothing_sigma_x":0,"processing_seconds":0}}
    control_start=time.perf_counter()
    previous,details=frozen.harmonic_transfer(raw,source,96,6,24)
    variants["frozen-harmonic96"]=frozen.compose(previous,source)
    records["frozen-harmonic96"]={"method":"frozen decoder-known source residual harmonic","radius_pixels":96,
                                   "smoothing_sigma_x":6,"residual_cap_255":24,"edges":details,
                                   "processing_seconds":time.perf_counter()-control_start}
    for method in args.methods:
        for radius in args.radii:
            name=f"direct-{method}{radius}"
            output,record=apply_direct(raw,source,radius,method,args.sigma,args.value_cap,args.gradient_cap)
            variants[name]=output
            records[name]=record
    for name,array in variants.items():
        directory=out/name
        directory.mkdir(exist_ok=True)
        path=directory/"composite.png"
        Image.fromarray(array).save(path)
        records[name].update(boundary_metrics(array,baseline,source,records[name]["radius_pixels"],args.sigma))
        records[name].update({"composite_path":str(path),"composite_sha256":sha(path),
                              "artifact_kind":"derived post-generation exact-source composite; not model raw"})
        if not records[name]["source_pixels_exact"] or not records[name]["far_exterior_pixels_exact"]:
            raise AssertionError(f"Source/far exterior preservation failed: {name}")
        (directory/"metrics.json").write_text(json.dumps(records[name],indent=2)+"\n")
    previews(out,variants)
    after={name:sha(Path(name)) for name in before}
    if before!=after:
        raise AssertionError("Input source/model artifact/frozen helper changed")
    suite={"status":"success","track":args.track,"kind":"CPU direct-boundary post-generation diagnostic",
           "raw_model_input_path":str(raw_path),"source_path":str(source_path),"source_rect_xyxy":list(RECT),
           "input_sha256_before":before,"input_sha256_after":after,"inputs_and_frozen_helper_unchanged":True,
           "radii_pixels":args.radii,"methods":args.methods,"smoothing_sigma_x":args.sigma,
           "preflight":preflight(),"variants":records,"runner_sha256":sha(Path(__file__)),
           "elapsed_seconds":time.perf_counter()-start,"gpu_calls":0,"model_weights_loaded":False,
           "no_geometric_warp_or_manual_line_geometry":True,"quality_verdict":"pending direct image inspection"}
    (out/"direct_compositor_metrics.json").write_text(json.dumps(suite,indent=2)+"\n")
    print(json.dumps({"status":"success","output":str(out),"preview":str(out/"full-comparison.png"),
                      "composites":len(variants),"elapsed_seconds":suite["elapsed_seconds"]}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track",choices=("track1","track2","track3"),default="track3")
    parser.add_argument("--raw",type=Path,default=DEFAULT_RAW)
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--radii",type=int,nargs="+",default=[32,64,96])
    parser.add_argument("--methods",choices=("harmonic","hermite","biharmonic","profile"),nargs="+",default=["harmonic","hermite","biharmonic"])
    parser.add_argument("--sigma",type=float,default=4)
    parser.add_argument("--value-cap",type=float,default=48)
    parser.add_argument("--gradient-cap",type=float,default=.35)
    parser.add_argument("--preflight",action="store_true")
    args=parser.parse_args()
    if args.preflight:
        cv2.setNumThreads(1)
        print(json.dumps(preflight()))
    else:
        if any(not 4<=radius<=320 for radius in args.radii):
            raise ValueError("Finite strip radius must remain between4 and320")
        run(args)
