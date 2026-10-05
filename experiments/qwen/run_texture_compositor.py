#!/usr/bin/env python3
"""CPU-only local texture-band finishing; derived images, never model raw.

Original source and distant exterior are copied exactly. Existing generated
texture bands receive locally estimated source/exterior gains, with fading and
source-connected/strong-edge protection. No source texture pixels are copied,
no coordinates are warped and no semantic content is generated here.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import run_direct_boundary_compositor as direct
import run_boundary_compositor as frozen

ROOT=Path(__file__).resolve().parents[2]
RECT=(0,320,512,832)
DEFAULT_RAW=direct.DEFAULT_RAW
DEFAULT_OUT=ROOT/"experiments/qwen/boundary_compositor_runs/2026-10-05/track3-top32-texture"
BANDS=("fine2-4","middle4-8","coarse8-16","coarse16-32")


def gray(rgb):
    return np.einsum("hwc,c->hw",rgb.astype(np.float32),np.array([.2126,.7152,.0722],np.float32))


def spectral_bands(luminance):
    """Even reflection makes FFT filtering nonperiodic at original borders.

    Raised-cosine radial crossovers give a partition of high-pass energy.
    Frequencies below the32px crossover retain their original base. Spatial
    gains/masks can change global Fourier phase; no phase-preservation claim.
    """
    height,width=luminance.shape
    reflected=np.concatenate([luminance,luminance[:,::-1]],axis=1)
    reflected=np.concatenate([reflected,reflected[::-1]],axis=0)
    transform=np.fft.rfft2(reflected)
    fy=np.fft.fftfreq(2*height)[:,None]
    fx=np.fft.rfftfreq(2*width)[None,:]
    frequency=np.sqrt(fx*fx+fy*fy)
    low=[]
    for cutoff in (.25,.125,.0625,.03125):
        octaves=np.log2(np.maximum(frequency,1e-30)/cutoff)
        progress=np.clip((octaves+.25)/.5,0,1)
        low.append(.5*(1+np.cos(np.pi*progress)))
    responses=[1-low[0],low[0]-low[1],low[1]-low[2],low[2]-low[3]]
    arrays=[np.fft.irfft2(transform*response,s=reflected.shape)[:height,:width].astype(np.float32)
            for response in responses]
    return np.stack(arrays),{"reflection":"even half-sample horizontal+vertical; no wrap at source edge",
                             "crossover_frequencies_cycles_per_pixel":[.25,.125,.0625,.03125],
                             "crossover_transition_octaves":.5,"base_below32px_crossover_unchanged":True}


def image_edges(rgb):
    y=gray(rgb)
    smooth=cv2.GaussianBlur(y,(0,0),.8)
    dx=cv2.Sobel(smooth,cv2.CV_32F,1,0,ksize=3)/8
    dy=cv2.Sobel(smooth,cv2.CV_32F,0,1,ksize=3)/8
    strength=np.sqrt(dx*dx+dy*dy)
    canny=cv2.Canny(np.clip(np.rint(smooth),0,255).astype(np.uint8),40,80)
    tangential_fraction=np.abs(dx)/np.maximum(strength,1e-6)
    return strength,canny>0,tangential_fraction


def protection(raw,source,edge_threshold=8):
    """Generic strong-edge core, plus source-connected edge components.

    Source seeds come from long connected Canny components touching its first
    or last8 known rows. Closing/dilation permits a few-pixel codec gap at the
    source join. No object/rail coordinates or source tangent are supplied.
    """
    composite=frozen.compose(raw,source)
    strength,edges,tangential=image_edges(composite)
    # Derive exterior gradients without convolving across the hard-pasted
    # source seam, so the seam itself is not mistaken for an object edge.
    for start,segment in ((0,raw[:320]),(832,raw[832:])):
        local_strength,local_edges,local_tangential=image_edges(segment)
        strength[start:start+320]=local_strength
        edges[start:start+320]=local_edges
        tangential[start:start+320]=local_tangential
    _,source_edges,source_tangential=image_edges(source)
    # Boundary-parallel tonal artifacts must not form a graph bridge from a
    # real crossing object to the entire source seam. Crossing edges have a
    # tangential-x gradient component; protect those in the immediate join.
    crossing_ok=np.ones(edges.shape,bool)
    crossing_ok[312:328]=tangential[312:328]>=.25
    crossing_ok[824:840]=tangential[824:840]>=.25
    edges&=crossing_ok
    count,labels,stats,_=cv2.connectedComponentsWithStats(source_edges.astype(np.uint8),8)
    seeds=np.zeros(composite.shape[:2],np.uint8)
    corridors=np.zeros(composite.shape[:2],bool)
    source_fits=[]
    accepted=[]
    for label in range(1,count):
        component=labels==label
        if stats[label,cv2.CC_STAT_AREA]<24:
            continue
        crosses_top=component[:8].any() and component[16:48].any()
        crosses_bottom=component[-8:].any() and component[-48:-16].any()
        if crosses_top or crosses_bottom:
            seed=component.copy()
            seed[8:-8]=False
            seed&=source_tangential>=.25
            seeds[320:832]|=seed.astype(np.uint8)
            accepted.append(int(label))
            # A graph can leak along a boundary-parallel artifact. Bound its
            # exterior support to a generic source-derived crossing corridor;
            # this is a protection mask, never an image warp or painted line.
            for side,crosses,rows in (("top",crosses_top,component[:48]),("bottom",crosses_bottom,component[-48:][::-1])):
                if not crosses:continue
                ys=[];xs=[]
                for y,row in enumerate(rows):
                    points=np.flatnonzero(row)
                    if len(points):ys.append(y);xs.append(float(np.median(points)))
                if len(ys)<12 or max(ys)-min(ys)<16:continue
                fit=np.polyfit(ys,xs,1)
                error=float(np.median(np.abs(np.asarray(xs)-np.polyval(fit,ys))))
                if error>2.5 or abs(fit[0])>6:continue
                distances=np.arange(1,321,dtype=np.float32)
                center=fit[1]-fit[0]*distances
                strip=np.abs(np.arange(512)[None,:]-center[:,None])<=12
                if side=="top":corridors[:320]|=strip[::-1]
                else:corridors[832:]|=strip
                source_fits.append({"side":side,"source_component":int(label),"x_per_inward_y":float(fit[0]),
                                    "boundary_x":float(fit[1]),"median_fit_error_px":error,"protection_halfwidth_px":12})
    # Dilation connects adjacent source/exterior edge fragments without
    # modifying image pixels. Strong edges elsewhere remain protected too.
    dilated=cv2.dilate(edges.astype(np.uint8),np.ones((7,7),np.uint8))
    _,joint_labels=cv2.connectedComponents(dilated,8)
    seed_labels=np.unique(joint_labels[seeds>0])
    seed_labels=seed_labels[seed_labels!=0]
    connected=np.isin(joint_labels,seed_labels)
    connected[:320]&=corridors[:320]
    connected[832:]&=corridors[832:]
    strong=((strength>=edge_threshold)&crossing_ok)|connected
    core=cv2.dilate(strong.astype(np.uint8),np.ones((7,7),np.uint8))>0
    distance=cv2.distanceTransform((~core).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    u=np.clip(distance/8,0,1)
    permission=(3*u*u-2*u*u*u).astype(np.float32)
    permission[320:832]=0
    meta={"source_edge_seed_components":accepted,"source_connected_joint_components":seed_labels.tolist(),
          "source_connected_pixel_count":int(connected.sum()),"strong_gradient_threshold_255_per_pixel":edge_threshold,
          "exact_edge_core_dilation_pixels":3,"soft_edge_fade_pixels":8,"canny_thresholds":[40,80],
          "minimum_source_component_pixels":24,"source_border_crossing_inward_depth_pixels":16,
          "boundary_parallel_seam_graph_exclusion_halfwidth_pixels":8,
          "minimum_boundary_tangential_gradient_fraction":.25,
          "exterior_gradient_context":"generated-only; no source-paste convolution",
          "source_crossing_protection_fits":source_fits}
    return permission,core,connected,meta


def weighted_power(values,allowed):
    samples=values[allowed]
    if len(samples)<64:
        return None
    # Trim extreme residual values to reduce leakage from unrecognized edges.
    limit=np.quantile(np.abs(samples),.95)
    return float(np.mean(np.clip(samples,-limit,limit)**2))


def local_gains(source_bands,generated_bands,source_ok,generated_ok,edge):
    width=source_bands.shape[-1]
    starts=list(range(0,width-64+1,32))
    centers=np.array([x+31.5 for x in starts],np.float64)
    powers=[]
    columns=[]
    for start in starts:
        src_y=slice(8,56) if edge=="top" else slice(-56,-8)
        gen_y=slice(-56,-8) if edge=="top" else slice(8,56)
        source_patch_mask=source_ok[src_y,start:start+64]
        generated_patch_mask=generated_ok[gen_y,start:start+64]
        row=[]
        info=[]
        for band in range(4):
            source_power=weighted_power(source_bands[band,src_y,start:start+64],source_patch_mask)
            generated_power=weighted_power(generated_bands[band,gen_y,start:start+64],generated_patch_mask)
            if source_power is None or generated_power is None:
                gain=1.0
                reason="insufficient unprotected texture pixels"
            else:
                gain=float(np.clip(np.sqrt((source_power+1e-6)/(generated_power+1e-6)),.6,1.6))
                reason="source/exterior local power; clipped gain0.6..1.6"
            row.append(gain)
            info.append({"band":BANDS[band],"source_power":source_power,"generated_power":generated_power,"gain":gain,"reason":reason})
        columns.append(row)
        powers.append({"x_half_open":[start,start+64],"source_valid_pixels":int(source_patch_mask.sum()),
                       "generated_valid_pixels":int(generated_patch_mask.sum()),"bands":info})
    samples=np.asarray(columns,dtype=np.float32)
    gains=np.stack([np.interp(np.arange(width),centers,samples[:,band]) for band in range(4)]).astype(np.float32)
    gains=np.stack([cv2.GaussianBlur(row[None],(0,0),sigmaX=12,sigmaY=0)[0] for row in gains])
    return gains,powers


def apply_texture(raw,source,banks,gains,permission,strength,radius,band_set):
    work=raw.astype(np.float32).copy()
    selected={"all":(0,1,2,3),"fine":(0,1),"middle":(1,2)}[band_set]
    statistics={}
    for edge,start,segment,allowed in (("top",0,raw[:320],permission[:320]),
                                      ("bottom",832,raw[832:],permission[832:])):
        bands=banks[edge]
        delta=sum((gains[edge][band][None,:]-1)*bands[band] for band in selected)*strength
        distance=np.arange(320,0,-1,dtype=np.float32) if edge=="top" else np.arange(1,321,dtype=np.float32)
        u=np.clip((distance-1)/(radius-1),0,1)
        taper=1-3*u*u+2*u*u*u
        taper[distance>=radius]=0
        delta*=allowed*taper[:,None]
        before=work[start:start+320].copy()
        work[start:start+320]+=delta[...,None]
        statistics[edge]={"delta_abs_p95_255":float(np.quantile(np.abs(delta),.95)),
                          "delta_max_abs_255":float(np.abs(delta).max()),
                          "pixels_changed_before_rounding":int(np.sum(np.abs(delta)>1e-4)),
                          "protected_core_delta_max":float(np.abs(delta[allowed==0]).max()) if np.any(allowed==0) else 0,
                          "clipped_channels_before_uint8":int(np.sum((work[start:start+320]<0)|(work[start:start+320]>255)))}
    return work,statistics


def finish(raw,source,permission):
    # This optional color layer shares the identical edge protection. Its
    # source-connected metal core is restored from the current texture input.
    profile,_=direct.apply_direct(raw,source,32,"profile",16)
    delta=profile.astype(np.float32)-raw
    output=raw+delta*permission[...,None]
    return output


def setup(raw,source,edge_threshold=8):
    permission,core,connected,meta=protection(raw,source,edge_threshold)
    source_bands,bank_meta=spectral_bands(gray(source))
    source_strength,_,_=image_edges(source)
    source_core=cv2.dilate((source_strength>=edge_threshold).astype(np.uint8),np.ones((7,7),np.uint8))>0
    source_ok=~source_core
    banks={}
    gains={}
    powers={}
    for edge,segment,allowed in (("top",raw[:320],permission[:320]),("bottom",raw[832:],permission[832:])):
        banks[edge],_=spectral_bands(gray(segment))
        gains[edge],powers[edge]=local_gains(source_bands,banks[edge],source_ok,allowed>.8,edge)
    return permission,core,connected,meta,banks,gains,powers,bank_meta


def preflight():
    # Neumann cosine eigenfunctions have known frequencies, so filtered
    # response identity is independent of source/model image statistics.
    x=np.arange(512,dtype=np.float32)
    frequencies=(.375,.1875,.09375,.046875)
    for selected,frequency in enumerate(frequencies):
        signal=np.cos(2*np.pi*frequency*(x+.5))
        image=np.broadcast_to(signal,(128,512)).copy()
        bands,_=spectral_bands(image)
        energies=np.mean(bands**2,axis=(1,2))
        if int(np.argmax(energies))!=selected or float(energies[selected]/energies.sum())<.995:
            raise AssertionError("Known cosine entered wrong spectral band")
        if not np.allclose(bands[:, :, ::-1],spectral_bands(image[:,::-1].copy())[0],atol=2e-5):
            raise AssertionError("Spectral filtering failed horizontal mirror oracle")
    # Synthetic texture has excess middle-frequency contrast and deficient
    # finest grain. Verify estimated gains have the expected opposite signs.
    fine=np.cos(2*np.pi*.375*(x+.5))
    middle=np.cos(2*np.pi*.1875*(x+.5))
    src_y=np.broadcast_to(120+8*fine+3*middle,(512,512)).copy()
    gen_y=np.broadcast_to(120+3*fine+8*middle,(320,512)).copy()
    src_bands,_=spectral_bands(src_y)
    gen_bands,_=spectral_bands(gen_y)
    estimated,_=local_gains(src_bands,gen_bands,np.ones((512,512),bool),np.ones((320,512),bool),"top")
    if float(estimated[0].mean())<1.3 or float(estimated[1].mean())>.8:
        raise AssertionError("Texture-gain frequency direction oracle failed")
    raw=np.repeat(gen_y[:,:,None],3,axis=2)
    full=np.full((1152,512,3),120,dtype=np.float32)
    full[:320]=raw
    full[832:]=raw
    source=np.repeat(np.clip(np.rint(src_y),0,255).astype(np.uint8)[:,:,None],3,axis=2)
    p,core,connected,meta,banks,gains,powers,_=setup(full,source)
    modified,_=apply_texture(full,source,banks,gains,p,.5,128,"all")
    output=frozen.compose(modified,source)
    baseline=frozen.compose(full,source)
    if not np.array_equal(output[320:832],source) or not np.array_equal(output[:192],baseline[:192]) or not np.array_equal(output[960:],baseline[960:]):
        raise AssertionError("Source/far-exterior preservation failed")
    # A sharp diagonal source-connected step exercises geometric edge safety.
    yy,xx=np.indices((1152,512))
    diagonal=np.where(xx+yy*.6>510,190,60).astype(np.float32)
    edge_raw=np.repeat(diagonal[:,:,None],3,axis=2)
    edge_source=edge_raw[320:832].astype(np.uint8)
    p,core,connected,meta,banks,gains,powers,_=setup(edge_raw,edge_source)
    modified,_=apply_texture(edge_raw,edge_source,banks,gains,p,.75,128,"all")
    color_finished=finish(modified,edge_source,p)
    output=frozen.compose(color_finished,edge_source)
    baseline=frozen.compose(edge_raw,edge_source)
    exterior=np.ones(core.shape,bool);exterior[320:832]=False
    protected=core&exterior
    if not protected.any() or not np.array_equal(output[protected],baseline[protected]):
        raise AssertionError("Source-connected sharp-edge core changed")
    # Same reflected texture statistics are a gain1 fixed point. This avoids
    # changing an already matched source/exterior texture distribution.
    rng=np.random.default_rng(616)
    tex=rng.uniform(100,140,(512,512,3)).astype(np.float32)
    ref=np.clip(np.rint(tex),0,255).astype(np.uint8)
    perfect=np.full((1152,512,3),120,dtype=np.float32)
    perfect[:320]=ref[:320][::-1]
    perfect[832:]=ref[-320:][::-1]
    p,_,_,_,banks,gains,_,_=setup(perfect,ref)
    if max(float(np.abs(gains[e]-1).max()) for e in gains)>.025:
        raise AssertionError("Matched reflected texture distribution did not yield gain1")
    return {"status":"passed","cosine_frequency_band_identity":True,"spectral_horizontal_mirror":True,
            "opposite_fine_middle_gain_direction":True,"source_and128px_far_exterior_exact":True,
            "sharp_source_connected_edge_exact_after_texture_and_color":True,"matched_reflected_texture_gain1":True,
            "no_model_or_gpu":True}


def previews(out,variants):
    font=ImageFont.load_default(size=15)
    full=Image.new("RGB",(256*len(variants),624),"#161a1e")
    draw=ImageDraw.Draw(full)
    for i,(name,image) in enumerate(variants.items()):
        picture=Image.fromarray(image)
        draw.text((256*i+5,8),name,fill="white",font=font)
        full.paste(picture.resize((256,576),Image.Resampling.LANCZOS),(256*i,36))
        for label,rect in {"water-top":(0,256,512,384),"asphalt-bottom":(0,768,512,896),"rail-top":(285,270,405,365)}.items():
            crop=picture.crop(rect)
            crop.save(out/name/f"{label}-1x.png")
            crop.resize((crop.width*4,crop.height*4),Image.Resampling.NEAREST).save(out/name/f"{label}-4x.png")
    full.save(out/"full-comparison.png")


def run(args):
    cv2.setNumThreads(1)
    start=time.perf_counter()
    raw_path=args.raw.resolve();out=args.output.resolve()
    if (out/"texture_metrics.json").exists():raise FileExistsError(out)
    shared=json.loads((ROOT/"experiments/evaluation/inputs/inputs.json").read_text())
    track=next(row for row in shared["tracks"] if row["id"]==args.track)
    source_path=Path(track["source_512"])
    source=np.asarray(Image.open(source_path).convert("RGB"),dtype=np.uint8)
    raw=np.asarray(Image.open(raw_path).convert("RGB"),dtype=np.float32)
    if raw.shape!=(1152,512,3) or source.shape!=(512,512,3):raise ValueError("Expected shared canvas/source geometry")
    before={str(p):direct.sha(p) for p in (raw_path,source_path,Path(track["original_jpg"]),Path(frozen.__file__),Path(direct.__file__))}
    out.mkdir(parents=True,exist_ok=True)
    p,core,connected,guard,banks,gains,powers,bank_meta=setup(raw,source,args.edge_gradient_threshold)
    Image.fromarray(np.rint(p*255).astype(np.uint8)).save(out/"texture-permission-mask.png")
    Image.fromarray(core.astype(np.uint8)*255).save(out/"exact-edge-core.png")
    Image.fromarray(connected.astype(np.uint8)*255).save(out/"source-connected-edges.png")
    variants={"baseline":frozen.compose(raw,source),"protected-profile16":frozen.compose(finish(raw,source,p),source)}
    records={"baseline":{"radius_pixels":128,"kind":"exact-source model raw composite"},
             "protected-profile16":{"radius_pixels":32,"kind":"source-connected protected color control"}}
    configurations=[(.25,64),(.5,128)]+([(1.0,128)] if args.full_strength else [])
    for band_set in ("all","fine","middle"):
        for strength,radius in configurations:
            name=f"{band_set}-s{int(strength*100)}-r{radius}"
            work,details=apply_texture(raw,source,banks,gains,p,strength,radius,band_set)
            variants[name]=frozen.compose(work,source)
            records[name]={"radius_pixels":radius,"strength":strength,"bands":band_set,"edge_statistics":details,
                           "kind":"derived CPU texture-band composite; not model raw"}
            if band_set=="all":
                finished_name=name+"-color"
                variants[finished_name]=frozen.compose(finish(work,source,p),source)
                records[finished_name]={**records[name],"protected_profile16_color":True}
    exterior=np.ones(core.shape,bool);exterior[320:832]=False
    protected=core&exterior
    for name,image in variants.items():
        directory=out/name;directory.mkdir(exist_ok=True)
        path=directory/"composite.png";Image.fromarray(image).save(path)
        radius=records[name]["radius_pixels"]
        record=records[name]
        record.update({"source_exact":bool(np.array_equal(image[320:832],source)),
                       "far_exterior_exact":bool(np.array_equal(image[:320-radius],variants["baseline"][:320-radius]) and np.array_equal(image[832+radius:],variants["baseline"][832+radius:])),
                       "protected_edge_core_exact":bool(np.array_equal(image[protected],variants["baseline"][protected])),
                       "composite_path":str(path),"composite_sha256":direct.sha(path),
                       "generated_mean_abs_change_255":float(np.abs(image.astype(float)-variants["baseline"].astype(float))[exterior].mean()),
                       "no_source_texture_pixel_copy_or_geometric_warp":True})
        if not record["source_exact"] or not record["far_exterior_exact"] or not record["protected_edge_core_exact"]:raise AssertionError(name)
        (directory/"metrics.json").write_text(json.dumps(record,indent=2)+"\n")
    previews(out,variants)
    after={name:direct.sha(Path(name)) for name in before}
    if before!=after:raise AssertionError("Input or frozen helper changed")
    suite={"status":"success","kind":"CPU local frequency-band finishing diagnostic","track":args.track,
           "raw_input":str(raw_path),"source_input":str(source_path),"source_rect_xyxy":list(RECT),
           "input_sha256_before":before,"input_sha256_after":after,"all_inputs_and_frozen_helpers_unchanged":True,
           "guard":guard,"exact_protected_exterior_pixels":int(protected.sum()),"spectral_bank":bank_meta,
           "local_gain_patches":powers,"gain_range":[.6,1.6],"gain_smoothing_sigma_x":12,
           "radii_pixels":[64,128],"strengths":sorted({strength for strength,radius in configurations}),"variants":records,
           "elapsed_seconds":time.perf_counter()-start,"runner_sha256":direct.sha(Path(__file__)),
           "preflight":preflight(),"gpu_calls":0,"source_texture_pixels_copied":False,
           "quality_verdict":"pending direct fullframe/crop and frozen feature review; spectral metric is not acceptance"}
    (out/"texture_metrics.json").write_text(json.dumps(suite,indent=2)+"\n")
    print(json.dumps({"status":"success","output":str(out),"preview":str(out/"full-comparison.png"),"variants":len(variants),"elapsed_seconds":suite["elapsed_seconds"],"protected_exterior_pixels":int(protected.sum())}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track",choices=("track1","track2","track3"),default="track3")
    parser.add_argument("--raw",type=Path,default=DEFAULT_RAW)
    parser.add_argument("--output",type=Path,default=DEFAULT_OUT)
    parser.add_argument("--edge-gradient-threshold",type=float,default=8)
    parser.add_argument("--full-strength",action="store_true")
    parser.add_argument("--preflight",action="store_true")
    args=parser.parse_args()
    cv2.setNumThreads(1)
    if args.preflight:print(json.dumps(preflight()))
    else:run(args)
