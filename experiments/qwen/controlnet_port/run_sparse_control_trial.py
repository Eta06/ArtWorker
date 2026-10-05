"""Matched experimental sparse structural-control teacher, full-canvas compute.

VAE-encode the complete map, then retain first64 control channels on known ROI
and optionally an input-guide support. Other structural latents are literal0;
mask/source65 channels stay exact. This spatial dropout is NOT the official
published global-control dropout. Zero input does not imply zero learned hints.
No installed runtime or preceding runner is modified. CPU preflight loads no
weights. Parent alone schedules GPU trials.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from run_trial import MODEL,REVISION
from control import CHECKPOINT_BYTES,CHECKPOINT_FILENAME,CHECKPOINT_REVISION,UPSTREAM_REVISION,checkpoint_header


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--track",choices=("track1","track2","track3"),default="track3")
    parser.add_argument("--steps",type=int,choices=(20,40),default=20)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--quantize",type=int,choices=(4,8),default=4)
    parser.add_argument("--control-quantize",type=int,choices=(4,8))
    parser.add_argument("--modes",nargs="+",choices=("mask-only","source-supported","tangent-supported"),default=["source-supported","tangent-supported"])
    parser.add_argument("--source-canny",type=Path)
    parser.add_argument("--tangent-canny",type=Path)
    parser.add_argument("--extrapolated-guide",type=Path)
    parser.add_argument("--support-dilation",type=int,default=64)
    parser.add_argument("--control-scale",type=float,default=1.0)
    parser.add_argument("--prompt",help="Complete text override; unchanged baseline prompt is default")
    parser.add_argument("--checkpoint",type=Path,default=ROOT/".build/models/qwen/controlnet_union"/CHECKPOINT_FILENAME)
    parser.add_argument("--known-bridge",action="store_true",help="Extra hard known-latent flow bridge; not untouched official solver")
    parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if len(set(args.modes))!=len(args.modes): parser.error("Modes must be unique")
    if not 0 <= args.control_scale <= 10: parser.error("Control scale must be finite and in[0,10]")
    if not 0 <= args.support_dilation <= 1152: parser.error("Support dilation must be in[0,1152]")
    suffix="-knownbridge" if args.known_bridge else ""
    suite_dir=args.output or ROOT/f"experiments/qwen/controlnet_runs/2026-10-05/{args.track}-sparse{args.steps}{suffix}"
    if not args.preflight and (suite_dir/"controlnet_metrics.json").exists():
        raise FileExistsError(f"Refusing to overwrite prior trial: {suite_dir}")
    suite_dir.mkdir(parents=True,exist_ok=True)
    begun=time.perf_counter()
    suite={"status":"preflight","track":args.track,"seed":args.seed,"steps":args.steps,
           "runs":{},"phases":{},"runner_sha256":file_sha256(__file__),
           "port_sha256":file_sha256(Path(__file__).with_name("control.py")),
           "model_repo":"Qwen/Qwen-Image-2.1","model_revision":REVISION,
           "control_repo":"alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union",
           "control_revision":CHECKPOINT_REVISION,"upstream_revision":UPSTREAM_REVISION,
           "source_conditioning":"text-only prefix; source in trained masked-image control channels",
           "compute_mode":"full canvas every step in both base and control chains",
           "growing_compute":False,"prefix_cache_enabled":False,"cfg":1.0,
           "adapter":None,"base_quantization":args.quantize,"control_quantization":args.control_quantize,
           "known_bridge_extension":args.known_bridge,"control_scale":args.control_scale,
           "sparse_conditioning_sha256":file_sha256(Path(__file__).with_name("sparse_conditioning.py")),
           "experimental_sparse_control":True,
           "sparse_semantics":"after full map VAE encoding, input control64 retained on support; no zero-hint claim"}

    def write(name="controlnet_metrics.json"):
        suite["elapsed_seconds"]=time.perf_counter()-begun
        suite["process_peak_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (suite_dir/name).write_text(json.dumps(suite,indent=2))

    def log(phase,**data):
        suite["phase"]=phase;suite.update(data);write()
        print(json.dumps({"phase":phase,"elapsed_seconds":round(suite["elapsed_seconds"],2),**data}),flush=True)

    try:
        import mlx.core as mx
        import numpy as np
        from PIL import Image
        from mlx.utils import tree_flatten
        from mflux.models.common.config.config import Config
        from mflux.models.common.vae.vae_util import VAEUtil
        from mflux.models.common.vae.tiling_config import TilingConfig
        from mflux.models.qwen21.reference import QwenImage21Edit
        from mflux.models.qwen21.reference.latent_creator.qwen_image21_latent_creator import QwenImage21LatentCreator
        from mflux.models.qwen21.reference.model.qwen_image21_transformer.layout import QwenImage21Layout
        from control import controlled_forward,load_control_branch
        from conditioning import encode_control_context,cpu_preflight
        from sparse_conditioning import prepare_sparse_support,apply_sparse_structural_context,cpu_preflight as sparse_cpu_preflight

        if args.preflight: mx.set_default_device(mx.cpu)
        shared=json.loads((ROOT/"experiments/evaluation/inputs/inputs.json").read_text())
        track=next(item for item in shared["tracks"] if item["id"]==args.track)
        width,height=shared["canvas_size"]
        rect=shared["source_rect_xyxy"]
        source=Image.open(track["source_512"]).convert("RGB")
        if (width,height)!=(512,1152) or source.size!=(512,512) or rect!=[0,320,512,832]:
            raise ValueError("Expected unchanged shared512-square centered in512x1152")
        map_dir=ROOT/f"experiments/qwen/controlnet_inputs/2026-10-05/{args.track}"
        source_map=args.source_canny or map_dir/"canny_known.png"
        tangent_map=args.tangent_canny or map_dir/"canny_extrapolated.png"
        guide_path=args.extrapolated_guide or map_dir/"extrapolated_only.png"
        map_paths={"mask-only":None,"source-supported":source_map,"tangent-supported":tangent_map}
        missing=[str(map_paths[mode]) for mode in args.modes if map_paths[mode] is not None and not map_paths[mode].is_file()]
        if "tangent-supported" in args.modes and not guide_path.is_file(): missing.append(str(guide_path))
        if not args.checkpoint.is_file() or args.checkpoint.stat().st_size!=CHECKPOINT_BYTES:
            missing.append(str(args.checkpoint))
        suite.update({"width":width,"height":height,"source_rect_xyxy":rect,
                      "source_512_path":track["source_512"],"source_512_sha256":file_sha256(track["source_512"]),
                      "map_paths":{mode:str(path) if path else None for mode,path in map_paths.items()},
                      "guide_path":str(guide_path),
                      "missing_files":missing})
        supports={};support_info={}
        for mode in args.modes:
            if mode=="tangent-supported" and not guide_path.is_file(): continue
            guide_rgb=Image.open(guide_path).convert("RGB") if mode=="tangent-supported" else None
            pixels,packed,info=prepare_sparse_support(np,Image,width,height,rect,guide_rgb,args.support_dilation)
            directory=suite_dir/mode;directory.mkdir(parents=True,exist_ok=True)
            support_path=directory/"support.png"
            Image.fromarray(pixels.astype(np.uint8)*255,mode="L").save(support_path)
            np.save(directory/"support_tokens.npy",packed)
            info.update({"support_path":str(support_path),"support_png_sha256":file_sha256(support_path),
                         "guide_file_sha256":file_sha256(guide_path) if guide_rgb is not None else None})
            (directory/"support.json").write_text(json.dumps(info,indent=2))
            supports[mode]=packed;support_info[mode]=info
        suite["sparse_support"]=support_info
        if args.checkpoint.is_file() and args.checkpoint.stat().st_size==CHECKPOINT_BYTES:
            header,header_sha=checkpoint_header(args.checkpoint)
            suite["checkpoint_header"]={"sha256":header_sha,"tensor_count":len(header)-int("__metadata__" in header),
                                         "payload_bytes":max(value["data_offsets"][1] for key,value in header.items() if key!="__metadata__")}
        if args.preflight:
            suite["conditioning_cpu_preflight"]=cpu_preflight()
            suite["sparse_cpu_preflight"]=sparse_cpu_preflight()
            parity_path=Path(__file__).with_name("cpu_parity.json")
            if parity_path.is_file():
                suite["port_cpu_parity"]=json.loads(parity_path.read_text())
            suite.update({"status":"preflight_files_ready" if not missing else "preflight_files_incomplete",
                          "model_weights_loaded":False,"device":"CPU",
                          "planned_nfe_per_arm":args.steps,"planned_base_block_forwards_per_arm":args.steps*32,
                          "planned_control_block_forwards_per_arm":args.steps*16,
                          "planned_target_token_forwards_per_arm":args.steps*(width//16)*(height//16)})
            write("preflight.json");print(json.dumps(suite,indent=2),flush=True);return
        if missing: raise FileNotFoundError(f"Trial files incomplete: {missing}")
        parity_path=Path(__file__).with_name("cpu_parity.json")
        if not parity_path.is_file(): raise RuntimeError("Run the CPU parity gate before loading checkpoint weights")
        parity=json.loads(parity_path.read_text())
        if parity.get("status")!="passed" or parity.get("control_port_sha256")!=suite["port_sha256"]:
            raise RuntimeError("Control port changed or lacks a matching passed CPU parity result")
        suite["port_cpu_parity_summary"]={"status":parity["status"],"max_abs_error":parity["max_abs_error"],
                                         "threshold":parity["threshold"],"path":str(parity_path)}

        descriptions=json.loads((ROOT/"experiments/flux2/prompts.json").read_text())
        prompt="One coherent portrait music album artwork. "+descriptions[args.track]
        if args.prompt is not None: prompt=args.prompt
        suite["prompt"]=prompt
        mx.set_cache_limit(512*1024**2);mx.set_memory_limit(28*1024**3)
        suite["metal_device"]=mx.device_info();suite["mlx_version"]=mx.__version__
        log("loading_base",status="running")
        start=time.perf_counter()
        model=QwenImage21Edit(model_path=str(MODEL),quantize=args.quantize,lora_paths=None)
        suite["phases"]["base_load_seconds"]=time.perf_counter()-start
        suite["base_load_mlx_peak_bytes"]=mx.get_peak_memory();mx.reset_peak_memory()
        log("text_conditioning")
        start=time.perf_counter()
        prompt_embeds,slots=model._encode_prompt(prompt,[])
        mx.eval(prompt_embeds)
        if bool(mx.any(slots)): raise AssertionError("Official control teacher must not have image-reference slots")
        del model.text_encoder;model.text_encoder=None;mx.clear_cache()
        suite["phases"]["text_conditioning_seconds"]=time.perf_counter()-start
        suite["text_encoder_deleted_before_control_load"]=True
        log("loading_control")
        start=time.perf_counter()
        branch,branch_info=load_control_branch(args.checkpoint,quantize=args.control_quantize)
        suite["control_checkpoint"]=branch_info
        suite["phases"]["control_load_seconds"]=time.perf_counter()-start
        suite["base_component_parameter_bytes"]={name:sum(value.nbytes for _,value in tree_flatten(getattr(model,name).parameters())) for name in ("transformer","vae")}
        suite["control_load_mlx_peak_bytes"]=mx.get_peak_memory();mx.reset_peak_memory()
        mx.set_memory_limit(25*1024**3)
        log("control_conditioning")
        start=time.perf_counter()
        contexts={};context_info={}
        for mode in args.modes:
            control_rgb=Image.open(map_paths[mode]).convert("RGB") if map_paths[mode] is not None else None
            if control_rgb is not None and control_rgb.size!=(width,height):
                raise ValueError("Control map must preserve finalcanvas aspect and geometry")
            context,info=encode_control_context(mx,np,Image,model.vae,QwenImage21LatentCreator,
                                               source,rect,width,height,control_rgb=control_rgb)
            context=context.astype(prompt_embeds.dtype);mx.eval(context)
            if context.shape!=(1,(height//16)*(width//16),129): raise AssertionError("Invalid129-channel control layout")
            baseline=np.asarray(context.astype(mx.float32)).copy()
            info["baseline_context_sha256_float32"]=hashlib.sha256(baseline.tobytes()).hexdigest()
            context=apply_sparse_structural_context(mx,np,context,supports[mode]);mx.eval(context)
            gated=np.asarray(context.astype(mx.float32))
            support_ids=supports[mode].reshape(-1)
            if not np.array_equal(gated[:,:,64:],baseline[:,:,64:]):
                raise AssertionError("Sparse control modified known mask or masked-source65 channels")
            if not np.array_equal(gated[:,support_ids,:64],baseline[:,support_ids,:64]):
                raise AssertionError("Sparse control changed retained control64 values")
            if np.any(gated[:,~support_ids,:64]!=0):
                raise AssertionError("Unsupported structural control64 must be literal0")
            info["sparse_support"]=support_info[mode]
            info["source65_exact_against_full_map_baseline"]=True
            info["supported_control64_exact_against_full_map_baseline"]=True
            info["unsupported_control64_literal_zero"]=True
            info["source65_sha256_float32"]=hashlib.sha256(gated[:,:,64:].tobytes()).hexdigest()
            info["map_sha256"]=file_sha256(map_paths[mode]) if map_paths[mode] is not None else None
            info["context_sha256_float32"]=hashlib.sha256(np.asarray(context.astype(mx.float32)).tobytes()).hexdigest()
            contexts[mode]=context;context_info[mode]=info
        source_contexts=[np.asarray(value[:,:,64:].astype(mx.float32)) for value in contexts.values()]
        if any(not np.array_equal(value,source_contexts[0]) for value in source_contexts[1:]):
            raise AssertionError("Matched modes changed known mask or masked-source codec conditioning")
        suite["known_mask_and_source_context_matched"]=True
        suite["control_conditioning"]=context_info
        suite["phases"]["control_conditioning_seconds"]=time.perf_counter()-start
        layout=QwenImage21Layout.create(slots,[(1,height//16,width//16)],model.transformer.axes)
        full_tokens=layout.target_tokens
        config=Config(model_config=model.model_config,num_inference_steps=args.steps,width=width,height=height,guidance=1.0)
        sigmas=config.scheduler.sigmas
        suite["sigmas"]=sigmas.tolist();suite["joint_prefix_tokens"]=layout.prefix_length
        mx.random.seed(args.seed)
        noise=mx.random.normal((1,full_tokens,64)).astype(prompt_embeds.dtype);mx.eval(noise)
        suite["noise_sha256"]=hashlib.sha256(np.asarray(noise.astype(mx.float32)).tobytes()).hexdigest()
        outputs={}

        for mode in args.modes:
            directory=suite_dir/mode;directory.mkdir(parents=True,exist_ok=True)
            context=contexts[mode]
            known_mask=context[:,:,64:65]>0.5
            known_clean=context[:,:,-64:]
            latents=noise
            record={"status":"running","mode":mode,"steps":args.steps,"seed":args.seed,
                    "transformer_calls":0,"base_block_calls":0,"control_block_calls":0,
                    "target_token_forward_sum":0,"step_records":[],"source_conditioning":suite["source_conditioning"],
                    "context_metadata":context_info[mode],"cfg":1.0,"known_bridge_extension":args.known_bridge,
                    "diagnostic_control":True,"shared_canvas_comparable":False,
                    "prefix_cache_enabled":False,"compute_mode":suite["compute_mode"],"growing_compute":False,
                    "noise_sha256":suite["noise_sha256"],"sigmas":suite["sigmas"],"phases":{}}
            record.update({"experimental_sparse_control":True,"control_scale":args.control_scale,
                           "sparse_support":support_info[mode]})
            suite["runs"][mode]=record;mx.reset_peak_memory();log("denoising_start",current_mode=mode)
            start=time.perf_counter()
            for step in range(args.steps):
                step_start=time.perf_counter()
                if args.known_bridge:
                    known_now=((1-sigmas[step])*known_clean.astype(mx.float32)+sigmas[step]*noise.astype(mx.float32)).astype(prompt_embeds.dtype)
                    latents=mx.where(known_mask,known_now,latents)
                timestep=(sigmas[step:step+1]*1000).astype(latents.dtype)/1000
                record["transformer_calls"]+=1;record["base_block_calls"]+=32;record["control_block_calls"]+=16
                record["target_token_forward_sum"]+=full_tokens
                velocity=controlled_forward(model.transformer,branch,latents,prompt_embeds,timestep,
                                             layout,context,control_scale=args.control_scale,cache=None)
                if velocity.shape!=latents.shape: raise AssertionError("Controlteacher target prediction shape mismatch")
                latents=(latents.astype(mx.float32)+(sigmas[step+1]-sigmas[step])*velocity.astype(mx.float32)).astype(prompt_embeds.dtype)
                if args.known_bridge:
                    known_next=((1-sigmas[step+1])*known_clean.astype(mx.float32)+sigmas[step+1]*noise.astype(mx.float32)).astype(prompt_embeds.dtype)
                    latents=mx.where(known_mask,known_next,latents)
                mx.eval(latents)
                if not bool(mx.all(mx.isfinite(latents))): raise FloatingPointError("Nonfinite controlteacher state")
                row={"step":step+1,"sigma":float(sigmas[step]),"next_sigma":float(sigmas[step+1]),
                     "model_timestep":float(np.asarray(timestep.astype(mx.float32))[0]),"target_tokens_forwarded":full_tokens,
                     "joint_query_tokens":layout.prefix_length+full_tokens,"base_blocks":32,"control_blocks":16,
                     "core_seconds":time.perf_counter()-step_start}
                record["step_records"].append(row)
                (directory/"metrics.json").write_text(json.dumps(record,indent=2))
                print(json.dumps({"phase":"denoising","mode":mode,**row}),flush=True)
            record["phases"]["denoising_seconds"]=time.perf_counter()-start
            record["denoising_mlx_peak_bytes"]=mx.get_peak_memory()
            final=np.asarray(latents.astype(mx.float32)).copy()
            np.savez_compressed(directory/"final_latents.npz",latents=final)
            record["final_latents_sha256"]=hashlib.sha256(final.tobytes()).hexdigest()
            if args.known_bridge and not np.array_equal(final[:,20*32:52*32],np.asarray(known_clean.astype(mx.float32))[:,20*32:52*32]):
                raise AssertionError("Optional known bridge failed sigma-zero source latent restoration")
            outputs[mode]=final;record["status"]="denoised";mx.clear_cache()
            log("denoising_complete",current_mode=mode,nfe=record["transformer_calls"])

        del branch,contexts,context,latents,velocity;mx.clear_cache()
        tiling=TilingConfig(vae_decode_tiles_per_dim=2,vae_decode_tile_size=512,vae_decode_overlap=4)
        for mode in args.modes:
            record=suite["runs"][mode];directory=suite_dir/mode
            start=time.perf_counter();mx.reset_peak_memory();log("decoding",current_mode=mode)
            packed=mx.array(outputs[mode]).astype(mx.float32)
            unpacked=QwenImage21LatentCreator.unpack_latents(packed,height,width)
            decoded=VAEUtil.decode(model.vae,unpacked,tiling);mx.eval(decoded)
            if decoded.ndim==5: decoded=decoded[:,:,0]
            if decoded.shape[1]==4:
                alpha=np.asarray(decoded[0,3])
                if not np.isfinite(alpha).all(): raise FloatingPointError("Nonfinite decoded alpha")
                Image.fromarray(np.clip((alpha+1)*127.5,0,255).round().astype(np.uint8),mode="L").save(directory/"alpha.png")
                record["decoded_alpha"]={"min_normalized":float(alpha.min()),"max_normalized":float(alpha.max()),
                                         "mean_normalized":float(alpha.mean()),"alpha_png_path":str(directory/"alpha.png")}
            rgb=np.asarray(decoded[0,:3].transpose(1,2,0))
            if rgb.shape!=(height,width,3) or not np.isfinite(rgb).all(): raise FloatingPointError("Invalid decodedteacher output")
            raw=Image.fromarray(np.clip((rgb+1)*127.5,0,255).round().astype(np.uint8),mode="RGB")
            raw.save(directory/"raw.png")
            error=np.abs(np.asarray(raw.crop(rect),dtype=np.int16)-np.asarray(source,dtype=np.int16))
            composite=raw.copy();composite.paste(source,(rect[0],rect[1]));composite.save(directory/"composite.png")
            if not np.array_equal(np.asarray(composite.crop(rect)),np.asarray(source)): raise AssertionError("Final original512 source pixels changed")
            record.update({"status":"success","source_pixels_exact":True,"final_latents_finite":True,"decoded_pixels_finite":True,
                           "raw_source_top16_mae_255":float(error[:16].mean()),"raw_source_bottom16_mae_255":float(error[-16:].mean()),
                           "raw_source_inner_mae_255":float(error[16:-16].mean()),"raw_source_mae_255":float(error.mean()),
                           "raw_path":str(directory/"raw.png"),"composite_path":str(directory/"composite.png"),
                           "output_sha256":file_sha256(directory/"composite.png"),"decoding_mlx_peak_bytes":mx.get_peak_memory()})
            record["phases"]["decode_seconds"]=time.perf_counter()-start
            (directory/"metrics.json").write_text(json.dumps(record,indent=2));mx.clear_cache()
        if any(record["transformer_calls"]!=args.steps for record in suite["runs"].values()): raise AssertionError("TeacherNFE mismatch")
        log("complete",status="success")
    except BaseException as exc:
        suite.update({"status":"failed","error_type":type(exc).__name__,"error":str(exc)})
        write();(suite_dir/"traceback.txt").write_text(traceback.format_exc());raise


if __name__=="__main__":
    main()
