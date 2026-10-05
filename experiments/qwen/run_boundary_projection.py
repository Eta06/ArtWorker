"""VAE-only post-denoising source-boundary latent projection diagnostic.

This is a derived-image repair experiment, not new sampling or LanPaint/DPS.
Only border latent rows can change. The original resized source is pasted back
exactly, as in the preceding experiments. All weights stay frozen. The optional
derivative objective is generic RGB finite-difference continuity, not a painted
rail, geometry detector, or proof of a correct scene outside the source.

The installed NumPy tiled decoder detaches gradients. This script implements
the same tile512 / overlap64 arithmetic in MLX and CPU-checks it against the
reference utility before any model is loaded. --preflight does no GPU work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / ".build/models/qwen/base"
DEFAULT_INPUT = ROOT / "experiments/qwen/geometry_runs/2026-10-05/track3/edge-context/previews/step06_predicted_clean.npz"


def validate_input_metadata(np, data, rect):
    """Accept retained full-canvas preview snapshots or exact terminal-state NPZs."""
    if not np.array_equal(data["final_canvas_ids"],np.arange(2304)):
        raise ValueError("Input IDs are not the final absolute full canvas")
    if "window_xyxy" in data:
        if not np.array_equal(data["window_xyxy"],[0,0,512,1152]):
            raise ValueError("Input preview window is not the complete final canvas")
        if "source_rect_xyxy" in data and not np.array_equal(data["source_rect_xyxy"],rect):
            raise ValueError("Input preview source rectangle differs from shared inputs")
        return "predicted-clean-preview"
    if "source_rect_xyxy" in data:
        if not np.array_equal(data["source_rect_xyxy"],rect):
            raise ValueError("Exact final source rectangle differs from shared inputs")
        return "exact-final"
    raise ValueError("Input requires a full window or validated source rectangle")


def original_raw_path(input_path, input_kind):
    return (input_path.parent if input_kind=="exact-final" else input_path.parent.parent)/"raw.png"


def checkpointed_vae_decode(mx, nn, vae, latents):
    """Reference decoder forward, recomputing mid/up blocks during backward.

    This changes graph memory management only. Real forward parity is checked
    against the untouched reference VAEUtil decoder before optimization.
    """
    value=vae._to_nhwc(latents)
    value=value*mx.array(vae._std,dtype=value.dtype)+mx.array(vae._mean,dtype=value.dtype)
    decoder=vae.decoder
    value=decoder.conv_in(vae.post_quant_conv(value))
    value=mx.checkpoint(decoder.mid_block)(value)
    for block in decoder.up_blocks:
        value=mx.checkpoint(block)(value)
    value=decoder.conv_out(nn.silu(decoder.norm_out(value)))
    return mx.clip(value,-1,1).transpose(0,3,1,2)[:,:,None]


def differentiable_tiled_decode(mx, np, latent, decode_fn, *, tile_px=512, overlap_px=64, scale=16):
    """Same tiles and cosine accumulation as reference VAETiler, without NumPy pixels."""
    b, _, t, h_lat, w_lat = latent.shape
    if b != 1 or t != 1:
        raise ValueError("This diagnostic requires a single still image")
    tile_lat = tile_px // scale
    overlap_lat = min(overlap_px // scale, tile_lat - 1)
    stride = tile_lat - overlap_lat
    height, width = h_lat * scale, w_lat * scale
    decoded_sum = None
    counts_np = np.zeros((1, 1, height, width), dtype=np.float32)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, 1, overlap_px, dtype=np.float32) * np.pi)
    for y in range(0, h_lat, stride):
        ye = min(y + tile_lat, h_lat)
        for x in range(0, w_lat, stride):
            xe = min(x + tile_lat, w_lat)
            if (y > 0 and ye - y <= overlap_lat) or (x > 0 and xe - x <= overlap_lat):
                continue
            tile = decode_fn(latent[:, :, :, y:ye, x:xe])
            if tile.ndim == 5:
                tile = tile[:, :, 0]
            eh, ew = (ye-y)*scale, (xe-x)*scale
            tile = tile[:, :, :eh, :ew].astype(mx.float32)
            wh = np.ones(eh, dtype=np.float32)
            ww = np.ones(ew, dtype=np.float32)
            ovh, ovw = min(overlap_px, eh-1), min(overlap_px, ew-1)
            if ovh > 0:
                if y > 0:
                    wh[:ovh] = ramp[:ovh]
                if ye < h_lat:
                    wh[-ovh:] = 1-ramp[:ovh]
            if ovw > 0:
                if x > 0:
                    ww[:ovw] = ramp[:ovw]
                if xe < w_lat:
                    ww[-ovw:] = 1-ramp[:ovw]
            weights = wh[:, None]*ww[None, :]
            weighted = tile * mx.array(weights)[None, None]
            padded = mx.pad(weighted, [(0, 0), (0, 0), (y*scale, height-ye*scale), (x*scale, width-xe*scale)])
            decoded_sum = padded if decoded_sum is None else decoded_sum+padded
            counts_np[:, :, y*scale:ye*scale, x*scale:xe*scale] += weights[None, None]
    if decoded_sum is None:
        raise AssertionError("No VAE tiles")
    return decoded_sum/mx.array(np.clip(counts_np, 1e-6, None))


def cpu_smoke(mx, np):
    from mflux.models.common.vae.vae_tiler import VAETiler
    import mlx.nn as nn
    mx.set_default_device(mx.cpu)
    # Three vertical tiles, exact production overlap pattern, small test scale.
    array = np.random.default_rng(17).normal(size=(1, 4, 1, 72, 32)).astype(np.float32)
    latent = mx.array(array)
    def mock_decode(value):
        return mx.repeat(mx.repeat(value, 2, axis=3), 2, axis=4)
    actual = differentiable_tiled_decode(mx, np, latent, mock_decode, tile_px=64, overlap_px=8, scale=2)
    expected = VAETiler.decode_image_tiled(latent=latent, decode_fn=mock_decode, tile_size=(64,64), tile_overlap=(8,8), spatial_scale=2)
    error = float(mx.max(mx.abs(actual-expected)).item())
    if error > 5e-7:
        raise AssertionError(f"Differentiable tiler disagrees with reference by {error}")
    grad = mx.grad(lambda z: mx.mean(differentiable_tiled_decode(mx, np, z, mock_decode, tile_px=64, overlap_px=8, scale=2)**2))(latent)
    grad_np = np.asarray(grad)
    if not np.isfinite(grad_np).all() or not np.any(grad_np):
        raise AssertionError("Differentiable tiler detached or invalid gradient")
    class TinyDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_in=nn.Conv2d(4,4,1)
            self.mid_block=nn.Sequential(nn.Linear(4,4),nn.Tanh())
            self.up_blocks=[nn.Sequential(nn.Linear(4,4),nn.Tanh())]
            self.norm_out=nn.LayerNorm(4)
            self.conv_out=nn.Conv2d(4,4,1)
        def __call__(self,x):
            x=self.mid_block(self.conv_in(x))
            for block in self.up_blocks:
                x=block(x)
            return self.conv_out(nn.silu(self.norm_out(x)))
    class TinyVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self._std=(1.1,0.9,1.2,0.8)
            self._mean=(0.1,-0.1,0.2,-0.2)
            self.post_quant_conv=nn.Conv2d(4,4,1)
            self.decoder=TinyDecoder()
        @staticmethod
        def _to_nhwc(value):
            return value[:,:,0].transpose(0,2,3,1)
        def decode(self,value):
            x=self._to_nhwc(value)
            x=x*mx.array(self._std,dtype=x.dtype)+mx.array(self._mean,dtype=x.dtype)
            image=self.decoder(self.post_quant_conv(x))
            return mx.clip(image,-1,1).transpose(0,3,1,2)[:,:,None]
    tiny=TinyVAE();tiny.freeze()
    small=latent[:,:,:,:4,:4]
    plain_value,plain_grad=mx.value_and_grad(lambda z:mx.mean(tiny.decode(z)**2))(small)
    checkpoint_value,checkpoint_grad=mx.value_and_grad(lambda z:mx.mean(checkpointed_vae_decode(mx,nn,tiny,z)**2))(small)
    mx.eval(plain_value,plain_grad,checkpoint_value,checkpoint_grad)
    checkpoint_forward_error=float(mx.abs(plain_value-checkpoint_value).item())
    checkpoint_gradient_error=float(mx.max(mx.abs(plain_grad-checkpoint_grad)).item())
    if checkpoint_forward_error>1e-7 or checkpoint_gradient_error>1e-6 or not np.any(np.asarray(checkpoint_grad)):
        raise AssertionError("Checkpointed decoder changed output/gradient")
    rect=[0,320,512,832]
    common={"final_canvas_ids":np.arange(2304)}
    preview={**common,"window_xyxy":np.asarray([0,0,512,1152])}
    final={**common,"source_rect_xyxy":np.asarray(rect)}
    if validate_input_metadata(np,preview,rect)!="predicted-clean-preview" or validate_input_metadata(np,final,rect)!="exact-final":
        raise AssertionError("Input metadata kind mismatch")
    if original_raw_path(Path("/arm/previews/step06.npz"),"predicted-clean-preview")!=Path("/arm/raw.png"):
        raise AssertionError("Wrong raw image for old preview")
    if original_raw_path(Path("/arm/final_latents.npz"),"exact-final")!=Path("/arm/raw.png"):
        raise AssertionError("Wrong raw image for exact final")
    rejected=0
    for invalid in ({**preview,"window_xyxy":np.asarray([0,128,512,1024])},
                    {**final,"source_rect_xyxy":np.asarray([0,321,512,833])},
                    {**final,"final_canvas_ids":np.arange(2304)[::-1]}):
        try:
            validate_input_metadata(np,invalid,rect)
        except ValueError:
            rejected+=1
    if rejected!=3:
        raise AssertionError("Malformed metadata accepted")
    return {"reference_tiler_max_error":error, "nonzero_finite_gradient":True, "model_weights_loaded":False,
            "checkpoint_forward_error":checkpoint_forward_error,"checkpoint_gradient_max_error":checkpoint_gradient_error,
            "metadata_cases":{"preview_and_exact_final_accepted":True,"correct_raw_paths":True,"invalid_cases_rejected":rejected},"device":"cpu"}


def rgb_image(np, Image, decoded):
    rgb = np.asarray(decoded)[0,:3].transpose(1,2,0)
    if not np.isfinite(rgb).all():
        raise FloatingPointError("Invalid decoded RGB")
    return Image.fromarray(np.clip((rgb+1)*127.5,0,255).round().astype(np.uint8), "RGB")


def crop_metrics(np, raw, source, rect):
    error = np.abs(np.asarray(raw.crop(rect),dtype=np.float32)-np.asarray(source,dtype=np.float32))
    pixels = np.asarray(raw,dtype=np.float32)
    original = np.asarray(source,dtype=np.float32)
    y0,y1 = rect[1],rect[3]
    # This is a generic one-pixel boundary derivative metric, not rail-angle accuracy.
    top = pixels[y0-1]-original[0]-(original[0]-original[1])
    bottom = pixels[y1]-original[-1]-(original[-1]-original[-2])
    return {"raw_source_top16_mae_255":float(error[:16].mean()), "raw_source_bottom16_mae_255":float(error[-16:].mean()),
            "raw_source_top32_mae_255":float(error[:32].mean()), "raw_source_bottom32_mae_255":float(error[-32:].mean()),
            "raw_source_inner_mae_255":float(error[32:-32].mean()), "raw_source_mae_255":float(error.mean()),
            "composited_boundary_derivative_top_mae_255":float(np.abs(top).mean()),
            "composited_boundary_derivative_bottom_mae_255":float(np.abs(bottom).mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",type=Path,default=DEFAULT_INPUT)
    parser.add_argument("--output",type=Path,default=ROOT/"experiments/qwen/projection_runs/2026-10-05/track3")
    parser.add_argument("--track",choices=("track1","track2","track3"),default="track3")
    parser.add_argument("--methods",default="consistency,derivative")
    parser.add_argument("--iterations",type=int,default=8)
    parser.add_argument("--learning-rate",type=float,default=0.015)
    parser.add_argument("--latent-collar",type=int,default=6,help="Each side of each source boundary in latent rows; known collar may also move")
    parser.add_argument("--source-band",type=int,default=32)
    parser.add_argument("--trust-weight",type=float,default=0.01)
    parser.add_argument("--derivative-weight",type=float,default=1.0)
    parser.add_argument("--max-delta",type=float,default=0.2)
    parser.add_argument("--checkpoint-decoder",action="store_true",help="Recompute decoder mid/up blocks in backward to lower activation memory")
    parser.add_argument("--preflight",action="store_true")
    args=parser.parse_args()
    if not 1 <= args.iterations <= 32 or not 1 <= args.latent_collar <= 10 or not 1 <= args.source_band <= 64:
        raise ValueError("Bounded diagnostic requires iterations<=32, collar<=10, source_band<=64")
    methods=args.methods.split(",")
    if not methods or any(k not in ("consistency","derivative") for k in methods):
        raise ValueError("Methods are consistency and derivative")
    import numpy as np
    import mlx.core as mx
    import mlx.nn as nn
    from PIL import Image, ImageDraw
    from mlx.utils import tree_flatten
    from mflux.models.common.weights.loading.weight_loader import WeightLoader
    from mflux.models.common.weights.loading.weight_applier import WeightApplier
    from mflux.models.qwen21.reference.model.qwen_image21_vae.vae import QwenImage21VAE
    from mflux.models.qwen21.reference.weights.qwen_image21_weight_definition import QwenImage21WeightDefinition
    from mflux.models.qwen21.reference.latent_creator.qwen_image21_latent_creator import QwenImage21LatentCreator
    from mflux.models.common.vae.tiling_config import TilingConfig
    from mflux.models.common.vae.vae_util import VAEUtil
    shared=json.loads((ROOT/"experiments/evaluation/inputs/inputs.json").read_text())
    track=next(t for t in shared["tracks"] if t["id"]==args.track)
    width,height=shared["canvas_size"]
    rect=shared["source_rect_xyxy"]
    if (width,height,rect)!=(512,1152,[0,320,512,832]):
        raise ValueError("Expected shared portrait geometry")
    source=Image.open(track["source_512"]).convert("RGB")
    data=np.load(args.input)
    packed=np.asarray(data["latents"],dtype=np.float32)
    if packed.shape != (1,2304,64) or not np.isfinite(packed).all():
        raise ValueError("Input must be finite full-canvas predicted-clean latent")
    input_kind=validate_input_metadata(np,data,rect)
    args.output.mkdir(parents=True,exist_ok=True)
    record={"method":"frozen-VAE post-denoise border-latent pixel-consistency projection", "input":str(args.input),
            "input_sha256":hashlib.sha256(args.input.read_bytes()).hexdigest(), "input_kind":input_kind, "track":args.track,"status":"preflight",
            "derived_postprocess":True,"new_transformer_calls":0,"weights_trained":False,"model_components_loaded":[],
            "source_rect_xyxy":rect,"source_sha256":hashlib.sha256(np.asarray(source).tobytes()).hexdigest(),
            "original_jpg_sha256_before":hashlib.sha256(Path(track["original_jpg"]).read_bytes()).hexdigest(),
            "parameters":{k:v for k,v in vars(args).items() if k not in ("input","output")},"runs":{}}
    started=time.perf_counter()
    def log(phase,**kwargs):
        global_updates=dict(kwargs)
        if "method" in global_updates:
            global_updates["current_method"]=global_updates.pop("method")
        record.update(global_updates,phase=phase,elapsed_seconds=time.perf_counter()-started,
                      process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        (args.output/("preflight.json" if args.preflight else "projection_metrics.json")).write_text(json.dumps(record,indent=2))
        print(json.dumps({"phase":phase,"elapsed_seconds":round(record["elapsed_seconds"],2),**kwargs}),flush=True)
    if args.preflight:
        record["cpu_smoke"]=cpu_smoke(mx,np)
        log("preflight_complete",status="preflight_passed")
        return
    if (args.output/"projection_metrics.json").exists():
        raise FileExistsError("Refusing to overwrite a previous projection run")
    try:
        mx.set_cache_limit(128*1024**2)
        mx.set_memory_limit(18*1024**3)
        log("load_vae",status="running")
        load_start=time.perf_counter()
        vae=QwenImage21VAE(json.loads((MODEL/"vae/config.json").read_text()))
        component=next(c for c in QwenImage21WeightDefinition.get_components() if c.name=="vae")
        weights=WeightLoader.load_single_local(component,MODEL)
        supplied=dict(tree_flatten(weights.components["vae"]))
        expected=dict(tree_flatten(vae.parameters()))
        if set(supplied)!=set(expected) or any(supplied[k].shape!=v.shape for k,v in expected.items()):
            raise ValueError("VAE weights fail exact key/shape validation")
        WeightApplier.apply_and_quantize_single(weights,vae,component,None)
        vae.freeze()
        mx.eval(vae.parameters())
        record["model_components_loaded"]=["vae"]
        record["load_seconds"]=time.perf_counter()-load_start
        record["vae_parameter_bytes"]=sum(v.nbytes for _,v in tree_flatten(vae.parameters()))
        del weights,supplied,expected
        mx.clear_cache()
        decode_fn=(lambda z:checkpointed_vae_decode(mx,nn,vae,z)) if args.checkpoint_decoder else vae.decode
        base=QwenImage21LatentCreator.unpack_latents(mx.array(packed),height,width)
        mask_np=np.zeros(base.shape,dtype=np.float32)
        for center in (rect[1]//16,rect[3]//16):
            mask_np[:,:,:,center-args.latent_collar:center+args.latent_collar]=1
        mask=mx.array(mask_np)
        source_tensor=mx.array(np.asarray(source,dtype=np.float32)/127.5-1).transpose(2,0,1)[None]
        source_top=source_tensor[:,:,:args.source_band]
        source_bottom=source_tensor[:,:,-args.source_band:]
        tiling=TilingConfig(vae_decode_tiles_per_dim=2,vae_decode_tile_size=512,vae_decode_overlap=4)
        log("baseline_decode")
        baseline_tensor=differentiable_tiled_decode(mx,np,base,decode_fn)
        reference_tensor=VAEUtil.decode(vae,base,tiling)
        mx.eval(baseline_tensor,reference_tensor)
        record["real_vae_tiler_max_error"]=float(mx.max(mx.abs(baseline_tensor-reference_tensor)).item())
        if record["real_vae_tiler_max_error"]>1e-5:
            raise AssertionError("Real differentiable VAE tiler mismatch")
        baseline=rgb_image(np,Image,baseline_tensor)
        baseline.save(args.output/"baseline-raw.png")
        baseline_composite=baseline.copy();baseline_composite.paste(source,(rect[0],rect[1]));baseline_composite.save(args.output/"baseline-composite.png")
        original_final=original_raw_path(args.input,input_kind)
        record["original_final_raw_path"]=str(original_final)
        if original_final.is_file():
            before_image=np.asarray(Image.open(original_final).convert("RGB"),dtype=np.int16)
            errors=np.abs(before_image-np.asarray(baseline,dtype=np.int16))
            record["snapshot_vs_previous_final_max_error_255"]=int(errors.max())
            record["snapshot_vs_previous_final_mae_255"]=float(errors.mean())
        record["baseline_metrics"]=crop_metrics(np,baseline,source,rect)
        baseline_pixels=np.asarray(baseline,dtype=np.float32)
        del reference_tensor,baseline_tensor
        mx.clear_cache()
        results=[("before",baseline_composite)]
        # Only top and middle tiles contain source boundaries. Their center seam
        # rows are outside overlapping pixels, so this objective has the exact
        # same decoder context as the differentiable/full reference final decode.
        def boundary_decodes(value):
            top=decode_fn(value[:,:,:,:32])[:,:3,0]
            bottom=decode_fn(value[:,:,:,28:60])[:,:3,0]
            return top,bottom
        for method in methods:
            directory=args.output/method;directory.mkdir(exist_ok=True)
            mx.reset_peak_memory()
            delta=mx.zeros_like(base);moment=mx.zeros_like(base);variance=mx.zeros_like(base)
            def objective(change):
                value=base+mask*change
                top,bottom=boundary_decodes(value)
                pixel_loss=(mx.mean((top[:,:,320:320+args.source_band]-source_top)**2)+
                            mx.mean((bottom[:,:,384-args.source_band:384]-source_bottom)**2))/2
                # Differing x-positions of edges in adjacent source rows provide
                # a local continuation cue without labeling or drawing a rail.
                top_derivative=top[:,:,319]-source_tensor[:,:,0]-(source_tensor[:,:,0]-source_tensor[:,:,1])
                bottom_derivative=bottom[:,:,384]-source_tensor[:,:,-1]-(source_tensor[:,:,-1]-source_tensor[:,:,-2])
                derivative=(mx.mean(top_derivative**2)+mx.mean(bottom_derivative**2))/2
                trust=mx.sum((mask*change)**2)/mx.sum(mask)
                return pixel_loss+args.trust_weight*trust+(args.derivative_weight*derivative if method=="derivative" else 0)
            value_and_grad=mx.value_and_grad(objective)
            arm={"status":"running","objective":"known source border pixel MSE + latent trust"+(" + one-pixel RGB boundary derivative MSE" if method=="derivative" else ""),"iteration_records":[]}
            record["runs"][method]=arm
            arm_start=time.perf_counter()
            for iteration in range(1,args.iterations+1):
                loss,gradient=value_and_grad(delta)
                gradient=gradient*mask
                moment=0.9*moment+0.1*gradient
                variance=0.999*variance+0.001*gradient**2
                corrected_m=moment/(1-0.9**iteration)
                corrected_v=variance/(1-0.999**iteration)
                delta=mx.clip(delta-args.learning_rate*corrected_m/(mx.sqrt(corrected_v)+1e-8),-args.max_delta,args.max_delta)*mask
                mx.eval(loss,delta,moment,variance)
                if not bool(mx.all(mx.isfinite(delta)).item()):
                    raise FloatingPointError("Invalid projection update")
                item={"iteration":iteration,"pre_update_loss":float(loss.item()),"delta_rms":float(mx.sqrt(mx.sum(delta**2)/mx.sum(mask)).item())}
                checkpoint_file=directory/"latest_iteration_state.npz"
                checkpoint_temporary=directory/"latest_iteration_state_tmp.npz"
                np.savez_compressed(checkpoint_temporary,delta=np.asarray(delta),moment=np.asarray(moment),variance=np.asarray(variance),
                                    iteration=iteration,input_sha256=record["input_sha256"])
                checkpoint_temporary.replace(checkpoint_file)
                item["state_checkpoint_path"]=str(checkpoint_file)
                arm["iteration_records"].append(item)
                log("projection",method=method,**item)
                del gradient,loss,corrected_m,corrected_v
                mx.clear_cache()
            projected=base+delta
            projected_np=np.asarray(projected.astype(mx.float32))
            base_np=np.asarray(base.astype(mx.float32))
            if not np.array_equal(projected_np[mask_np==0],base_np[mask_np==0]):
                raise AssertionError("Changed latent outside border collar")
            arm["latent_outside_collar_exact"]=True
            arm["projection_seconds"]=time.perf_counter()-arm_start
            arm["mlx_projection_peak_bytes"]=mx.get_peak_memory()
            final_tensor=differentiable_tiled_decode(mx,np,projected,decode_fn);mx.eval(final_tensor)
            raw=rgb_image(np,Image,final_tensor);raw.save(directory/"raw.png")
            composite=raw.copy();composite.paste(source,(rect[0],rect[1]));composite.save(directory/"composite.png")
            if not np.array_equal(np.asarray(composite.crop(rect)),np.asarray(source)):
                raise AssertionError("Original source pixels changed")
            arm.update(crop_metrics(np,raw,source,rect),source_pixels_exact=True,finite=True,status="success")
            raw_pixels=np.asarray(raw,dtype=np.float32)
            diff=np.abs(raw_pixels-baseline_pixels)
            distance=max(128,args.latent_collar*16)
            arm["raw_generated_far_from_collar_change_mae_255"]=float(np.concatenate((diff[:rect[1]-distance].reshape(-1),diff[rect[3]+distance:].reshape(-1))).mean())
            np.savez_compressed(directory/"projected_latents.npz",latents=np.asarray(QwenImage21LatentCreator.pack_latents(projected)),source_rect_xyxy=rect)
            results.append((method,composite))
            log("arm_complete",method=method)
            del delta,moment,variance,projected,projected_np,base_np,final_tensor
            mx.clear_cache()
        # Actual untouched final composites and separate nearest-neighbor crop.
        montage=Image.new("RGB",(len(results)*280,670),(15,18,20));draw=ImageDraw.Draw(montage)
        closeup=Image.new("RGB",(len(results)*620,260),(15,18,20));cdraw=ImageDraw.Draw(closeup)
        for index,(label,image) in enumerate(results):
            draw.text((index*280+12,8),label,fill="white")
            montage.paste(image.resize((272,612)),(index*280+4,28))
            cdraw.text((index*620+12,8),label,fill="white")
            # Source boundary is at y320, crop has both generated and source.
            crop=image.crop((250,302,400,350)).resize((600,192),Image.Resampling.NEAREST)
            closeup.paste(crop,(index*620+10,32))
        montage.save(args.output/"final-comparison.png")
        closeup.save(args.output/"rail-join-4x.png")
        record["original_jpg_sha256_after"]=hashlib.sha256(Path(track["original_jpg"]).read_bytes()).hexdigest()
        if record["original_jpg_sha256_after"]!=record["original_jpg_sha256_before"]:
            raise AssertionError("Original cover file changed")
        log("complete",status="complete")
    except Exception as error:
        log("failed",status="failed",error=repr(error),traceback=traceback.format_exc())
        raise


if __name__=="__main__":
    main()
