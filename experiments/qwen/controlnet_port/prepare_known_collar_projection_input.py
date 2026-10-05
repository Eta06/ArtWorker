"""CPU-only metadata adapter for a retained exact ControlNet final tensor.

Older localized-control outputs saved packed latents alone. The VAE projection
diagnostic also requires absolute final IDs and a source rectangle. This copies
the exact tensor and original raw PNG into a new isolated directory, records
their provenance, and adds only validated geometry metadata. No pixels or
latent values are changed; no model is imported or loaded.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
DEFAULT=ROOT/"experiments/qwen/controlnet_runs/2026-10-05/track3-localized40-outpaint/tangent-canny/localized-hints/final_latents.npz"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",type=Path,default=DEFAULT)
    parser.add_argument("--output",type=Path,default=ROOT/"experiments/qwen/projection_inputs/2026-10-05/track3-localized40-known-collar")
    args=parser.parse_args()
    import numpy as np
    from PIL import Image
    original=args.input.resolve()
    metrics_path=original.parent/"metrics.json"
    metrics=json.loads(metrics_path.read_text())
    shared=json.loads((ROOT/"experiments/evaluation/inputs/inputs.json").read_text())
    source_record=next(t for t in shared["tracks"] if t["id"]=="track3")
    rect=shared["source_rect_xyxy"]
    if shared["canvas_size"]!=[512,1152] or rect!=[0,320,512,832]:
        raise ValueError("Unexpected shared projection geometry")
    if metrics.get("status")!="success" or not metrics.get("final_latents_finite") or not metrics.get("source_pixels_exact"):
        raise ValueError("Only a completed finite exact-source ControlNet result can be adapted")
    with np.load(original) as data:
        latents=np.asarray(data["latents"])
    if latents.shape!=(1,2304,64) or latents.dtype!=np.float32 or not np.isfinite(latents).all():
        raise ValueError("Expected finite exact packed float32 reference-codec tensor")
    tensor_sha=hashlib.sha256(latents.tobytes()).hexdigest()
    if tensor_sha!=metrics["final_latents_sha256"]:
        raise ValueError("Tensor does not match completed generation metrics")
    raw=original.parent/"raw.png"
    composite=original.parent/"composite.png"
    if sha(raw)!=metrics["raw_sha256"] or sha(composite)!=metrics["composite_sha256"]:
        raise ValueError("Saved source images do not match completed generation metrics")
    source=np.asarray(Image.open(source_record["source_512"]).convert("RGB"))
    if not np.array_equal(np.asarray(Image.open(composite).convert("RGB"))[320:832],source):
        raise ValueError("Saved composite does not preserve the resized source exactly")
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite a projection input directory")
    args.output.mkdir(parents=True)
    destination=args.output/"final_latents.npz"
    np.savez_compressed(destination,latents=latents,final_canvas_ids=np.arange(2304,dtype=np.int32),
                        source_rect_xyxy=np.asarray(rect,dtype=np.int32),sigmas=np.asarray(metrics["sigmas"],dtype=np.float32))
    shutil.copy2(raw,args.output/"raw.png")
    with np.load(destination) as adapted:
        if not np.array_equal(adapted["latents"],latents):
            raise AssertionError("Metadata adaptation changed exact final tensor")
    report={"status":"passed","device":"CPU only","models_loaded":False,"gpu_work":False,
            "input":str(original),"input_file_sha256":sha(original),"input_metrics_sha256":sha(metrics_path),
            "exact_final_tensor_sha256":tensor_sha,"latent_values_bitwise_unchanged":True,
            "adapted_input":str(destination.resolve()),"adapted_file_sha256":sha(destination),
            "source_rect_xyxy":rect,"absolute_final_ids_valid":True,
            "original_raw_png_sha256":sha(raw),"copied_raw_png_sha256":sha(args.output/"raw.png"),
            "source_pixels_exact_in_original_composite":True,
            "purpose":"Metadata-only exact-final adapter; no VAE projection or quality result"}
    if report["copied_raw_png_sha256"]!=report["original_raw_png_sha256"]:
        raise AssertionError("Original raw pixels changed during input preparation")
    (args.output/"metadata_adapter.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"status":"passed","gpu_work":False,"output":str(args.output.resolve())}))


if __name__=="__main__":
    main()
