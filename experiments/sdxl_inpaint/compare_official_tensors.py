"""Streaming CPU tensor-byte comparison without constructing pretrained models."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import struct
import time

ROOT = Path(__file__).resolve().parents[2]

def header(path):
    with path.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        data = json.loads(stream.read(length))
    data.pop("__metadata__", None)
    return data, 8 + length

def tensor_digest(path, entry, data_start, cast_f32_to_f16=False):
    begin, end = entry["data_offsets"]
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(data_start + begin)
        remaining = end - begin
        while remaining:
            data = stream.read(min(4 * 1024 * 1024, remaining))
            if not data:
                raise EOFError(str(path))
            remaining -= len(data)
            if cast_f32_to_f16:
                import numpy as np
                import torch
                values = np.frombuffer(data, dtype="<f4").copy()
                data = torch.from_numpy(values).to(device="cpu", dtype=torch.float16).numpy().tobytes()
            digest.update(data)
    return digest.hexdigest()

def compare(left, right, cast_right=False):
    a, off_a = header(left); b, off_b = header(right)
    rows = []
    for name in sorted(set(a) & set(b)):
        shape_equal = a[name]["shape"] == b[name]["shape"]
        target_dtype_equal = a[name]["dtype"] == ("F16" if cast_right else b[name]["dtype"])
        digest_a = tensor_digest(left, a[name], off_a)
        digest_b = tensor_digest(right, b[name], off_b, cast_right)
        rows.append({"name":name,"shape_equal":shape_equal,"left_dtype":a[name]["dtype"],
                     "right_stored_dtype":b[name]["dtype"],"right_cast_to_F16":cast_right,
                     "left_tensor_sha256":digest_a,"right_tensor_sha256_after_cast":digest_b,
                     "same":shape_equal and target_dtype_equal and digest_a == digest_b})
    return {"left":str(left),"right":str(right),"common_tensors":len(rows),
            "only_left":sorted(set(a)-set(b)),"only_right":sorted(set(b)-set(a)),
            "all_common_equal":all(e["same"] for e in rows),
            "mismatch_names":[e["name"] for e in rows if not e["same"]],"tensors":rows}

def main():
    started=time.perf_counter()
    import torch
    torch.set_default_device("cpu")
    torch.set_num_threads(4)
    results={}
    for name in ("text_encoder","text_encoder_2"):
        results[name]=compare(ROOT/f".build/models/sdxl-inpaint-fp16/{name}/model.fp16.safetensors",
                              ROOT/f".build/models/sdxl-promax-official/base/{name}/model.fp16.safetensors")
    results["vae"]=compare(ROOT/".build/models/sdxl-inpaint-fp16/vae/diffusion_pytorch_model.fp16.safetensors",
                           ROOT/".build/models/sdxl-promax-official/vae_fix/diffusion_pytorch_model.safetensors",True)
    result={"device":"CPU","model_weights_loaded":False,"streaming_raw_file_tensors_only":True,
            "seconds":time.perf_counter()-started,"comparisons":results,
            "decision":"Keep the fresh exact official assembly; comparison does not modify or substitute components."}
    path=ROOT/"experiments/sdxl_inpaint/official_tensor_comparison.json"
    path.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"path":str(path),"seconds":result["seconds"],"comparisons":{
        k:{n:v[n] for n in ("common_tensors","only_left","only_right","all_common_equal","mismatch_names")}
        for k,v in results.items()}},indent=2),flush=True)

if __name__=="__main__":
    main()
