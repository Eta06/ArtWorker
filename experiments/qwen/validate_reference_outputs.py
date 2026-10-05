#!/usr/bin/env python3
"""Independent CPU validation of completed reference-prefix teacher artifacts.

Reads saved real conditioning/latent arrays; never loads weights or uses GPU.
Integrity passing does not establish image quality or independently replay all
runtime denoiser forwards. Original metrics/images are retained unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from evaluate_boundary import ROOT, rgb
from validate_controlnet_outputs import white_diagnostic


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def array_sha(a):return hashlib.sha256(a.tobytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite",type=Path,required=True)
    args=parser.parse_args()
    p=args.suite.resolve()
    s=json.loads((p/"reference_control_metrics.json").read_text())
    if s.get("status")!="success" or s.get("phase")!="complete":raise ValueError("Reference suite is incomplete")
    shared=p/"shared"
    conditioning=np.load(shared/"reference_conditioning.npz")
    prefix=conditioning["source_latents"]
    prompt=conditioning["prompt_embeds"]
    slots=conditioning["image_slots"]
    noise=np.load(shared/"target_noise.npz")["noise"]
    source=rgb(s["source_512_path"])
    rgba=np.array(Image.open(p/"source_rgba.png").convert("RGBA"))
    layout=s["layout"]
    common={"saved_source_prefix_shape":prefix.shape==(1,1024,64),
            "source_prefix_hash":array_sha(prefix)==s["reference_encoding"]["source_prefix_sha256"],
            "prefix_latents_finite":bool(np.isfinite(prefix).all()),
            "actual_square_rgba_shape":rgba.shape==(512,512,4),
            "actual_square_rgb_exact":bool(np.array_equal(rgba[...,:3],source)),
            "source_opaque_alpha":bool(np.all(rgba[...,3]==255)),
            "source_file_hash":sha(s["source_512_path"])==s["source_512_sha256"],
            "source_rgb_array_hash":array_sha(source)==s["source_uint8_sha256"],
            "source_rgba_array_hash":array_sha(rgba)==s["source_rgba_uint8_sha256"],
            "prompt_embedding_hash":array_sha(prompt)==s["prompt_embeds_sha256"],
            "prompt_finite":bool(np.isfinite(prompt).all()),
            "image_slots_hash":array_sha(slots)==s["image_slots_bool_sha256"],
            "image_slots_bool_and256":slots.dtype==np.bool_ and int(slots.sum())==256,
            "prefix_text_count":layout["prefix_text_tokens"]==len(slots)-int(slots.sum())==173,
            "joint_prefix_count":layout["joint_prefix_tokens"]==1024+173==1197,
            "image_context_count":layout["context_rows"]==1024+2304==3328,
            "saved_noise_shape":noise.shape==(1,2304,64),"noise_hash":array_sha(noise)==s["noise_sha256"],
            "noise_finite":bool(np.isfinite(noise).all())}
    original=json.loads((ROOT/"experiments/evaluation/inputs/inputs.json").read_text())
    common["original_jpeg_hashes_unchanged"]=all(sha(t["original_jpg"])==t["original_jpg_sha256"] for t in original["tracks"])
    contexts={};context_checks={}
    for mode,info in s["control_conditioning"].items():
        z=np.load(shared/f"{mode}_contexts.npz")
        target,padded,support=z["target_context"],z["padded_image_context"],z["structural_support"]
        known=np.zeros((72,32),np.float32);known[20:52]=1
        keep=support.reshape(-1)
        contexts[mode]=target
        context_checks[mode]={"target_shape":target.shape==(1,2304,129),
                              "target_hash":array_sha(target)==info["target_context_sha256_float32"],
                              "padded_shape":padded.shape==(1,3328,129),
                              "padded_hash":array_sha(padded)==info["padded_image_context_sha256_float32"],
                              "reference129_literal_zero":bool(np.all(padded[:,:1024]==0)),
                              "target_context_exact_after_padding":bool(np.array_equal(padded[:,1024:],target)),
                              "target_known_mask_exact":bool(np.array_equal(target[0,:,64].reshape(72,32),known)),
                              "reference_known_mask_zero":bool(np.all(padded[:,:1024,64]==0)),
                              "unsupported_structural64_zero":bool(np.all(target[:,~keep,:64]==0)),
                              "support_known_tokens_all_true":bool(support.reshape(72,32)[20:52].all()),
                              "source65_preserved_runtime_assertion":info["source65_exact_against_baseline"]}
    checks={};white={}
    steps=s["steps"]
    for name,r in s["runs"].items():
        if r.get("status")!="success":raise ValueError("Reference arm incomplete")
        d=p/name
        latents=np.load(d/"final_latents.npz")["latents"]
        raw,composite=rgb(d/"raw.png"),rgb(d/"composite.png")
        rows=r["step_records"]
        context=contexts[r["mode"]]
        known=context[0,:,64]>.5
        input_example=np.concatenate([prefix,latents],axis=1)
        checks[name]={"nfe_count":len(rows)==r["transformer_calls"]==steps,
                      "base_blocks":sum(row["base_blocks"] for row in rows)==r["base_block_calls"]==steps*32,
                      "control_blocks":sum(row["control_blocks"] for row in rows)==r["control_block_calls"]==steps*16,
                      "target_token_count":sum(row["target_tokens_forwarded"] for row in rows)==r["target_token_forward_sum"]==steps*2304,
                      "reference_image_token_count":sum(row["reference_image_tokens_forwarded"] for row in rows)==r["reference_image_token_forward_sum"]==steps*1024,
                      "joint_query_count":sum(row["joint_query_tokens"] for row in rows)==r["joint_query_token_forward_sum"]==steps*(1197+2304),
                      "each_step_prefix_runtime_assertion":all(row["source_prefix_input_exact"] for row in rows),
                      "exact_prefix_concatenation_cpu":bool(np.array_equal(input_example[:,:1024],prefix)),
                      "input_assembly_shape":input_example.shape==(1,3328,64),
                      "same_prefix_hash":r["source_prefix_sha256"]==array_sha(prefix),
                      "target_latent_shape":latents.shape==(1,2304,64),
                      "target_latents_finite":bool(np.isfinite(latents).all()),
                      "target_latents_hash":array_sha(latents)==r["final_latents_sha256"],
                      "final_known_target_exact_from_saved_context":bool(np.array_equal(latents[:,known],context[:,known,65:])) if s["known_bridge_extension"] else True,
                      "raw_image_shape":raw.shape==(1152,512,3),"composite_shape":composite.shape==(1152,512,3),
                      "source_pixels_exact":bool(np.array_equal(composite[320:832],source)),
                      "raw_hash":sha(d/"raw.png")==r["raw_sha256"],
                      "composite_hash":sha(d/"composite.png")==r["composite_sha256"]}
        white[name]=white_diagnostic(raw)
    if not all(common.values()) or not all(all(c.values()) for c in context_checks.values()) or not all(all(c.values()) for c in checks.values()):
        raise AssertionError("Reference artifact validation failed")
    report={"status":"passed","quality_status":"integrity only; independent full image review required",
            "common":common,"contexts":context_checks,"arms":checks,"raw_near_white_diagnostic":white,
            "input_counts_per_forward":{"target":2304,"reference_images":1024,"text":173,"joint_queries":3501,"packed_images":3328},
            "runtime_prefix_limit":"Saved source and CPU assembly verified; equality in every actual model forward is a recorded checked runtime assertion, not independently replayed denoising.",
            "prefix_hint_limit":"Literal-zero reference context does not itself disable biased hints; prefix-hints-disabled helper gates completed full text/image prefix hints. Mechanism is covered by pinned tiny CPU tests, not a stored per-layer real-model hint trace.",
            "validator_sha256":sha(__file__),"suite_metrics_sha256":sha(p/"reference_control_metrics.json")}
    (p/"independent_validation.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"status":report["status"],"common_checks":len(common),"arms":list(checks),"counts":report["input_counts_per_forward"],"white":white},indent=2))


if __name__=="__main__":main()
