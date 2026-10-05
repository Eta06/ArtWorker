#!/usr/bin/env python3
"""CPU-only independent state, latent and exact-pixel projection audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from run_saved_top32_checkpoint_projection import image_texture_profiles


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def image(path):
    return np.asarray(Image.open(path).convert("RGB"))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",type=Path,required=True)
    args=parser.parse_args()
    directory=args.run.resolve()
    metrics=json.loads((directory/"projection_metrics.json").read_text())
    assert metrics["status"]=="complete" and metrics["weights_trained"] is False and metrics["new_transformer_calls"]==0
    assert metrics["model_components_loaded"]==["vae"] and metrics["real_vae_tiler_max_error"]==0
    assert metrics["snapshot_vs_previous_final_max_error_255"]==0
    input_path=Path(metrics["input"])
    assert sha(input_path)==metrics["input_sha256"]
    provenance=json.loads(input_path.with_name("input_metadata.json").read_text())
    assert sha(input_path)==provenance["copied_npz_file_sha256"]==sha(provenance["original_final_npz"])
    assert sha(input_path.with_name("raw.png"))==provenance["copied_raw_file_sha256"]==sha(provenance["original_raw_png"])
    with np.load(input_path,allow_pickle=False) as data: packed=data["latents"].copy()
    base=packed.reshape(1,72,32,64).transpose(0,3,1,2)[:,:,None]
    mask=np.zeros(base.shape,dtype=bool)
    collar=metrics["parameters"]["latent_collar"]
    for center in (20,52): mask[:,:,:,center-collar:center+collar]=True
    shared=json.loads((Path(__file__).resolve().parents[2]/"experiments/evaluation/inputs/inputs.json").read_text())
    track=next(t for t in shared["tracks"] if t["id"]=="track3")
    source=image(track["source_512"])
    assert hashlib.sha256(source.tobytes()).hexdigest()==metrics["source_sha256"]
    assert sha(track["original_jpg"])==metrics["original_jpg_sha256_before"]==metrics["original_jpg_sha256_after"]
    baseline=image(directory/"baseline-raw.png")
    assert np.array_equal(baseline,image(input_path.with_name("raw.png")))
    assert image_texture_profiles(np,Image.fromarray(baseline))==metrics["baseline_raw_texture_profiles"]
    baseline_comp=image(directory/"baseline-composite.png")
    assert np.array_equal(baseline_comp[320:832],source)
    assert np.array_equal(baseline_comp[:320],baseline[:320]) and np.array_equal(baseline_comp[832:],baseline[832:])
    arms={}
    for method,arm in metrics["runs"].items():
        steps=metrics["parameters"]["iterations"]
        assert arm["status"]=="success" and len(arm["iteration_records"])==steps
        assert [r["iteration"] for r in arm["iteration_records"]]==list(range(1,steps+1))
        with np.load(directory/method/"latest_iteration_state.npz",allow_pickle=False) as state:
            fields={k:state[k].copy() for k in state.files}
        assert int(fields["iteration"])==steps and str(fields["input_sha256"])==metrics["input_sha256"]
        assert str(fields["source_sha256"])==metrics["source_sha256"] and str(fields["runner_sha256"])==metrics["runner_sha256"]
        assert str(fields["method"])==method and json.loads(str(fields["parameters_json"]))==metrics["parameters"]
        for name in ("delta","moment","variance"):
            assert fields[name].shape==base.shape and fields[name].dtype==np.float32 and np.isfinite(fields[name]).all()
            assert np.all(fields[name][~mask]==0)
        assert np.max(np.abs(fields["delta"]))<=metrics["parameters"]["max_delta"]+1e-7
        assert int(fields["observed_mlx_peak_bytes"])==arm["iteration_records"][-1]["mlx_peak_since_arm_reset_bytes"]
        with np.load(directory/method/"projected_latents.npz",allow_pickle=False) as final:
            actual=final["latents"].copy()
            assert final["source_rect_xyxy"].tolist()==[0,320,512,832]
        projected=(base+fields["delta"])[:,:,0].transpose(0,2,3,1).reshape(1,2304,64)
        assert np.array_equal(actual,projected)
        assert np.array_equal(projected.reshape(1,72,32,64).transpose(0,3,1,2)[:,:,None][~mask],base[~mask])
        raw=image(directory/method/"raw.png"); composite=image(directory/method/"composite.png")
        assert raw.shape==composite.shape==(1152,512,3)
        assert np.array_equal(composite[320:832],source)
        assert np.array_equal(composite[:320],raw[:320]) and np.array_equal(composite[832:],raw[832:])
        assert sha(directory/method/"raw.png")==arm["raw_png_sha256"] and sha(directory/method/"composite.png")==arm["composite_png_sha256"]
        assert image_texture_profiles(np,Image.fromarray(raw))==arm["raw_texture_profiles"]
        error=np.abs(raw.astype(np.int16)-baseline.astype(np.int16))
        distance=max(128,collar*16)
        far=np.concatenate((error[:320-distance].reshape(-1),error[832+distance:].reshape(-1)))
        changed=far!=0
        arms[method]={"iterations":steps,"atomic_state_valid":True,"state_arrays_finite":True,
                      "projected_latents_equal_base_plus_saved_delta":True,"outside_collar_latents_exact":True,
                      "source_pixels_exact":True,"final_exterior_equals_raw":True,
                      "far_pixel_values_exact":bool(not changed.any()),"far_mae_255":float(far.mean()),
                      "far_max_change_255":int(far.max()),"far_changed_channel_fraction":float(changed.mean()),
                      "max_abs_latent_delta":float(np.max(np.abs(fields["delta"]))),
                      "saved_spectral_and_row_metrics_reproduced":True,
                      "raw_png_sha256":sha(directory/method/"raw.png"),"composite_png_sha256":sha(directory/method/"composite.png")}
        assert abs(float(far.mean())-arm["raw_generated_far_from_collar_change_mae_255"])<1e-7
    result={"status":"passed","device":"CPU NumPy/Pillow only; no model/GPU imports",
            "input_npz_raw_and_original_jpg_unchanged":True,"exact_baseline_pixels":True,
            "runner_sha256":metrics["runner_sha256"],"arms":arms,
            "mlx_observed_peak_across_scopes_bytes":metrics["mlx_observed_peak_across_scopes_bytes"],
            "quality_verdict":"not established by integrity checks","matched_checkpoint_off_control_run_performed":False,
            "weights_trained":False,"script_sha256":sha(__file__)}
    output=directory/"independent_state_pixel_integrity.json"
    output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"status":"passed","source_and_latent_scope_exact":True,"arms":arms,"output":str(output)},indent=2))


if __name__=="__main__": main()
