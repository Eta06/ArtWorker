#!/usr/bin/env python3
"""CPU-only diagnostics for exact projection input and descriptive metrics."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

import run_saved_top32_checkpoint_projection16 as instrumented

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def node(path, function):
    tree = ast.parse(Path(path).read_text())
    found = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function)
    return ast.dump(found, include_attributes=False)


def main():
    here = Path(__file__).parent
    old = here / "run_saved_top32_checkpoint_projection.py"
    new = here / "run_saved_top32_checkpoint_projection16.py"
    unchanged = ("validate_input_metadata", "original_raw_path", "checkpointed_vae_decode",
                 "differentiable_tiled_decode", "cpu_smoke", "rgb_image", "crop_metrics", "boundary_decodes", "objective")
    for name in unchanged:
        assert node(old, name) == node(new, name), f"Original mathematical function changed: {name}"
    preflight_path = here / "projection_runs/2026-10-05/track3-saved-top32-checkpoint16-preflight-v2/preflight.json"
    preflight = json.loads(preflight_path.read_text())
    assert preflight["status"] == "preflight_passed"
    assert preflight["parameters"]["iterations"] == 16 and preflight["parameters"]["checkpoint_decoder"] is True
    assert preflight["parameters"]["methods"] == "consistency,derivative"
    assert preflight["runner_sha256"] == sha(new) and preflight["four_update_reference"]["runner_sha256"] == sha(old)
    provenance_path = here / "projection_inputs/2026-10-05/track3-saved-top32/input_metadata.json"
    provenance = json.loads(provenance_path.read_text())
    assert sha(provenance["copied_final_npz"]) == sha(provenance["original_final_npz"]) == provenance["copied_npz_file_sha256"]
    assert sha(Path(provenance["copied_final_npz"]).with_name("raw.png")) == sha(provenance["original_raw_png"]) == provenance["copied_raw_file_sha256"]
    with np.load(provenance["copied_final_npz"], allow_pickle=False) as copied, np.load(provenance["original_final_npz"], allow_pickle=False) as original:
        assert set(copied.files) == set(original.files)
        for key in copied.files:
            assert copied[key].dtype == original[key].dtype and np.array_equal(copied[key], original[key])
    constant = Image.fromarray(np.full((1152,512,3),128,dtype=np.uint8))
    constant_metrics = instrumented.image_texture_profiles(np, constant)
    assert all(v["high_frequency_power"] == 0 for v in constant_metrics["raw_patch_spectra"].values())
    yy,xx = np.indices((1152,512))
    alternating = np.where((yy+xx)%2,1,-1)
    def texture(amplitude, offset=128):
        return Image.fromarray(np.repeat((offset+amplitude*alternating).astype(np.uint8)[...,None],3,axis=2))
    weak = instrumented.image_texture_profiles(np, texture(10))
    stronger = instrumented.image_texture_profiles(np, texture(20))
    offset = instrumented.image_texture_profiles(np, texture(10,133))
    tests = []
    for name, values in weak["raw_patch_spectra"].items():
        power = values["high_frequency_power"]
        gain = stronger["raw_patch_spectra"][name]["high_frequency_power"]/power
        offset_ratio = offset["raw_patch_spectra"][name]["high_frequency_power"]/power
        assert abs(gain-4)<1e-5 and abs(offset_ratio-1)<1e-5
        tests.append({"patch":name,"double_amplitude_power_ratio":gain,"constant_offset_power_ratio":offset_ratio})
    raw = Image.open(provenance["original_raw_png"]).convert("RGB")
    actual = instrumented.image_texture_profiles(np, raw)
    assert actual == instrumented.image_texture_profiles(np, raw.copy())
    assert all(len(v["rows_y"])==16 and len(v["adjacent_rgb_mae"])==15 for v in actual["raw_two_sided_row_profiles"].values())
    reference = preflight["four_update_reference"]
    assert reference["automatic_resume"] is False
    for path, expected in reference["retained_file_sha256"].items():
        assert sha(path) == expected
    for method in ("consistency", "derivative"):
        with np.load(Path(reference["directory"])/method/"latest_iteration_state.npz",allow_pickle=False) as state:
            assert int(state["iteration"])==4
            for key in ("delta","moment","variance"):
                assert state[key].shape==(1,64,1,72,32) and np.isfinite(state[key]).all()
    result = {"status":"passed","device":"CPU NumPy/Pillow only; no model weights or GPU",
              "original_mathematical_function_ast_exact":list(unchanged),"input_npz_and_raw_files_exact":True,
              "all_array_values_and_dtypes_exact":True,"bounded_steps":16,"methods":"consistency,derivative",
              "decoder_checkpoint_enabled":True,"constant_image_high_frequency_zero":True,"metric_power_scaling_tests":tests,
              "actual_raw_metric_repeat_exact":True,"physical_temperature_measured":False,
              "checkpoint_memory_benefit_measured":False,"model_weights_trained":False,
              "source_code_sha256":{str(new.name):sha(new),str(old.name):sha(old),"prepare_saved_top32_projection_input.py":sha(here/"controlnet_port/prepare_saved_top32_projection_input.py")},
              "preflight_sha256":sha(preflight_path),"input_metadata_sha256":sha(provenance_path),"script_sha256":sha(__file__)}
    output = here / "saved_top32_projection16_cpu_validation.json"
    output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"status":"passed","objective_ast_exact":True,"input_files_exact":True,"output":str(output.resolve())},indent=2))


if __name__ == "__main__":
    main()
