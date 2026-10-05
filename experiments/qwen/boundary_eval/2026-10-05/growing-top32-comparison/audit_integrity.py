#!/usr/bin/env python3
"""Independent disk/NumPy audit. No MLX import, model loads, or mutations to trials."""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[5]
TRIAL = ROOT / "experiments/qwen/controlnet_runs/2026-10-05/track3-growing-top32-6step"
BASELINE = TRIAL.with_name("track3-growing-localized6")
PORT = ROOT / "experiments/qwen/controlnet_port"
OUT = Path(__file__).with_name("independent_validation.json")
HEIGHTS = [640, 896, 1152, 1152, 1152, 1152]
checks = []
arrays_report = {}
files_report = {}


def file_sha(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    files_report[str(path)] = {"sha256": value, "bytes": path.stat().st_size}
    return value


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def check(name, ok, detail=None):
    result = {"name": name, "passed": bool(ok)}
    if detail is not None:
        result["detail"] = detail
    checks.append(result)
    return bool(ok)


def image(path, mode="RGB"):
    with Image.open(path) as handle:
        return np.asarray(handle.convert(mode)).copy()


def load_npz(path):
    with np.load(path, allow_pickle=False) as values:
        result = {name: values[name].copy() for name in values.files}
    file_sha(path)
    for name, value in result.items():
        arrays_report[f"{path}:{name}"] = {"sha256": array_sha(value), "shape": list(value.shape),
            "dtype": str(value.dtype), "finite": bool(np.isfinite(value).all())}
    return result


def bf16_values(value):
    bits = np.asarray(value, dtype=np.float32).view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def active_ids(height):
    y0 = (1152 - height) // 2
    return np.arange((y0 // 16) * 32, ((y0 + height) // 16) * 32, dtype=np.int32), [0, y0, 512, y0 + height]


def sampling_nodes(path):
    retained = {"ids", "index", "added", "active", "active_context", "known_mask_np", "known_mask", "known_clean",
        "latents", "before", "model_input", "timestep", "prediction", "advanced", "clean", "previous_clean", "previous_ids"}
    tree = ast.parse(path.read_text())
    loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For)
        and isinstance(node.target, ast.Tuple)
        and [getattr(item, "id", None) for item in node.target.elts] == ["step", "active_height"])
    result = []
    for node in loop.body:
        if isinstance(node, ast.Assign) and any(isinstance(item, ast.Name) and item.id in retained
            for target in node.targets for item in ast.walk(target)):
            result.append(node)
        elif isinstance(node, ast.If) and any(isinstance(item, ast.Name) and item.id == "previous_ids" for item in ast.walk(node.test)):
            result.append(node)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and ast.unparse(node.value) == "mx.eval(latents, clean)":
            result.append(node)
    return ast.dump(ast.Module(body=result, type_ignores=[]), include_attributes=False), len(result)


def main():
    launch = json.loads((TRIAL / "launch.json").read_text())
    suite = json.loads((TRIAL / "growing_top32_metrics.json").read_text())
    old = json.loads((BASELINE / "growing_localized_metrics.json").read_text())
    summary = {"status": "running", "device": "CPU NumPy/Pillow only; no MLX imported", "trial_directory": str(TRIAL),
        "baseline_directory": str(BASELINE), "launch_status_read_first": launch,
        "checks": checks, "array_inventory": arrays_report, "file_inventory": files_report,
        "limitations": [
            "This is a saved-artifact/static-code audit; GPU execution is not repeated.",
            "Per-step source-prefix input equality was checked by the runner immediately before each forward. Actual per-block hidden states were not retained, so this audit does not independently establish a hidden-state trace.",
            "Per-step active context/hint hashes are reconstructed from exact saved full tensors and absolute IDs, then matched to the runner's logged actual gathers. Per-step context tensor files and RoPE values were not retained.",
            "Predicted-clean previews retain actual latent arrays, IDs, windows, and sigma, and match source-preserving decode outputs. Pre-step states and velocities were not saved, so the predicted-clean arithmetic cannot be independently recomputed.",
            "NPZ preview container/array hashes computed here are current provenance; the earlier run did not persist historical preview-latent hashes.",
            "Checkpoint/base weight payloads and GPU determinism were not reverified. Header/config/runtime provenance is retained; same-scale baseline was not rerun.",
            "The older cached geometry baseline was already numerically inexact; no exact reproduction of that pipeline is claimed.",
            "Visual quality and boundary/tangent evaluation are separate from this integrity audit."]}
    try:
        check("launch exited successfully", launch.get("status") == "success" and launch.get("exit_code") == 0)
        check("suite completed successfully", suite.get("status") == "success" and suite.get("phase") == "complete")
        launch_records = []
        for line in (TRIAL / "process.log").read_text().splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                launch_records.append(item)
        check("launch terminal log ends in success", launch_records[-1] ==
            {"phase":"complete", "elapsed_seconds":round(suite["elapsed_seconds"], 2), "status":"success"})
        check("launch log contains exactly twelve denoiser calls", sum(item.get("phase") == "denoising" for item in launch_records) == 12)
        keys = ["tangent-canny/scale0.5", "tangent-canny/scale1"]
        check("exactly two requested completed arms", set(suite["runs"]) == set(keys) and suite["control_scales"] == [0.5, 1.0])
        check("bounded elapsed time", suite["elapsed_seconds"] < suite["max_seconds"], suite["elapsed_seconds"])
        check("saved baseline reused without rerun", suite["actual_saved_gpu_baseline_reused"] and not suite["actual_gpu_baseline_rerun_performed"])
        check("no new inference encodes or RNG draws", suite["inference_encodes_performed"] == 0 and suite["inference_rng_draws_performed"] == 0)
        check("payload revalidation limitation declared", not suite["checkpoint_payload_hash_reverified"] and not suite["base_weight_payload_hashes_reverified"])
        runner = PORT / "run_growing_top32_control_trial.py"
        check("current frozen runner hash", file_sha(runner) == suite["runner_sha256"])
        check("launch runner hash agrees", launch["code_sha256"].get(str(runner)) == suite["runner_sha256"])
        for name, expected in suite["source_code_sha256"].items():
            check(f"current helper/script hash {name}", file_sha(PORT / name) == expected)
        for name, item in suite["runtime_dependency_hashes"].items():
            check(f"current runtime hash {name}", file_sha(item["path"]) == item["sha256"])
        check("runtime equals retained baseline", suite["runtime_dependency_hashes"] == old["runtime_dependency_hashes"])
        cpu_gate = PORT / "growing_known_collar_cpu_validation.json"
        gate = json.loads(cpu_gate.read_text())
        check("passed CPU gate hash current", file_sha(cpu_gate) == suite["growing_top32_cpu_validation_sha256"] and gate["status"] == "passed")
        for field, filename in [("helper_sha256", "growing_known_collar_saved_inputs.py"),
            ("script_sha256", "test_growing_known_collar_cpu.py"), ("frozen_growing_helper_sha256", "growing_localized_control.py"),
            ("frozen_localized_helper_sha256", "localized_reference_control.py")]:
            check(f"CPU gate {field} current", gate[field] == file_sha(PORT / filename))
        check("CPU gate runtime hashes current", gate["runtime_dependency_hashes"] == suite["runtime_dependency_hashes"])
        before, count_before = sampling_nodes(PORT / "run_growing_localized_control_trial.py")
        after, count_after = sampling_nodes(runner)
        check("retained twenty sampling statements AST exact", before == after and count_before == count_after == 20)
        tree = ast.parse(runner.read_text())
        protocol_node = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "frozen_protocol" for target in node.targets))
        protocol = ast.literal_eval(protocol_node.value)
        for key in protocol:
            check(f"retained recipe exact {key}", suite[key] == old[key])
        summary["retained_recipe_keys"] = list(protocol)
        baseline_recipe_names = suite["saved_baseline_validation"]["recipe_keys_exact"]
        summary["saved_helper_recipe_scope"] = {
            "baseline_validation_list_size":len(baseline_recipe_names),
            "direct_top_level_equal_names":[key for key in baseline_recipe_names if key in suite and suite[key] == old[key]],
            "intentionally_different_names":[key for key in baseline_recipe_names if key in suite and suite[key] != old[key]],
            "baseline_only_names":[key for key in baseline_recipe_names if key not in suite],
            "note":"The helper's 49-item list validates the retained baseline. The current top32 suite copies a separate 51-field protocol; those copied fields are independently compared above. Executed scales differ intentionally, and four baseline metadata fields are not copied into the new suite."}

        retained_claims = suite["saved_baseline_validation"]["persisted_file_sha256"]
        baseline_count = external_count = 0
        for path, expected in retained_claims.items():
            candidate = Path(path)
            inside = candidate.is_relative_to(BASELINE)
            baseline_count += int(inside); external_count += int(not inside)
            check(f"retained historical file unchanged {path}", candidate.is_file() and file_sha(candidate) == expected)
        check("retained claim set has 91 files", len(retained_claims) == 91)
        summary["retained_historical_files"] = {"total": len(retained_claims), "baseline_artifacts": baseline_count,
            "external_code_runtime_inputs_configs": external_count,
            "claim_set_sha256": hashlib.sha256(json.dumps(retained_claims, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "baseline_files_without_historical_claim": sorted(str(path.relative_to(BASELINE)) for path in BASELINE.rglob("*")
                if path.is_file() and str(path) not in retained_claims)}
        inputs = json.loads((ROOT / "experiments/evaluation/inputs/inputs.json").read_text())
        track = next(item for item in inputs["tracks"] if item["id"] == "track3")
        check("original JPEG unchanged", file_sha(track["original_jpg"]) == track["original_jpg_sha256"]
            == suite["saved_baseline_validation"]["original_jpeg_sha256"])
        source = image(suite["source_512_path"])
        source_rgba = image(suite["source_512_path"], "RGBA")
        check("current source PNG unchanged", file_sha(suite["source_512_path"]) == suite["source_512_sha256"])
        check("source is 512 square", source.shape == (512, 512, 3))
        check("RGB source pixel hash", array_sha(source) == suite["source_uint8_sha256"])
        check("RGBA source pixel hash and opaque alpha", array_sha(source_rgba) == suite["source_rgba_uint8_sha256"] and np.all(source_rgba[..., 3] == 255))
        check("saved trial source pixels exact", np.array_equal(image(TRIAL / "source_rgba.png", "RGBA"), source_rgba))
        for mode, map_path in suite["map_paths"].items():
            check(f"current control map hash {mode}", file_sha(map_path) == suite["map_sha256"][mode])
        check("current guide file hash", file_sha(suite["guide_path"]) == suite["localized_hint_weights"]["guide_file_sha256"])

        loaded, baseline_loaded = {}, {}
        shared_files = ["reference_conditioning.npz", "edge_known_target.npz", "target_noise.npz", "tangent-canny_contexts.npz", "effective_hint_weights.npz"]
        for name in shared_files:
            loaded[name], baseline_loaded[name] = load_npz(TRIAL / "shared" / name), load_npz(BASELINE / "shared" / name)
        for name, expected in suite["shared_file_sha256"].items():
            check(f"shared recorded container hash {name}", file_sha(TRIAL / "shared" / name) == expected)
        for filename in shared_files[:-1]:
            check(f"shared tensor keys preserved {filename}", loaded[filename].keys() == baseline_loaded[filename].keys())
            for field, value in loaded[filename].items():
                check(f"saved actual tensor exact {filename}:{field}", np.array_equal(value, baseline_loaded[filename][field]))
                check(f"shared tensor finite {filename}:{field}", np.isfinite(value).all())
                if value.dtype == np.float32:
                    check(f"saved BF16 roundtrip exact {filename}:{field}", np.array_equal(value, bf16_values(value)))
        reference = loaded["reference_conditioning.npz"]
        context = loaded["tangent-canny_contexts.npz"]["target_context"]
        padded = loaded["tangent-canny_contexts.npz"]["padded_image_context"]
        support = loaded["tangent-canny_contexts.npz"]["structural_support"]
        known = loaded["edge_known_target.npz"]["known"]
        edge_guess = loaded["edge_known_target.npz"]["edge_guess"]
        noise = loaded["target_noise.npz"]["noise"]
        hint = loaded["effective_hint_weights.npz"]
        old_hint = baseline_loaded["effective_hint_weights.npz"]
        expected_shapes = [("source_latents", reference["source_latents"], (1,1024,64)), ("prompt_embeds", reference["prompt_embeds"], (1,429,4096)),
            ("image_slots", reference["image_slots"], (429,)), ("context129", context, (1,2304,129)), ("padded_context129", padded, (1,3328,129)),
            ("support", support, (1,2304,1)), ("edge_known", known, (1,1024,64)), ("edge_guess", edge_guess, (1,2304,64)), ("noise", noise, (1,2304,64))]
        for name, value, shape in expected_shapes:
            check(f"actual saved shape {name}", value.shape == shape)
        check("source-prefix array hash", array_sha(reference["source_latents"]) == suite["reference_encoding"]["source_prefix_sha256"])
        check("prompt embeddings array hash", array_sha(reference["prompt_embeds"]) == suite["prompt_embeds_sha256"])
        check("vision slot array hash and 256 slots", array_sha(reference["image_slots"]) == suite["image_slots_bool_sha256"] and reference["image_slots"].dtype == bool and reference["image_slots"].sum() == 256)
        prefix = 1024 + len(reference["image_slots"]) - int(reference["image_slots"].sum())
        check("derived fixed prefix length", prefix == suite["layout"]["joint_prefix_tokens"] == 1197)
        check("noise array hash", array_sha(noise) == suite["noise_sha256"])
        check("edge-known array hash", array_sha(known) == suite["edge_known_target_sha256"])
        codec_difference = np.abs(known-context[:,640:1664,65:])
        check("edge-known codec differs from masked-source129 codec", not np.array_equal(known,context[:,640:1664,65:]))
        summary["distinct_edge_known_and_control_source_codec"] = {"arrays_equal":False,
            "max_abs":float(codec_difference.max()), "mean_abs":float(codec_difference.mean())}
        edge_ids = ((np.clip(np.arange(72),20,51)-20)[:,None]*32+np.arange(32)[None,:]).reshape(-1)
        check("edge guess clips original known codec raster rows", np.array_equal(edge_guess, known[:,edge_ids]))
        mask = np.zeros((1,2304,1), dtype=np.float32); mask[:,640:1664] = 1
        check("known mask exact rows20:52", np.array_equal(context[:,:,64:65],mask))
        check("reference image context has literal zero129", np.all(padded[:,:1024] == 0) and np.array_equal(padded[:,1024:], context))
        check("mask/source65 conditioning unchanged", np.array_equal(context[:,:,64:], baseline_loaded["tangent-canny_contexts.npz"]["target_context"][:,:,64:]))
        check("structural64 conditioning unchanged", np.array_equal(context[:,:,:64], baseline_loaded["tangent-canny_contexts.npz"]["target_context"][:,:,:64]))
        check("unsupported structural64 literal zero", np.all(context[:,~support.reshape(-1),:64] == 0))
        check("support boolean and known source retained", support.dtype == bool and np.all(support[:,640:1664]))
        check("support .npy equals native saved support", np.array_equal(np.load(TRIAL/"shared/support_tokens.npy",allow_pickle=False),support))
        check("support PNG unchanged", file_sha(TRIAL/"shared/support.png") == file_sha(BASELINE/"shared/support.png"))
        for field in ["target_weights", "requested_target_weights"]:
            check(f"hint shape {field}", hint[field].shape == (1,2304,1))
            changed = np.flatnonzero((hint[field] != old_hint[field]).reshape(-1))
            check(f"only hint IDs640:704 changed {field}", np.array_equal(changed,np.arange(640,704)))
            check(f"top32 scalar one {field}", np.all(hint[field][:,640:704] == 1))
            check(f"all other known hints zero {field}", np.all(hint[field][:,704:1664] == 0))
            check(f"unknown hints exact saved values {field}", np.array_equal(hint[field][:,np.r_[0:640,1664:2304]],old_hint[field][:,np.r_[0:640,1664:2304]]))
        check("saved old effective hint exact", np.array_equal(hint["old_target_weights"],old_hint["target_weights"]))
        check("effective hint matches requested BF16 rounding", np.array_equal(hint["target_weights"],bf16_values(hint["requested_target_weights"])))
        check("full prefix hint weights literal zero", hint["joint_weights"].shape == (1,3501,1) and np.all(hint["joint_weights"][:,:prefix] == 0))
        check("joint target weights aligned", np.array_equal(hint["joint_weights"][:,prefix:],hint["target_weights"]))
        check("effective target hint array hash", array_sha(hint["target_weights"]) == suite["localized_hint_weights"]["effective_target_weights_float32_sha256"])
        check("requested target hint array hash", array_sha(hint["requested_target_weights"]) == suite["localized_hint_weights"]["target_weights_float32_sha256"])
        check("joint hint array hash", array_sha(hint["joint_weights"]) == suite["localized_hint_weights"]["effective_joint_weights_float32_sha256"])
        old_pixels = np.load(BASELINE/"shared/hint_pixel_weights_float32.npy",allow_pickle=False)
        pixels = np.load(TRIAL/"shared/hint_pixel_weights_float32.npy",allow_pickle=False)
        expected_pixels = old_pixels.copy(); expected_pixels[320:352] = 1
        check("only source top32 pixel weights changed", np.array_equal(pixels,expected_pixels))
        check("pixel nearest sample reproduces requested weights", np.array_equal(pixels[::16,::16].reshape(1,2304,1),hint["requested_target_weights"]))
        check("target .npy equals requested saved weights", np.array_equal(np.load(TRIAL/"shared/hint_target_weights_float32.npy",allow_pickle=False),hint["requested_target_weights"]))
        check("pixel hint array hash", array_sha(pixels) == suite["localized_hint_weights"]["pixel_weights_float32_sha256"])
        check("positive hint tokens equals97+64", int((hint["target_weights"]>0).sum()) == suite["localized_hint_weights"]["positive_hint_target_tokens"] == 161)
        sigma_nodes = np.array([1,.9375,.875,.75,.5,.25],dtype=np.float64)
        mu = .5 + (.9-.5)*(2304-256)/(8192-256)
        sigmas = np.r_[math.exp(mu)/(math.exp(mu)+(1/sigma_nodes-1)),0].astype(np.float32)
        check("original target2304 shifted six sigmas exact", np.array_equal(sigmas,np.asarray(suite["sigmas"],dtype=np.float32)))
        retained_arrays = {**reference, "target_context":context, "padded_image_context":padded, "support":support,
            "noise":noise, "edge_known_target":known, "edge_guess":edge_guess,
            "old_effective_target_hint":old_hint["target_weights"], "old_effective_joint_hint":old_hint["joint_weights"],
            "old_requested_target_hint":old_hint["requested_target_weights"], "requested_top32_hint":hint["requested_target_weights"],
            "expected_effective_top32_hint":hint["target_weights"], "expected_effective_top32_joint_hint":hint["joint_weights"], "sigmas":sigmas}
        for name,expected in suite["saved_baseline_validation"]["persisted_array_sha256"].items():
            check(f"retained sixteen-array historical hash {name}",array_sha(retained_arrays[name]) == expected)
        summary["restored_inference_tensors"] = {"all_common_tensor_arrays_exact_against_saved": True,
            "hint_changed_ids_half_open": [640,704], "hint_changed_tokens": 64,
            "prefix_tokens": prefix, "target_tokens": 2304, "known_tokens": 1024,
            "known_collar_tokens": 64, "other_known_hint_tokens_zero": 960,
            "unknown_positive_hint_tokens_retained": int((old_hint["target_weights"]>0).sum()),
            "effective_hint_sum": float(hint["target_weights"].sum()), "requested_hint_sum": float(hint["requested_target_weights"].sum())}

        arms = {}
        for key in keys:
            directory = TRIAL/key
            arm = suite["runs"][key]
            persisted_arm = json.loads((directory/"metrics.json").read_text())
            check(f"per-arm metrics exact suite {key}", persisted_arm == arm)
            check(f"per-arm completed {key}", arm["status"] == "success")
            for name,expected in [("steps",6),("seed",42),("cfg",1.0),("base_block_calls",192),("control_block_calls",96),
                ("transformer_calls",6),("target_token_forward_sum",12288),("reference_image_token_forward_sum",6144),
                ("joint_query_token_forward_sum",19470),("active_heights",HEIGHTS)]:
                check(f"per-arm protocol/count {key}:{name}", arm[name] == expected)
            for name in ["source_prefix_sha256","noise_sha256","known_target_sha256","sigmas","prompt"]:
                check(f"per-arm original input exact {key}:{name}", arm[name] == old["runs"][key][name])
            check(f"per-arm caches disabled {key}", not arm["base_prefix_cache_enabled"] and not arm["control_prefix_cache_enabled"])
            check(f"per-arm source-prefix observed checks declared {key}", arm["source_prefix_exact_each_forward"] and all(row["source_prefix_input_exact"] for row in arm["step_records"]))
            check(f"per-arm six actual step records {key}", len(arm["step_records"]) == 6)
            step_details = []
            previous_count = 0
            for index,(row,active_height) in enumerate(zip(arm["step_records"],HEIGHTS)):
                ids,window = active_ids(active_height)
                context_hash = array_sha(context[:,ids])
                hint_hash = array_sha(hint["target_weights"][:,ids])
                expected = {"step":index+1,"sigma":float(sigmas[index]),"next_sigma":float(sigmas[index+1]),
                    "active_height":active_height,"active_window_xyxy":window,"target_tokens_forwarded":len(ids),
                    "future_target_tokens_absent":2304-len(ids),"newly_activated_tokens":len(ids)-previous_count,
                    "input_image_latent_tokens":1024+len(ids),"control_image_rows":1024+len(ids),
                    "base_joint_query_tokens":prefix+len(ids),"control_joint_query_tokens":prefix+len(ids),
                    "prefix_recomputed_both_chains":True,"base_blocks":32,"control_blocks":16,
                    "known_source_hint_zero_outside_top32":True,"top32_hint_scalar_one_tokens":64,
                    "top32_hint_absolute_ids":[640,704],"source_prefix_input_exact":True,
                    "active_context_sha256":context_hash,"active_hint_weights_sha256":hint_hash}
                for name,value in expected.items():
                    check(f"step actual gather/layout/count {key}:{index+1}:{name}",row[name] == value)
                terminal_rows = [item for item in launch_records if item.get("phase") == "denoising"
                    and item.get("arm") == key and item.get("step") == index+1]
                check(f"step launch log exact observed row {key}:{index+1}", len(terminal_rows) == 1
                    and {name:value for name,value in terminal_rows[0].items() if name not in {"phase","arm"}} == row)
                check(f"step active known hint invariants {key}:{index+1}",
                    np.all(hint["target_weights"][:,ids[(ids>=640)&(ids<704)]]==1)
                    and np.all(hint["target_weights"][:,ids[(ids>=704)&(ids<1664)]]==0))
                step_details.append({"step":index+1,"absolute_target_ids_half_open":[int(ids[0]),int(ids[-1])+1],
                    "window_xyxy":window,"target_tokens":len(ids),"future_targets_absent":2304-len(ids),
                    "active_context_sha256_recomputed":context_hash,"active_hint_weights_sha256_recomputed":hint_hash})
                previous_count = len(ids)
            final = load_npz(directory/"final_latents.npz")["latents"]
            baseline_final = load_npz(BASELINE/key/"final_latents.npz")["latents"]
            check(f"final tensor shape/finite {key}", final.shape == (1,2304,64) and np.isfinite(final).all())
            check(f"final latent array hash {key}", array_sha(final) == arm["final_latents_sha256"])
            check(f"final known target codec exact at sigma0 {key}",np.array_equal(final[:,640:1664],known))
            check(f"final known target restoration declaration {key}",arm["known_target_exact_at_sigma_zero"])
            difference = np.abs(final-baseline_final)
            comparison = arm["saved_same_scale_baseline_comparison"]
            recomputed = {"final_latents_exact":bool(np.array_equal(final,baseline_final)),"max_abs":float(difference.max()),"mean_abs":float(difference.mean())}
            for name,value in recomputed.items():
                check(f"same-scale numerical difference reproduced {key}:{name}",comparison[name] == value)
            check(f"same-scale baseline final hash {key}",array_sha(baseline_final) == comparison["baseline_final_latents_sha256"])
            check(f"same-scale baseline not rerun {key}",comparison["actual_gpu_baseline_rerun_performed"] is False)
            raw,composite = image(directory/"raw.png"),image(directory/"composite.png")
            check(f"raw/composite geometry {key}",raw.shape == composite.shape == (1152,512,3))
            check(f"raw/composite container hashes {key}",file_sha(directory/"raw.png") == arm["raw_sha256"] and file_sha(directory/"composite.png") == arm["composite_sha256"])
            check(f"final hardpaste source byte exact {key}",np.array_equal(composite[320:832],source))
            check(f"final exterior raw pixels preserved {key}",np.array_equal(raw[:320],composite[:320]) and np.array_equal(raw[832:],composite[832:]))
            error = np.abs(raw[320:832].astype(np.int16)-source.astype(np.int16))
            maes = {"raw_source_top16_mae_255":float(error[:16].mean()),"raw_source_bottom16_mae_255":float(error[-16:].mean()),
                "raw_source_inner_mae_255":float(error[16:-16].mean()),"raw_source_mae_255":float(error.mean())}
            for name,value in maes.items():
                check(f"raw source reconstruction metric {key}:{name}",arm[name] == value)
            check(f"retained baseline raw hash {key}",file_sha(BASELINE/key/"raw.png") == comparison["baseline_raw_sha256"])
            check(f"retained baseline composite hash {key}",file_sha(BASELINE/key/"composite.png") == comparison["baseline_composite_sha256"])
            check(f"previews exactly steps2/4/6 {key}",[item["step"] for item in arm["previews"]] == [2,4,6])
            previews = []
            for record in arm["previews"]:
                step = record["step"]; active_height = HEIGHTS[step-1]; ids,window = active_ids(active_height)
                stem = f"step{step:02d}_predicted_clean"
                preview_file = directory/"previews"/(stem+".npz")
                preview = load_npz(preview_file)
                for name,expected in [("final_canvas_ids",ids),("window_xyxy",np.asarray(window)),("sigma",np.asarray(sigmas[step-1]))]:
                    check(f"genuine preview position/sigma {key}:{step}:{name}",np.array_equal(preview[name],expected))
                check(f"genuine preview active latent shape {key}:{step}",preview["latents"].shape == (1,len(ids),64) and np.isfinite(preview["latents"]).all())
                local_known = (ids>=640)&(ids<1664)
                check(f"preview known codec restored {key}:{step}",np.array_equal(preview["latents"][:,local_known],known))
                check(f"preview metadata actual path/window {key}:{step}",Path(record["latent_path"]) == preview_file and record["active_window_xyxy"] == window and record["active_height"] == active_height)
                p_raw,p_comp = image(record["raw_path"]),image(record["composite_path"])
                y0 = 320-window[1]
                check(f"preview raw/composite active image size {key}:{step}",p_raw.shape == p_comp.shape == (active_height,512,3))
                check(f"preview raw/composite recorded hash {key}:{step}",file_sha(record["raw_path"]) == record["raw_sha256"] and file_sha(record["composite_path"]) == record["composite_sha256"])
                check(f"preview source hardpaste exact {key}:{step}",np.array_equal(p_comp[y0:y0+512],source))
                check(f"preview exterior raw pixels preserved {key}:{step}",np.array_equal(p_raw[:y0],p_comp[:y0]) and np.array_equal(p_raw[y0+512:],p_comp[y0+512:]))
                check(f"preview definition honest {key}:{step}",record["definition"] == "pre-step z_sigma-sigma*velocity; edge-known clean target restored")
                previews.append({"step":step,"active_height":active_height,"window_xyxy":window,"local_source_y":y0,
                    "latent_array_sha256_current":array_sha(preview["latents"]),"latent_container_sha256_current":file_sha(preview_file),
                    "final_latents_exact":bool(np.array_equal(preview["latents"],final))})
            arms[key] = {"status":arm["status"],"step_gathers":step_details,"nfe":arm["transformer_calls"],
                "base_blocks":arm["base_block_calls"],"control_blocks":arm["control_block_calls"],
                "target_token_forwards":arm["target_token_forward_sum"],"reference_image_token_forwards":arm["reference_image_token_forward_sum"],
                "final_known_codec_exact":bool(np.array_equal(final[:,640:1664],known)),"source_pixels_exact":bool(np.array_equal(composite[320:832],source)),
                "same_scale_final_latent_difference":recomputed,"raw_source_maes":maes,"previews":previews}
        summary["arms"] = arms
        total = {"nfe":sum(arm["nfe"] for arm in arms.values()),"base_blocks":sum(arm["base_blocks"] for arm in arms.values()),
            "control_blocks":sum(arm["control_blocks"] for arm in arms.values()),"target_token_forwards":sum(arm["target_token_forwards"] for arm in arms.values()),
            "reference_image_token_forwards":sum(arm["reference_image_token_forwards"] for arm in arms.values())}
        check("suite totals12NFE/24576targets/12288reference",total == {"nfe":12,"base_blocks":384,"control_blocks":192,"target_token_forwards":24576,"reference_image_token_forwards":12288})
        summary["recomputed_totals"] = total
        summary["elapsed_seconds_runner"] = suite["elapsed_seconds"]
        summary["older_cached_geometry_comparison"] = suite["saved_baseline_validation"]["older_cached_geometry_comparison"]
        cached_report = summary["older_cached_geometry_comparison"]
        cached_path = Path(cached_report["path"])/"previews/step06_predicted_clean.npz"
        cached = load_npz(cached_path)["latents"]
        baseline_zero = load_npz(BASELINE/"tangent-canny/scale0/final_latents.npz")["latents"]
        cached_diff = np.abs(cached-baseline_zero)
        check("older cached geometry latent file hash retained",file_sha(cached_path) == cached_report["old_latents_file_sha256"])
        check("older cached geometry remains numerically inexact", bool(np.array_equal(cached,baseline_zero)) == cached_report["exact"] is False)
        check("older cached geometry max difference reproduced",float(cached_diff.max()) == cached_report["max_abs"])
        check("older cached geometry mean difference reproduced",float(cached_diff.mean()) == cached_report["mean_abs"])
        check("no unsupported hidden-state preservation claim",suite["known_target_hidden_trace_preserved"] is False and suite["saved_baseline_validation"]["unsaved_gpu_trace_claim"] is False)
        for path in sorted(TRIAL.rglob("*")):
            if path.is_file():
                file_sha(path)
        summary["status"] = "passed" if all(item["passed"] for item in checks) else "failed"
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = {"type":type(exc).__name__,"message":str(exc)}
    summary["checks_total"] = len(checks)
    summary["checks_passed"] = sum(item["passed"] for item in checks)
    summary["failed_checks"] = [item for item in checks if not item["passed"]]
    summary["audit_script_sha256"] = file_sha(Path(__file__))
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({"status":summary["status"],"checks":summary["checks_total"],"passed":summary["checks_passed"],
        "failed":summary["failed_checks"],"error":summary.get("error"),"output":str(OUT),
        "recomputed_totals":summary.get("recomputed_totals")},indent=2))
    if summary["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
