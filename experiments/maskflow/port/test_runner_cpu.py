"""CPU orchestration gate, using synthetic tiny model components only.

This runs the real ``execute`` entry point, schedule/CFG/source bridge, pixel
blend, retained safetensors checkpoints, and resume validation. Full pretrained
weights, model quality, GPU memory and GPU speed are deliberately untested.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, redirect_stdout
from datetime import datetime, timezone
import gc
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import weakref
from unittest.mock import patch

import mlx.core as mx

# Select CPU before importing any code that could create MLX arrays.
mx.set_default_device(mx.cpu)

import numpy as np
from PIL import Image
import torch

import maskflow_adapter as adapter
import maskflow_checkpoint as checkpoint
import maskflow_conditioning as conditioning
import maskflow_loading as loading
import maskflow_math as math_port
import maskflow_transformer as transformer_port
import maskflow_vae as vae_port
import run_maskflow_trial as runner
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.qwen.model.qwen_text_encoder import qwen_text_encoder as text_module
from mflux.models.qwen.model.qwen_text_encoder import qwen_vision_transformer as vision_module
from mflux.models.qwen.model.qwen_text_encoder import qwen_vision_language_encoder as vl_module
from mflux.models.qwen.tokenizer import qwen_vision_language_processor as processor_module
from mflux.models.qwen.tokenizer import qwen_vision_language_tokenizer as tokenizer_module
from mflux.utils.runtime_memory import RuntimeMemory

HERE = Path(__file__).resolve().parent
SOURCE_FILES = [HERE / name for name in (
    "run_maskflow_trial.py", "maskflow_checkpoint.py", "maskflow_math.py",
    "maskflow_loading.py", "maskflow_conditioning.py", "maskflow_adapter.py",
    "maskflow_transformer.py", "maskflow_vae.py", "test_runner_cpu.py")]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def float_array(value):
    mx.eval(value)
    return np.asarray(value.astype(mx.float32))


class FakeEncoder:
    pass


class FakeMaskFlowEncoder(FakeEncoder):
    pass


class Probe:
    def __init__(self, name):
        self.name = name
        self.events = []
        self.weakrefs = []
        self.forward_calls = []
        self.poisson_calls = []
        self.encode_calls = []
        self.requested_devices = []
        self.memory_limits = []
        self.checkpoint_steps = []
        self.cfg_encodes = 0
        self.live_vl = None
        self.adapter_applied = False

    def require(self, condition, message):
        if not condition:
            raise AssertionError(message)

    def track(self, object_, name):
        self.weakrefs.append((name, weakref.ref(object_)))
        return object_

    def vl_absent(self):
        gc.collect()
        return all(reference() is None for _, reference in self.weakrefs)

    def assert_vl_absent(self, phase):
        self.require(self.vl_absent(), f"VL objects retained at {phase}")
        self.events.append(f"vl_absent:{phase}")


def run_fixture(directory, *, resume_from=None):
    probe = Probe("resume" if resume_from else "fresh")
    directory.mkdir()
    args = SimpleNamespace(output_dir=directory, track="track3", fill="black",
                           prompt_file=None, seed=42, resume_from=resume_from,
                           memory_limit_gib=26.0, cache_limit_gib=1.0)
    original_set_device = mx.set_default_device
    original_load = mx.load
    original_save_checkpoint = checkpoint.save_checkpoint
    original_validate = checkpoint.validate_checkpoint
    original_restore = checkpoint.restore_checkpoint
    captured_checkpoint = directory / "step20-receipt.json"

    def select_cpu(requested):
        probe.events.append("device_request")
        probe.requested_devices.append(str(requested))
        original_set_device(mx.cpu)

    def memory_limit(value):
        probe.memory_limits.append(value)

    def cpu_gates():
        probe.events.append("cpu_gates")
        return {"mocked_cpu_gate": "controlled-test-receipt"}

    def payload_gates():
        probe.events.append("payload_gates")
        return {"manifest_sha256": "controlled-test-receipt", "verified_files": 0}

    class FakeText:
        def __init__(self):
            probe.events.append("construct:text")
            self.encoder = probe.track(FakeEncoder(), "text_encoder")
            probe.track(self, "text")

    class FakeVision:
        def __init__(self):
            probe.events.append("construct:vision")
            probe.track(self, "vision")

    class FakeVL:
        def __init__(self, encoder):
            probe.events.append("construct:vl")
            self.encoder = encoder
            probe.track(self, "vl")

    class FakeProcessor:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
            probe.track(self, "processor")

    class FakeTokenizer:
        def __init__(self, processor, max_length, use_picture_prefix):
            probe.require(max_length == 1024 and use_picture_prefix, "Tokenizer construction changed")
            self.processor = processor
            probe.track(self, "tokenizer")

    class FakeRawTokenizer:
        def __init__(self):
            probe.track(self, "raw_tokenizer")

    def tokenizers(*_args):
        probe.events.append("load:tokenizers")
        return {"qwen": SimpleNamespace(tokenizer=FakeRawTokenizer())}

    def encode_cfg(**kwargs):
        probe.events.append("encode:both_cfg")
        probe.cfg_encodes += 1
        probe.require(isinstance(kwargs["qwen_vl_encoder"], FakeVL), "VL was not constructed")
        probe.require(kwargs["prompt"] and kwargs["negative_prompt"] == "", "CFG prompt policy changed")
        inputs = kwargs["vlm_conditions"]
        probe.require(list(inputs) == ["source", "mask"], "VLM condition order changed")
        probe.require(inputs["source"].shape == (1, 3, 576, 256), "VLM geometry changed")
        probe.require(not np.array_equal(inputs["source"], inputs["mask"]), "VLM source/mask collapsed")
        pm = mx.full((1, 5, 4), 0.75, dtype=mx.bfloat16)
        nm = mx.full((1, 3, 4), -0.375, dtype=mx.bfloat16)
        return {"pm": {"prompt_embeds": pm, "prompt_embeds_mask": mx.array([[1, 1, 1, 1, 0]])},
                "nm": {"prompt_embeds": nm, "prompt_embeds_mask": mx.array([[1, 1, 0]])},
                "metadata": {"mocked": True, "branches": {"pm": "distinct positive", "nm": "distinct empty-negative"}}}

    class FakeVAE:
        def __init__(self):
            probe.events.append("construct:vae")
            probe.assert_vl_absent("vae_construction")

        def encode(self, image):
            probe.require(image.shape == (1, 3, 1, 1152, 512), "VAE input geometry changed")
            probe.encode_calls.append(float_array(image[:, :, :, ::128, ::128]).tolist())
            probe.events.append(f"vae_encode:{len(probe.encode_calls)}")
            small = image[:, :, :, ::8, ::8]
            return mx.concatenate([small] * 6, axis=1)[:, :16].astype(mx.bfloat16)

        def decode(self, latents):
            probe.events.append("vae_decode")
            probe.require(latents.shape == (1, 16, 1, 144, 64), "VAE output geometry changed")
            # Deterministic synthetic decode is deliberately distinguishable
            # from the source even where the latent source projection is exact.
            pixels = mx.repeat(mx.repeat(latents[:, :3], 8, axis=3), 8, axis=4)
            return mx.clip(pixels * 0.5 + 0.125, -1, 1).astype(mx.bfloat16)

    class FakeTransformer:
        def __init__(self, zero_cond_t):
            probe.events.append("construct:transformer")
            probe.require(zero_cond_t is True, "2511 zero_cond_t not requested")
            probe.assert_vl_absent("dit_construction")

        def parameters(self):
            return {"synthetic": mx.array([0.125], dtype=mx.bfloat16)}

        def __call__(self, **kwargs):
            probe.require(probe.adapter_applied, "Adapter missing before forward")
            hidden = kwargs["hidden_states"]
            prompt_embeds = kwargs["encoder_hidden_states"]
            mask = kwargs["encoder_hidden_states_mask"]
            branch = "pm" if prompt_embeds.shape[1] == 5 else "nm"
            probe.require(hidden.shape == (1, 6912, 64), "Joint target/source/mask token geometry changed")
            probe.require(kwargs["cond_image_grid"] == [(1, 72, 32)] * 2, "Reference RoPE grids changed")
            probe.require(kwargs["config"].height == 1152 and kwargs["config"].width == 512,
                          "Full-native transformer canvas changed")
            probe.require(float_array(prompt_embeds).mean() == (0.75 if branch == "pm" else -0.375),
                          "CFG branches/embeddings collapsed")
            probe.require(mask.shape == (1, 5 if branch == "pm" else 3), "CFG masks collapsed")
            known_refs = float_array(hidden[:, 2304:])
            refs_digest = hashlib.sha256(known_refs.tobytes()).hexdigest()
            probe.forward_calls.append({"branch": branch, "sigma": kwargs["t"],
                                        "refs_sha256": refs_digest})
            channel = mx.arange(64, dtype=mx.float32)[None, None] / 256
            return (hidden[:, :2304].astype(mx.float32) * 0.03125
                    + prompt_embeds[0, 0, 0].astype(mx.float32) * 0.125
                    + channel + kwargs["t"] * 0.0625).astype(mx.bfloat16)

    def load_component(_base, name, constructor, **_kwargs):
        probe.events.append(f"load:{name}")
        object_ = constructor()
        return object_, {"component": name, "mocked": True, "real_weights_loaded": False}

    def adapter_load(path, *args_, **kwargs):
        if Path(path).name == "latest.safetensors":
            probe.events.append("load:synthetic_adapter")
            return {"synthetic": mx.array([1], dtype=mx.bfloat16)}
        return original_load(path, *args_, **kwargs)

    def adapter_apply(model, weights, header):
        probe.require(isinstance(model, FakeTransformer), "Adapter target changed")
        probe.require(list(weights) == ["synthetic"] and header == {"mocked": True}, "Adapter mock crossed paths")
        probe.events.append("apply:adapter")
        probe.adapter_applied = True
        return {"mocked": True, "scale": 1.0}

    def skip_poisson(xt, velocity, sigma, source, mask, noise, height, width, **kwargs):
        probe.require((height, width) == (1152, 512) and kwargs["backend"] == "mlx", "Poisson call geometry changed")
        probe.require(xt.shape == source.shape == mask.shape == noise.shape == (1, 2304, 64),
                      "Poisson state inventory changed")
        probe.poisson_calls.append(float(sigma.item()))
        # Poisson math is independently parity-tested. Here its FP32-return
        # boundary is preserved while avoiding 50 full-resolution Jacobi loops.
        return velocity.astype(mx.float32)

    def save_state(output, step, arrays, metadata):
        probe.checkpoint_steps.append(step)
        result = original_save_checkpoint(output, step, arrays, metadata)
        if step == 20:
            shutil.copyfile(output / "checkpoint.json", captured_checkpoint)
        return result

    def validate_state(path, protocol):
        probe.events.append("validate:checkpoint")
        return original_validate(path, protocol)

    def restore_state(receipt):
        probe.events.append("restore:checkpoint")
        return original_restore(receipt)

    def memory_snapshot(phase, **_kwargs):
        probe.events.append(f"snapshot:{phase}")
        if phase in ("vl_released_after_both_cfg_conditions", "teacher_and_adapter_loaded_vl_absent"):
            probe.assert_vl_absent(phase)
        return SimpleNamespace(to_metadata=lambda: {"phase": phase, "mocked_memory_telemetry": True})

    with ExitStack() as stack:
        replacements = [
            (runner, "verify_cpu_gates", cpu_gates), (runner, "verify_download_receipts", payload_gates),
            (mx, "set_default_device", select_cpu), (mx, "set_memory_limit", memory_limit),
            (mx, "load", adapter_load), (mx, "clear_cache", lambda: None),
            (mx, "get_active_memory", lambda: 0), (mx, "get_peak_memory", lambda: 0),
            (RuntimeMemory, "apply_mlx_cache_limit", lambda value: probe.events.append("set:cache_limit")),
            (RuntimeMemory, "snapshot", memory_snapshot),
            (loading, "load_saved_component", load_component),
            (transformer_port, "MaskFlowTransformer", FakeTransformer),
            (vae_port, "MaskFlowVAE", FakeVAE),
            (conditioning, "MaskFlowEncoder", FakeMaskFlowEncoder),
            (conditioning, "encode_cfg_branches", encode_cfg),
            (adapter, "apply_maskflow_adapter", adapter_apply),
            (adapter, "read_safetensors_header", lambda _path: {"mocked": True}),
            (TokenizerLoader, "load_all", tokenizers),
            (text_module, "QwenTextEncoder", FakeText),
            (vision_module, "VisionTransformer", FakeVision),
            (vl_module, "QwenVisionLanguageEncoder", FakeVL),
            (processor_module, "QwenVisionLanguageProcessor", FakeProcessor),
            (tokenizer_module, "QwenVisionLanguageTokenizer", FakeTokenizer),
            (math_port, "apply_poisson_to_prediction", skip_poisson),
            (checkpoint, "save_checkpoint", save_state),
            (checkpoint, "validate_checkpoint", validate_state),
            (checkpoint, "restore_checkpoint", restore_state),
        ]
        for object_, name, value in replacements:
            stack.enter_context(patch.object(object_, name, value))
        with redirect_stdout(io.StringIO()):
            runner.execute(args)
    probe.require(probe.events[:2] == ["cpu_gates", "payload_gates"], "Authorization gates did not precede device/loading")
    probe.require(probe.requested_devices == [str(mx.gpu)], "Execution request was not intercepted exactly once")
    probe.require(mx.default_device() == mx.cpu, "Actual device escaped CPU")
    probe.require(probe.memory_limits == [int(26 * 1024**3)], "Memory bound did not propagate")
    return probe, json.loads((directory / "metrics.json").read_text()), captured_checkpoint


def check(name, condition, details=None):
    if not condition:
        raise AssertionError(name)
    return {"name": name, "passed": True, **({"details": details} if details else {})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "runner_cpu_validation.json")
    args = parser.parse_args()
    tested_hashes = {str(path): sha(path) for path in SOURCE_FILES}
    report = {"schema_version": 1, "status": "running", "device": "cpu", "gpu_used": False,
              "real_model_weights_loaded": False, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "mocked": ["component weight loaders", "VL/DiT/VAE models", "adapter payload",
                         "CPU/payload authorization receipts", "GPU memory telemetry", "Poisson solver"],
              "real": ["runner.execute", "input/canvas/mask preparation", "Torch CPU BF16 initial noise",
                       "MLX schedule and CFG", "Euler source/noise bridge", "pixel blend/hard paste",
                       "atomic safetensors checkpoints", "checkpoint hash/protocol/resume validation"],
              "checks": [], "source_sha256": tested_hashes}
    checks = report["checks"]
    try:
        with tempfile.TemporaryDirectory(prefix="maskflow-runner-cpu-") as directory:
            temp = Path(directory)
            fresh, fresh_metrics, step20 = run_fixture(temp / "fresh")
            resumed, resume_metrics, _ = run_fixture(temp / "resume", resume_from=step20)
            checks.append(check("fresh staging order VL -> branches -> release -> VAE -> adapter/DiT", 
                fresh.events.index("encode:both_cfg") < fresh.events.index("snapshot:vl_released_after_both_cfg_conditions")
                < fresh.events.index("construct:vae") < fresh.events.index("construct:transformer")
                < fresh.events.index("apply:adapter"), fresh.events))
            checks.append(check("VL weak references absent before VAE and DiT", fresh.vl_absent()))
            checks.append(check("both independently cached CFG branches; exactly 100 fresh forwards",
                fresh.cfg_encodes == 1 and len(fresh.forward_calls) == 100
                and [x["branch"] for x in fresh.forward_calls] == ["pm", "nm"] * 50))
            checks.append(check("source/mask conditioning distinct and reference tokens unchanged",
                len(fresh.encode_calls) == 2 and fresh.encode_calls[0] != fresh.encode_calls[1]
                and len({x["refs_sha256"] for x in fresh.forward_calls}) == 1))
            schedule = math_port.sigma_grid(50, 2304)["sigmas"]
            checks.append(check("exact 50-step native sigma schedule and source bridge",
                np.allclose([x["sigma"] for x in fresh.forward_calls[::2]], schedule[:-1], atol=1e-7, rtol=0)
                and len(fresh.poisson_calls) == 50 and fresh_metrics["processed_mask_zero_latents_exact"]
                and len(fresh_metrics["step_records"]) == 50
                and [x["step"] for x in fresh_metrics["step_records"]] == list(range(1, 51))))
            expected_steps = [0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
            checks.append(check("atomic checkpoints retain original full sampling state",
                fresh.checkpoint_steps == expected_steps
                and len(list((temp / "fresh").glob("state-step*.safetensors"))) == len(expected_steps)))
            initial_path = temp / "fresh/initial_noise_2511.npy"
            seeded = torch.randn((1, 16, 1, 144, 64), generator=torch.Generator(device="cpu").manual_seed(42),
                                 dtype=torch.bfloat16).float().numpy()
            checks.append(check("new 2511 Torch CPU BF16 grid noise and hash",
                np.array_equal(np.load(initial_path), seeded)
                and fresh_metrics["initial_noise_sha256"] == sha(initial_path)
                and "no 2.1 noise reuse" in fresh_metrics["noise_backend"]
                and resume_metrics["initial_noise_sha256"] == fresh_metrics["initial_noise_sha256"]))
            checks.append(check("resume from step20 restores both branches and skips VL and VAE encodes",
                resumed.cfg_encodes == 0 and not resumed.encode_calls
                and "load:text_encoder" not in resumed.events and "load:tokenizers" not in resumed.events
                and resumed.events.index("validate:checkpoint") < resumed.events.index("device_request")
                and len(resumed.forward_calls) == 60
                and [x["branch"] for x in resumed.forward_calls] == ["pm", "nm"] * 30
                and [x["step"] for x in resume_metrics["step_records"]] == list(range(21, 51))))
            products = ["raw_decoded.png", "native_softblend.png", "composite.png", "processed_mask.png"]
            checks.append(check("fresh and step20 resumed products are byte identical",
                all(sha(temp / "fresh" / name) == sha(temp / "resume" / name) for name in products)))
            original = np.array(Image.open(fresh_metrics["source_file"]).convert("RGB"))
            native = np.array(Image.open(temp / "fresh/native_softblend.png"))
            composite = np.array(Image.open(temp / "fresh/composite.png"))
            raw = np.array(Image.open(temp / "fresh/raw_decoded.png"))
            mask = np.array(Image.open(temp / "fresh/processed_mask.png"))
            checks.append(check("separate raw/native/composite files; full source512 hard paste exact",
                raw.shape == native.shape == composite.shape == (1152, 512, 3)
                and mask.shape == (1152, 512) and not np.array_equal(raw, native)
                and not np.array_equal(native[320:832], original)
                and np.array_equal(composite[320:832], original)
                and np.array_equal(composite[:320], native[:320])
                and np.array_equal(composite[832:], native[832:])))
            inside_rows = fresh_metrics["inside_source_soft_rows"]
            checks.append(check("native processed mask permits exact 24px inner boundary softness",
                inside_rows == list(range(320, 344)) + list(range(808, 832))
                and fresh_metrics["maximum_inside_boundary_softness_pixels"] == 24,
                {"source_soft_rows": inside_rows, "soft_rows_count": len(inside_rows)}))
            receipt = json.loads(step20.read_text())
            protocol = dict(receipt["protocol"])
            protocol["seed"] += 1
            try:
                checkpoint.validate_checkpoint(step20, protocol)
            except ValueError as error:
                changed_protocol_rejected = "seed" in str(error)
            else:
                changed_protocol_rejected = False
            checks.append(check("resume rejects changed sampling protocol", changed_protocol_rejected))
            bad_payload = temp / "altered.safetensors"
            shutil.copyfile(receipt["state_path"], bad_payload)
            with bad_payload.open("r+b") as handle:
                handle.seek(-1, 2)
                old = handle.read(1)
                handle.seek(-1, 2)
                handle.write(bytes([old[0] ^ 1]))
            receipt["state_path"] = str(bad_payload)
            bad_receipt = temp / "altered-receipt.json"
            bad_receipt.write_text(json.dumps(receipt))
            try:
                checkpoint.validate_checkpoint(bad_receipt, receipt["protocol"])
            except ValueError as error:
                corrupted_rejected = "state bytes changed" in str(error)
            else:
                corrupted_rejected = False
            checks.append(check("resume rejects changed state payload bytes", corrupted_rejected))
            report["fresh_summary"] = {"forwards": len(fresh.forward_calls), "steps": 50,
                                       "cfg_encodes": fresh.cfg_encodes, "vae_encodes": len(fresh.encode_calls),
                                       "checkpoints": fresh.checkpoint_steps}
            report["resume_summary"] = {"start_step": 20, "forwards": len(resumed.forward_calls),
                                        "steps": 30, "cfg_encodes": resumed.cfg_encodes,
                                        "vae_encodes": len(resumed.encode_calls), "checkpoints": resumed.checkpoint_steps}
            report["mock_output_sha256"] = fresh_metrics["output_sha256"]
        checks.append(check("all tested sources unchanged during integration gate",
                            tested_hashes == {str(path): sha(path) for path in SOURCE_FILES}))
        report["status"] = "passed"
        report["limitation"] = "Mocked CPU orchestration confirms neither pretrained teacher quality nor GPU execution/memory."
    except Exception as error:
        import traceback
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "checks": len(checks), "report": str(args.output),
                      **({"error": report["error"]} if "error" in report else {})}), flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
