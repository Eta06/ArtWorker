# Task-specific Klein outpainting

`fal/flux-2-klein-4B-outpaint-lora` is pinned and SHA-256 verified by `download_manifest.json`. Its native BFL-format adapter is 76,039,072 bytes. The model card uses green exterior, the instruction `Fill the green spaces according to the image` and scale 1.1. Its example calls a base edit endpoint while its base-model metadata names distilled 4B; our base and distilled arms remain separate.

## Runtime bug found by the experiment

The Swift MLX checkout is pinned by the existing `experiments/flux2/README.md`. Its BFL LoRA mapper sent `time_in.in_layer` and `time_in.out_layer` to non-existent `inLayer`/`outLayer` paths. Native embeddings use `linear1`/`linear2`. The unpatched verbose run applied only 86/88 mapped layers. `bfl-timestep-mapping.patch` fixes those two paths; the rebuilt runtime applies 88/88. The observed jump from green/unfilled exterior to filled background is evidence that this loader bug mattered. It is not proof of final outpaint quality.

Apply the patch inside the pinned Swift checkout and rebuild the CLI with its adjacent MLX resource bundle. Use one build job and the memory supervisor. `docs/stages/cinar.md` records the failed `-packagePath` build invocation and successful build from the correct checkout directory. Full tensor coverage is required by `run_trial.py`; a partially applied adapter fails the trial.

## Reproduce

```sh
experiments/qwen/.venv/bin/python experiments/klein_outpaint/run_trial.py \
  --track track3 --describe \
  --output experiments/klein_outpaint/runs/2026-10-06/NEW-RUN

# A separate base/CFG arm, not the same model with a longer display animation:
experiments/qwen/.venv/bin/python experiments/klein_outpaint/run_trial.py \
  --track track3 --model klein-4b-base --steps 28 --guidance 4 --describe \
  --output experiments/klein_outpaint/runs/2026-10-06/NEW-BASE-RUN
```

Each run uses the common source/canvas, records the binary hash, runs under the serial memory supervisor, saves raw output and separately pastes back the exact source square. Raw-source MAE and composite integrity do not decide boundary quality. The optional `--vae standard` arm isolates decoder selection relative to the default small decoder. The standard VAE's pinned source and verified hash are in `standard_vae_manifest.json`; local canonical cache paths/hardlinks must be set up before calling it. Otherwise the runtime may try to resolve missing assets.

Payloads, covers, logs and rendered images stay local and outside normal Git. Licenses come from the pinned upstream source, not the runtime registry's generic metadata. Existing experiments remain intact.

## Bounded pre-quantized path

`prepare_native_q4.py` streams one BF16 tensor at a time into Swift-native affine INT4/group64. QKV projections are split and final scale/shift order matches the pinned runtime. The full 387-key/shape/dtype checkpoint checks passed in the actual Swift loader, with no dense fallback. It leaves the source payload and mtime unchanged and refuses overwrites.

Runtime LoRA merging still caused a 6.66 GiB sampled footprint and was stopped. `--bake-outpaint` creates a **separate** local model root, adds all 88 mapped FP16 adapter pairs at scale 1.1 before one INT4 quantization, and tags the checkpoint `lora_baked=true`. This differs from the old quantize → runtime merge → requantize path and must not be presented as a matched causal ablation or a newly trained model. Original caches remain intact.

`sequential-decode-memory.patch` includes Klein Base in the conservative phase-cache profiles. Its opt-in `ARTWORKER_RELEASE_TRANSFORMER_BEFORE_DECODE=1` releases the completed transformer before final VAE decoding. Resident unbaked hosts retain upstream lifetime behavior. The runner allows this switch only for the baked-adapter experiment; otherwise reloading after adapter release could lose the in-memory merge.

```sh
python3 scripts/run_bounded_model.py --output .build/NEW-BAKED-EXPORT \
  --max-gib 2 --timeout 120 -- experiments/qwen/.venv/bin/python \
  experiments/klein_outpaint/prepare_native_q4.py --model base --bake-outpaint

experiments/qwen/.venv/bin/python experiments/klein_outpaint/run_trial.py \
  --model klein-4b-base --steps 12 --guidance 1 --vae standard \
  --baked-outpaint --require-prequantized --release-before-decode --max-gib 6 \
  --output experiments/klein_outpaint/runs/2026-10-06/NEW-RUN
```

The 28-step generic/CFG1 run completed in 234.22 seconds / 5.39 GiB sampled footprint. Raw source MAE fell to 3.76, but the combined recipe changes cannot isolate which factor caused that improvement. Twelve steps completed in 79.82 seconds; these are measured Mac runs, not mobile or warm-cache speed guarantees. Direct boundary and cross-cover review is recorded in the stage results.

## Geometry diagnostics

`global_registration.py` fits CPU ECC translation/rigid/affine transforms using the known source only. On the earlier base28 described output its global fit was already within 0.52–2.20 pixels; global registration did not solve the exterior rail mismatch. Reflected border pixels and color transfer are recorded explicitly. It is postprocessing, not another model.

The legacy rail evaluator's generated corridor was frozen to an earlier poor probe and misses the improved generic arm. `experiments/qwen/evaluate_boundary.py --corridor source` offers a separately identified corridor derived from the original source fit alone. Legacy default remains numerically unchanged. Both profile records remain preserved; selected ridge points must be inspected, and measurements from different profiles are not interchangeable.
