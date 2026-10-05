# Known top-collar hint probe — CPU ready, no GPU result

This isolated probe asks whether retaining trained ControlNet hints on the known source's top 32 pixels improves the generated rail's width and position at the join. The preceding localized40 result is a coherent scene but keeps all known-source target hints zero. Enabling a narrow known strip can provide an internal source-contour signal during transformer attention while preserving the final original source pixels.

New files are copied from the frozen localized runner/helper. The originals remain unchanged:

- Runner SHA256: `3a67d04d8da1eac90f37ede81a4d98083b9bd3b2a93feab08172ed795996e7ad`.
- Helper SHA256: `2aa4dee1c5b4d64d119d58b8943f5d224b3119fc0ae8054c9d4b85d069fbd722`.

## Matched arms

| Arm | Known source top hints | Other known source hints | Unknown hints | Prefix hints |
|---|---|---|---|---|
| `none` | zero | zero | unchanged input-guide cosine support within 64 px | zero |
| `top32` | scalar 1 on pixel rows 320–351 / packed rows 20–21 | zero | unchanged input-guide cosine support within 64 px | zero |
| Optional `top64` | scalar 1 on pixel rows 320–383 / packed rows 20–23 | zero | unchanged input-guide cosine support within 64 px | zero |

The scalar is separately declared by `--known-collar-weight` in [0,1]. A zero scalar or `none` reproduces the prior gate exactly. The source/reference/control representations, the 65 source-conditioning channels, sparse tangent structural64 support, actual-square image/text reference, prompt, seed42 noise, target-only scheduler, active known bridge and all 40 steps stay the same. No bottom collar, source interior hint, generated-image geometry inference, source painting, warp, or additional cache is introduced.

All 16 control hints are computed on the complete original joint before multiplying the scalar field. The copied forward's executable AST matches the old forward exactly after ignoring its name/docstring. Prefix hints stay zero, including the text and 1024 reference-image tokens. Known target hidden states intentionally may change inside a transformer forward; their target latent bridge is restored with the unchanged original context before and after each Euler step. The original 512 source is pasted back exactly after decoding. This is a 40-step full-canvas diagnostic, with no growing-compute or sparse-execution claim.

## CPU evidence

[Known-collar test script](test_known_collar_reference_cpu.py) and [passed validation](known_collar_cpu_validation.json) cover:

- Actual 512×1152 top32 changes exactly 64 target tokens in packed rows 20/21. Top64 changes only rows 20–23. Unknown cosine-support weights are byte-identical to the frozen builder.
- All 16 raw hints are unchanged before gating; all 32 prefix block traces match the old/base path. Only intended top-collar hints become nonzero; deeper source, bottom source and outside-guide unknown hints remain zero.
- Collar0/scalar0 reproduce the old forward. All-zero hints or control scale0 reproduce the base prediction. Inputs, reference latents and context129 remain unmodified.
- Two FP32 full-trace cases, BF16 forward parity, six FP32/BF16 bridge cases at sigma 1/.5/0 and 22 invalid-input checks pass. BF16 traces are excluded because the frozen trace hook directly converts BF16 tensors to NumPy; BF16 forward comparisons cast losslessly to F32.
- Existing port/reference/localized and new known-collar dependency hashes are current.

[Main preflight](../controlnet_runs/2026-10-05/track3-known-collar32-preflight/preflight.json) is `preflight_files_ready`, with no missing files, no real model weights loaded and all four CPU gates current. [Production field parity](../controlnet_runs/2026-10-05/track3-known-collar32-preflight/field_parity.json) compares saved prior weights directly: `none` is byte-identical, and top32 changes only the intended 64 known tokens to 1. Syntax checks and an independent read-only runner audit pass. The prior none final-latent SHA is `df1fa78c76720e09c4d8bbb7f2c3e7f76595a6860efab266b7f25229fa50a979`; its raw PNG SHA is `b47c8d1f86b345fa2f1490aed71c973a1890171f6f9a44502e1c1480646561c9`. The future none arm must be checked against these before attributing quality differences.

```sh
experiments/qwen/.venv/bin/python experiments/qwen/controlnet_port/run_known_collar_reference_control_trial.py \
  --track track3 --steps 40 --collar-arms none top32 --known-collar-weight 1 \
  --modes tangent-canny --structural-support tangent --known-bridge \
  --prompt-style outpaint --hint-shape cosine --hint-radius 64 --control-scale 1 \
  --output /absolute/path/to/fresh-known-collar-trial
```

The GPU command is prepared but has not been run. The preceding localized arm took approximately 415 seconds for 40 denoiser calls, so the pair needs a serialized bound appropriate to roughly 80 calls plus shared setup. Actual quality must be inspected on raw and exact-source final PNGs and evaluated independently from source preservation. The `none` result should be checked against the retained localized40 baseline before interpreting a difference.

## Optional VAE derivative projection input

The stronger localized40 final tensor is finite F32 with shape (1,2304,64), compatible with the existing Qwen reference VAE. Its older NPZ contains latents alone. [A CPU-only metadata adapter](prepare_known_collar_projection_input.py) created an isolated exact copy with validated final IDs/source rectangle and an identical raw PNG:

`experiments/qwen/projection_inputs/2026-10-05/track3-localized40-known-collar/final_latents.npz`.

The [adapter evidence](../projection_inputs/2026-10-05/track3-localized40-known-collar/metadata_adapter.json) verifies the completed generation's tensor/image hashes and exact resized source pixels. Existing checkpointed derivative projection [CPU preflight](../projection_runs/2026-10-05/track3-localized40-checkpoint-preflight/preflight.json) passes on this input. No VAE projection was executed, and memory reduction remains unmeasured. The original projection's completed consistency arm did not fix rail geometry; its derivative arm timed out with no final image at a 40.60-GiB autodiff peak. That failure remains documented in [PROJECTION_RESULTS.md](../PROJECTION_RESULTS.md).
