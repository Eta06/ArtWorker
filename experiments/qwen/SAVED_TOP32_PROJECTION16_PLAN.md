# Sixteen-update latent-optimization diagnostic

The separate [16-update runner](run_saved_top32_checkpoint_projection16.py) retains the validated checkpointed VAE decoder, objective, mask, casts and Adam mathematics of the frozen four-update runner. Its SHA256 is `b64dc876d0eea0208fccaa53260eebf35c9ae3e7c047580620fb184f2ae628b8`. The frozen four-update runner remains `06a58842753de3b17577395fcb5fbb231294f0117b90554697ed15f73e4624d6`.

Both consistency and derivative start from the same original saved-top32 NPZ and zero Adam state. Each performs16 updates; this repeats the first4, preserves an additional atomic iteration04 snapshot and compares its delta/moment/variance shapes, dtypes and bytes against the completed four-update run, then continues12 updates. The comparison is explicitly diagnostic, not an abort guard. No automatic resume, new transformer sampling or model-weight training occurs.

[CPU preflight-v2](projection_runs/2026-10-05/track3-saved-top32-checkpoint16-preflight-v2/preflight.json) and [CPU validation](saved_top32_projection16_cpu_validation.json) pass. [Test script](test_saved_top32_projection16_cpu.py) binds exact NPZ/raw/source/reference-state metadata and unchanged decoder/objective AST. Independent audit verifies all mathematical updates, state-comparison semantics and protected/fresh output guards. Reference4 NPZ provenance, shape/dtype/finite arrays and parameter equality are checked before writes.

The [four-update actual report](projection_runs/2026-10-05/track3-saved-top32-checkpoint4-instrumented-v2/ACTUAL_REPORT.md) justifies a bounded depth test: both losses still decrease and delta magnitude is only about0.060 versus cap0.2. It does not establish improved texture. Consistency4 loses reliable near-join edge extraction; derivative4 reduces the final-row jog but retains a local bend, water/ripple boundary and bottom grain mismatch. Those artifacts require direct review after16 rather than acceptance from lower loss.

```sh
experiments/qwen/.venv/bin/python experiments/qwen/run_saved_top32_checkpoint_projection16.py \
  --input experiments/qwen/projection_inputs/2026-10-05/track3-saved-top32/final_latents.npz \
  --track track3 --methods consistency,derivative --iterations 16 --checkpoint-decoder \
  --output experiments/qwen/projection_runs/2026-10-05/track3-saved-top32-checkpoint16-instrumented
```

Parent launched this command in a serialized360-second supervisor. Launch files are outside the required empty output directory. Runtime, memory, first4 parity and16-update quality remain unknown until completion. Actual first-run failures, the old40.60GiB/incomplete derivative attempt and all four-update outputs remain unchanged.

Independent completion checks should run [the CPU state/pixel audit](audit_saved_top32_projection.py), inspect preserved first4 states and saved spectra/profiles, and use the same frozen source-tangent evaluator thresholds. Outside-collar latent equality and final-source equality do not imply distant decoded pixels stay exact; four updates already caused max1/255 distant channel changes. Physical Celsius temperature is still unavailable from the OS thermal-warning probe.
