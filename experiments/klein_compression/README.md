# Koza compression experiments

Scope: reduce the existing baked Klein base4B outpaint pipeline, checking matched outputs on the three local album covers. This is post-training quantization and fixed-instruction specialization, not a newly distilled ArtWorker model. Media and model payloads stay ignored.

Apply `lowbit-fixed-prompt.patch` after the two Cinar runtime patches to `flux-2-swift-mlx` at commit `315c27e83187909bc5e1f880b4850918342d891b`. Build the native CLI with one Xcode build job and the existing resource guard. MLX Swift is pinned by that checkout; tensor-wise exports use the existing Python MLX environment.

The cached FP32 conditioning tensor replaces the pinned Qwen3-4B encoder only for `Fill the green spaces according to the image`, Klein base4B, CFG1, without prompt upsampling or interpretation. The reference image still enters the VAE and transformer. Checksums, recipe metadata and tensor shape are verified before accepting the cache. Arbitrary prompts still need an encoder. The native embedding-export command requires the pinned local encoder; the manifest records its revision and the produced payload checksum.

```sh
# One-time specialization; bounded on the development Mac, not on the phone.
experiments/qwen/.venv/bin/python scripts/run_bounded_model.py \
  --output experiments/klein_compression/export_receipts/fresh-fixed-prompt \
  --max-gib 6 --timeout 120 -- \
  .build/flux2-native/Build/Products/Release/Flux2CLI artworker-embedding \
  --encoder-path .build/models/flux2/lmstudio-community/Qwen3-4B-MLX-4bit \
  --output .build/models/artworker-fixed-fill/embedding.safetensors

# Independent quantization from the preserved FP16 source+adapter, not old INT4.
experiments/qwen/.venv/bin/python scripts/run_bounded_model.py \
  --output experiments/klein_compression/export_receipts/fresh-int3 \
  --max-gib 2 --timeout 120 -- experiments/qwen/.venv/bin/python \
  experiments/klein_outpaint/prepare_native_q4.py --model base --bake-outpaint --bits 3

# Matched road/sea cover trial. Use seed42 for track1 and track2.
experiments/qwen/.venv/bin/python experiments/klein_outpaint/run_trial.py \
  --track track3 --model klein-4b-base --steps 12 --guidance 1 --seed 17 \
  --vae standard --max-gib 6 --require-prequantized --baked-outpaint \
  --release-before-decode --bits 4 \
  --fixed-embedding .build/models/artworker-fixed-fill/embedding.safetensors \
  --output experiments/klein_compression/runs/fresh-track3-int4
```

Existing outputs/export files are preserved and must not be overwritten; use fresh receipt directories. Generation is serialized with the existing GPU lock, pressure and swap checks. A successful native load or exact pasted source rectangle does not establish outpaint quality. `compare_outputs.py` checks matched recipes, compares raw generated pixels outside the source rectangle, and produces an ignored contact sheet for visual review. The dated stage report records acceptance and failures.

`--require-prequantized` now also sets a native guard: an invalid or missing cache fails before the large dense-weight fallback. The mixed profile always requires a validated cache. This prevents a broken compact checkpoint from silently materializing the BF16 source.

The uniform INT3 export above is a rejected experiment: all three covers showed major additional visual damage. The encoder-free INT4 recipe is the safe compression baseline, with exact raw parity on all three matched covers. That preserves existing baseline mistakes too; it does not establish general outpainting quality.

Checkpoint storage is separate from peak working memory and runtime latency. Component totals do not constitute a portable iPhone bundle: the current upstream cache loader still resolves the preserved BF16 source for provenance/fingerprint validation. A standalone manifest-backed loader, Core ML/Metal deployment and physical-device measurements remain work to do. No upstream payload or new trained checkpoint is published here.

`--quant-profile mlp3 --bits 4` is a separate mixed-precision experiment: double-block feed-forward matrices and single-block attention/output projection matrices use INT3; attention-input, timestep, modulation and remaining matrices use INT4. The single-block output projection combines attention and feed-forward output, so this is not an isolated MLP ablation. Exports live in a distinct `flux2-outpaint-baked-mlp3` root. Native loading requires the exact profile tag and builds the same per-layer structure for key/shape validation; it refuses a uniform or unvalidated checkpoint for this profile. It is a quality experiment, not an accepted smaller replacement until reviewed.

`quant_probe.py` separately checks a small Python-MLX/Swift-MLX fixture. All bit depths decode exactly, but the tested INT3 matrix product differs: max absolute 0.033203125, relative L2 0.00105108. It fails the recorded 0.001 compatibility threshold; INT2/INT4 pass exactly. This is retained as a failed check, not a quality pass. The initial exact-product check failed too; a follow-up wrapper attempt hit a NameError, corrected before the complete all-bit record. Numerical compatibility and actual cover quality are distinct. This cross-version difference remains a confound when assigning the INT3 image failure solely to bit depth.
