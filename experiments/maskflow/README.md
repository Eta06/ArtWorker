This isolated local teacher follows [MaskFlow](https://github.com/ReyChiaro/MaskFlow/tree/3125f1ecd72f5a4e068c5a1954e9d6a8df38d443) standard `latest.safetensors`: 50 steps, text CFG4, mask CFG1, per-token norm rescale, native SoftPoisson, then the source/noise bridge. It uses the actual Qwen-Image-Edit-2511 RGB codec and zero-condition-time DiT. It does not import the current Qwen2.1 environment or modify its files.

The public base and adapter are pinned and fully checksum verified in `download_manifest.json`. Base `AbstractFramework/qwen-image-edit-2511-4bit@dbc6d597c1316fc34c0f7726904121a284a69c0b` is mixed Q4/Q8. Adapter `ReyChiaro/MaskFlow@f18f0828ab6a40a7f3456feace7a33bebe8c511e` is the unchanged 1,200-key latest checkpoint. The quantized converter omitted a learned output-normalization bias; the original pinned 12,288-byte BF16 tensor is retained in a separate sidecar with exact HTTP range provenance. The original full 982 MB shard was not downloaded or checksum verified.

Frozen MLX-Gen source is `runtime/mlx-gen@99fb94dd3eaa9dd1931cd3cd8eae1ae3e20f2ef3` in a separate `.venv`. Isolated compatibility classes fix padded text RoPE width, the learned bias, BF16 visual inputs, and model-dtype codec affine operations. Strict loading allows only 52 value-preserving VAE RMS singleton-axis reshapes. Public downloaded base/adapter bytes remain unchanged.

The protocol keeps the original 512 square at y320:832 on a 512×1152 canvas, with declared black unknown pixels. Source and processed mask are both VLM and independently VAE encoded. The official VLM FastTorch preprocessing receives BF16 float images and performs its configured second `/255` rescale; PNG substitution changes conditioning. The actual frozen prompt produces positive/negative embedding widths 556/397. DiT processes 6,912 image tokens twice per step, or 100 forwards for the standard recipe.

The official dilated/blurred mask reaches 24 pixels inside each source boundary. `raw_decoded.png` saves the decoded teacher result; `native_softblend.png` saves the official processed-mask blend. `composite.png` separately pastes the complete original 512 square exactly. Both native and presentation results must be reviewed for geometry at the join; the hard paste is not proof that the model aligned the scene.

CPU receipts under `port/*_cpu_validation.json` cover executable official math, all 60 tiny Transformer layers, actual full-model header coverage, adapter mapping and tiny Q4 arithmetic, VLM preprocessing, codec dtype, strict loading, atomic checkpoints, and mocked runner orchestration. These establish compatibility and control flow, not pretrained image quality or mobile speed. Estimated static memory after releasing VL is 14.815 GiB; real load/activation peak is unmeasured until the serialized hardware run.

Run the complete weight-free CPU preflight from the ArtWorker root:

```sh
experiments/maskflow/.venv/bin/python experiments/maskflow/port/run_maskflow_trial.py \
  --output-dir experiments/maskflow/runs/2026-10-05/track3-preflight-all \
  --require-gates --require-full-payload
```

Only the parent serial queue may launch this bounded hardware trial after that preflight passes:

```sh
experiments/maskflow/.venv/bin/python experiments/maskflow/port/run_bounded_trial.py \
  --timeout-seconds 3600 -- \
  --output-dir experiments/maskflow/runs/2026-10-05/track3-latest50 \
  --track track3 --fill black --seed 42 \
  --memory-limit-gib 26 --cache-limit-gib 1 --execute-gpu
```

`metrics.json` records staged physical/MLX memory, actual component and conditioning times, every step time, both CFG forward times, and peaks. VL is deleted after both prompt branches materialize and before VAE/DiT denoising. Checkpoints retain source/mask/noise/current latents and both CFG caches at steps 0, 1, and every five steps. A bound or interruption leaves an atomic `checkpoint.json` and unique retained state files. Resume validates identical source, prompt, seed, protocol, code hashes and state bytes, skips VL reload, and continues the same 50-step trajectory:

```sh
experiments/maskflow/.venv/bin/python experiments/maskflow/port/run_bounded_trial.py \
  --timeout-seconds 3600 -- \
  --output-dir experiments/maskflow/runs/2026-10-05/track3-latest50-resume \
  --resume-from experiments/maskflow/runs/2026-10-05/track3-latest50/checkpoint.json \
  --track track3 --fill black --seed 42 \
  --memory-limit-gib 26 --cache-limit-gib 1 --execute-gpu
```

MaskFlow code/adapters and MLX-Gen are MIT licensed; original and selected 2511 base are Apache2.0. This is the standard latest teacher. Older paired SFT/DMD few-step variants require their own matched checkpoints and isolated recipe; latest is not mixed with an older residual.
