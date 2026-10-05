# Native task-specific Qwen outpaint trials

This experiment tests the September 27 Qwen 2.1 Outpaint V1/V2 adapters separately from the previous ControlNet/known-latent-mask experiments. Read `docs/stages/cinar.md` for actual visual verdicts. Execution success, adapter coverage and exact-source compositing do not establish good outpainting.

## Checkpoints and license

- Full original: `Qwen/Qwen-Image-2.1` at `790c92633540aa0cb11d9abf19eb46d861714758`; existing `experiments/qwen/base-manifest.json` records source files.
- V1/V2: `ausboss/Qwen-Image-2.1-Outpaint-LoRA` at `449336db42ff074aee970ba0facc0ac0feb77863`; downloaded SHA-256 values verified in `download_manifest.json`.
- Optional step-distilled adapter: Viggle v0.3 rank128 at `009a44a895ef85f7e643c80fdca9543795248867`; `turbo_v03_manifest.json` records verified hash. Its author's six-node schedule is required. Combining it with the outpaint adapter is an unvalidated experiment, not a jointly trained model.

These use the upstream Qwen Research license. Our Apache-2.0 code does not change the third-party checkpoint license. No weights are in normal Git or released as an ArtWorker checkpoint.

The community `mlx-community/Qwen-Image-2.1-mflux-q4` pack at `746a58556820933a2df5c75887a2570f1ad200c0` was inspected and its download stopped: its text-only export lacks the visual tower and edit component configs. Its Apache metadata also does not override Qwen's upstream terms. Do not treat it as a ready image-edit checkpoint.

## Native export and bounded loading

`prepare_native_q4.py` reads one tensor from the cached original at a time. It exports affine 4-bit matrices in groups of 64, retains sensitive diffusion modulation/time/output-norm tensors, preserves the full visual tower, converts the VAE layout to native float32 and retains the upstream license. This is quantization, not distillation. A streaming safetensors writer avoids keeping an entire dense shard in RAM. The completed export is 10,277,260,600 bytes of weights; manifests include per-shard hashes.

`low_memory_loader.py` replaces packed linear/embedding structures with small placeholders before applying the actual weights. It checks every key and shape. Unlike quantizing the randomly initialized dense model, it avoids materializing large temporary weights. The runner loads the text/vision encoder, releases it after conditioning, loads/releases the VAE for reference encoding, loads the diffusion transformer, then releases it before reloading the VAE for tiled decoding. A real first general-loader run was automatically stopped for elevated system pressure; that failure is retained.

Run from the repository root, using the existing Qwen environment and cached full original:

```sh
python3 scripts/run_bounded_model.py --output .build/qwen-native-q4-export-run \
  --max-gib 6 --timeout 600 -- experiments/qwen/.venv/bin/python \
  experiments/qwen_outpaint/prepare_native_q4.py

experiments/qwen/.venv/bin/python experiments/qwen_outpaint/run_trial.py \
  --preflight --output .build/qwen-outpaint-preflight

python3 scripts/run_bounded_model.py \
  --output experiments/qwen_outpaint/runs/2026-10-06/NEW-RUN \
  --max-gib 12 --timeout 600 -- experiments/qwen/.venv/bin/python \
  experiments/qwen_outpaint/run_trial.py --track track3 --adapter v1 --steps 25 \
  --output experiments/qwen_outpaint/runs/2026-10-06/NEW-RUN
```

Use fresh export/run directories. The export refuses overwrites and an incomplete export cannot pass preflight. Runtime is the reference editor in the locally patched MFLUX checkout pinned by `experiments/qwen/runtime.json`; earlier runtime modifications remain prerequisites, so this is not a claim that stock MFLUX works identically.

## Recipe and evaluation

The input uses opaque `#808080` exterior and a centered source square. Reference and output use exactly the same geometry. The prompt begins with the adapter's trigger. Factual scene captions are stored separately from legacy instructions. CFG is 1; no known-latent clamp or extra noise mask is applied. Native flow Euler is recorded explicitly; it is not claimed identical to ComfyUI's `simple` scheduler. Raw RGB is retained separately from an exact-source composite. MAE reports the raw source square, while exact-source equality describes the compositing operation only. Neither metric accepts seam quality.

Defaults are 512×1152. `--width 640 --height 1440` is a separate resolution experiment, not a matched baseline. For Turbo, pass `--turbo v03-r128 --steps 6`; both adapters are unbaked at their recorded scales. Watch the resource guard and review global geometry, boundary textures and subjects in both raw and composite images.

All runs share the GPU lock and sample RSS, macOS footprint, pressure and swap. These are sampled limits, not an instantaneous kernel-enforced cap. Mac timings and memory do not prove iPhone deployment or Neural Engine support.

## Single-use encoder offloading investigation

`--stream-encoder` evaluates and releases each language and visual layer after use. The encoder cannot be reused for another prompt. Reduced actual reference classes, affine Q4/FP16, with and without deepstack injections, produced zero maximum absolute error; `streaming_encoder_validation.json` records that limited scope. This is offloading, not distillation or pruning.

Actual full conditioning still triggered system pressure before denoising. `repack_encoder.py` additionally copies tensor payload bytes unchanged into per-layer safetensors under the ignored cache. Its manifest records hashes, sizes and source provenance; it refuses overwrites. `--layerwise-encoder --stream-encoder` selects that separate layout. Repacking also failed to avoid full-model conditioning pressure in the first real run, so it is not a successful low-memory solution. Phase tracing showed full vision/language completion; tiled reference encoding also completed, with the next pressure stop during transformer loading. `--ondemand-encoder` reads and validates only each consumed component, rather than eagerly opening every shard. `--tiled-encode` uses 256-pixel tiles / 64-pixel overlap and changes reference encoding, so it is a distinct recipe. Reduced checkpoint-backed language/vision/embedding parity passes exactly when rotary buffers match the reference; generated rotary buffers are not checkpoint payloads. Do not raise the host budget or infer an iPhone speedup from reduced-class arithmetic parity.

Use fresh guard subdirectories separate from output images, a 6 GiB inference gate, and normal-pressure enforcement. All failed receipts remain available. No full teacher/student or mobile performance claim follows from this investigation. `--conditioning-only` writes an ignored safetensors cache, preserving BF16, and exits; `--condition-cache PATH` validates its SHA, source input, prompt and dimensions in a fresh process. Phase splitting is a resource experiment, not streamed output. The first NumPy cache attempt rejected BF16 and is retained as a failed attempt.


The completed preparation cache was measured at 3.84 s / 2.20 GiB. Fresh diffusion still failed: 4.80 s / 6.02 GiB, before a denoising step. `repack_encoder.py --component transformer` creates a separate identical-payload layout; `--sequential-transformer` validates every key/shape with lazy placeholders, then materializes one shard at a time. Reduced checkpoint-backed parity is exact, but the real separate process also exceeded the gate at 6.08 GiB. Neither workaround supplies a new accepted image. A completed `conditioning_only` receipt is preparation only, not a completed image run. All pending first-pass runtime and quality caveats remain.
