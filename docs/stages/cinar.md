# Çınar — task-specific outpaint comparison

2026-10-06. Stage `cinar` was selected by the primary agent. No new subagents were started. This stage produced a better local road/sea prototype, reproducible model comparisons and resource fixes. It did **not** produce an accepted universal teacher, a trained ArtWorker student, an iPhone benchmark or true spatial streaming.

## Findings from actual images

The strongest local candidate is **Klein base4B + task outpaint LoRA, baked before INT4, standard VAE, fixed fill instruction, CFG1, 12 steps, seed17**. At 512×1152 it finished in **79.85 seconds / 5.27 GiB sampled peak footprint** on M4 Max. Its sea, coast, road and barrier continue much more coherently than the earlier described/CFG4 runs. The selected source-derived rail fit still has approximately −3.22 px near offset and −2.72° far angle error. Slight texture/color transitions remain. This is the better of two tested seeds, not broad independent teacher acceptance.

A separate color-only exterior collar slightly harmonizes the join while preserving the original square exactly. Local optical flow and tangent warps did not materially improve this candidate; keep them as failed comparison arms. The local preview is `experiments/klein_outpaint/compositor_trial/2026-10-06/base12-seed17/color/composite.png`; artwork is intentionally excluded from public Git.

| Actual arm | Wall time | Peak footprint | Direct visual verdict |
|---|---:|---:|---|
| Klein base28, baked Q4, sequential decode, seed42 | 234.22 s | 5.39 GiB | Good scene continuation; visible barrier angle kink |
| Same recipe, 12 steps, seed42 | 79.82 s | 5.33 GiB | Similar quality; same residual geometry issue |
| Same recipe, 4 steps, seed42 | 27.63 s | 5.21 GiB | Rejected: checkerboard/mesh exterior textures |
| Same 12-step recipe, road/sea seed17 | 79.85 s | 5.27 GiB | Best local candidate; residual seam and small rail mismatch |
| Same 12-step recipe, black/moon cover | 80.29 s | 5.27 GiB | Clean exterior; largely trivial black padding |
| Same 12-step recipe, person/T-shirt cover | 79.73 s | 5.27 GiB | Rejected: printed bag becomes real bag; invented clothing graphics |
| Qwen2.1 task V1, native Q4, legacy caption | 186.43 s | 7.92 GiB | Rejected: sea becomes asphalt at upper boundary |
| Qwen2.1 task V2, native Q4, legacy caption | 144.84 s | 7.77 GiB | Similar scene continuation failure |
| MI-GAN 28 MB CPU pipeline, three covers | 0.51–0.66 s | 0.36–0.40 GiB | Rejected: wrong textures/structures and deformed clothing |
| MI-GAN ten 64-row expansions | 2.07 s | 0.80 GiB | Rejected: banding and accumulated scene drift |

Full machine-readable attempt receipts are indexed in [cinar-runs.json](cinar-runs.json). Raw model output is evaluated separately from exact-source compositing. Low source MAE, finite pixels, correct adapter coverage and execution completion are numerical/integrity checks, not visual acceptance. Smaller resolution runs are separate experiments.

## Implementation and failure record

- The native Klein BFL mapping silently skipped two `time_in` pairs. The patch fixes `linear1/linear2`; all 88 pairs then apply. Mapping alone improved fill but did not fix geometry.
- Standard VAE originally failed twice because its cache path/config were incomplete. Fixed cache lookup completed runs, but decoder choice alone did not remove the boundary error.
- Original dense base load and runtime LoRA merge caused excessive temporary memory. Tensor-wise native Q4 export completed in 6.21 s / 0.71 GiB. A separate baked adapter export completed in 6.85 s / 1.06 GiB. The latter merges FP16 adapter/source **before one quantization** and differs from the earlier quantize/merge/requantize operation.
- Native Swift loader validated all 387 key/shape/dtype entries. Baked transformer is 2,180,050,216 bytes. Source payloads remain untouched; this is not newly trained ArtWorker weights.
- The first baked 28-step run reached all diffusion steps but the final decoder exceeded its 6 GiB gate, so no output was accepted. The opt-in sequential decode patch releases the transformer and fixes conservative cache profiles before decoding. One-job build completed in 28.61 s / 1.28 GiB footprint. Subsequent 28/12/4-step runs completed within the gate.
- Recipe improvement changes several factors together: base checkpoint, fixed prompt, CFG, bake/quant order, decoder and memory strategy. This is not a causal ablation proving one factor caused every visual improvement.
- Qwen's community Q4 pack lacked its visual tower and edit configs. Download was stopped and original complete weights were exported tensor-wise: 10,277,260,600 bytes in 25.83 s / 3.56 GiB. Sequential component loading produced actual V1/V2 outputs; both failed visual review.
- Viggle v0.3 six-step rank128 adapter was downloaded and hashed. Its combined outpaint attempt and later factual-caption/lower-resolution Qwen attempts hit **system memory pressure before denoising**. They provide no Turbo quality or inference-speed result. Combining adapters is not jointly trained distillation.
- Single-use Qwen language and vision layer release matches reduced reference classes exactly in FP16/Q4, including deepstack outputs. Full real conditioning still hit pressure (6.14 s / 4.97 GiB); Layerwise repacking alone also hit pressure. True on-demand layer loading and 256-pixel tiled reference encoding completed conditioning in a separate process in **3.84 s / 2.20 GiB**. Its cache preserves BF16 with safetensors, validates source/prompt/geometry and SHA, and stays ignored. NumPy cache serialization initially rejected BF16; the first root embedding loader missed quantized root keys; both failures are retained. Fresh-process diffusion still exceeded its 6 GiB gate: 4.80 s / 6.02 GiB; sequential transformer-shard loading also stopped at 5.12 s / 6.08 GiB, before denoising. Therefore the new preparation improvements do not establish a completed new Qwen output. Reduced checkpoint-backed decoder/vision/root-embedding/sequential-loader parity is exact after matching generated rotary buffers. The first reduced on-demand test omitted that buffer parity and failed; it was corrected. Reduced parity does not establish full VLM quality or mobile performance.
- MI-GAN upstream pipeline rounded a few source pixels; outputs were retained, and exact-source composites were added separately. The ONNX pipeline includes upstream crop/resize/postprocess. It is a compact 2023 baseline, not a new 2026 model.
- Global translation/rigid/affine registration found only small whole-image shifts and did not fix locally wrong rails. Previous flow/color arms can bend geometry or add halos.
- The legacy rail search corridor excluded the improved rail. It remains the default, unchanged. Optional `--corridor source` derives its search path solely from the original source before inspecting outputs. Its metrics are a different profile, cannot be silently compared to old ones, and selected ridge annotations must be checked for parallel-edge bias. Near-zero offset can coexist with a visibly wrong angle.
- Initial unsupported Swift `-packagePath` build invocation, boundary/compositor CLI/path mistakes and failed exports/runs are preserved in their receipts. Nonzero exits in older guard versions may say `complete`; assess exit code as well as status. Current guard records nonzero exits explicitly.

## Models, licenses and next work

[Klein outpaint](https://huggingface.co/fal/flux-2-klein-4B-outpaint-lora), [Qwen task adapters](https://huggingface.co/ausboss/Qwen-Image-2.1-Outpaint-LoRA), [Viggle Turbo](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo) and [MI-GAN](https://github.com/Picsart-AI-Research/MI-GAN) are pinned with SHA manifests in their experiment folders. Klein/standard VAE are Apache-2.0; Qwen exports/adapters retain Qwen Research terms; MI-GAN code and weight licenses were separately read as MIT. No third-party payloads, local covers/music or credentials enter normal Git. No own trained checkpoint is released.

Alibaba-PAI Z-Image Turbo Fun Union2.1's updated task-control recipe warrants a correctly implemented inpaint baseline. Its Lite 2.02 GB / full 6.71 GB are **additional** control weights, not the entire pipeline. Native installed MFLUX control conditioning is not verified to match that mask recipe; no download/inference result is claimed. Inspected original HiDream-O1 totals about 35.22 GB and was not launched on this 36 GiB Mac. MaskFlow remote revision is unchanged from the earlier failed latest50 run; it was not blindly repeated.

The current full Klein components are about **4.61 GB**, not just the 2.18 GB transformer; Qwen package is about **10.28 GB**. Quantization, fewer steps and a smaller student architecture are separate. See [mobile/student direction](cinar-mobile-direction.md): first obtain accepted teacher pairs on licensed held-out image families, then investigate a compact conditioner and 0.4–1B student with boundary/geometry losses and 4–8-step distillation. Fixed fill embeddings could remove the large text encoder but have not been implemented. General iPhone support needs actual old/low-memory and recent Pro measurements with music playback, thermal and battery constraints.

True expanding-region generation remains secondary. A reveal animation and MI-GAN repeated strips do not establish single-generation spatial streaming; that needs sampling/training with a spatial activation schedule and quality tests.

## Resource and publication controls

One heavy process group under `.build/outpaint-gpu.lock`; approximately 100 ms samples of RSS, macOS `phys_footprint`, pressure and swap. RSS and footprint overlap and are not summed. Stop on pressure level2, sampled budget excess, swap growth over 256 MiB or timeout; only our group is terminated. Initial runs used 12 GiB; later inference uses 6 GiB, repacks/reduced validation 1 GiB and exports/builds separate budgets. Normal pressure is not unused RAM. Sampling can miss instantaneous peaks. No other applications were closed.

Publication follows exactly one changed file per `cinar:` commit and one grouped push, under the existing Git memory guard with no arbitrary inter-commit delay. Model/media payloads remain local and recoverable. [ChatGPT Space stage](https://chatgpt.com/space/page_22376bf12c8c8191a430855b3e2d0d15) records actual attempts, limits and next work.
