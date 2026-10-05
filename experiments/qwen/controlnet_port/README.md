# Isolated trained structural-control teacher port

This directory ports the Alibaba PAI Qwen-Image-2.1 Fun ControlNet-Union branch into the existing MLX reference execution. The installed runtime and previous experiments remain unchanged. The first teacher uses the full canvas; it does not demonstrate center-out token growth or iPhone inference.

## Verified inputs and architecture

- [Checkpoint](https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union/tree/8a4702014d4dabb5f896fcba917e2ee0a961465f): `Qwen-Image-2.1-Fun-Controlnet-Union.safetensors`, **7,550,979,904 bytes**, expected SHA256 `65d6b66d734da9e7ff5ef04e7db3a133553a52a3f29a7fcb3e9cce8fa21dcfcd`.
- Header inspected with two HTTP ranges only: 180 BF16 tensors; 20,280-byte JSON header; 7,550,959,616 tensor payload bytes. No weight payload was downloaded by that inspection.
- [VideoX reference code](https://github.com/aigc-apps/VideoX-Fun/tree/4b7b6402a1e0f0406bd6801fb66c0a00bd922621): 32 base transformer blocks, 16 control blocks, hint injection after base layers `[0,2,...,30]`, model width4096 and32 heads of128 dimensions. Only the first control block has `before_proj`; every control block has `after_proj`, both biased.
- Control input has **129 channels**: control latents64, inverted nearest-neighbor mask1, masked-source latents64. Pixel mask white regenerates; packed mask value1 means **known**. Unknown RGB is zero in normalized `[-1,1]` space (midgray pixels), while alpha is appended as opaque1. Missing structural control means literal zero64 latent channels.

## Faithful first teacher

The official control pipeline encodes text without an image-reference prefix. The central source is supplied through the masked-source control channels. Use this conditioning first; extra square reference tokens are an experimental extension, not part of the official control pipeline. Every step recomputes the full joint stream, with prefix cache disabled, matching the current VideoX pipeline.

The control branch is CFG-distilled. Published examples use **CFG1 and40 Euler steps**. A20-step trial is a declared reduction; CFG4 changes the trained inference regime and doubles conditional/unconditional work. Use the base resolution-shifted scheduler, including its terminal configuration; the Turbo six-node scheduler belongs only to the separate adapter path.

Exact final central pixels require compositing the resized original source after decoding. The faithful first solver does not clamp known target latents between steps; a known-latent flow bridge is a separately declared extension. Inspect the raw decoded source boundary and the exact-source composite because hard compositing can expose learned reconstruction drift.

## Port and numerical acceptance

`control.py` reuses existing MLX Qwen block/normalization/RoPE modules and implements the trained control chain plus whole-joint hint injection. It strictly validates all180 checkpoint keys and shapes before loading. No reference runtime file is modified. Load the branch after text encoding and encoder deletion to avoid retaining the large text encoder alongside the new BF16 branch. OptionalQ4/Q8 control quantization is a separate approximation; the129-input projection remains unquantized because its input is not divisible by64.

The control chain first computes all hints from the original joint input, then the base chain consumes them after each configured layer. It must not interleave the two chains or add hints before base blocks. Each control and base block receives identical per-token modulation, absolute RoPE and block-causal prefix segments.

CPU parity with tiny random weights against the pinned upstream Torch block code is required before a GPU trial. Those tests must exercise nonzero trained-style projections, structural input scatter, attention, modulation and both control scales. Weight-free arithmetic parity does not establish actual image quality or quantized/full-checkpoint parity. Real outputs must still preserve source pixels, remain finite and pass direct visual inspection.

## Later growth port

Once the full teacher produces a useful structure, retain the final-canvas RoPE and gather target latents and129-channel control context using exactly the same active absolute raster IDs. Future target tokens must be omitted from both chains. The current port accepts a caller-supplied layout and context, enabling this path without re-centering positions. The inactive regions' precomputed control VAE latents do not themselves constitute denoiser work, but full control-image VAE encoding must be included in compute accounting.

## Completed CPU gates and next command

`cpu_parity.json` reports four meaningful float32 cases against exact AST-extracted pinned upstream Torch classes and model forward: text-only targets, prompt padding, interleaved image references and custom control injection positions. Maximum control block/hint difference was `9.54e-7`; maximum complete output difference was `1.049e-5` against a `2e-5` threshold. Setting control strength to zero reproduces the existing MLX base forward exactly. Nonzero random control projections change the output, so this is not an all-zero no-op test. Equivalent CPU attention/RMSNorm/timestep helper substitutions and source hashes are recorded. No checkpoint or GPU was used.

The conditioning CPU gate and a separate read-only review passed normalized pixel values, opaque alpha, mask inversion, the129-channel order, source bytes and absolute target gathers. The parent subsequently ran the actual trained checkpoint. All dtype/shape/finite/source-exact checks passed, but every20-step completion failed full-scene quality. Tangent Canny materially improved the rail; white exterior and incompatible coast remained. A40-step tangent trial with known bridge also failed. See [CONTROLNET_RESULTS.md](../CONTROLNET_RESULTS.md) for direct-view evidence and measured costs.

The initial runner defaults to BF16 control on the existing Q4 base,20 steps,CFG1 and no known-latent bridge. It saves raw and exact-source final composites, source-border errors, real NFE and both base/control block counts. Shared conditioning, encoder deletion and control load are logged. All modes share the same masked source and original noise; the modes differ only in supplied structural control.

```sh
experiments/qwen/.venv/bin/python experiments/qwen/controlnet_port/run_control_trial.py --track track3 --preflight
experiments/qwen/.venv/bin/python experiments/qwen/controlnet_port/run_control_trial.py --track track3 --steps 20 --modes mask-only source-canny tangent-canny --output /absolute/fresh/suite
```

`tangent-canny` uses the independently prepared automatic Canny continuation map. Its inferred lines are a fallible structural hypothesis, not source ground truth. The runner neither draws lines over the generated output nor modifies the original source. Use a new output directory for every trial. A passed CPU gate confirms numerical portability, not image quality, quantized checkpoint parity or mobile deployment.

## Sparse structural-input diagnostic

`run_sparse_control_trial.py` is a separate copied runner. After complete control-map VAE encoding, `sparse_conditioning.py` keeps control64 on the known source and optional64px Euclidean dilation of accepted unknown guide pixels. Unconstrained structural64 becomes literal zero; source65 remains exact. White support means retain, packed with the same floor-nearest coordinates as the known mask. This is experimental local input dropout: official training drops whole control maps globally, and zero input does not imply zero trained hints. The base/control chains still process the entire target on every step.

The first40-step tangent-supported diagnostic with known bridge completed: white margins disappeared, but duplicated rail/sea/road exterior still failed full composition. It matches the completed40-step full-control known-bridge run except spatial control64 support. See the results report. Preflight loads no weights and saves actual support PNG, packed NPY, hashes and CPU polarity/zero/keep/gather tests. Optional `--control-scale` and `--prompt` are recorded deviations; preserve defaults for matched diagnostics. Alpha is saved during decode if present.

```sh
experiments/qwen/.venv/bin/python experiments/qwen/controlnet_port/run_sparse_control_trial.py --track track3 --steps 40 --known-bridge --modes tangent-supported --preflight --output /absolute/fresh/preflight
```
