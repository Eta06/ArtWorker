# Qwen-Image-2.1 local masked outpaint experiment

This experiment uses the official checkpoint with native MLX reference editing
through MFLUX. It performs flow Euler mask blending at each denoising step,
then composites the original resized cover pixels back exactly. This is **not
LanPaint**. Results must not be described as a LanPaint benchmark.
The reference encoder receives the padded 512×1152 canvas; the known-region
latent also comes from that canvas. The shared mask leaves the central
512×512 source at y=320 unchanged. Track descriptions are shared with the
Klein experiment, with an additional explicit expansion instruction.

Pinned base: `Qwen/Qwen-Image-2.1@790c92633540aa0cb11d9abf19eb46d861714758`.
Pinned turbo adapter: `Viggle/Qwen-Image-2.1-viggle-turbo@bb26a0f38e5fe6c124aaccc9187a87eed5d9ed13`,
`Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors`.
The additional `r256` adapter is the publisher's recommended full-rank variant;
`r128` is its smaller truncated variant. Variant folders remain distinct.
The adapter remains a runtime branch rather than being baked into quantized
weights. Turbo uses its documented six-node sigma schedule.

Model files live under ignored `.build/models/qwen`. The Python environment is
isolated in `experiments/qwen/.venv`. Player source files are untouched.

```sh
experiments/qwen/.venv/bin/python experiments/qwen/run_trial.py --preflight
experiments/qwen/.venv/bin/python experiments/qwen/run_trial.py --track track1 --steps 40 --quantize 4
experiments/qwen/.venv/bin/python experiments/qwen/run_trial.py --track track1 --turbo --quantize 4
experiments/qwen/.venv/bin/python experiments/qwen/run_trial.py --track track3 --turbo --adapter-rank 256 --quantize 4
experiments/qwen/.venv/bin/python experiments/qwen/run_trial.py --track track3 --turbo --adapter-rank 256 --known-padding edge --quantize 4
experiments/qwen/.venv/bin/python experiments/qwen/run_spatial_trial.py --preflight --track track3
experiments/qwen/.venv/bin/python experiments/qwen/run_spatial_trial.py --track track3
```

The scripts record phase timings, MLX allocator peak, process peak RSS, exact
source preservation, revisions, seed, scheduler values and errors. Mac runtime
measurements do not establish iPhone feasibility.
On-load conversion from the original BF16 checkpoint is a separate peak from
conditioning, denoising, and decoding. The script releases the text/vision
encoder after prompt conditioning, then releases the denoiser before VAE
decoding. The saved metrics distinguish those phases.

`RESULTS.md` describes nine successful standard trials. The separate
`run_spatial_trial.py` diagnostic runs matched FULL/GROW loops with one model
load and a fixed square-source prefix. GROW omits future target rows from the
transformer and adds them within its single six-step schedule. Actual target
forwards drop 22.2%, measured core time drops 17.2% in one pair, but final outer
band quality fails. Predicted-clean snapshots are decoded after both loops;
the experiment does not measure live UI streaming. `SPATIAL_RESULTS.md` and
`TOKEN_GROWTH_NOTE.md` preserve these limits and actual artifact paths.

References:

- https://huggingface.co/Qwen/Qwen-Image-2.1
- https://github.com/mflux-community/mflux/tree/main/src/mflux/models/qwen21/reference
- https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo
