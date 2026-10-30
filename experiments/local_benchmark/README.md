# Local ten cover outpaint benchmark

Safran compares the installed model families on ten frozen, locally available
original album covers. It does not alter the old experiment inputs, teacher
outputs, review decisions or training manifests. All media, titles, selections,
per-job commands and measurements remain under the ignored `results/` directory.

## Protocol

- One 512-square RGB source per cover, centered at `[0,320,512,832]` in a
  512×1152 target, seed 42. The same seed is not identical noise across runtimes.
- Fifteen arms: MI-GAN, DreamLite Mobile and Base, Mobile-O with latent
  restoration, FLUX.1 Fill Q4, native SDXL inpainting, official-component
  ProMax, Klein distilled/base Q4, distilled Q4 plus outpaint adapter, and
  five baked-base variants (Q4 with encoder, cached Q4/Q3/Q2 and mixed MLP3).
- The LoRA is an adapter, not an independent generator. Quantized variants are
  separate arms of a family. Source BF16 files are backing weights used by the
  native cache loader, not additional mobile recipes counted as new models.
- Mask-capable models receive explicit masks. Klein's outpaint adapter receives
  its original green-placeholder trigger; native instruction models receive the
  recorded extension instruction. This compares complete working recipes, not a
  controlled causal effect of architecture with identical conditioning.
- Raw outputs and exact-source composites are retained separately. No color
  harmonization, seam retouching or synthetic structure is applied. A successful
  paste or a low source reconstruction error is not a visual-quality ranking.
- Legacy adapters keep an internal `track1` label; each external `coverNN` has
  its own manifest and SHA256. This alias never changes the historical track1.

## Run

```sh
experiments/qwen/.venv/bin/python experiments/local_benchmark/benchmark.py prepare \
  --folder experiments/local_benchmark/results/DATE/STAGE
experiments/qwen/.venv/bin/python experiments/local_benchmark/benchmark.py run \
  --folder experiments/local_benchmark/results/DATE/STAGE
```

The serial queue holds its own nonblocking lock and the existing global heavy-job
lock. It requires three normal macOS pressure samples before each launch.
After explicit user authorization, `--allow-warning-pressure` permits warning
level 2 for this benchmark only. Three samples are still required; critical
levels, the process ceiling and the swap-growth stop remain enforced. All other
callers of `run_bounded_model.run` still require normal pressure by default.
Inference retains the previously used **6 GiB** RSS/physical-footprint ceiling;
MI-GAN uses **1 GiB** after two recorded 512 MiB resource stops. The API queue's
512 MiB limit is not an image-diffusion budget. The current benchmark recipe
allows at most 1 GiB swap growth per non-MI-GAN job (256 MiB for MI-GAN), and
adds a 25% system free-memory floor, sampled once per second. RSS/footprint,
pressure and swap are sampled every 100 ms. Other supervisor callers retain
their previous 256 MiB swap default and have no implicit warning allowance.
No other apps are
closed and no remote host or paid image endpoint is used. A waiting queue exits
with a paused status after two hours without adequate memory.

Each job is a fresh process, so measured times include cold loading. A capacity
or runtime failure on the first cover blocks the remaining nine for that arm;
these are explicitly untested, not nine poor images. A system-pressure/swap stop
pauses the entire queue. An interrupted running job is never silently replayed.
Inspect its receipt before resuming. Previously completed jobs are skipped.
`--retry-resource-arm NAME` records a resource-stopped attempt in history and
uses a fresh numbered output directory. It never replays successful jobs or
overwrites old receipts. `--low-memory-first` schedules the smaller cached
Klein recipes before larger instruction editors. CUDA/unsupported-runtime
failures still block an arm rather than triggering blind retries.
`--arms NAME ...` runs a controlled subset without consuming other pending jobs.

The optional `portrait80` Klein decoder recipe applies
`portrait-vae-tiling.patch` to the existing pinned/patched Swift checkout and
requires a guarded one-job rebuild. It decodes this exact 512×1152 output as two
640×512 tiles with 128 px overlap, with their join inside the pasted source.
The default decoder is unchanged without the environment flag. This changes
decoder computation and can change pixel values; it is a separately recorded
memory-saving recipe, not a demonstrated equivalent output to full-frame decode.

The subsequent `low-memory-execution.patch` adds a separate opt-in
`ARTWORKER_LOW_MEMORY=1` profile. It evaluates/clears after each transformer block
and caps phase caches at 64 MiB. The benchmark arm field `low_memory: true`
selects it and requires the native confirmation log; unmodified native runtimes
are rejected. Without the flag, the upstream/past experimental execution stays
unchanged. See [Firuze](../../docs/stages/firuze.md) for actual execution status.

`state.json` is the live queue status, `summary.json` contains only execution
measurements and `comparisons/*.jpg` contain completed composite contact sheets.
Every successful leaf has `raw.png`, `composite.png`, `result.json`, and a guarded
launch receipt. Final visual assessment still requires inspecting actual
extensions, seams, repeated subjects and unwanted lettering.

## Validation

```sh
experiments/qwen/.venv/bin/python -m unittest discover \
  -s experiments/local_benchmark -p 'test_*.py'
```

The CPU tests exercise pressure refusal, interrupted-job refusal, capacity-failure
handling and the distinction between raw source error and exact compositing.
They do not load pretrained checkpoints or establish output quality.

See [the Safran stage](../../docs/stages/safran.md) and
[current candidate research](NEW_MODEL_RESEARCH.md).
