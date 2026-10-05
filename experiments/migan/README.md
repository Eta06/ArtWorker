# MI-GAN compact CPU baseline

Source: [Picsart AI Research MI-GAN](https://github.com/Picsart-AI-Research/MI-GAN), code revision `2b793c5ece43f4253e32d4afc257120a5deed6f5`. Both `LICENSE` and **separate** `LICENSE-WEIGHTS` were read; both are MIT. The official README links the author's complete ONNX pipeline. `download_manifest.json` pins that export and verifies its SHA-256. The model file is **28,079,181 bytes**, with preprocessing/postprocessing included; it has no giant text encoder or diffusion loop.

This is a compact mobile-oriented baseline from ICCV 2023, not a claim that it is a newly released 2026 model or the best outpaint model. It is relevant to the full-system footprint and existing teacher/student inpainting direction. Author-reported model-only mobile timings do not substitute for the complete pipeline measurements here.

Runtime: isolated `.build/migan-venv`, ONNX Runtime 1.30.0, two intra-op CPU threads and one inter-op thread. Dependency versions are recorded by `requirements.lock`. CPU provider only; no Core ML or real iPhone run is claimed.

```sh
python3 scripts/run_bounded_model.py \
  --output experiments/migan/runs/2026-10-06/NEW-RUN --max-gib 2 --timeout 60 -- \
  .build/migan-venv/bin/python experiments/migan/run_trial.py --track track3 \
  --output experiments/migan/runs/2026-10-06/NEW-RUN
```

The full pipeline takes the same 512×1152 canvas and a binary **known=255 / unknown=0** mask. Its output already blends original pixels; `pipeline.png` is not an uncomposited generator prediction. The first three full pipeline outputs were **not byte-exact** in the original square, despite the pipeline's known-area blend. The runner separately reports raw source MAE and pastes the exact source into `composite.png`; old pipeline outputs are preserved. Cold supervised jobs completed in 0.51–0.66 seconds with sampled footprint peaks of 0.36–0.40 GiB, but the difficult road cover hallucinated lettering/structure and texture; small footprint does not establish acceptable quality.

An additional `--method strips --band 64` diagnostic used ten sequential 512-square working windows, copying only each predicted 64-row strip. It completed in 2.07 seconds / 0.80 GiB sampled footprint but accumulated scene drift and visible bands. This is iterative local filling with more model calls, not the user's desired single-pass spatial streaming. It is retained as a failed quality attempt.

No checkpoint was fine-tuned, no ArtWorker trained weight was released, and local covers/outputs remain excluded from public Git. The upstream payload's MIT license and full pipeline size are preserved separately from any future student training/release decisions.
