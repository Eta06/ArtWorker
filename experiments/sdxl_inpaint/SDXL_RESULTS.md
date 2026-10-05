# Native SDXL inpaint 0.1: valid execution, severe visual failure

The first parent-scheduled native 20-step trial fails complete-image quality. Both raw and exact-source composite were directly viewed: almost-black upper and lower extensions, a short curving rail near the upper boundary, and an incoherent lower shoe/body-like patch. The original source is exact after paste, but the generated exterior does not continue the scene. Execution and numerical integrity passed separately.

The public official FP16 pipeline is downloaded and strictly verified at revision `115134f363124c53c7d878647567d04daf26e41e`: 18 files, 6,941,218,469 bytes, four LFS SHA256 checks and fourteen supporting Git blob checks, zero issues. CPU verification took 4.573 s. Payload remains isolated at `.build/models/sdxl-inpaint-fp16`; older models and the installed runtime were preserved.

- Verified manifest SHA256: `8d8d4d1834a4c3b7137e46d66e015cfd348fa77ea678f830974968272ce50f2c`.
- Frozen runner SHA256: `2a636b4c34cc1f2c714569dc1f2b5e48f45382dc79951ebdcbb675d3461261f0`.
- Strict CPU preflight SHA256: `5a4c4d88f123623a4e9c65e58818057d9b3ae4e15da6c3348c4d2c724427109e`; status `preflight_files_ready`, no pretrained model loaded.

Six meaningful CPU groups cover native image/mask threshold and polarity, exact source geometry and zero normalized unknown masked-source pixels, nonuniform nearest mask downsampling, native nine-channel concatenation and latent bypass, strength/timestep trimming, and a random tiny SDXL-style U-Net's two-branch CFG forward contract. Independent read-only audits found no wrapper, shape or native-RNG-order blocker. These checks do not establish real MPS inference or image quality.

The executed first trial is native full-canvas 512×1152, source `[0,320,512,832]`, seed 42, 20 requested steps, CFG 8, strength 0.99. The supplied Euler scheduler trims this to 19 actual U-Net calls / 38 CFG branch samples. Its learned input is noisy4 + mask1 + masked-source4. Source pixels are pasted separately after retaining raw decode; native nine-channel generation does not hard-clamp known latents.

The positive exterior prompt matches the tested Fill arm, but cross-model initial conditions do **not** match. SDXL uses the frozen shared RGB0 unknown canvas; Fill built RGB128 unknown. At strength 0.99 the native SDXL initial canvas is fully VAE encoded and mixed with sampled noise. Only source/mask/geometry are shared; the noise streams and full-canvas bytes differ. All actual states, posterior/noise draw order, scheduler timesteps, calls, sampled MPS memory and raw/composite hashes will be logged during a parent-scheduled trial.

The model card documents inpainting fine-tuning with synthetic masks; dedicated outpainting, source-rail continuity, spatial-growth savings and iPhone performance remain unproven. The public license identification is CreativeML Open RAIL++-M. [Official model card](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1), [exact metadata/source audit](SOURCES.md).

Exact launch (parent only, serialized GPU and bounded supervisor):

```sh
.build/dreamlite-venv/bin/python experiments/sdxl_inpaint/run_sdxl_trial.py --track track3 --steps 20 --guidance 8 --strength 0.99 --seed 42 --output experiments/sdxl_inpaint/runs/2026-10-05/track3-20-cfg8-strength099-seed42
```

`launch_ready.json` retains exact argv and verification hashes. This report's author has not launched GPU inference.

## Actual result and initialization audit

Run: `runs/2026-10-05/track3-20-cfg8-strength099-seed42/`. Total 25.163 s; native pipeline 18.624 s, observed denoising 14.988 s, summed U-Net calls 14.909 s, raw decode 0.932 s, model load 2.113 s, strict rehash 2.386 s. Sampled MPS maxima are 6,959,199,488 bytes current allocations / 10,794,549,248 bytes driver allocations; these are sampled values, not a true peak profiler.

Actual timesteps are 901,851,…,51,1, scheduler begin index 1, initial sigma 8.390687. Saved mask is `[2,1,144,64]`, with 0 in source rows40:104 and 1 above/below. The first U-Net input mask channel equals that saved mask exactly. Actual initial noise and all recorded states are finite; top unknown noise standard deviation is 1.014 and initialized latent standard deviation is 8.637. Recorded VAE/posterior states confirm native full-canvas posterior → noise → masked-source posterior order. Independent CPU reconstruction also matches actual `initial_latents = encoded_full_canvas + sigma * sampled_noise` exactly. All ten retained NPZ arrays are finite and match their recorded hashes. There is no inverted mask, skipped U-Net, absent noise or compositor blackening in these checks. Detailed evidence and unfiltered join crops are retained in the run's `independent_validation.json` and PNGs.

Independently reconstructed native half-precision postprocessing of the saved raw decode equals the raw PNG exactly. In the exterior, 93.308% of upper pixels and 85.183% of lower pixels have all RGB channels below16; mean RGB is 3.839/5.058, versus source 52.305. This quantifies an actual generated failure, not a preview artifact. Raw source MAE all/top16/bottom16/inner is 9.034/10.091/6.203/9.093 on0–255; source exact and outside/raw pixel equality both pass after compositing.

- Raw PNG SHA256: `94a5662306b0843cad2a46983c2fbe8b446f600d80b4498fbabd424153f3070f`.
- Composite PNG SHA256: `b2b51e7d9c886f01d1d4cb51c49cc99a4225708ceca6d5c5b7aac16e73875409`.
- Actual initial noise SHA256(float32 expansion of native F16): `84d0897db02d9376ea1f5ca23d4512a05b12df44de265d16030420ad9041579b`.

The encoded black-init prior is a hypothesis, not a proven cause. Original runner and tests remain frozen. No claim is made that the native task architecture, short Mac runtime or CPU gate solves image quality or iPhone deployment.

## Executed native strength1 pure-noise probe

Run: `runs/2026-10-05/track3-20-cfg8-strength1-seed42/`; new frozen-copy runner SHA256 `9dce19ce14c3ec0f9bd1e711b440e827b92f9b6b6ddee9361622fca57d05c508`. The actual trace confirms 20 calls / 40 CFG branch samples, scheduler begin index 0, pure noise multiplied by native init sigma 11.073581, no full initial-image VAE encode, and exactly one masked-source VAE posterior encode. Noise→masked-source posterior is the changed native RNG path; seed 42 does not match the earlier .99 noise. The real saved noise hash matches the independent CPU prediction `0b77214495cb763414bc7e7585e2c0a552f91869f9dcf1cef00a282bdf005211`.

Both raw and composite were independently viewed. The black exterior disappears, but a large tan upper plane and bent rail replace the expected lake continuation; lower grass and a bright new post replace uninterrupted asphalt. The raw known region is substantially re-rendered/brighter (MAE 68.351 on 0–255), causing a strong brightness seam when the exact source is pasted back. **Complete-image quality remains failed.** Initialization, initial sigma/timestep coverage and sampled noise all change together, so this result does not isolate black RGB prior as the sole cause.

Total 24.472 s; pipeline 16.807 s, observed denoising 14.820 s, summed U-Net calls 14.746 s, decode 1.115 s. Sampled MPS current/driver maxima are 6,959,199,488/11,862,130,688 bytes. Source exact and generated-exterior/raw equality pass separately. Raw PNG SHA256 `3c74089d36a776097de6a687d08284a2d77c30070a0c47d7f0897192d50a1391`; composite SHA256 `3f1717608255013f9083556a7c9fd1df644488cbb2229871f274e5a543bfee95`.

The next task-conditioned candidate is the official ProMax repaint path with a four-channel SDXL base and per-step source-latent anchoring. Its pinned public assembly has separately passed file verification, and the native class matches all 863 checkpoint keys/shapes on CPU/meta. No ProMax generated result is established by that readiness gate. [Official recipe/compatibility plan](PROMAX_FEASIBILITY.md).
