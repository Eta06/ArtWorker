# ProMax repaint: official semantics audit

Audited 2026-10-05 after the native task-7 run completed. This audit is read-only except this report and the earlier feasibility note's numeric-identity update; no runner, runtime or manifest changed and no GPU was used. The nearly black unknown area is an observed failed visual result, not yet a diagnosis of the checkpoint's quality.

**Main divergence:** the author demo masks a complete real photograph but still passes the complete photograph as the initial `image`. Our test passes a cover on a black expanded canvas. At `strength=0.9999`, both pipelines encode that entire initial image and add noise, so the author demo retains information about the masked target region in initial latents; our unknown region instead contributes black-canvas latents. This is an official-component/parameter baseline, not identical unknown-region initialization.

## Frozen evidence

- Runner: `run_promax_trial.py`, SHA256 `ecdd952376fbf04569f8ebc37c61e1cde00a367c7de15a93e2c00d753623bd20`.
- Native Diffusers 0.39 Union inpaint pipeline: SHA256 `a4fcc80425c0e40477c6c89ee75e528f24dc0df7aeafd9c904bb8cc08239db8b`.
- [Pinned author demo](https://github.com/xinsir6/ControlNetPlus/blob/b48420576eac63c04388cb65fb74513cbd17405a/promax/controlnet_union_test_outpainting.py): SHA256 `e3a7f8cb56ac5dbc952e3c58c37797934eec201cf1604e275abc4c3e6d370fb8`.
- [Pinned author pipeline](https://github.com/xinsir6/ControlNetPlus/blob/b48420576eac63c04388cb65fb74513cbd17405a/pipeline/pipeline_controlnet_union_inpaint_sd_xl.py): SHA256 `42b3ba7bb88a0686269a1dc5869f14520fdaa396d1e7f4511d813bc47f8a6b69`.
- Observed run: `runs/2026-10-05/track3-30-repaint7-seed42/metrics.json`; 29 ControlNet, U-Net and scheduler calls; source-latent replacement verified on every step. Technical `status=success` does not establish visual success.

## What matches, and what differs

| Item | Exact author semantics | Frozen native trial | Assessment |
|---|---|---|---|
| Control RGB | Copy the complete RGB image, zero only masked pixels (demo 89–99); control processor has `do_normalize=False` (pipeline 238–240) | Known RGB retained, unknown black (runner 168–170); actual CFG tensor `[2,3,1152,512]` checked in [0,1] | Matches. Do not use a -1 sentinel or make the control unknown area gray. |
| Image/VAE range | Separate full image normalized to [-1,1]; masked-source image multiplied by known mask (pipeline 1503–1507,1534) | Black padded initial image is normalized to -1 outside; separate masked-source unknown RGB is zero in normalized space | Same preprocessing; different full-image unknown content. Control black [0,1] and masked-source zero [-1,1] are different correct tensors. |
| Mask | Mask generator's unknown area becomes white 255, known black; grayscale conversion and binarization (demo 91–100; pipeline 235–237) | White 255 above/below cover, black on cover; exact geometry asserted (runner 164–166) | Polarity matches. |
| Task | Legacy eight-slot list, repaint in slot 7, one-hot `[0,0,0,0,0,0,0,1]` | Native condensed `control_image=[RGB]`, `control_mode=[7]`; model receives `[batch,8]` one-hot and index list [7] | Correct API translation; not a mode mismatch. |
| Initialization | `image=original_img` remains the complete real photograph; only the control image is blacked (demo 95,113) | `image=canvas` has real source center and RGB [0,0,0] unknown area (runner 173,500–505; metrics) | Material input-prior divergence at strength below 1. |
| Resize/mask extent | Whole target photograph resized to approximately 1024² area, dimensions floored to multiples of 8 (demo 82–86); random masked outer strips, padding 6–30% per selected side | Fixed 512×1152, 512-square source; each unknown top/bottom strip is 320px, about 27.8% of canvas height | Deliberate benchmark resolution/geometry difference. No square-only aspect-ratio prohibition follows from the author's axis-parallel-boundary comment. |
| Prompt/CFG | Positive prompt is a placeholder, not a reproducible scene prompt; demo provides a nonempty anatomy/quality negative prompt. Pipeline default CFG is 5 (1159) | Detailed lake/guardrail/dark-asphalt prompt (runner 30–34), `negative_prompt=None`, CFG 5 | CFG matches; negative conditioning differs. No official concrete scene prompt to reproduce. |
| Scheduler/strength | Euler ancestral from ordinary SDXL scheduler; 30 requested steps; pipeline default strength 0.9999 (1155), with floor-based timestep trim (1015–1023) | Same scheduler source/type, 30 requested, strength 0.9999; actual 29 steps, begin index 1, first timestep 925 | Parameter semantics match. Native additionally sets scheduler begin index explicitly. |
| Randomness | Demo creates a random seed variable but leaves the generator argument commented out | CPU generator seed 42 passed into VAE posterior, initial noise and ancestral sampler; forwarding verified | Deliberate deterministic trial; not bitwise reproduction of demo. |
| Preservation | Ordinary four-channel branch re-noises known source at the next timestep, then clamps it after each solver step (pipeline 1756–1769) | Same operation independently verified from saved tensors every step | Matches; decoded source pixels still need a separate source-only composite. |

## Is strength 1 justified?

**Yes, as a declared initialization ablation, not as the unchanged author default.** Both the author's pipeline (1490,921–926) and native implementation (1526,931–933) explicitly support `strength == 1.0`: initial latents become pure noise multiplied by scheduler initial sigma. The four-channel pipeline still encodes the source for later known-region clamping; it does not discard the source anchor or black repaint control.

The observed 0.9999 schedule skips timestep 958 and starts at 925 with sigma **9.5435857773**. The initial state is `image_latents + sigma * noise`; after the scheduler's model-input scaling, the initial-image term has coefficient approximately **0.10421**. That is a remaining latent prior, not a proof that it caused the black result. Strength 1 uses all **30 iterations**, starts at **958**, and removes the black unknown-region prior from the initial solver state. For this leading schedule the expected pure-noise initial sigma is approximately **11.52033409**. The extra solver step changes the ancestral RNG trajectory; the same seed alone does not make outputs otherwise compute-matched.

The frozen runner currently forbids strength 1 at lines 101–102. Its owner must explicitly permit and label this ablation, update initialization metadata, and log all 30 actual calls; this audit does not edit it. Keep mask, control RGB, source rectangle, task, CFG, prompt and scheduler fixed for the first diagnostic run. If black output persists, then test negative conditioning and author-area resolution separately and examine numerical model parity/activation statistics. None of the input differences alone establishes a model-quality cause or guarantees a repair.
