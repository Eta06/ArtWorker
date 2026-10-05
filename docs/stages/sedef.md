# Sedef — first public archive

Date: 2026-10-06 (Europe/Istanbul). The word was selected randomly by the primary agent. This stage establishes publication and progress tracking; it archives existing September/October experiments rather than claiming they were newly run during Sedef.

- Repository: https://github.com/Eta06/ArtWorker
- ChatGPT Space: https://chatgpt.com/space/page_eeabd8711884819189a662a8a09e9290
- Stage Page: https://chatgpt.com/space/page_704d1c7aeed0819191f996f1a6442ec2
- One changed file per commit, `sedef: <file>` subjects, one grouped push.
- Primary agent handles this publication. Future delegation remains capped at two subagents total, with the allowed models and no descendants defined in [AGENTS.md](../../AGENTS.md).

## Product goal and priorities

Extend square album artwork to a phone-shaped canvas while keeping its original pixels and making scene boundaries continuous. Compare quality, speed, working memory and model footprint. Investigate a compact mobile model and distillation after identifying an acceptable teacher. Real center-outward progressive sampling remains desirable but is secondary to these priorities.

The SwiftUI player is a separate prototype. It currently uses a blurred copy of the original cover; no model is bundled. There is no accepted production outpainting pipeline, trained ArtWorker checkpoint or physical-iPhone inference benchmark.

## Archived evidence

These are historical local Mac runs, generally on one challenging road/sea cover. Different runtimes, resolutions, conditioning, token budgets and measurement boundaries prevent using this table as a fair speed ranking. Times refer to the respective reports.

| Experiment | Recorded result | Quality / scope |
| --- | --- | --- |
| Qwen 2.1 + Union ControlNet, source top32 collar, 40 steps | About 430 s supervisor elapsed. Local rail endpoint offset improved from −17.404 to −2.086 px. | Water/asphalt texture seams and a small final-row rail jog remain. [Direct review](../../experiments/qwen/boundary_eval/2026-10-05/controlnet-saved-top32-comparison/REVIEW.md). No complete quality pass. |
| Genuine growing Qwen Turbo/control, 6 steps | Two-arm suite about 146.52 s; future target tokens absent before activation. | One arm improves endpoint placement but has a +7.72° angle error and doubled/pinched rail. Untrained Turbo/control combination. [Review](../../experiments/qwen/boundary_eval/2026-10-05/growing-top32-comparison/REVIEW.md). |
| Native FLUX.1 Fill dev Q4, 50 steps | Full-canvas arms about 347/315 s; split upper/lower context about 321 s combined. | Duplicate objects/lettering in the first arm; alternative conditioning leaves implausible geometry or asphalt stripes. [Report](../../experiments/flux1_fill/FILL_RESULTS.md). |
| Native SDXL inpainting 0.1 | About 25–26 s, 19/20 actual timesteps. | Black exterior or incoherent tan/grass/rail completions. [Report](../../experiments/sdxl_inpaint/SDXL_RESULTS.md). |
| Native SDXL Union ProMax repaint | About 46 s, 29 actual timesteps. | Exterior predominantly black. Conditioning checks do not establish quality or isolate the whole cause. [Report](../../experiments/sdxl_promax/PROMAX_RESULTS.md). |
| MaskFlow latest, mixed Q4/Q8 2511 base, 50 steps | Actual run completed: 1,889.729 s elapsed; observed peak about 16.08 GiB. | Misaligned rail, changed water texture and dark lower asphalt seam. [Root visual review](../../experiments/maskflow/runs/2026-10-05/track3-latest50/ROOT_VISUAL_REVIEW.md). Not the official BF16 backend or a phone benchmark. |
| VAE latent projection / CPU compositors | Projection 4/16 updates and source-preserving post-processing were tried. | More optimization can worsen the rail despite lower loss. Tone/seam changes do not establish scene continuity. [Projection report](../../experiments/qwen/PROJECTION_RESULTS.md). |

Original numerical integrity and execution receipts remain in the experiment directories. Exact pasted source pixels, finite arrays, checksum matches and numerical port parity establish those properties only. They do not establish visual acceptance. Historical images and latent arrays stay local; their hashes and text reports are published.

Prepared alternatives such as the Fill footer-conditioning arm, ProMax pure-noise probe and older matched MaskFlow few-step pair are not promoted to completed runs. The older fast MaskFlow pair was researched, not downloaded and evaluated as a matched pair. No new model inference or training is performed by this publication stage.

## Next research stage

1. Define a bounded multi-cover evaluation set with scene continuity and preservation criteria, and consistent hardware/timing boundaries.
2. Compare genuinely different task-matched models and applicable few-step variants, including faithful upstream protocols. Separate teacher-quality trials from mobile feasibility trials.
3. Measure phone/device-tier memory, time, thermal behavior and storage before claiming broad iPhone support. Mac timings cannot substitute for these measurements.
4. Only then choose a teacher/student route, distillation objective and permitted training data. Keep progressive sampling as a separately evaluated feature.

See [model artifacts](../MODEL_ARTIFACTS.md), [distillation plan](../DISTILLATION_PLAN.md), [deployment research](../IOS_OUTPAINT_DEPLOYMENT.md) and [third-party notices](../../THIRD_PARTY_NOTICES.md).
