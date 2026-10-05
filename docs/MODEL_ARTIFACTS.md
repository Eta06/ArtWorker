# Model artifacts and reproducibility

No ArtWorker-trained or distilled checkpoint has been released. Current experiments use third-party pretrained weights, local ports, quantized conversions and sampler/compositor changes. Latent optimization is not model training.

## What this repository stores

Store the source repository ID, immutable revision, source license, file sizes, expected digests and actual verification receipts. The experiment directory also records the runtime, prompts, scheduler, mask/conditioning protocol, seed, calls, measurements and visual verdict. Report preparation separately from actual inference.

| Model family | Existing records |
| --- | --- |
| Qwen-Image 2.1, Turbo and trained Union ControlNet | [Model research](OUTPAINT_MODEL_RESEARCH.md), [ControlNet provenance](../experiments/qwen/CONTROLNET_RESULTS.md) and experiment-specific manifests |
| FLUX.1 Fill dev Q4 | [Native Fill report](../experiments/flux1_fill/FILL_RESULTS.md), verified file manifest in [model manifests](../experiments/model_manifests/) |
| SDXL inpaint and Union ProMax | [Download manifest](../experiments/sdxl_inpaint/download_manifest.json), [ProMax download manifest](../experiments/sdxl_inpaint/promax_download_manifest.json), verified file snapshots in [model manifests](../experiments/model_manifests/) |
| Qwen-Image-Edit-2511 + MaskFlow latest | [Verified download manifest](../experiments/maskflow/download_manifest.json), [runtime and source pins](../experiments/maskflow/README.md) |
| Other mobile candidates | [Trial results](OUTPAINT_TRIAL_RESULTS.md), [deployment research](IOS_OUTPAINT_DEPLOYMENT.md) and each experiment's status records |

Manifest verification is historical evidence at the recorded time, not a fresh check of today's remote or local files. Absolute local paths in historical records identify original artifacts; adapt paths in your own checkout. Model payloads, local caches, private input/output images and latent arrays remain outside normal Git. The files have not been deleted locally.

GitHub [blocks normal Git files over 100 MiB](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github). Store multi-GB pretrained weights at their upstream pinned sources. Do not duplicate them into this repository merely to track an experiment.

## Future ArtWorker weight release

For a checkpoint we actually train, record the training code commit, teacher and base revisions, dataset permissions and lineage, objective, resolution/mask distribution, hardware, steps and evaluation set. Distillation needs a teacher that passes quality review first. Publish a model card with latency, memory and quality measured on named hardware, including failures and limitations, plus file checksums.

Choose an appropriate model artifact host or GitHub release only after checking the checkpoint's redistribution terms and storage limits. An open-source code repository does not automatically make a derived checkpoint freely redistributable. No paid hosting, model upload or new training job has been configured in this publication stage.

## Current license distinctions

Project code: Apache-2.0, with retained third-party notices. MaskFlow code/adapters: MIT, separate from its Qwen base. The chosen 2511 conversion declares Apache-2.0; Qwen 2.1 control checkpoint terms are separately recorded as Qwen Research. FLUX.1 Fill dev uses non-commercial terms. SDXL uses its model card's OpenRAIL++ terms. Check each pinned model card and license for the exact intended use rather than applying the root code license to weights.
