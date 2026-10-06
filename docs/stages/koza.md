# Koza — measured model compression

6 October 2026. The safe compression recipe removes the full text encoder for the fixed outpaint instruction: inference-component weights fall from **4,611,193,511 to 2,363,900,089 bytes (48.736% smaller)**. All three matched cover outputs are raw-pixel identical to the original recipe. More aggressive quantization was actually exported and tested, with failures retained. This does not finish the phone goal: inference is still slow and memory-heavy, and the current native loader is not a standalone mobile bundle.

The user redirected work away from the previous player/simulator integration request to model compression. The unfinished development bridge is preserved under ignored `.build/player-integration-wip/`; no player or simulator work ran. No subagents started. No trained/distilled ArtWorker checkpoint was created or released.

## Actual component sizes and decisions

| Recipe | Transformer bytes | Text conditioning bytes | VAE bytes | Total bytes | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Original baked INT4 + Qwen3 encoder | 2,180,050,216 | 2,263,022,417 | 168,120,878 | 4,611,193,511 | Cinar baseline, existing human-cover failure |
| INT4 + fixed FP32 instruction | 2,180,050,216 | 15,728,995 | 168,120,878 | 2,363,900,089 | Selected compression baseline; exact parity on 3 covers |
| Selective MLP/output INT3, remaining INT4 | 1,979,510,048 | 15,728,995 | 168,120,878 | 2,163,359,921 | Experimental; road/black promising, human clothing changes |
| Uniform INT3 | 1,695,608,016 | 15,728,995 | 168,120,878 | 1,879,457,889 | Rejected on all 3 covers |
| Uniform INT2 | 1,211,165,728 | 15,728,995 | 168,120,878 | 1,395,015,601 | Rejected on road/sea cover; other covers not run |

These are model-component storage totals, excluding application code, local research caches and preserved source weights. They are not peak working-memory totals or a deployable iPhone package. The upstream cache loader still resolves the BF16 source directory and its file fingerprint, even when it never materializes those source tensors. A portable manifest-backed loading path remains necessary.

## Fixed instruction specialization

The pinned Qwen3 encoder is run once to export the exact FP32 `[1,512,7680]` conditioning tensor for `Fill the green spaces according to the image`. The source image continues to condition the VAE and transformer. This removes an encoder component, not image-transformer parameters, and does not support arbitrary text prompts. Native guards check the checksum, shape, pinned encoder revision and fixed recipe (Klein base4B, CFG1, no upsampling/interpretation).

The first export hit its 4 GiB process limit after 1.100 s, without a completed cache. The separate export under the standard existing 6 GiB development budget completed in 1.389 s, peak sampled footprint about 4.044 GiB. This one-time desktop preparation is not phone inference. The cache manifest pins revision/hash and records parity; the payload remains local.

The first encoder-free image attempt stopped automatically on system memory pressure after 0.957 s. The cached-embedding branch had skipped the text-phase GPU cache-limit initialization, leaving reference-image encoding with the default cache budget. That branch now applies the conservative phase cache limit. The fresh run completed and matched the Cinar output exactly. The failed receipt was preserved; pressure was allowed to return to normal before continuing.

## Matched cover quality

All generation comparisons use the same fixed prompt, 512×1152 canvas, source rectangle `[0,320,512,832]`, baked outpaint adapter scale 1.1, 12 steps, CFG1, standard VAE and release-before-decode. Seeds: track1/track2 42, track3 17. Baselines are the Cinar sequential-decode INT4 runs, with the original pinned text encoder. Raw generated pixels outside the source rectangle are compared; the original rectangle is pasted identically into each visualization. Exact pasted source pixels alone are not a quality test.

- **Encoder-free INT4:** all raw pixels match on all three covers, outside MAE/RMSE zero. It preserves the original human-cover mistake where the printed tote-bag graphic becomes a real bag. Quality preservation relative to this baseline is established for these three pairs; general model quality is not accepted.
- **Uniform INT3:** sea/road blur and lost guardrail continuation, green fog even on the black cover, and green exterior/hard seams on the human cover. Rejected; smaller bytes and valid native tensors do not count as success.
- **Selective `mlp3`:** road/sea structure continues and black padding changes only slightly. Sea/asphalt texture and tones differ. The human cover invents a different bag shape and lower-shirt graphics. Retained as an opt-in research candidate, not a generally accepted default replacement.
- **Uniform INT2:** the first road/sea result is speckled noise outside the source. Rejected immediately instead of spending more runs on the other two covers.

| Comparison | Outside MAE /255 | Outside RMSE /255 | Entire raw image exact |
| --- | ---: | ---: | --- |
| track1-int3-fixed | 36.5967 | 40.3168 | no |
| track1-int4-fixed | 0.0000 | 0.0000 | yes |
| track1-mlp3-fixed | 1.5046 | 2.0368 | no |
| track2-int3-fixed | 39.3327 | 52.3320 | no |
| track2-int4-fixed | 0.0000 | 0.0000 | yes |
| track2-mlp3-fixed | 11.4226 | 22.8348 | no |
| track3-int2-fixed | 54.9122 | 69.0509 | no |
| track3-int3-fixed | 37.0776 | 44.0258 | no |
| track3-int4-fixed | 0.0000 | 0.0000 | yes |
| track3-mlp3-fixed | 8.0785 | 11.2651 | no |

The pixel metrics describe differences, not semantic correctness. Visual verdicts are recorded separately in each comparison JSON; ignored local contact sheets retain the actual media.

## Native compatibility and safety

The source FP16/BF16 weights plus all 88 mapped adapter pairs are merged before one independent low-bit quantization; the old INT4 payload is not requantized or overwritten. Uniform exports use affine group64. The selective profile uses INT3 for double-block feed-forward matrices and single-block output projections (the latter combine attention and feed-forward output), INT4 elsewhere. Separate model roots/profile metadata prevent silent uniform/mixed confusion. Native key/shape/dtype validation uses the same per-layer quantization structure.

A small cross-version fixture compares Python MLX 0.32.3 with Swift MLX 0.31.6. All bit depths decode exactly. INT2/INT4 matrix products are exact; INT3 differs by max 0.033203125 and relative L2 0.00105108, failing the recorded 0.001 compatibility threshold. The initial zero-tolerance product check also failed. A follow-up wrapper run hit a NameError and was corrected; its failure remains in the receipts. The all-bit check remains failed. This rounding/runtime difference is a confound when assigning image failure solely to bit depth; it does not excuse the observed bad images or promote the mixed candidate.

`--require-prequantized` now forbids native dense fallback. An intentionally wrong mixed profile against the uniform cache was rejected before loading the BF16 source or producing an image; `cache-rejection-probe/verification.json` records the expected negative test. This is a verified rejection, not an inference-quality success. The embedding/low-bit runtime patch applies cleanly to the preserved pre-Koza snapshot and the final native build succeeded.

The wrapper also checks the embedding against the pinned compression-manifest checksum before native launch. A mismatched fixture was rejected with no model launch; `embedding-checksum-rejection/verification.json` records this separate negative test.

## Measured Mac execution

M4 Max, 36 GiB. One guarded heavy job at a time; builds limited to 3 GiB, tensor-wise exports 2 GiB, inference 6 GiB. Every process run samples physical footprint/RSS, system pressure and swap; the guard stops its process group on excess. Completed image runs stayed at pressure level 1 with no swap growth. The initial pressure stop is recorded above. Sampling is not a kernel cap and can miss short peaks. RSS and physical footprint overlap and must not be added.

| Run | Execution | Seconds | Peak sampled footprint GiB | Peak sampled RSS GiB |
| --- | --- | ---: | ---: | ---: |
| track1-int3-fixed | completed | 91.181 | 4.813 | 1.827 |
| track1-int4-fixed | completed | 92.539 | 5.265 | 2.265 |
| track1-mlp3-fixed | completed | 73.412 | 5.077 | 2.091 |
| track2-int3-fixed | completed | 90.744 | 4.806 | 1.827 |
| track2-int4-fixed | completed | 92.359 | 5.499 | 2.276 |
| track2-mlp3-fixed | completed | 110.910 | 5.078 | 2.092 |
| track3-int2-fixed | completed | 71.173 | 4.361 | 1.376 |
| track3-int3-fixed | completed | 96.209 | 5.432 | 1.828 |
| track3-int4-fixed-cachelimit | completed | 79.768 | 5.233 | 2.278 |
| track3-int4-fixed-parity | failed | 0.957 | 5.306 | 2.265 |
| track3-mlp3-fixed | completed | 71.530 | 5.078 | 2.092 |

The selected encoder-free INT4 recipe still takes **79.77–92.54 s** and peaks at **5.23–5.50 GiB**. The original encoder was already released before image inference, so removing it mainly reduces shipped component storage rather than the dominant image-stage peak. Mixed precision timings vary from 71.53 to 110.91 s; one run per cover on a live development Mac does not establish a consistent speedup. No physical-iPhone performance or general-phone support is claimed.

All preparation, build, compatibility and image receipts remain under `experiments/klein_compression/`; [machine-readable summary](../../experiments/klein_compression/compression-summary.json), [selected recipe](../../experiments/klein_compression/chosen_recipe.json), [reproduction commands](../../experiments/klein_compression/README.md), and pinned conversion manifests in `experiments/klein_outpaint/`. Model/media payloads remain ignored. Code/API reference: [MLX quantization](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.nn.quantize.html).

## Next work

Keep the encoder-free INT4 recipe as the lossless compression reference, while retaining the smaller mixed candidate for controlled recovery/calibration experiments. Resolve/version-align INT3 matmul behavior before making numerical-equivalence claims. More covers and seeds are needed for general visual acceptance; the existing human-cover error must not be taught to a student as correct output.

The much smaller phone goal requires fewer image-transformer parameters and task/recovery training or distillation, not just a lower checkpoint bit count. The existing [100–400M student research budget](../DISTILLATION_PLAN.md) is still an engineering target, not a trained checkpoint or quality result. Preserve these three album covers as held-out evaluation and use permissioned paired/full-frame images with synthetic crop/mask conditions for training. Build a validated teacher/data pilot, then compare a smaller student against this fixed baseline before step distillation or mobile deployment. No student training, paid cloud workload, app integration, physical phone inference or true spatial streaming ran in Koza.
