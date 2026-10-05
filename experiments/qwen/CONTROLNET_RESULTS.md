# Trained ControlNet teacher: verified runtime, failed outpaint quality

The real trained teacher does **not** pass the complete-composition gate. Automatic tangent Canny conditioning substantially improves the upper guardrail connection, but introduces an incompatible upper coastline and broad white exterior regions. Preserving the original central pixels does not make these completions acceptable. Both faithful40 and40 known-bridge fail. Sparse structural input removes the white panels but still duplicates rail/sea/road exterior, so it also fails.

## Matched 20-step comparison

Suite: `controlnet_runs/2026-10-05/track3-20/`. All three raw images and all three exact-source composites were inspected directly. Same resized source, prompt, full512×1152 target grid, seed42, original noise, baseQ4, trained controlBF16, CFG1, no Turbo adapter, no prefix cache. This is a **full-canvas teacher**: all2304 target tokens and96 text-prefix tokens are present from step1. It does not establish center-out generation, reduced growing compute, or phone performance.

| Mode | Direct visual verdict | Denoise / decode seconds | Raw source MAE, top16 / bottom16 / all |
|---|---|---:|---:|
| mask-only | Upper rail changes direction; sea/rail/road are repeated below the source. Scene continuity fails. |128.740 /5.074|5.215 /4.069 /3.730|
| source-canny | Separate upper rail and disconnected vehicle/road forms; pale top strip and almost white lower exterior. Fails. |154.795 /1.573|5.304 /3.665 /3.602|
| tangent-canny | Upper rail follows the original diagonal and connects much better. Small local seam, invented coast/forest above, pale top strip and white lower exterior remain. Partial structural improvement; complete result fails. |140.814 /1.589|5.085 /3.629 /3.599|

Each arm performs20 full transformer evaluations,640 base-block calls,320 control-block calls and46,080 target-token forwards. The three-arm suite takes447.911s including shared load/conditioning and decoding. Shared phases: base load9.186s, text conditioning1.091s, control load1.157s, control conditioning2.242s. Each arm's observed MLX denoising peak is14,226,883,348bytes (~13.25GiB). Control load peak is13,060,998,984bytes (~12.16GiB); resident parameters include4,148,486,144bytes base transformer,1,350,961,616bytes VAE and7,550,959,616bytes control branch. These are local Mac measurements, not an iOS estimate.

Independent file checks confirm all three saved `[1,2304,64]` float32 latent arrays are finite, their recorded hashes match, both PNGs are512×1152, and each composite source crop equals the512-square source **byte for byte**. Exactness comes from pasting the original resized source after decode. The first20-step solver does not hard-clamp known target latents. Raw-source MAE measures learned reconstruction drift, not generated-exterior quality; lower MAE alone cannot promote an arm.

The white bands already exist in raw RGB and are not added by the final source paste or an alpha-to-white compositor: the runner saves decoded RGB directly. This does not identify their cause. Baseline decoding did not retain alpha diagnostics; the new sparse runner does.

## 40-step known-bridge follow-up

Suite: `controlnet_runs/2026-10-05/track3-40-knownbridge/`, tangent-canny only. The source latent region is bridged before and after each Euler step to the same source-conditioned clean latents plus original noise at the current sigma. This is an explicitly added solver constraint, not the untouched published sampler.

The composite was independently viewed: the rail connects well, but the top water changes into an invented coast and the top/bottom white panels remain severe. The image fails. Denoising267.802s, decode4.898s, suite287.934s;40NFE,1,280 base-block calls,640 control-block calls,92,160 target-token forwards. Peak14,225,050,722bytes. Raw source top/bottom16 MAE increases to9.309/7.402 despite final exact central compositing. The masked-source VAE latents are reconstruction conditioning, and hard bridging them does not guarantee a seamless full-canvas VAE decode.

## Faithful 40-step follow-up without bridge

Suite: `controlnet_runs/2026-10-05/track3-40/`, tangent-canny only. Same original noise, prompt, conditioning and40-step schedule as the bridge run, with the bridge disabled. Raw and composite images were independently viewed and their saved latents and composite hashes checked. Source crop remains byte-exact after paste; latent array is finite.

The rail is again much better, but the incompatible upper coastline and white top/bottom remain. More denoising alone does not fix this composition. Denoising253.795s, decode4.869s, suite273.827s;40NFE,1,280 base-block calls,640 control-block calls,92,160 target-token forwards. MLX peak14,224,458,596bytes. Raw source top/bottom/all MAE5.284/3.751/3.617.

Native and bridge outputs are similar but **not identical**: composite pixel MAE1.0384/255, latent MAE0.11172. In both images, rows0–73 and868–1151 have more than95% of their width with every RGB channel>240. Native total such-white-pixel fraction31.1237%, bridge31.1312%. This threshold is a diagnostic for the visible white bands, not a quality score. The native result removes hard bridging as their necessary cause. It does not exclude conditioning, base quantization, task mismatch or another shared behavior.

## Provenance and hashes

- Base: `Qwen/Qwen-Image-2.1`, revision`790c92633540aa0cb11d9abf19eb46d861714758`, localQ4.
- [Control checkpoint](https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union/tree/8a4702014d4dabb5f896fcba917e2ee0a961465f), revision`8a4702014d4dabb5f896fcba917e2ee0a961465f`;7,550,979,904file bytes; parent verified full SHA256`65d6b66d734da9e7ff5ef04e7db3a133553a52a3f29a7fcb3e9cce8fa21dcfcd`.
- [VideoX reference](https://github.com/aigc-apps/VideoX-Fun/tree/4b7b6402a1e0f0406bd6801fb66c0a00bd922621),32 base blocks,16 trained control blocks with hints after base layers0,2,…30. Full129-channel control context, all hints recomputed each step, no stale prefix cache.
- Port SHA256`f4017d5989b5a97db5783cc6f100938ffa1f8a17de92587a1608db5418dd9d2a`; baseline runner SHA256`10c6dd5693437f5619bb2d45bcc1d1650829035cb70fd474aa56c762cfb50ec4`.
- Source PNG SHA256`ddc835523a27f455b999673ee2460b31a1b6b93b86ef0420e6f37ff8ea0f0157`; source rectangle`[0,320,512,832]`.
- Shared original-noise float32 SHA256`a23986b82c69e27372626fc4c4f415c96036a8fa2deb50981d1c560a64c2a23d`.
- Input Canny PNG SHA256`8d511f6f8d7197340a0ced5e43209377304eb57d1464322531469d67ed9a7788`; tangent Canny PNG`77ef3e3fc7967a54de11b7a8d60985eb5d09ed25f28f65c1bc407960e37e39d8`. Tangents are automatic source-derived guesses: four parallel lines,256px maximum Euclidean extension; no output-derived guides or lines pasted over generated pixels.

| Composite | SHA256 |
|---|---|
|20 mask-only|`462faf7a4ac900ce95cfa2004511f0ffe3ab9a4f84ff3cd2439e0df44bf82a17`|
|20 source-canny|`02b8c005fcb8930017aa013727be9aec62dc4dfbd029fc4260e5a340a6d4f55e`|
|20 tangent-canny|`d916df456924c2c301376ef44e7048344e93bc74f7a791f8fdafed63597f4399`|
|40 tangent-canny + bridge|`22ffce55dd2617be862115838e0711981b6050d6bc225d31db0e3914b075beb4`|
|40 tangent-canny native|`e2ad9a179665942508e919fc9ffe628be5a88b984b9e1e516c474207cd82a154`|
|40 tangent-supported + bridge|`1d19c88c406d4ed4b1c5cb0ea6afcb8ff5b106c49c4e7a0e9fdad32cc515ec3e`|

The CPU port gate compares tiny nonzero random weights with exact pinned upstream Torch blocks/forward. Maximum block-chain/hint error9.54e-7; complete prediction error1.049e-5 below2e-5. Scale0 reproduces the local base exactly. Real checkpoint shape/dtype/finite checks passed during these runs. Numerical portability and successful execution are distinct from visual acceptance.

## Official conditioning/training evidence and next bounded experiment

Missing structural control is a global literal-zero latent64 sentinel. Supplying black RGB instead gives normalized RGB−1 with alpha+1 and VAE-encodes a real control image; they are distinct inputs. [Inference preparation](https://github.com/aigc-apps/VideoX-Fun/blob/4b7b6402a1e0f0406bd6801fb66c0a00bd922621/videox_fun/pipeline/pipeline_qwenimage21_control.py#L636-L668)

The public training script drops **all** structural control latents in a sample with10% probability. Structural maps receive the target transform and remain unmasked; the generation mask modifies only the source RGB for its separate latent stream. Partial Canny maps with black exterior therefore describe edge-free exterior, not locally absent control. [Global dropout](https://github.com/aigc-apps/VideoX-Fun/blob/4b7b6402a1e0f0406bd6801fb66c0a00bd922621/scripts/qwenimage21_fun/train_control.py#L1454-L1488), [paired transforms](https://github.com/aigc-apps/VideoX-Fun/blob/4b7b6402a1e0f0406bd6801fb66c0a00bd922621/scripts/qwenimage21_fun/train_control.py#L1214-L1225)

Published single-image training masks select full regeneration70%, interior rectangle20%, ellipse5%, circle5%; there is no explicit known-center/exterior-outpaint branch. Full-grid flow MSE predicts the original whole target, not a seam-specific objective. The script's later loss variable named`mask` filters large residual outliers rather than the inpaint region. [Mask recipe](https://github.com/aigc-apps/VideoX-Fun/blob/4b7b6402a1e0f0406bd6801fb66c0a00bd922621/videox_fun/data/utils.py#L85-L107), [ellipse/circle](https://github.com/aigc-apps/VideoX-Fun/blob/4b7b6402a1e0f0406bd6801fb66c0a00bd922621/videox_fun/data/utils.py#L140-L155), [whole-target loss](https://github.com/aigc-apps/VideoX-Fun/blob/4b7b6402a1e0f0406bd6801fb66c0a00bd922621/scripts/qwenimage21_fun/train_control.py#L1558-L1604)

This supports a **task-distribution mismatch hypothesis**. It does not establish the released checkpoint's actual training history or prove the white-band cause; mask-only also fails without the partial black structural map.

The sparse experiment is `controlnet_port/run_sparse_control_trial.py`: keep encoded structural64 on the entire known ROI plus optional64px Euclidean dilation of accepted unknown tangent guide pixels, and replace structural64 elsewhere with literal zeros. Preserve mask1+masked-source64 exactly. The first arm is40-step tangent-supported with known bridge, preserving the completed40-step known-bridge noise, prompt, source, solver and full-canvas calls; only control64 support gating changes. Source-supported remains available as a later arm. Actual support1024 versus1121 of2304 tokens. The helper passes CPU all-keep equality, all-drop literal-zero, exact source65 preservation, correct polarity, floor-nearest packing, nonmonotonic absolute-ID gathering and independent Euclidean-circle tests. Pixel/packed supports and hashes are saved.

Spatially zeroing control input is itself **experimental**: official dropout is global, and zero input does not zero branch biases, merge state, attention or learned hints. This is neither a trained spatial gating guarantee nor a growth implementation. GPU scheduling remains with the parent; no GPU trial was launched by this report's author.

Independent sparse CPU validation is recorded in `controlnet_port/sparse_cpu_parity.json`: five radii0/1/3/16/64 match an integer-squared Euclidean oracle and Torch nearest packing;16 float32/BF16 gating cases match Torch exactly(error0), preserve source65 bytes including signed zero, preserve dtype, and gather duplicate/nonmonotonic absolute IDs correctly. The frozen sparse runner SHA256 is`a409b723933c27a3f1053c05ae03b83a0aabbb4390c4bb152bbe44a20f89c549`; helper SHA256`458a8eda69b31600365360822a25369bf479ba86246fdb110b07a546fbb12b0f`.

## Completed sparse40 result

Suite: `controlnet_runs/2026-10-05/track3-sparse40-knownbridge/`, tangent-supported. Raw, exact-source composite and saved alpha were independently viewed. White top/bottom panels disappear: no pixel has all RGB channels>240. Alpha remains nearly opaque (normalized min0.94677, mean0.99798, max1). The source-adjacent diagonal rail remains improved, but the upper exterior adds an unrelated horizontal rail/road and the lower exterior repeats sea/rail/road instead of continuing asphalt. **Full composition still fails.**

This matched trial associates the white-panel behavior with globally encoded black structural exterior; it does not establish sparse gating as a fix, and does not establish a general causal mechanism across covers/seeds. Freed exterior returns to the scene-repetition failure already visible in mask-only conditioning.

Denoising269.675s, decode4.234s, suite289.162s;40NFE,1,280 base-block calls,640 control-block calls,92,160 target-token forwards. Peak14,245,907,554bytes. Raw source top/bottom/all MAE9.739/8.197/3.560. Source pixels are exact after paste, latents finite, saved latent and composite hashes independently match. Recorded baseline full-map context hash`e0a1d20832efd6bfa54dbb61efe192ee608a6d7aa49d8b8e752bfbda4695419e`; final sparse context hash`569b4efc8be563d7315869a376927c57e3c8e775cc1566841caf694f5f6cf44f`. In-run tests confirm source65 unchanged, supported control64 unchanged and unsupported control64 literal zero. The first trial alone does not justify another seed as acceptance or deployment.

## Native masked-model fallback, researched October5

**FLUX.1 Fill is the most practical next local task-matched baseline**, because the installed MFLUX runtime already implements it. This is a feasibility ranking, not a claim of superior observed quality. The12B Fill model is trained for inpaint/outpaint and receives noisy target64+masked-source64+packed full-resolution mask256=384 channels every step. This directly distinguishes known source from generated exterior without an auxiliary partial Canny map. It still does not guarantee unchanged decoded source pixels or a seamless paste. [BFL model card](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev), [official conditioning](https://github.com/black-forest-labs/flux/blob/802fb4713906133fcbd0d8dc5351620ca4773036/src/flux/sampling.py#L98-L145), [MFLUX Fill](https://github.com/mflux-community/mflux/blob/main/src/mflux/models/flux/README.md#-fill)

Public maintainer conversion `mflux-community/flux-1-dev-fill-mflux-q4`, pinned revision`eebfbaa12c95107169452c7d22622e04771192a3`, is ungated. Eight safetensors total9,613,055,254bytes;19 repository files total9,619,362,389bytes. Transformer6,699,773,861bytes, T52,679,190,473, CLIP69,436,741, VAE164,654,179. The original BFL BF16 transformer is23,804,922,408bytes and gated. The Q4 conversion retains **FLUX.1 dev Non-Commercial** terms; this prototype is research-only. Public download was authorized and started separately, without token changes or a license click-through. [Conversion](https://huggingface.co/mflux-community/flux-1-dev-fill-mflux-q4/tree/eebfbaa12c95107169452c7d22622e04771192a3), [official files](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev/tree/358293da0354175698b67ec8299acf928313a78a)

The new isolated runner under `experiments/flux1_fill/` follows native MFLUX Fill with official quality-example50steps/guidance30, full2304-token canvas, raw decode and exact-source-only final paste. It deletes CLIP/T5 after prompt encoding and transformer before raw decode. Conditioning CPU tests cover every latent/pixel-mask offset, mask polarity, source rows20:52 and exact normalized-zero unknown RGB.

Download completed with CLI exit0. All19 files totaling9,619,362,389bytes independently match expected sizes; all eight LFS SHA256 and eleven ordinary-file Git object IDs match. `.build/models/flux1-fill-q4.manifest.json` has`status=verified` and actual per-file digests; verification took3.494s. Strict offline CPU preflight at `experiments/flux1_fill/preflight/track3-verified/preflight.json` reports files-ready, no weights loaded, six conditioning groups passed, CLIP tokens`[1,77]` and T5`[1,512]`. Runner SHA256`e63fac021c3e3849baeac7aeb5f41ae44c27a3cbda38a12520963c8955d8b6b3`; preflight SHA256`a45302d44f43030324b0cb4bda7113947a52403a7b3a14d24b267cafbc99e7dc`. These gates establish file integrity and runtime readiness, not generated quality.

The parent subsequently ran two native Fill50 arms. First: new yellow lettering above and a second car/person scene below, despite improved source-adjacent rail; core339.515s, total347.407s. The matched exterior-only prompt removes those repetitions, but creates a curved bridge-like rail across the water and a hard lower asphalt texture cut; core307.361s, total314.918s. Both are **quality failures**, with50NFE/115,200 target-token forwards, exact source, unchanged raw exterior and finite states. The task-matched architecture alone did not solve this cover. [Actual Fill report](../flux1_fill/FILL_RESULTS.md). No inference was launched by this report's author.

Second feasible baseline is native [SDXL inpainting0.1](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1), pinned revision`115134f363124c53c7d878647567d04daf26e41e`, ungated,6,938,041,649bytes of FP16 weights (18 complete pipeline files total6,941,218,469bytes). Its learned9-channel input includes target4+mask1+masked-source4. Existing DreamLite environment already has Diffusers0.39/Torch2.14. Model-card starting point20steps/CFG8/strength0.99 entails19 effective timesteps on that scheduler. The license is the card's OpenRAIL++ terms; exact-pixel source compositing remains separate. Parent-authorized download completed, all files passed exact checksum verification, and isolated CPU preflight passed. No SDXL GPU trial has been launched by this author. [SDXL readiness and comparison limits](../sdxl_inpaint/SDXL_RESULTS.md).

Third, [Fooocus](https://github.com/lllyasviel/Fooocus) combines an SDXL base with a learned5→320 head(52,602bytes) andv2.6 patch(1,323,362,033bytes). Its anchoring, synthetic-refiner schedule and blending must be ported together; simply loading the patch is not equivalent. It is a larger integration task than the already supported Fill baseline, and the official Mac support is not thoroughly tested. [Official patch/head](https://huggingface.co/lllyasviel/fooocus_inpaint/tree/74bbcc070e55219adb9b6c3b0d035b34e3697d1d)
