# Genuine six-step growing localized ControlNet review

All three completed scale arms preserve a coherent overall scene, but none fixes the source-adjacent guardrail. Steps 2 and 4 remain visibly blurred and broken at the rail; the final images retain a kink, changing metal profile and extra edge at the source join. Increasing control from 0 to 0.5 or 1 changes the rail but does not produce a seamless result. Visual acceptance fails for every arm.

This review independently inspected all nine saved predicted-clean previews and all three final images in both raw and source-exact composite forms, plus close crops of the rail and both source borders. No GPU inference, sampler change or metric retuning was performed.

The independent CPU integrity validator passed 551 checks with zero failures. Integrity passes separately from the failed visual acceptance described here.

## Actual frames

Each column below is old edge-context, new scale0, new scale0.5 and new scale1. These are actual decoded active windows. Step 2 has height 896 and absolute window y128–1024; no future top/bottom image was painted into that preview. Steps 4 and 6 and the final image have height 1152.

![Step 2 actual composites](step02-full-composite.png)

![Step 4 actual composites](step04-full-composite.png)

![Step 6 actual composites](step06-full-composite.png)

![Completed final composites](final-full-composite.png)

The full scene stays recognizable, with water and vegetation above and asphalt below. There are no new white end margins or repeated full road/lake panels. The source itself is restored exactly in every composite. At step 2 the generated water and lower asphalt are very smooth, with obvious source-border texture transitions. The upper rail is a blurred changing profile; scale1 adds the most visibly broken fragments. Step 4 still has the same join problem. Step 6 resolves much of the overall texture, but not the local source-adjacent geometry.

![All actual rail crops, raw then composite for each saved stage](rail-all-frames-raw-composite.png)

![Both sides of the upper source border at every saved stage](upper-border-all-frames-raw-composite.png)

![Both sides of the lower source border at every saved stage](lower-border-all-frames-raw-composite.png)

All crop coordinates are final absolute coordinates, transformed into the active preview's local window. The rail crop covers x260–420, y286–350 and is enlarged four times using nearest neighbor. External yellow marks identify the original source border; they are outside the crop pixels. Raw frames show decoded inner-source changes around the rail/post. Restoring the original source removes those inner changes, while retaining the exterior rail discontinuity. The bottom asphalt transition is less conspicuous in the final than in early frames but does not establish correct geometry at the top.

The corresponding full raw frames were also directly viewed and are retained in [step 2 raw](step02-full-raw.png), [step 4 raw](step04-full-raw.png), [step 6 raw](step06-full-raw.png) and [final raw](final-full-raw.png).

## Fixed source-boundary diagnostic

The unchanged source-only local tangent diagnostic reports `invalid-feature` for all three final scale arms and the old edge-context baseline. Their detected near paths are discontinuous, curved or locally ambiguous beyond the frozen fit confidence threshold. Tangent/endpoint values remain null, not zero. Direct crop inspection separately identifies the failed join; the null itself is not an image-quality verdict.

![Source-derived selected and alternative paths](source-tangent-selected-features.png)

[REPORT.md](REPORT.md) and [metrics.json](metrics.json) retain all final candidates, paths, source fit and invalid reasons. The source-only corridor was not shifted to follow a preferred output.

Applying the same diagnostic to actual active preview pixels yields invalid status in every new preview except scale1 step4. That provisional detected path has a +7.420° near tangent difference, +13.612 px extrapolated horizontal endpoint difference and +12.012 px final-four-row offset. Its image is visibly malformed; extraction is not a quality pass. The [step4 scale1 feature crop](step04-scale1-feature-diagnostic/source-tangent-selected-features.png) identifies what was selected. [preview_geometry_diagnostics.json](preview_geometry_diagnostics.json) records all stages. No absent future pixels participate in this local diagnostic.

## Actual growth and computation

| Step | Active pixels | Absolute target token rows | Target tokens | Future target tokens absent | Joint query tokens per chain |
|---|---|---|---:|---:|---:|
| 1 | 512 × 640, y256–896 | 16–55 | 1280 | 1024 | 2477 |
| 2 | 512 × 896, y128–1024 | 8–63 | 1792 | 512 | 2989 |
| 3–6 | 512 × 1152 | 0–71 | 2304 each | 0 | 3501 each |

The fixed prefix has 1024 reference-image and 173 text tokens. Every new arm recomputes that full prefix on every step in both the base and control chains. Per arm, the recorded actual totals are six transformer calls, 192 base block calls, 96 control block calls, 12,288 target token forwards, 6144 reference-image token forwards and 19,470 joint queries per chain. A full 2304-target raster on all six calls would forward 13,824 target tokens; this activation geometry saves 1536 target forwards, or 11.11%, while retaining the full prefix work. Actual active nonzero hint positions are 60 on step1 and 97 on each later step, gathered from the same saved input-derived final-raster field. This is real target growth from the source outward, followed by full-canvas refinement once the third step activates the complete raster.

Scale0 still computes the full control branch before multiplying completed hints by zero. It is a geometry/input control arm and not a bare-base timing benchmark. Real per-arm core times were 54.780 s, 57.290 s and 58.399 s for scales 0, 0.5 and 1. The complete three-arm wrapper took 211.185 s; suite-internal elapsed was 210.342 s. Preview decode and loading costs are separate from core times.

The trained Union branch and Viggle r256 six-step adapter were not jointly trained. This experimental combination uses the old global six-sigma schedule and predicted-clean frontier insertion. No quality acceptance, distillation result or iPhone performance follows from completing these mechanics.

## Integrity and old scale0 comparison

The separate CPU audit is saved in [independent_validation.json](independent_validation.json), with 551 passed checks and no failures. It checks actual source/prompt/noise/context/gate arrays and hashes, reconstructed absolute windows and conditioning gathers, preview IDs and sigma nodes, final known-source target latents, exact composite source pixels, recorded call totals and current code/test provenance. Saved per-step context and hint hashes reproduce exactly from the actual full conditioning arrays and absolute IDs. Desired cosine weights and their BF16 effective rounding also reproduce exactly. Actual per-layer model states/hints and RoPE arrays were not persisted for each inference call; those mechanics are supported by checked runtime shapes/assertions and matching pinned CPU tests, not independent replay of real GPU forwards. Two synthetic unfinished-suite refusal checks passed without writing a result report.

The old and new arms share the recorded actual-square source-prefix hash, initial-noise hash, edge-context known-target codec hash, prompt, seed, adapter/model revisions, sigma schedule and activation geometry. The old baseline did not save its original full conditioning/noise/prompt tensors, so equality for those is recorded-hash provenance rather than direct comparison of two saved tensors. Its known block and preview state are independently inspectable.

Scale0 is close to the old image by direct inspection, but it is not byte-identical. Same-stage step6 predicted-clean latents differ by a maximum 0.0625 and mean 0.002002. Actual final raw pixels differ by a maximum 11/255 and mean 0.149037/255; source-exact composites differ by mean 0.145549/255. These small pixel averages describe numerical proximity, not quality. The new integrated final and its step6 predicted-clean state are independently verified identical for all three arms, including latents and decoded raw pixels. The old final PNGs also match their old step6 preview PNGs, although a separate old integrated final latent array was not persisted. These comparisons are retained in [scale0_baseline_pixel_and_latent_comparison.json](scale0_baseline_pixel_and_latent_comparison.json). Cached/recomputed prefix behavior and numerical execution order can contribute; the comparison does not attribute the difference solely to caching.

The old implementation caches its fixed prefix after the first call and records 13,485 total base queries; the new full-prefix recomputation records 19,470 per chain. Their total prefix work is therefore not the same. All three new scales use the same growth geometry, conditioning and input noise, allowing their completed-hint strength effect to be compared without treating old/new implementation differences as exact model parity.
