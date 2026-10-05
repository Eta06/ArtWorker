# Saved top32 collar: local rail improvement, remaining texture failure

The original large guardrail step is substantially improved. The generated outer metal edge now joins much closer to the source, and the extra broad upper band seen in localized40 is largely removed. This is a real local geometry improvement in the model output. A small near-border profile/blur irregularity remains, and the water and asphalt seams still prevent seamless whole-image acceptance.

## Actual edge identity and profile

![Untouched paired raw and composite rail crops](rail-raw-composite.png)

These are actual pixels at x260–420, y286–354, enlarged five times with nearest neighbor. Raw is above; source-exact composite below. External marks identify y320 without painting the rail. Localized40 has a distinct projecting outer metal band above the source. Top32 brings that band's outer edge toward the original rail and makes the local profile much closer; it does not depend on selecting an interior parallel line as the source's outer edge.

![Fixed source-derived selected features](source-tangent-selected-features.png)

Yellow follows the generated outer ridge; green follows the original outer ridge. Purple remains a separate interior ridge, with lower confidence near the border. The unchanged evaluator reproduces:

| Composite | Near tangent difference | Extrapolated horizontal endpoint difference | Last-four-row median horizontal offset |
|---|---:|---:|---:|
| localized40 | +0.300° | −17.404 px | −17.591 px |
| saved top32 | +0.341° | −2.086 px | −0.975 px |

Top32 has 15 of 16 strong near rows, no near corridor-edge hit and no comparably scored alternative path. This establishes a traceable outer feature, not an exact or accepted join. Row y318 lacks a confident edge, and the weak selected y319 point is x316.274, approximately +3.930 px from the source tangent at that row. Thus the favorable local fit still hides a small final-row jog/blur. The untouched crop is decisive: the large misplaced band is improved, while the metal profile is not perfectly uniform across the border. No new universal rail-width score or output-specific corridor was introduced.

The detailed confidence/path records remain in [metrics.json](metrics.json) and [REPORT.md](REPORT.md). Both retain `quality_pass: false`.

## Remaining scene transitions

![Actual upper source border](upper-border-raw-composite.png)

The water retains a horizontal blue/cyan band and abrupt ripple-detail change at the source top. Vegetation is a plausible continuous dark mass but does not conceal this boundary. Its apparent improvement is not comparable to the rail correction. The full-width adjacent-row luminance change actually rises from 15.084 to 19.295/255; this describes photometry and is not a universal quality score.

![Actual bottom source border](bottom-border-raw-composite.png)

The lower asphalt still changes into a darker, coarse mottled texture. The raw image has a rough horizontal reconstruction seam; pasting back the original square retains the exterior tonal/texture break. The corresponding adjacent-row luminance change is 5.553 versus 5.652/255. Direct full and crop review finds no material texture-seam improvement over localized40.

![Actual full composites](full-composite-comparison.png)

The scene remains coherent, with no extra person/car, obvious duplicated full scene or white-end collapse. Both actual full raw images were also inspected and are retained in [full-raw-comparison.png](full-raw-comparison.png). A separate CPU-only reviewer independently confirmed the remaining water/asphalt transitions from the four original full PNGs.

## Scope and provenance

The already completed [independent saved-input audit](../../../controlnet_runs/2026-10-05/track3-saved-top32-outpaint/independent_saved_top32_integrity.json) passed. It verifies bit-exact restored source latents, prompt embeddings/image slots, target/padded contexts, noise and sigmas against localized40; the only effective hint changes are target IDs640–703, the 64 known-source cells covering the top32 px strip at y320–352. Those hints have weight1. The final source pixels and known latent bridge remain exact. All exterior pixels come from the new raw model output. The old baseline was not regenerated in this run, and repeated-run GPU determinism was not measured.

This paired one-track, one-seed experiment supports the source collar as a targeted correction for the original rail failure here. It does not establish generalization to other covers or seamless whole-image quality. The run still uses 40 transformer calls and 92,160 target-token forwards on the full canvas; core denoising is 410.329 s and supervisor elapsed is 430.195 s. It is not a growing-compute or iPhone-performance result.

This review performed direct image/feature inspection and created untouched comparison crops. It did not repeat the broad audit, run GPU inference, change the evaluator, modify saved model outputs or edit the Player.
