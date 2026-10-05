# Direct source/exterior compositor on the top32-conditioned ControlNet output

The pretrained ControlNet output with the top32 known-collar inference configuration has stronger rail alignment, but a horizontal water/vegetation transition and lower asphalt transition remain. The previous decoder-source residual compositor does not address the full exterior/source color mismatch. This new **CPU post-generation** experiment matches the actual generated exterior directly against the original source edge. It improves those photometric joins, while retaining the original source and distant generated scene exactly. It is **not** a complete seam/texture fix and none of the derived images is labeled model raw. No new top32 checkpoint was trained.

The strongest conservative photometric candidate in this CPU comparison is a **32-pixel per-row color-profile correction with x smoothing sigma16**. Sigma8 is a middle candidate; sigma4 reduces the adjacent-row metric farther but changes exterior gradient contrast more. The subsequent frozen source-tangent evaluator rejects **both sigma8 and sigma16 as invalid-feature**, because their source-adjacent metal edge loses sufficient detector confidence. Thus neither is accepted as a geometry-preserving final result. Actual water-wave/grain frequency and faint vegetation/rail/asphalt transitions remain visible.

## Controlled input and preservation

Runner: `run_direct_boundary_compositor.py`.

Actual unchanged model input:
`controlnet_runs/2026-10-05/track3-saved-top32-outpaint/tangent-canny/top32/raw.png`.

The shared source remains the original resized 512 square centered at `(0,320,512,832)` in a 512×1152 canvas. Every output pastes this square exactly. Only generated RGB values within the declared 32/64/96-pixel top/bottom strip change; farther exterior pixels remain exact. There is no image warp, source feathering, painted line, hand-selected object mask or manually supplied rail geometry. Diagnostic crop coordinates never enter the correction algorithm.

The earlier `run_boundary_compositor.py` is imported read-only for the frozen decoder-residual harmonic control and exact-source composition. Its hash, the source PNG/JPEG hash and actual model raw hash remain unchanged. No frozen evaluator, runtime, helper, model file or Player file was edited. All saved intermediates are declared **derived post-generation composites**, not new model inference.

## Methods and regularization

The first matrix uses sigma4, value-residual cap ±48 RGB levels and regularized outward gradient cap ±0.35 RGB levels per pixel. For top and bottom separately, the first generated row is distance1 and the original source edge is distance0. Source outward gradient is the negative of its eight-row inward least-squares derivative, using the correct opposite physical-y directions at the two boundaries. The desired first exterior color is source-edge color plus one outward step. This avoids the incorrect edge-copy constraint on a nonzero affine gradient.

- **Direct harmonic,32/64/96:** source-to-first-generated residual with finite-strip harmonic extension, zero at the outer strip boundary. This is C0 only.
- **Direct Hermite,32/64/96:** clamped cubic color residual, with the discrete first-to-second generated-row residual gradient matched before regularization/caps; residual value and continuous derivative vanish at the far end.
- **Spectral biharmonic,32/64/96:** clamped finite-strip value/continuous-gradient constraints, high spatial modes decay through the strip. This reduces the vertical tone marks compared with holding every x pattern through a cubic strip, but does not solve different image textures.
- **Frozen harmonic96:** the previous known-decoder-source residual method, sigma6 and cap24, on the same unchanged new model input.

Direct first-row correction improves the metric, but a large original exterior first-to-second-row color jump remains. Eight-row derivative estimation plus caps deliberately does not force that photographic jump into an unlimited long extrapolation. Long Hermite variants can introduce vertical color marks or a dark vegetation-side halo. These variants are diagnostic and rejected as the finishing default.

The additional **profile** method addresses each generated row's low-frequency color directly. It continues a capped source-normal color gradient for the first two exterior rows, then smoothly saturates that gradient's color prior at ten pixels of equivalent distance. It blends each row's actual generated color toward that prior using a cubic weight with full weight for the first two rows and zero value/derivative at the far end. Only an additive color residual is applied; textures/coordinates are not resampled or copied. Smoothing sigma4/8/16 and32/64/96 strip controls are saved. Smoothing/caps make these **regularized color targets**, not exact pixelwise source/outside C0/C1 continuity. The profile also intentionally changes a long affine color continuation after two pixels; the nonzero-affine fixed-point claim is scoped to the other residual methods.

## Actual results

All methods are applied to the same model raw and source. Adjacent-row RGB jump uses the saved pixels on either side of the source boundary, scale0–255. It is photometric only and does not establish geometry/texture quality.

| Output | Upper adjacent-row RGB jump | Lower adjacent-row RGB jump | Exterior collar horizontal-gradient energy ratio |
|---|---:|---:|---:|
| Exact-source baseline | 20.528 | 5.818 | 1.000 |
| Frozen decoder harmonic96 | 14.663 | 5.259 | 1.003 |
| Direct C0 harmonic32, sigma4 | 7.665 | 4.220 | 1.002 |
| Direct profile32, sigma4 | 7.641 | 4.105 | 0.947 |
| Direct profile32, sigma8 | 9.688 | 4.775 | 0.990 |
| Direct profile32, sigma16 | 10.460 | 5.066 | 1.000 |

Do not rank the methods by the adjacent-row metric alone. Direct C0 methods have a smaller first-row jump while retaining a next-row band and producing less acceptable vegetation/rail-side tonality. Profile32 sigma16 reduces the low-frequency first-exterior target mismatch from17.132 to1.125, and the first-to-second exterior gradient mismatch from17.246 to2.277, **both measured at the same sigma16**. Its lower equivalents improve2.841→0.591 and1.389→0.479. Sigma4/8 metrics use their own smoothing scale and are not directly comparable to sigma16 target-error values.

Profile32 sigma16 has no pre-quantization RGB channels below0 or above255. Actual source-normal gradient caps affect56.45% of top channels and15.43% of bottom channels; actual per-row value caps affect1.335% of top channels and0% of bottom. These are the profile's true cap counts, distinct from the unused earlier gradient-residual diagnostic. Sigma4/8 and profile64/96 also have zero pre-quantization clipping in this trial. This supports bounded color processing; it does not establish a perceptually seamless image.

Full512×1152 images, untouched1× crops and nearest-neighbor4× rail/top-water/bottom-asphalt crops were viewed directly:

- Profile32 reduces the sharp upper photometric line more than the frozen control, without an observed new rail displacement/direction change.
- Water wave/grain frequency still changes at the original source boundary. Additive color cannot match those different scene textures.
- A faint vegetation/rail-source stripe and lower asphalt texture change remain.
- Profile64/96 spread a lighter water patch farther outward. The32-pixel variant changes a smaller scene region and is the preferred conservative candidate.
- Sigma4 changes some local gradient contrast more; sigma8 is a middle choice, while sigma16 preserves the saved exterior collar contrast most closely.
- A contrast-based line detector can change feature eligibility after color adjustment. No improvement in such a score alone should be called restored rail topology. These crops do not replace the root's frozen near-ridge feature-identity review.

The root subsequently ran that frozen review: sigma8 lacks80% near-row strong-metal coverage and both sigma8/sigma16 lack three strong edge rows immediately before the source. Tangent and displacement values are invalid, not improved. The code applies no coordinate warp, but color/contrast appearance can still degrade the rail feature. The selected-feature and untouched crop images were viewed directly. [Frozen source-tangent review](boundary_eval/2026-10-05/direct-profile-source-tangent/REPORT.md). A later color/texture finishing stage must protect source-connected strong edges and pass this same frozen gate; these two photometric candidates do not do so.

![Same model input, frozen control and profile16 variants](boundary_compositor_runs/2026-10-05/track3-top32-profile16/full-comparison.png)

![Profile32 sigma16 fullwidth top boundary, actual pixels4x](boundary_compositor_runs/2026-10-05/track3-top32-profile16/direct-profile32/water-top-4x.png)

![Profile32 sigma16 rail, actual pixels4x](boundary_compositor_runs/2026-10-05/track3-top32-profile16/direct-profile32/rail-top-4x.png)

![Profile32 sigma16 bottom asphalt, actual pixels4x](boundary_compositor_runs/2026-10-05/track3-top32-profile16/direct-profile32/asphalt-bottom-4x.png)

## Fixtures, validation and limits

Meaningful CPU fixtures passed: known affine source/exterior sign and first-row value above/below; Hermite's first-two-row affine oracle; zero far correction and a clamped taper; decoded-known-source independence; constant-scene fixed points; seamless nonzero-affine residual fixed point; cosine mean and x-mirror symmetry; profile-specific opposite-sign/magnitude first-two-row affine oracle, constant scene, mirror/mean and exact source/far exterior. Tests use independent affine/cosine scene properties rather than reproducing the implementation's output equations. Profile is explicitly excluded from the global nonzero-affine fixed-point property because of its saturated color prior.

An independent read-only reviewer checked the spectral equations, sign/indexing and actual saved profile images. It independently verified source/far-exterior equality, reported no visible new rail displacement, and favored profile32 over64. It rejected perfect acceptance because the wave/grain and faint vegetation/asphalt seam remain. Its cap-statistics concern was corrected by separately recording profile-specific actual counts, with saved pixels unchanged; profile-specific fixtures were added and pass.

The four suites save22 composite records,16 distinct hashes. Every exact-source baseline and frozen harmonic control reproduces across suites. All saved output hashes, source pixels, declared far exterior and original model/source/frozen-helper hashes validate. No GPU calls, model-weight load or new inference is involved.

On this Mac single CPU thread, profile32 sigma16 computation took3.7ms excluding image load/save; profile64/96 took5.4/6.9ms. The11-output first matrix took0.86s including PNGs/crops/montages; the5-output profile16 matrix0.44s. Sigma4/8 one-radius suites each took about0.34s while run concurrently, so their wall timings are not a standalone speed comparison. These are Mac diagnostics, not Swift/CoreImage or iPhone performance evidence.

The runner refuses to overwrite completed suites. No model raw or original source image was replaced, and no Player integration or production acceptance is claimed.

- [All22 saved-output validation](boundary_compositor_runs/2026-10-05/track3-top32-direct-validation.json).
- [First C0/C1 method matrix](boundary_compositor_runs/2026-10-05/track3-top32-direct/direct_compositor_metrics.json).
- [Profile sigma16 complete metrics](boundary_compositor_runs/2026-10-05/track3-top32-profile16/direct_compositor_metrics.json).
- [Profile sigma8 complete metrics](boundary_compositor_runs/2026-10-05/track3-top32-profile8/direct_compositor_metrics.json).
- [Profile sigma4 complete metrics](boundary_compositor_runs/2026-10-05/track3-top32-profile4/direct_compositor_metrics.json).
- [Candidate profile32 sigma16 composite](boundary_compositor_runs/2026-10-05/track3-top32-profile16/direct-profile32/composite.png).
- [Middle candidate profile32 sigma8 composite](boundary_compositor_runs/2026-10-05/track3-top32-profile8/direct-profile32/composite.png).
