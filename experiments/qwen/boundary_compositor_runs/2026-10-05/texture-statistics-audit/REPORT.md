# Independent same-material texture diagnosis

This read-only CPU audit examined the actual saved-top32 model raw, its exact-source composite and the existing profile32 sigma16 finishing candidate. It did not edit a model, evaluator, helper, original asset or existing runner. New diagnostics are isolated in this folder; all three input file hashes remain unchanged. The source square is the fixed resized512 reference, not the original720 crop.

## Actual1× observations

The full512×1152 composite, existing4× crops and the new untouched1× seam/ROI crops were viewed directly. At the upper boundary the generated water has a conspicuous more rounded/corrugated appearance than the source's thin diagonal wave strokes. The profile changes color but does not establish the same wave structure. The bottom-right generated asphalt is more coarse/organic than the source; the bottom-left mismatch is much smaller. A single full-width texture gain would therefore affect an already similar left asphalt patch unnecessarily. The rail/vegetation is a different class of structured edge and must not participate in a texture RMS estimate.

## Measurements and their limits

Rectangles are diagnostic only and never enter a compositor algorithm. Water48 uses x8–104, source y320–368 and generated y272–320. A second water32 control uses x8–136, y320–352/y288–320. Asphalt uses source y768–832 and generated y832–896 at x8–136 and x368–496. These paired patches exclude rail, subject and logo. Each grayscale patch is independently plane detrended and Hann-windowed. Fourier powers below are luma-squared diagnostics, not a seamlessness score.

| ROI/band | Source power | Generated power |
|---|---:|---:|
| Water48,2–4px wavelength |14.48|9.61|
| Water48,4–8px |29.19|50.86|
| Water48,8–16px |23.66|25.87|
| Left asphalt,2–4px |7.25|6.35|
| Left asphalt,4–8px |6.69|6.44|
| Left asphalt,8–16px |2.37|2.34|
| Right asphalt,2–4px |6.54|5.84|
| Right asphalt,4–8px |6.64|9.27|
| Right asphalt,8–16px |2.63|4.97|

The water32 control agrees with the important direction: source/generated powers13.52/9.32 at2–4px and28.18/56.49 at4–8px. Coarser numbers differ substantially between the32/48px windows and should not be treated as stable estimates. ROI height poorly resolves wavelengths above16px. Radial power also conflates orientation: the water gradient-tensor anisotropy is0.597 source vs0.709 generated, while right asphalt is0.263 vs0.372. Scalar radial equalization cannot recover different orientation/co-occurrence statistics.

Five full-resolution difference-of-Gaussian band RMS gains (source/generated, sigmas0,.7,1.4,2.8,5.6,11.2) are:

| ROI |0–.7|.7–1.4|1.4–2.8|2.8–5.6|5.6–11.2|
|---|---:|---:|---:|---:|---:|
| Water48 |.974|.870|.860|.704|.815|
| Water32 |.875|.838|.827|.698|.791|
| Left asphalt |.948|1.015|.936|.893|1.126|
| Right asphalt |.888|.825|.711|.706|.868|

These wide overlapping bands are not interchangeable with disjoint Fourier annuli. In particular the finest water Laplacian RMS ratio near1 does not imply that2–4px power is already matched: the band overlaps several spectral regions. The largest Gaussian scales also have reflected ROI-border contamination. These gains are an oracle diagnostic, not recommended production parameters.

## Consequences for a texture branch

Positive gains can attenuate the existing overpowered generated middle scales while retaining their band patterns. They cannot invent a missing frequency or turn rounded water ripples into the source's diagonal stroke geometry. Even with all frequencies present, marginal power matching says nothing about joint phase/texture co-occurrence. A bounded branch may improve visual compatibility, but should be rejected if it simply smooths away the waves, adds sharpened halos, moves rail features or introduces a taper line.

“Preserves phase” needs qualification. A positive constant multiplier preserves an individual linear band's Fourier phase, but unequal overlapping-band gains can move reconstructed zero crossings through interference. Multiplying by a spatial taper/edge-protection mask creates sidebands and changes global Fourier phase. “Retains generated band structure without pixel warping” is the defensible description.

Protect the complete strong-edge profile, not just the rail centerline: include both metal edges, their width and a filter-support guard band. A coarse structural gradient is preferable to treating every grain gradient as a rail. Statistics near rail/vegetation/text/car must be excluded from both source and generated estimates with enough guard distance for the chosen filter; if a valid patch is too small, skip the affected coarse octave. A fixed global gain from the entire boundary strip would be dominated by semantically unrelated edges.

## Meaningful fixtures

An independent math reviewer proposed these properties, distinct from reproducing implementation formulas:

1. Random RGB and irregular/narrow allowed rings: source and far exterior byte exact, immutable inputs, finite output.
2. Source sinusoids at f1+f2, generated only f1: constant scaling cannot recover f2; any mask-induced sideband is modulation, not recovered texture. Zero-energy bands must not amplify epsilon noise.
3. Isolated sinus under positive constant gain: unchanged complex phase and normalized band correlation1. Do not impose global phase equality on a masked result.
4. Identical texture plus unrelated rail/ramp outside the usable ROI: estimated gains unchanged when filter-support exclusions are valid.
5. Rail crossing a permitted ring: retain normal edge position, width, overshoot and ghost-lobe count. A centerline-only metric is inadequate.
6. Vertical mirror of image/source/masks: mirrored output and identical gains; signed vertical derivatives invert.
7. All gains1: exact identity before source replacement, including odd dimensions and pyramid alignment.

These are recommendations; this audit did not run the forthcoming compositor's fixtures or accept its output. [Numerical statistics](statistics.json), [actual1× ROI comparison](roi-comparison-1x.png), [actual1× upper seam](top-boundary-1x.png), [actual1× lower seam](bottom-boundary-1x.png).
