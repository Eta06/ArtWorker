Hints on the top 32 source rows give **scale 1 a modest local rail improvement**. Its join has a gentler kink than the saved same-scale growing output, and the selected feature follows the metal rail rather than the vertical post. The rail still looks pinched/spliced with doubled edges, and water/asphalt texture seams remain. Scale .5 retains a pronounced curved kink. Neither arm is accepted as finished outpainting.

The real serial suite completed successfully: launch **146.52 s**, sampling suite **145.66 s**. Each arm recorded **6 NFE, 192 base blocks, 96 control blocks, 12,288 target-token forwards and 6,144 reference-image-token forwards**. Active heights were 640, 896, then 1152 four times; future target counts were 1024, 512, then zero. Both branches used the same active absolute IDs, with a full fixed prefix recomputed each step.

[Independent validation](/Users/emir/Documents/ChatGPT/ArtWorker/experiments/qwen/boundary_eval/2026-10-05/growing-top32-comparison/independent_validation.json) passed **719/719 checks**. Saved source prefix, prompt/slots, noise, all 129 control channels, support, edge-known/edge-guess and seven sigmas match the completed growing baseline exactly. Only requested/effective hint IDs **640:704** changed to scalar 1; unknown cosine weights within 64 pixels remain exact, prefix weights remain zero, and the other 960 known-source weights remain zero. All 91 retained historical hashes match: 53 baseline artifacts plus 38 provenance files.

Final known latents equal the saved edge-context codec exactly. Final and preview composites contain the exact shared source pixels, and their exterior pixels equal the corresponding raw decodes. All 12 per-step control/hint gathers match independently reconstructed absolute IDs. Successful runtime guards record an exact fixed source-prefix input for each forward. Hidden-state and per-step velocity traces were not retained, so they were not independently replayed. Weight payloads were not rehashed in this audit, and no GPU rerun was performed.

The [frozen source-tangent evaluator](/Users/emir/Documents/ChatGPT/ArtWorker/experiments/qwen/boundary_eval/2026-10-05/growing-top32-comparison/source-tangent/metrics.json) used unchanged source-derived coordinates and thresholds:

| Arm | Ridge extraction | Endpoint dx | Tangent difference |
|---|---|---:|---:|
| Saved zero-source-hint .5 | Invalid curved/discontinuous fit | — | — |
| Top 32, .5 | Feature extracted | +2.782 px | +37.585° |
| Saved zero-source-hint 1 | Invalid curved/discontinuous fit | — | — |
| Top 32, 1 | Feature extracted | −0.998 px | +7.720° |

Scale .5 crosses the extraction residual cutoff: old p90 **2.548 px**, new **2.278 px**, cutoff **2.5 px**. Its angle remains wrong, so extraction status is not a quality win. Scale 1’s local ridge is cleaner (p90 **0.880 px**) and agrees with the visual improvement. The fit covers only generated rows 304:320; it does not establish whole-rail continuity, metal width or global quality. Old invalid fits provide no validated numerical angle ranking.

Same-scale final latent differences were reproduced against the saved uncached growing outputs:

| Control scale | Maximum absolute difference | Mean absolute difference |
|---|---:|---:|
| .5 | 0.671875 | 0.01637012884 |
| 1 | 1.043212890625 | 0.02491877228 |

Common input restoration is exact; output differences are expected from the changed hint field. Repeated-baseline GPU determinism remains unmeasured. The older scale 0 comparison with the original cached geometry pipeline was already inexact (max 0.0625, mean 0.00200196309); this audit does not claim equivalence to that cached implementation.

Both final raw/composites and all six actual steps 2/4/6 previews were viewed. Steps 2/4 are blurry pre-step predicted-clean estimates, with visible texture bands. Step 2 contains only the actual 896-row window y128:1024; grey bands in the review sheet identify future regions outside that support. Final frames preserve one coherent car/shore/road scene without duplicated car panels or new exterior text, but the joins remain visible. This is an untrained Turbo/control combination on one cover/seed, with no mobile-performance or general quality claim.

Actual review artifacts: [rail join at 6×](/Users/emir/Documents/ChatGPT/ArtWorker/experiments/qwen/boundary_eval/2026-10-05/growing-top32-comparison/rail-join-6x.png), [asphalt join at 2×](/Users/emir/Documents/ChatGPT/ArtWorker/experiments/qwen/boundary_eval/2026-10-05/growing-top32-comparison/asphalt-join-2x.png), [steps 2/4/6](/Users/emir/Documents/ChatGPT/ArtWorker/experiments/qwen/boundary_eval/2026-10-05/growing-top32-comparison/step-progression.png), [artifact provenance](/Users/emir/Documents/ChatGPT/ArtWorker/experiments/qwen/boundary_eval/2026-10-05/growing-top32-comparison/review_artifact_provenance.json).
