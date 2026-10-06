# Esinti — fresh cover renders, 6 October 2026

User requested fresh outpainting of all three local album covers using the latest selected model. No weights, runtime, quantization settings or app code changed in this stage. No subagents were started.

## Recipe and measured results

Used `experiments/klein_compression/chosen_recipe.json`: Klein base4B, uniform INT4, baked outpaint LoRA scale 1.1, standard VAE, fixed cached FP32 instruction, 12 steps, CFG 1, 512×1152. Component weights total 2,363,900,089 bytes. Required prequantized loading and transformer release before final decode remained enabled. Measurements are Mac M4 Max results, not iPhone benchmarks.

| Cover | Seed | Wall time (s) | Peak sampled physical footprint (GiB) | Raw parity with prior selected recipe | Local output folder |
| --- | --- | --- | --- | --- | --- |
| track1 | 42 | 89.53 | 5.262 | exact | `experiments/klein_compression/runs/2026-10-06/esinti-track1-retry1` |
| track2 | 42 | 130.30 | 5.264 | exact | `experiments/klein_compression/runs/2026-10-06/esinti-track2` |
| track3 | 17 | 96.26 | 5.264 | exact | `experiments/klein_compression/runs/2026-10-06/esinti-track3` |

All three composites preserve the original center rectangle [0,320,512,832] exactly. Raw outputs match the prior selected INT4 renders pixel-for-pixel, including generated areas. Similarity verifies repeatability, not semantic quality.

## Actual failure and retry

The first track1 launch (`esinti-track1`) stopped after 1.485 seconds at macOS memory pressure level 2, exit -15. Sampled peak physical footprint was 4.680 GiB. The guard stopped the process; no raw render was generated. Failure receipts were preserved. After pressure returned to level 1, a separate retry succeeded without raising resource limits or changing code. Other applications were not closed. All inference ran serially with a 6 GiB sampled footprint limit, pressure checks and swap-growth protection. Sampling is not a kernel-enforced memory cap.

## Visual inspection and limitations

Inspected all three fresh composite images directly. Track1 keeps the dark surround. Track2 still mistakes the printed tote graphic for a real bag and extends clothing incorrectly; this remains a failed human-cover case. Track3 extends the sea, rail and road, with visible texture and local continuation imperfections. No general quality acceptance, new student checkpoint, distillation, phone integration or phone performance claim is made.

Fresh `composite.png` images are delivered inline in the chat. Images, local artwork and model payloads stay ignored and local; only receipts, comparisons and this stage record are published. Upstream weight pins and licenses remain in existing manifests.

Next work remains reducing runtime memory and improving semantic quality before using teacher outputs for student training. The first-pressure-stop behavior warrants VAE/transformer lifetime investigation; that runtime change was not needed or made for this delivery.
