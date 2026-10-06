# Mercan — restore finishing consistency and improve the human-cover candidate

6 October 2026. The user identified top/bottom square color seams and preferred the earlier Cinar road/sea image. No subagents were started; weights and compression remained unchanged.

## Diagnosis verified from saved pixels

The Esinti INT4 raw road/sea render is pixel-identical to the selected Cinar seed17 raw render, as already recorded in the matched compression comparison. Cinar's smoother publicized local preview used an exterior-only harmonic color collar. Esinti delivery hard-pasted the original square without that finishing step. This omission explains the actual delivery difference; it is not evidence of INT4 degradation. Model-decoded source pixels differ in tone from the original, and exact-source pasting exposes that difference at the boundary.

Reapplying the exact Cinar parameters (96-row exterior strip, horizontal sigma6, residual cap24) to the new Esinti raw render reproduces the older Cinar preview pixel-for-pixel. Original source pixels and exterior pixels beyond the strip remain exact. The accepted near-black Dua Lipa surround bypasses correction and stays pixel-identical. The generic bypass uses the source/decoded boundary rows' 99th percentile <=8; it is not a per-cover name check.

## CPU finishing comparison

Exact-source hard paste, the restored Cinar96 collar and a wider192 collar were actually evaluated on all three saved INT4 outputs. Color96 was retained: wide192 showed no compelling visual improvement and changes more generated pixels. MLP3 saved road and human outputs were also finished and directly inspected; color joins soften but the wrong bag/clothing content remains. MLP3 is still experimental and is not the selected model.

| Cover | Arm | Top adjacent-row RGB jump (0–255) | Bottom jump | Finishing ms |
| --- | --- | --- | --- | --- |
| track1 | exact | 4.62 | 4.81 | 54.51 |
| track1 | cinar96 | 4.62 | 4.81 | 44.13 |
| track1 | wide192 | 4.62 | 4.81 | 40.66 |
| track2 | exact | 8.86 | 24.45 | 39.60 |
| track2 | cinar96 | 2.30 | 20.40 | 48.57 |
| track2 | wide192 | 2.29 | 20.38 | 47.37 |
| track3 | exact | 8.38 | 4.72 | 35.55 |
| track3 | cinar96 | 7.25 | 3.61 | 50.29 |
| track3 | wide192 | 7.25 | 3.61 | 50.51 |
| track2 | seed17 + color96 | 2.32 | 12.07 | 64.81 |
| track2 MLP3 | color96 | 2.02 | 21.62 | 56.75 |
| track3 MLP3 | color96 | 7.76 | 3.51 | 46.32 |

Row jumps are descriptive photometric measurements. They mix natural edges, texture and geometry and do not certify semantic quality. Color correction cannot invent missing clothing, fix a rail angle, or correct a bag-print misunderstanding. Full-size and comparison images were visually inspected, not judged only from these measurements.

## New actual human-cover model trial

One different seed (17) was tried with the same Klein base4B uniformINT4 baked adapter, fixed embedding, standard VAE, 12 steps and CFG1. Compared to seed42, it keeps the printed bag on the T-shirt instead of extending it into a hanging real bag. The left shirt extension and cyan background also look more coherent. There are still source-boundary details, a long white shirt and an invented dark shape at the bottom edge. This is a better local candidate selected from two seeds, not a trained fix or a held-out quality guarantee.

The first seed17 attempt stopped after 1.437 seconds at system memory pressure, exit -15; peak sampled footprint 4.697 GiB. Receipts were preserved. Pressure returned to level1 and a separate retry completed in **84.33 seconds / 5.366 GiB**. Existing 6GiB/pressure/swap guards remained enabled. No unrelated apps were closed.


## Implementation and verification

`experiments/klein_outpaint/finish_output.py` wraps the existing harmonic transfer, validates geometry and protects original/distant pixels. `run_trial.py` now defaults to `--finish color`; `--finish exact` retains the former hard paste. `raw.png` remains unchanged, `exact_composite.png` preserves the uncorrected baseline, and `composite.png` is the finished delivery. Finishing metadata enters the run receipt. The recipe records selected illustrative seeds42/17/17 and this finishing policy; weights still total 2,363,900,089 bytes.

Five CPU tests passed: decoder tone drift, original/far-exterior preservation, horizontal texture retention, black bypass, exact-mode reproduction and geometry rejection (protection checks share one test). Real-cover checks also passed; the restored Cinar road preview is exact. The new seed17 native generation started before the runner edit, so its finishing was separately executed with the new helper. No extra full GPU run was made solely to retest the CLI integration; runner syntax was validated. `validation.json` contains code hashes and test output.

All old outputs are preserved. New PNG previews remain local/ignored; code, measurements and failed/successful receipts are published with one file per `mercan:` commit and one grouped guarded push. No source artwork, music or model payloads enter Git. Stage details are also reconciled into the ArtWorker Space Page.

Next: test more licensed human/background families and source-boundary aware conditioning/training before distilling. A better seed is not a general solution; color finishing does not replace semantic quality work. General teacher acceptance, a trained ArtWorker student and physical iPhone measurements are still absent.
