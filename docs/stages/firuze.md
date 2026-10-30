# Firuze bounded low memory model execution

30 October 2026. Follow-up to Safran's installed-model comparison, authorized by
the user to run in the current Mac RAM conditions. The ten source covers,
model weights, completed raw/composite outputs and all human review records
are preserved. No new model family, paid endpoint, remote host or subagent is
used. This is execution tuning, not training or distillation.

## Reason for the change

Safran completed ten MI-GAN images and one cached INT4 Klein image. The next
Klein cover stopped twice during denoising on swap-growth guards. Its later
attempt stopped after 74.70 s, with 5.23 GiB peak physical footprint and 1 GiB
swap-growth budget. System free memory was still 29%, but the swap guard
correctly stopped the process. Failed receipts and partial logs remain separate
from completed images; they are not quality rejections.

The limits are not increased for this follow-up: one heavy job, 6 GiB process
budget (1 GiB for MI-GAN), 1 GiB swap-growth stop (256 MiB for MI-GAN), 25%
reported system free-memory floor and critical-pressure refusal. The supervisor
default for other callers remains normal-only/256 MiB.

## Opt in native profile

`low-memory-execution.patch` applies after the existing pinned runtime patches
and Safran's portrait decoder patch. Only `ARTWORKER_LOW_MEMORY=1` activates it:
transformer evaluation/clearing after every block rather than the Mac's prior
four-block preset, and at most 64 MiB phase caches rather than 256/512 MiB.
The baseline remains unchanged without the flag. Different graph boundaries
may affect floating-point fusion, so numerical equivalence is not claimed.

The benchmark records `low_memory` and the executable fingerprint and rejects
a missing native activation log. A fresh retry uses a new output directory;
completed cases are skipped. Pre-change portrait80 source and binary backups
remain private. The fixed 512×1152 tiled decoder is retained.

Nine CPU-only tests passed, including a negative test that rejects a native
runtime lacking the low-memory confirmation. They do not establish image
quality. A fresh one-job native build is guarded at 3 GiB with the explicit
warning-pressure allowance and 25% headroom floor. Compilation and generated
image measurements are pending at this checkpoint.

The build subsequently completed in 43.44 s / 1.63 GiB peak RSS / 1.14 GiB
physical footprint. The first INT4 retry confirmed the new native profile but
stopped during loading at critical pressure level 4 after 1.71 s (4.25 GiB peak
physical footprint), before denoising or output. It does not establish a
successful memory improvement. The guard remained active and the process was
terminated. The heavier arm was held; a controlled cached INT2-only pilot was
then launched with a lower **4.5 GiB process ceiling**, the same swap/headroom
limits, and the same ten frozen source covers. Completed images remain 11 at
that checkpoint. No failed attempt is counted as an image or a quality score.

The first low-memory INT2 image then completed in **111.21 s**, with **3.92 GiB**
sampled physical footprint / **1.39 GiB** RSS, minimum reported free memory
**38%**, and zero swap growth. Output dimensions and native activation logs
passed. Visual inspection of its raw-source-restored composite shows strong
colored noise in the extensions, even on the easy dark control. This is a
completed but weak-looking result, not a human training rejection. No claim of
numerical equivalence or a controlled causal memory improvement is made between
the preceding INT4 and this INT2 recipe.

There are now 12 completed benchmark images: MI-GAN ten, INT4 Klein one, INT2
Klein one. The ten-cover INT2 subset continues under its 4.5 GiB ceiling.
A private one-shot continuation waits for that subset to finish, then requires
three normal-pressure samples with at least 40% reported free memory before
starting the remaining pending recipes. The repeatedly resource-stopped exact
cached INT4 recipe is held as resource-limited, not marked visually bad. The
continuation never retries a resource-guard stop or a completed job. Any critical
pressure/headroom/swap stop pauses work for inspection rather than looping.
