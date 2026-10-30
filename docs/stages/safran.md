# Safran local model comparison

30 October 2026. Ten existing original cover images were frozen for a new local
outpaint comparison. Fifteen model/recipe arms produce a planned 150-case queue.
The old three-cover inputs, approved teacher outputs, all human choices and
training manifests were preserved. No subagents, remote compute, paid API calls
or new trained checkpoints were used.

## Prepared comparison

The common geometry is a 512-square source at `[0,320,512,832]` in a 512×1152
target. The seed is 42; random tensors are not assumed equivalent across model
families. The source selection includes portrait/text, illustration, landscape,
framed photography, vivid green graphics and flat-color design. Names, source
hashes, derived inputs and commands remain in private results.

Arms include MI-GAN, DreamLite Mobile/Base, Mobile-O with latent restoration,
FLUX.1 Fill Q4, SDXL inpainting, ProMax, and Klein distilled/base/outpaint recipes
with Q4/Q3/Q2/mixed-MLP3 and cached conditioning variants. A LoRA is not counted
as a standalone generator. Native BF16 backing files are not extra mobile
recipes. The benchmark compares supported recipes rather than identical
conditioning capabilities; prompts, mask methods and step counts are recorded.

Every completed case must preserve a raw output and a separate exact-source
composite. The composite cannot serve as evidence that a model itself preserved
the source. No exterior color harmonization is applied. Model quality remains
unranked until real generated extensions and joins are visually inspected.

## Execution and limits

The installed Mac has 36 GiB. The serial queue uses the existing inference guard
at at most 6 GiB RSS/physical footprint, 512 MiB for MI-GAN, normal macOS pressure
only and the unchanged swap-growth stop. A capacity/runtime failure is recorded
once; remaining covers for that recipe are explicitly blocked, not counted as
quality failures. A system-pressure or swap stop pauses the entire queue.

At the recorded checkpoint, **0/150 generated outputs**: macOS reported pressure
level **2**, while the guard requires **1**. The detached queue is alive and
waiting for three normal samples. It is not running a model. It waits at most
two hours before a paused status. No other apps were closed or protections
relaxed. This is a prepared, resource-blocked comparison, not a completed one.

## Validation

Five CPU-only benchmark tests passed, including elevated-pressure refusal,
interrupted-job replay refusal, capacity-failure propagation, pressure-stop
behavior and separation of raw source error from composited preservation.
Existing FLUX Fill, SDXL and ProMax CPU conditioning/plumbing gates also passed.
Those gates use no pretrained generation and establish no visual quality result.

The nine-file publication batch was also attempted with the existing 256 MiB Git
guard. It refused before the first `git add` because pressure remained 2. No
commit or push was made; the per-file publication list remains private for a
later guarded run. The progress Page records the prepared/blocked status.

The four historical runners gained an optional input-manifest argument; their
default shared input path and track labels remain unchanged. Private per-cover
manifests safely bind the legacy track1 alias to each external benchmark ID.

## New model research

[Detailed primary-source research](../../experiments/local_benchmark/NEW_MODEL_RESEARCH.md)
identifies FLUX 3 Image Klein (15 October, 5.6B image transformer plus encoder)
and a published Apple MLX/outpaint-adapter path. Moebius is a smaller 0.2B masked
completion candidate with public weights, but its CUDA-oriented source and
conflicting Hub/GitHub license metadata need resolution. BlazeEdit's 195M
text-free outpainting architecture is relevant; public weights were not located.
Iris 3B, Fibo 2.0 Edit Turbo, LongCat Edit Turbo and Z-Image release availability
were checked separately. No new model payload was downloaded or run.

## Remaining work

Let the prepared queue run only after pressure normalizes, inspect its failures
and real outputs, and assess all completed ten-cover comparisons visually.
Preserve the user's final quality decisions. No winner, phone performance or
training-readiness claim follows from the current preparation and CPU checks.

## User authorized controlled warning level execution

The user subsequently explicitly authorized controlled execution with the
current RAM situation. The benchmark gained an opt-in warning-level allowance
(`1` or `2`) while critical pressure, the 6 GiB inference ceiling, MI-GAN's
512 MiB budget, 256 MiB swap-growth stop and the serial lock remain enforced.
The supervisor's default for every other caller stays normal-only. Eight CPU
safety tests cover the opt-in, continued critical/swap refusal and preserved
retry receipts.

The first actual MI-GAN attempt stopped at 535.73 MiB sampled RSS (475.28 MiB
physical footprint). Disabling ONNX allocation arenas/pattern caching did not
help: a separate preserved retry stopped at 595.94 MiB RSS (184.55 MiB footprint).
These overlapping/shared-memory measures are not summed. Neither produced a
benchmark output; remaining MI-GAN cases are resource-blocked, not quality
rejections.

DreamLite Mobile's first loading attempt stopped on swap growth after 5.30 s,
with 3.09 GiB sampled peak physical footprint. It did not complete generation.
The queue then resumed with low-memory Klein recipes first, under the same
process and swap limits. All failed attempts remain available separately; no
teacher dataset or user decision was changed.

The controlled MI-GAN budget was then set to 1 GiB, still below the 6 GiB
inference ceiling. All ten covers completed: mean cold worker time **0.577 s**,
largest sampled RSS/footprint maximum **0.686 GiB**. Raw and exact composites
are preserved. Contact-sheet inspection shows visibly discontinuous joins,
invented dark shapes/lettering and unrelated objects in several extensions;
runtime success does not make it a quality winner. Human votes remain untouched.

The first cached Q2 Klein completed all 12 denoising steps, then stopped on
swap growth during final VAE decode at 91.30 s / 5.32 GiB sampled physical
footprint, before a final image. To reduce that phase rather than remove its
guard, an opt-in Swift patch uses the upstream tiled decoder on an exact fixed
grid: latent 144×64, tile size 80, overlap 16, two vertical tiles. Independent
grid arithmetic yields exactly 1152×512 output; other dimensions are refused.
The tile join is inside the source rectangle, but tiled/global decoder
normalization can still change generated colors/details. All Klein arms in the
new resource-recipe revision record this option explicitly. The previous
runtime source and binary were backed up before patching; the default path
remains full-frame. A single-job 3 GiB guarded rebuild was launched with the
explicit warning-pressure allowance; final compilation and image results must
be checked before accepting the new recipe.

## Latest measured checkpoint

The guarded native rebuild completed in 128.79 s, with peak sampled RSS
1.73 GiB / physical footprint 1.20 GiB. The benchmark's explicitly recorded
resource recipe now permits 1 GiB swap growth per non-MI-GAN job, adds a 25%
system-free-memory floor, and retains the 6 GiB process limit and critical
pressure refusal. MI-GAN remains 1 GiB process / 256 MiB swap. All other callers
keep the supervisor's original normal-only / 256 MiB default.

The first cached INT4 Klein image completed with the tiled decoder: **81.05 s**,
**5.29 GiB** sampled peak physical footprint, **2.31 GiB** RSS, zero swap growth,
and minimum observed system free memory **32%**. Native output is 512×1152.
Its source is the easy dark-background control; this single success does not
rank the model on the other covers. Current complete count is **11** (ten
MI-GAN, one Klein). The following Klein cover stopped on the 1 GiB swap-growth
guard during denoising after 54.95 s, not a completed output. All its receipts
are preserved. System pressure subsequently returned to normal (1), with 50%
reported free memory. Resuming that resource-stopped case under the same guard
is appropriate after this observed state change; successful cases are skipped.
