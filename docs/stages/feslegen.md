# Feslegen — parallel teacher calls with serial, isolated source audits

2026-10-11. The user requested concurrent API generation after observing the
serial queue's low throughput. The main/fallback teachers, frozen old-canvas
prompt, reference images and quality settings remain unchanged.

## Safe handoff and concurrency

A pause marker allowed the active old request to finish. The next benchmark
exited with code 75 before creating a receipt, loading credentials or sending
any API request. Its exact deferred item was restored to pending only after
verifying that exit and absence of a receipt. No in-flight paid request was
terminated or blindly repeated. The old worker stopped with 69 Sunburst images,
one Nano image and two pending Nano fallbacks.

`--workers 3` uses a three-thread dispatcher to overlap network requests. One
coordinator checks account credit and key limit and reserves $0.10 for **each
slot of the whole wave before launching it**. All calls drain before the next
wave, even when one fails. Remaining funds automatically reduce available slots;
unknown outcomes or unexpected prices stop subsequent waves. The input/output
directories and receipts are distinct per cover, and state updates remain serial.

Source registration runs one at a time in short-lived processes, releasing
OpenCV/SIFT allocations after every image. The earlier serial process retained
these allocations between calls and reached a sampled RSS peak of 471.92 MiB.
Separating registration from network work keeps the existing 512 MiB process-group
guard, memory-pressure and swap-growth stops. No local generative model or new
agent was launched. Three workers are explicit; the CLI default remains one.

The status command now reads an atomic state snapshot without requesting the
mutation lock, so it can report progress while a worker is running. A pause
marker is checked between parallel waves and by each child before API access.
Code 75/no-receipt is the only automatically deferred case; other absent receipts
remain uncertain until reconciled.

## Real parallel pilot

Three new Sunburst requests started concurrently. Provider-response durations
were 27.845, 34.831 and 39.713 seconds; all three completed successfully. Each
reported charge remained **$0.048952**, total **$0.146856**. The full guarded job,
including serial source audits and the next Google fallback submission, completed
in **51.10 seconds**. Summing request times is not wall time, and concurrency does
not change the price per output or establish exterior/source quality acceptance.

Sampled peak RSS: **293.30 MiB**; physical footprint: **241.08 MiB**. Pressure
remained normal and swap growth stayed within the guard. RSS and footprint
overlap; these are samples, not a kernel-enforced ceiling.

Snapshot after pilot: 72 completed Sunburst images, one completed Nano image,
two Nano fallbacks submitted and 925 unattempted sources. Recorded cumulative
OpenRouter spend is **$3.524544**, using known receipts/account movement as before.
The full parallel production continuation has not launched at this publication
snapshot; the private state and Space entry record its subsequent launch.

## Nano protocol pilot and continued fallback processing

The first native Google batch returned one 768×1376 image. Its raw output and
original square were directly inspected; the two-reference protocol worked,
the source was detected with free alignment, and the completed usage implied
**1.109092335 TRY** at the saved conversion assumption. This is an estimate,
not a verified Google invoice charge. Source SSIM was 0.88944 and visible-source
fraction 0.99805; neither is an unchanged-pixel percentage or user approval.

Protocol review is recorded in private state, enabling further two-item fallback
batches within the existing reservation budget. The next two definitive Sunburst
fallbacks were submitted in one batch; their remote job ID and unresolved budget
reservation are saved. Unused Google funds still do not cause Nano to become a
primary teacher. Training admission remains subject to the user's visual review.

## Verification and publication

Twenty offline tests passed. New tests verify that a whole wave's budget is
reserved, that three dispatches actually overlap, that insufficient credit/key
limits reduce or prevent dispatch, and that a pause exits before any API access.
Existing prompt/reference and Google cost regression tests passed. Tests used
local fixtures and mocks, not paid services. Whitespace checks passed.

Publication contains six small files, one changed file per commit, one guarded
push. Source art, account balances, private queue state, job IDs and credentials
are excluded. Existing outputs and receipts remain intact. No student training,
checkpoint release or general quality approval occurred.
