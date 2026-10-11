# Lavanta — explicit credit rejection and recovery to Nano

Date: 2026-10-11.

## Observed failure

The live status check found the Reyhan worker stopped after a three-request
Sunburst wave received HTTP 402. Each receipt explicitly named insufficient
OpenRouter account credits, with `limit_source: openrouter_credits`. The account
still had a positive residual balance; the provider nevertheless rejected those
requests. The old coordinator treated these as unresolved technical errors,
preventing the planned automatic transition to Nano. This was an execution bug,
not a completed handoff. The guard stopped on child exit code one, not RAM.
Its sampled peak RSS was approximately 451.38 MiB.

At the initial check, 313 Sunburst and 32 Nano images had completed: 345 total.
Five Nano no-image outputs were quarantined. Three already-submitted Nano batch
jobs covered five more sources. Sunburst known charges totaled $15.321976 and
matched the initial-to-current account credit movement at the recovery check.
The actual account balance remains in the private recovery receipt.

## Fix and recovery

A strict classifier recognizes only an HTTP-error receipt with status 402,
parsed provider error code 402, the explicit OpenRouter-credit metadata, and an
insufficient-credit message. Receipts with output images, a generation identifier
or a nonzero reported charge do not use this shortcut. Other errors retain the
existing uncertainty/reconciliation behavior.

The parallel coordinator drains the current wave, retains its receipts and
known costs, and records `primary_budget_rejected` sources plus a primary budget
stop. Mixed ambiguous errors still block progression. The existing teacher
pipeline then switches to its persisted Nano-remaining phase. Credit-rejected
covers are prioritized before other untouched covers, after ordinary fallbacks.
There is no automatic Sunburst POST retry or parameter/quality reduction.

For the three existing rejected sources, receipts were inspected and the known
charge sum reconciled to the live account movement before changing their private
state. A full private pre-recovery queue snapshot was retained. The coordinator
was relaunched directly in Nano-remaining phase; existing remote Google jobs are
retrieved without resubmission. Native batch reservations, the original Google
spending cap, frozen prompt, two references, 1K output settings and source audits
remain in place. No funds were added or spending limits increased.

## Validation and limitations

25 offline queue tests passed. Added coverage verifies the explicit-credit
classifier, that a credit-rejected wave drains and stops primary dispatch without
another POST, and that rejected sources enter remaining Nano work ahead of
untouched sources. Malformed errors, generic errors and charged/output-bearing
responses do not qualify. `git diff --check` passed.

The new guard retains the sampled 512 MiB RSS/footprint limit, system pressure
and swap-growth checks, and a 24-hour local runtime limit. Google usage-based
cost estimates are not a directly verified invoice balance. Paid POSTs are not
blindly retried, and missing usage retains its conservative reservation.

This stage repairs and resumes the already-authorized plan. It does not claim
all covers are generated, the Google jobs are finished, or examples are accepted
for training. Visual review still matters; SSIM is not unchanged-pixel percentage.
No student training or checkpoint release occurred.
