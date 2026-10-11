# Reyhan — resume primary teacher, then spend remaining Google budget

Date: 2026-10-11.

## Authorized production policy

The user resumed the full production queue: use Sunburst through OpenRouter
within the existing account/user spending limit; use native Google Nano Banana
batch for definite primary refusals, then for other untouched covers once the
Sunburst budget stops. No OpenRouter image batch substitution is attempted.
The original spending caps are retained; no top-up or budget increase is assumed.

The Susam direct 100-cover trial plan remains in private state as history and is
marked superseded. Its 16 already-submitted covers are collected without another
generation. The 84 unsubmitted trial covers return to normal primary priority.
The source queue's completed/no-image/submitted states prevent duplicate spending.

## Pipeline implementation

One coordinator holds the private queue lock. Sunburst runs in three parallel
network slots with whole-wave reservations and serial isolated source audits.
Between primary groups, the coordinator collects existing Google jobs and sends
only definite refusal/no-image fallbacks. On the persisted primary budget-stop
condition, it transitions to Nano for remaining covers, prioritizing fallbacks.
Resuming that persisted phase does not restart Sunburst.

Nano remains 1K, 9:16, with the frozen old-canvas prompt and two references. At
most eight two-cover native batch jobs are outstanding. Each job reserves its
maximum output-token cost with the existing conversion buffer; complete usage
permits releasing excess reservation. Missing usage, failed provider jobs and
uncertain submissions retain reservations. Uncertain paid requests are never
automatically repeated. With no outstanding jobs and insufficient budget, the
pipeline stops instead of sending another generation.

The same pause markers stop new work. Completed and remote submitted work are
preserved. Account and key preflight reservations can leave a small unspent
amount when another safe primary request will not fit; zero remaining balance
is not promised. Google's invoice balance is not directly verified by the API.

## Validation and launch

The combined offline queue suites passed 22 tests. New tests cover fallback-only
Nano routing during the primary phase, refusal priority in the remaining phase,
automatic budget handoff, resuming without restarting primary generation,
uncertain submission blocking, and preserving jobs during pause. Tests do not
call paid services. `git diff --check` passed.

Both previous guards had exited before the pause markers were removed. A new
Reyhan coordinator was launched under the memory guard with a sampled 512 MiB
RSS/footprint limit, memory-pressure and swap-growth checks, and a 24-hour local
runtime limit. Existing provider jobs remain recoverable if the local process
stops. Local launch receipts, balances, provider job identifiers, images and
credentials remain ignored and are not public repository payloads.

This stage resumes production; it does not claim the full corpus is generated
or accepted for training. Source registration remains free to translate/scale;
SSIM is not an unchanged-pixel percentage. Visual quality and training acceptance
still require review. No student training or checkpoint release occurred.
