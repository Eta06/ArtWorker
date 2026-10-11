# Pelit — recover from a read-only account timeout

2026-10-11. Live status inspection found the parallel worker had exited after an
HTTP 408 on `GET /credits`, after the last three images and their source audits
completed. No paid POST was ambiguous. The memory guard reported normal pressure
and a sampled peak RSS of 356.75 MiB, below its unchanged 512 MiB limit.

## Reconciliation and repair

All 131 completed Sunburst outputs were checked against their receipt hashes
and free-alignment report hashes. One audit-path field was absent from the saved
queue snapshot because the failing credit GET happened before the final state
write; the already-existing report was verified and its path restored. No image
or audit was generated again. Completed primary receipt costs sum to $6.412712,
matching the live account movement exactly. The old displayed aggregate was one
three-image wave behind, not evidence of three unpaid outputs.

Account credit/key GETs now retry transient 408/429/5xx or transport timeouts at
most three times with 2/4 second backoff. The helper accepts only `credits` and
`key`; it cannot retry a paid POST. Authentication failures are not retried.
Known charges and completed audit paths are saved before the final account GET;
an exhausted read retry records `account_query_paused` before stopping. Existing
source, prompt, quality, reservation and no-blind-POST-retry rules are unchanged.

The remaining native Google batch was retrieved without generating again. Four
Nano images are now completed. Total Google usage-based estimate is 4.3951673025
TRY at the existing planning exchange rate; actual invoice charge remains unknown.
Snapshot before continuation: 131 Sunburst + 4 Nano outputs, all pending human
training acceptance. There are no unresolved primary requests to repeat.

## Verification and publication

Twelve queue tests passed, including two new regressions: transient account GET
retry is read-only, and authentication errors do not retry. Existing concurrent
dispatch, wave reservation, source-order and unknown-outcome checks passed.
Tests used mocks and did not call paid services. Whitespace checks passed.
Google collection/audit took 3.13 seconds, sampled RSS 226.98 MiB / footprint
174.11 MiB; these measures overlap and are not added.

Publication consists of three files, one changed file per commit and one guarded
push. Artwork, account balances, credentials and remote job IDs remain local.
The subsequent three-worker continuation is recorded in the private state and
Space; this snapshot does not claim that the whole production queue completed.
New agents, model changes and student-training runs: zero.
