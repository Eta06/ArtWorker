# Yonca — fresh Nano Banana retry

11 October 2026, Türkiye. The user requested another Nano Banana attempt after
changing their setup. Three serial requests use `google/gemini-nano-banana-2.1`
/ 1K and the same Filiz sources: Aklın Hep Bende, Karambol and Gripin. Each sends
one unchanged square reference with the original short extension prompt and
`aspect_ratio: 9:16`. No green diagnostic canvas is transmitted. Reference counts,
input mode and transmitted-file checksums were verified from fresh receipts.

All three requests returned **HTTP 429, Google AI Studio quota exceeded**, with
no image. This differs from the earlier 402 depleted-prepayment response, but
does not establish that billing, model access or generation now works. The
provider error asks to check plan/billing and references rate limits; its message
does not identify the exhausted quota or when it resets. No transient-only cause
is inferred. The current endpoint remains Google AI Studio.

New charge **$0**, account delta **$0**, fully reconciled; zero unknown charges.
See the [anonymous ledger](../../experiments/openrouter_outpaint/yonca-costs.json).
No automatic POST retry, routing/settings change, purchase or background monitor.
No other model was called and no quality/preservation scores exist for this run.

The bounded run finished in 4.866 seconds, sampled peak RSS **133.13 MiB** and
physical footprint **83.63 MiB**, under the 1.5 GiB ceiling with continuous pressure
checks. Existing system swap was present. No memory/pressure stop, weight download,
training, simulator work or subagent. Historical images, receipts and votes remain
untouched. Private receipts and the error-only local gallery are retained under
`experiments/openrouter_outpaint/results/2026-10-11/yonca/` and are ignored by Git.

The two public files are one file per `yonca:` commit, grouped in one guarded
publication push. Record publication verification and the new 429 condition in
the existing ArtWorker Space Page. No credentials, private source IDs or media.

Next: determine the active model/project quota before another user-authorized
attempt. Google's rate limits can involve per-minute, per-day, image and spending
limits, and apply at project level; a 429 alone does not tell us which applies
here. [Google rate-limit documentation](https://ai.google.dev/gemini-api/docs/rate-limits).
