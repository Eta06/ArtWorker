# Menekse — local output preferences

6 October 2026. Added per-output **Beğendim / Beğenmedim** choices to the
existing Zeytin gallery. The user chooses visual quality. No new model calls,
generations, billing requests, weight changes or subagents were used. The
reconciled Zeytin ledger and its 91 raw images are retained unchanged.

Each output has a mutually exclusive like/dislike state; pressing the active
button clears the decision. An output without an image cannot be rated. Ratings
are identified by model, native profile, cover and raw-image SHA256, so a newly
generated image does not inherit a previous output's vote. Browser localStorage
preserves decisions after reload; storage errors are shown visibly. Same-origin
tabs synchronize their selections through storage events.

The liked-model list contains each model with at least one liked output once,
with accepted cover/settings and a count of disliked outputs from that model.
A single liked setting does not imply acceptance of all settings or both covers.
Model, cover and decision filters allow browsing accepted/rejected/undecided
results. The user can view a copyable model-list text export or a full JSON
preference backup, with file-download links. Export text uses Europe/Istanbul
local time; machine metadata uses ISO UTC. These local choices are not uploaded
to GitHub, Space or a provider, and do not automatically select a teacher.

## Live verification

Verified in the existing in-app browser:

- One like adds the correct model, cover and native profile.
- Reload retains that decision.
- Changing it to dislike removes the model when it has no other liked output.
- Clicking the active decision clears it.
- Liking two settings of one model yields one model and two liked outputs.
- A mixed like/dislike retains one model with separate accepted/rejected counts.
- The liked-only filter shows the one accepted image in the mixed example.
- Model text export includes both accepted profiles and their costs.
- JSON backup includes accepted and rejected output identities and settings.
- Verification selections were removed, leaving zero test votes.

The first programmatic download attempt timed out in the in-app browser. A
copyable export dialog was added and verified directly; native file-download
completion remains unverified in that browser. File links are still offered.
No JSON import is implemented. Browser storage is tied to the current origin,
including the server port; clearing browser data or changing that origin loses
those stored decisions, so the UI offers a backup.

The standalone renderer uses the existing ledger and makes no network calls.
The normal report generator shares that renderer, so regenerating a report
retains this interface. Python compilation and whitespace checks passed. No
visual quality verdicts were assigned by the implementation or its UI checks.

## Publication

Five changed files, each in its own menekse: commit, grouped into one guarded
push. Source media, generated images, credentials and user preferences remain
outside normal Git. Stage details are also recorded on the ArtWorker Space Page.
