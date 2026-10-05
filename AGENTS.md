# Agent cost limits

These limits are direct user instructions from 2026-10-05.

- Prefer doing the work in the primary agent.
- At most two subagents may run at once across the entire task tree, including descendants.
- Allowed subagent models: `gpt-6.1-sol` with reasoning effort `medium`, or `gpt-6-luna` with an appropriate reasoning effort.
- Use Luna for simple work and research.
- Subagents must not spawn further agents. Delegation is controlled by the primary agent.
- Do not create separate agents merely to repeat reviews, report writing, or verification that the primary agent can perform.
- Preserve existing experiments and distinguish visual quality from successful execution or numerical integrity checks.

# Publication and experiment records

- Exactly one changed file per commit, including the initial import. Group commits into one push per publication batch.
- Choose a new single-word stage name yourself. Examples supplied by the user are not assigned names. The first publication stage is `sedef`.
- Commit subjects use `<stage>: <file or concise change>`; no AI trailers.
- Record each stage under `docs/stages/` and in the ArtWorker ChatGPT Space Page. Record actual attempts, failures, measurements, limitations and next work.
- Public repository: https://github.com/Eta06/ArtWorker
- Progress Page: https://chatgpt.com/space/page_eeabd8711884819189a662a8a09e9290
- Track third-party weights by pinned source, license and download/checksum manifests. Preserve upstream licenses. Do not put model payloads, caches, local music/artwork or credentials in normal Git.
- Publish our own trained weights only with a model card, checksums and a reviewed redistribution license. No trained ArtWorker checkpoint has been released yet.
