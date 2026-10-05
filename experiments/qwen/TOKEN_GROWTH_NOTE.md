# Actual target-token growth in one Qwen2.1 sampler

Read-only implementation analysis against MFLUX commit
`25661d8e853996fec9134998b995edf97369bfbd`. The initial architecture analysis
and layout smoke used CPU arrays without loading weights. A separate matched
FULL/GROW GPU experiment was then executed; its actual results are below.

The reference editing path can accept a changing number of target latent
tokens during one sampler loop. Future target tokens can be absent from the
transformer input and attention, rather than merely receiving masked Euler
updates. This requires a custom sampler; the supplied editing loop keeps a
fixed target shape throughout.

## Fixed source prefix and stable coordinates

Encode the true 512×512 RGBA source with `_encode_prompt(prompt, [source])` and
the VAE once. Its prefix image shape is `(1,32,32)` and it contributes 1,024
latent tokens. Do not encode the gray 512×1152 padded canvas as the prefix:
those additional gray reference tokens would describe future canvas area.
The checkpoint processor accepts the square source's 262,144-pixel area;
its range is 65,536–16,777,216, patch size 16 and merge size 2.

Build a full final layout once with source `(1,32,32)` and target `(1,72,32)`.
The source rotary height/width coordinates are `[-16,15]`; final target
coordinates are `[-36,35] × [-16,15]`. The centered source aligns with target
rows `[20,52)`. The source and target have different frame coordinates by
the checkpoint's reference-image convention. Keep both fixed; spatial
alignment does not require forcing their frame IDs equal.

Maintain target state keyed by final-canvas raster IDs `y*32+x`. At each
expansion, preserve existing active states and insert newly activated rows.
For every forward call create an active layout, then overwrite its target
RoPE rows with a gather from the full final layout:

```python
# Conceptual mapping, not a tested runtime helper.
active_layout = QwenImage21Layout.create(
    slots, [(1, 32, 32), (1, active_height, active_width)], axes
)
P = full_layout.prefix_length
active_layout.rope = tuple(
    mx.concatenate([r[:P], r[P + active_final_ids]], axis=0)
    for r in full_layout.rope
)
# Input contains only source-prefix latents plus actually active target latents.
prediction = transformer(
    mx.concatenate([source_latents, active_latents], axis=1),
    prompt_embeds, global_sigma, active_layout, prefix_cache,
)
```

Existing raster IDs must keep exactly the same positions and noise samples
after growth. The standard layout factory only describes contiguous
rectangles and requires a multiple of four target tokens. Symmetric centered
rectangles already preserve existing centered rotary coordinates; gathering
from final-canvas coordinates also supports asymmetric growth safely.
Arbitrary annular or sparse active sets need a custom layout constructor.

The first call builds the reference/text prefix cache. Every later call uses
the same prefix cache and the current active target count. The cached path
projects only the last `layout.target_tokens` input rows. Source/text prefix
tokens attend only to their own earlier prefix segments and receive t=0
modulation, so their per-layer K/V do not depend on target geometry or sigma.
Target tokens attend to the constant prefix and all currently active target
tokens. Earlier generated target K/V cannot be retained unchanged: active
target state and mutual attention change every denoising step.

## Noise schedule and model limitation

Keep one monotonically decreasing sigma schedule for the entire loop. Do not
restart the scheduler or recompute dynamic sigma shifting at every shape
change. The checkpoint schedule uses image sequence length for dynamic
shifting, so a practical controlled prototype should choose the final target
length at initialization and document this choice.

At sigma `s`, the flow marginal is of the form
`z_s = (1-s)*x_0 + s*epsilon`. Adding fresh unit noise late at `s < 1` is not
that marginal. Adding `s*epsilon` still omits the unknown clean-image term.
An edge/extrapolation guess for `x_0` creates a runnable initializer but not a
proof of correct conditional sampling. This mismatch becomes larger for late
expansion. Activating all outer bands early while sigma remains near one is
the least invasive first experiment, but leaves a shorter growth phase.

The supplied checkpoint applies one global target sigma: the transformer
builds `[sampled_sigma, 0]` timestep rows, and a boolean target mask chooses
sampled sigma for all target tokens, zero for all prefix tokens. It cannot
currently represent a mature inner region at low local sigma and a fresh
outer region at high local sigma. Passing a vector of local times without
rewriting row selection produces the wrong broadcasting semantics.

Tokenwise time embeddings and tokenwise Euler steps can be added in code,
but code support alone does not establish quality. A reliable asynchronous
center-out generator should be trained or distilled on nested activation
masks with tokenwise noise levels and the same stable spatial coordinates.
Teacher trajectories must cover that conditioning/noise pattern, rather
than only final-image pairs. Few-step LoRA distillation does not by itself
teach this new sampling distribution.

One invocation with target insertion is honestly a custom hybrid sampler:
it integrates existing states between expansion events and changes state
dimension at those events. It is not the unchanged fixed-dimensional flow
ODE used by the official checkpoint.

## Executed paired probe

`run_spatial_trial.py` ran one shared model load and two independent six-forward
loops using the recommended Viggle r256 adapter at q4. Both arms share the
actual square source prefix, prompt, source encoding, absolute noise table,
and final-canvas sigma schedule. Each arm builds its own fresh prefix cache
once, preserving a matched cache construction cost. Measured joint prefix
length is 1,197 tokens, of which 1,024 are source-image latents.

FULL forwards 2,304 target tokens per step. GROW's centered active canvas is
512×640, 512×896, then 512×1152, two steps per size; token counts are 1,280,
1,792 and 2,304. Runtime assertions verify that future target rows are absent
from model inputs and outputs, prefix cache geometry stays fixed, and the
final known-source latent is exact. Newly admitted rows use the untrained
edge-extrapolated clean latent plus appropriately scaled same-seed noise;
the scheduler never restarts.

The actual target-forward sum falls from 13,824 to 10,752, a 22.2% reduction.
Measured core denoising falls from 29.38s to 24.33s, 17.2% in one pair.
Execution order, thermal state and kernel warm-up remain timing limitations.
Final GROW quality fails: outer bands contain grid/ringing patterns, an extra
guardrail and small invented symbols. FULL's square-prefix result is coherent.
This supports executable compute reduction, not reliable center-out quality.

True predicted-clean latent snapshots `z_sigma - sigma*v` are captured at
steps 2/4/6, with the known source restored. Their VAE decode occurs only after
both loops and after releasing the denoiser. Capture and preview decoding time
are separated from denoising. This is not a live UI streaming benchmark.
Earlier inner regions continue refining; no completed/frozen ring is claimed.
The final failed frame is retained along with intermediates and raw outputs.

`SPATIAL_RESULTS.md` and `spatial_runs/q4/track3/paired_metrics.json` contain the
results. Both records are explicitly diagnostic controls and excluded from the
standard common-canvas comparison.

## Code evidence

Paths are relative to `.build/model-research/mflux/src/mflux/`.

- `models/qwen21/model/qwen21_transformer/qwen21_layout.py:22–38`: target count,
  fourfold image-slot expansion, prefix length and target mask.
- Same file `43–60`: frame progression and centered height/width rotary
  coordinates; `69–77`: the explicit layout fields.
- `models/qwen21/model/qwen21_transformer/qwen21_transformer.py:185–201`:
  variable target count and cache-only target projection; `213–231`: sampled
  sigma plus t=0 row, layer cache, and target-only output.
- `models/qwen21/model/qwen21_transformer/qwen21_attention.py:84–105`:
  cached prefix concatenation, prefix extraction, and causal/block-segment
  attention independent of target tokens.
- `models/qwen21/model/qwen21_transformer/qwen21_transformer_block.py:108–118`:
  boolean target/prefix timestep row selection.
- `models/qwen21/reference/latent_creator/qwen_image21_latent_creator.py:8–16`:
  one raster token per 16×16 pixels; `24–28`: supported size/step validation.
- `models/qwen21/variants/edit/qwen_image_21_edit.py:89–120`: fixed source
  encodings, fixed layout, source+target input, global Euler schedule.
- Local checkpoint `processor/preprocessor_config.json` and
  `scheduler/scheduler_config.json`: processor range and dynamic schedule
  metadata.
