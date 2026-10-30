# Compact image editing candidates checked 30 October 2026

Parameter counts for image generators frequently omit the text encoder, VAE,
vision encoder or prompt-expansion model. Download footprint, peak working
memory, native masked completion and visual outpaint quality are separate tests.
None of the models below has generated a new ArtWorker benchmark output in this
stage. Repository metadata was queried without downloading model payloads.

## FLUX 3 Image Klein

[BFL's release](https://bfl.ai/blog/flux-3-image-klein) is dated 15 October
2026. It provides a 5.6B image transformer, with a separate Qwen3-VL-4B encoder,
base and distilled releases, image references and bounding-box control. The
[official model card](https://huggingface.co/black-forest-labs/flux-3-image-klein)
requires expanded scene captions rather than treating a short instruction as a
sufficient prompt. This is a potential larger quality reference, not a
demonstrated small phone bundle.

The [MFLUX implementation](https://github.com/mflux-community/mflux/blob/main/src/mflux/models/flux3/README.md)
supports Apple MLX, low-RAM loading and quantization. It documents an
[outpaint LoRA](https://huggingface.co/ausboss/FLUX-3-Image-Klein-Outpaint-LoRA)
using gray padding. The community
[Q4 conversion listing](https://huggingface.co/mflux-community/flux-3-image-klein-mflux-bf16)
reports 5.3 GB; that is a published storage figure, not our measured inference
memory. Its Flux Kommunity license must be read for the intended product use;
it is not the Apache-2.0 license of our Klein 4B baseline.

Observed official distilled revision:
`337337621b8526e120f2275d250242f4a6bedd75`; the Hub reports gated access.

## Moebius

[Official code and weights](https://github.com/hustvl/Moebius) were released
18 June 2026. This is a newly identified candidate for this project, not a model
released after October's Klein work. It is approximately 0.2B/0.22B, specialized
for masked image completion, and does not need a large text encoder. Our proposed
use as an outpaint student requires border-mask tests; inpainting demonstrations
alone do not establish album-cover continuation quality.

The [official Hub repository](https://huggingface.co/hustvl/Moebius) at
`cd01f47fb648219d3fa605806c5ce00b713faa5e` has four approximately 905 MB F32 UNet
checkpoints. Only one plus the separately sourced VAE is needed for a trial;
the 3.37 GiB complete repository is not the minimal package. The official code
revision inspected is `b88d462bacb9af6e7128a3b4cc4a07418bedfd61`. Its requirements
include CUDA flash-linear-attention, so its stock runner is not verified on MPS.
An Apple-compatible implementation/export is a distinct engineering task.

License metadata conflicts: the inspected GitHub LICENSE is Apache-2.0 and the
README says code/weights Apache-2.0, while the official Hub card labels the
weights MIT. Preserve both records and resolve before weight redistribution.
No payload was downloaded, executed, re-licensed or redistributed in this stage.

## BlazeEdit

The [Google-authored paper](https://arxiv.org/html/2605.28067v1), submitted
27 May 2026, describes a text-free 195M image-to-image model with explicit
outpainting among its tasks. The paper reports 290 ms on a Pixel 10; that is
author-reported hardware performance, not an ArtWorker or iPhone result. This
architecture is directly relevant to the compact-student goal. No public weight
download or executable release was located in the paper and targeted searches,
so it cannot currently be presented as an installed benchmark candidate.

## Iris 3B

[The publisher's model card](https://huggingface.co/speridlabs/iris-3b)
describes a 3B pixel-space generator with a separate frozen 4B encoder. It avoids
a VAE, but that does not remove the text encoder. Released downstream fine-tunes
cover depth and restoration/upscaling; a native outpaint checkpoint was not
established. Apache-2.0, observed revision
`adc05b909fb05d4276d26aae5a52b36406b2ef26`. Research interest, not a drop-in
replacement for our masked outpaint recipe.

## Other editing models

- [Fibo 2.0 Edit Turbo](https://huggingface.co/briaai/Fibo-2.0-edit-turbo)
  is an 8B four-step editor with mask/reference support. The publisher lists
  image expansion as coming soon. Gated, noncommercial-weight terms; observed
  revision `9a24ee6810a782dd6aad4639fadcf48db0957ef4`. It is a larger research
  candidate, not a confirmed native outpaint/mobile replacement.
- [LongCat Image Edit Turbo](https://github.com/meituan-longcat/LongCat-Image)
  was released 3 February 2026, so it is not new since our October experiments.
  The family is 6B, with an eight-step editing version under Apache-2.0.
  Official Hub revision `6a7262de5549f0bf0ec54c08ef7d283ef41f3214` is accessible;
  the whole repository totals 27.31 GiB, not a minimal quantized package.
- [Z-Image's official repository](https://github.com/Tongyi-MAI/Z-Image)
  still marks the Edit and Omni-Base checkpoints as to be released in its model
  table. A text-to-image Turbo checkpoint is not evidence of native editing.

## Next comparison

Run the installed fifteen recipes on the frozen ten covers when Mac memory
pressure is normal. Use the actual outputs to select a quality baseline. Then
prioritize Moebius for a small masked student and FLUX 3 Klein plus its outpaint
adapter for a newer quality reference, subject to runtime/weight access and
memory checks. Keep BlazeEdit as architecture research until weights are found.
These are investigation priorities, not quality rankings from an unrun test.
