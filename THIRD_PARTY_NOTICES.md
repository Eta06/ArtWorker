# Third-party notices

The root Apache-2.0 license applies to project-authored material. It does not replace upstream licenses or grant rights to third-party model weights, datasets, music or album covers.

| Material | Source / pinned revision | Terms and retained attribution |
| --- | --- | --- |
| Qwen ControlNet MLX architecture adaptation in `experiments/qwen/controlnet_port/control.py` | [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun/tree/4b7b6402a1e0f0406bd6801fb66c0a00bd922621), `4b7b6402a1e0f0406bd6801fb66c0a00bd922621` | Apache-2.0. The file identifies the upstream architecture and local adaptation. [Upstream license](https://github.com/aigc-apps/VideoX-Fun/blob/4b7b6402a1e0f0406bd6801fb66c0a00bd922621/LICENSE). The control checkpoint has separate Qwen Research terms. |
| MaskFlow research code snapshots and adaptations | [ReyChiaro/MaskFlow](https://github.com/ReyChiaro/MaskFlow/tree/3125f1ecd72f5a4e068c5a1954e9d6a8df38d443), `3125f1ecd72f5a4e068c5a1954e9d6a8df38d443` | MIT, copyright (c) 2026 Chill. Full notice retained in [the reference directory](experiments/maskflow/research/2026-10-05/code/LICENSE); it also covers substantial adaptations of this reference. |
| MLX-Gen external runtime used by MaskFlow | `99fb94dd3eaa9dd1931cd3cd8eae1ae3e20f2ef3`; origin recorded in [runtime documentation](experiments/maskflow/README.md) | MIT. Runtime checkout and dependencies are not published in this repository; obtain their licenses with the upstream installation. |
| MFLUX, Diffusers, Torch, MLX and Swift package dependencies | Pins, environment lock files and package-resolution records under each experiment | External packages retain their own licenses. They are imported dependencies, not relicensed ArtWorker code. |
| Downloaded Hugging Face model cards and configuration/reference metadata | Source repository IDs and revisions in adjacent research/download manifests | Upstream reference material; retain its stated terms and attribution. Reading or publishing metadata is not permission to redistribute the model payload. |

The local music collection and all associated artwork/output images are excluded from this publication. Reproduction requires your own permitted input images and separately obtained models. Historical reports contain local artifact paths and hashes; the corresponding private media and binary arrays are not included.

See [MODEL_ARTIFACTS.md](docs/MODEL_ARTIFACTS.md) before distributing a weight file or using a model in a product. In particular, the evaluated FLUX.1 Fill dev conversion has non-commercial model terms; ArtWorker's Apache-2.0 code license does not remove that restriction.
