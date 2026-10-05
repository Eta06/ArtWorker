# Manual footer-conditioning hypothesis probe

The completed native local-context lower Fill output contains two pale road stripes. This isolated lower-only experiment tests whether white album footer lettering in the known source context contributes to that invention. It is not a generic footer detector, growing sampler, or quality-approved result.

Only conditioning RGB pixels in hand-selected original-source box `[238,470,275,509]` are changed. They are replaced byte-for-byte by same-row asphalt from `[191,470,228,509]`, 47 pixels left. The actual logo bright pixels were inspected inside `[241,473,271,506]`. The rendered source remains the original, including its lettering.

- The native generation mask remains exact: 128 known rows and 320 unknown rows in a 512×448 lower patch.
- The saved lower float32 noise, original prompt, native 50-step / guidance-30 schedule and all four denoising operations are retained. The changed source is encoded as an entire native patch; VAE latent changes may spread beyond the small input RGB edit.
- Native prompt encodings must equal saved baseline values exactly before sampling; actual native dtypes are recorded. Native Metal sigma values must equal all 51 saved values exactly. CPU sigma regeneration differs by one float32 ULP, so CPU values are not substituted.
- Actual unmodified model `raw.png` is saved before original-source assembly. The full display composite uses unchanged original source, reused baseline upper320, and newly generated lower320; it is not a new full-canvas raw generation.
- Planned inference is one lower call: 50 NFE, 44,800 target-token forwards, 950 joint and 1,900 single block forwards. No model inference was run in the CPU preparation.

`ready.json` contains the bounded root-only GPU command. `preflight/preflight.json` records static/model-header/CPU fixture gates and retained baseline hashes. `independent_baseline_audit.json` independently checks the actual previous lower inputs and replay requirements.

CPU fixtures cover changed pixels / unchanged far pixels, native normalized input and packed-mask polarity, exact saved noise restoration, native Euler arithmetic, actual native-operation AST, assembly orientation, preserved raw values and rejection of mismatched known source context. Inference success and source integrity will not establish visual quality.
