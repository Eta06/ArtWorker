# ArtWorker

Open research on fast, compact album-cover outpainting for iPhone, with a native SwiftUI music-player prototype. We compare image quality, latency, memory and model footprint before choosing a mobile model. True center-outward progressive generation remains an experimental goal.

There is no released trained ArtWorker model or visually accepted production outpainting pipeline yet. Mac inference measurements are not iPhone performance results.

- [Public repository](https://github.com/Eta06/ArtWorker)
- [ChatGPT Space progress Page](https://chatgpt.com/space/page_eeabd8711884819189a662a8a09e9290)
- [Sedef: first publication and experiment archive](docs/stages/sedef.md)
- [Koza: fixed-instruction and low-bit compression results](docs/stages/koza.md)
- [Model artifacts and redistribution](docs/MODEL_ARTIFACTS.md)

## Contribution and publication workflow

Each commit changes exactly one file and uses a stage prefix, for example `sedef: README.md`. Collect the commits and push once per batch. Enable the local checks with `git config core.hooksPath .githooks`. For a reviewed JSON list of repository-relative files, `python3 scripts/commit_stage.py --phase sedef --files-file <list.json>` creates one commit per file and never pushes automatically. Record experiment outcomes in the stage document and linked Space Page.

Project-authored code is licensed under [Apache 2.0](LICENSE). Third-party code and model weights retain their own terms; see [third-party notices](THIRD_PARTY_NOTICES.md). Local music, album covers, model binaries and runtime caches are excluded from Git.

## Music player

Native iPhone music player, built with SwiftUI and AVAudioPlayer. The player uses an edge-to-edge artwork atmosphere, warm white controls, and fine 2 pt playback/volume rails with 44 pt touch targets. This first version plays the three local MP3s from `~/test-player/`, including their embedded album artwork. It has playback, scrubbing, previous/next, shuffle, repeat-one, session favorites, volume, and a queue sheet. Playback does not start automatically on launch.

Open `ArtWorker.xcodeproj` in Xcode and run the ArtWorker scheme on an iPhone simulator. To run on a physical iPhone, choose your signing team in Xcode.

## Local media

Media is copied into `Player/Resources/` and excluded from Git. Original files remain untouched. The bundled covers have a square album image centered in a 16:9 export; `Track.artwork` removes the side padding for display only.

To reimport the three files (requires ffmpeg/ffprobe):

```sh
python3 scripts/import_tracks.py ~/test-player
```

## Build

```sh
xcodebuild -project ArtWorker.xcodeproj -scheme ArtWorker \
  -configuration Debug -sdk iphonesimulator \
  -derivedDataPath .build CODE_SIGNING_ALLOWED=NO build
```

## Outpainting experiments

Downloaded local models and real Mac inference are kept under `experiments/`, outside the player. See the [measured trial results](docs/OUTPAINT_TRIAL_RESULTS.md), [distillation plan](docs/DISTILLATION_PLAN.md), and [iOS deployment research](docs/IOS_OUTPAINT_DEPLOYMENT.md). The minimal [iOS SDK compile probe](experiments/ios_sdk/README.md) builds successfully; it has not run model inference on a physical phone.

The [October 5 spatial quality controls](experiments/qwen/QUALITY_ABLATION_RESULTS.md) compare final images from the same six-step model with earlier target activation and two insertion initializers. The severe outer grid artifacts improve on the road/sea cover, while composition and boundary continuity still need validation. The subsequent [matched boundary-context probe](experiments/qwen/GEOMETRY_RESULTS.md) reduces the guardrail kink but leaves an angle mismatch and a more visible lower asphalt seam. These are custom sampler experiments, not a bundled player feature.

The ongoing [boundary-fix comparison](experiments/qwen/BOUNDARY_FIX_RESULTS.md) records actual sampler, decoder, adapter, compositor and trained ControlNet trials, including failed and incomplete arms. Structural guidance improves the local rail direction, but the first trained full-canvas outputs introduce blank exterior regions. Full-frame quality and true growing computation are evaluated separately; none of these findings is an iPhone performance result.

## Player model integration

No AI model or generated imagery is bundled into the player yet. The current edge-to-edge background is a blurred copy of the cover, including the status bar area. `PlayerView.artwork(size:)` is the visual surface to extend with progressive outpainting; original artwork should remain a separate, preserved layer.

This prototype has not been tested on physical devices. Lock-screen remote controls, background audio entitlement, audio interruption handling, persistent favorites, and user file import are not included yet.
