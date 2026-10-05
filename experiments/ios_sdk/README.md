# iOS outpainting SDK compile probe

This isolated Swift package checks whether the public MediaGenerationKit local backend and image/mask API compile for iOS Simulator. It does not modify Player, download model weights, or run inference. A successful compile would establish only SDK integration feasibility, not quality, memory use or actual iPhone performance.

Pinned SDK revision: `8868a9685d9c299816f43ef53efd455ffca437f0`.

## Actual build result — 2026-09-30

**BUILD SUCCEEDED** with Xcode 27 / iOS Simulator 27.0, for **arm64 and x86_64**. The final optional native mask / RGBA automatic-mask source was recompiled in an incremental build after editing; both object files are newer than the source. `validation.json` records the source hash and pinned dependency revisions; `Package.resolved` retains the complete resolved graph.

This verifies compilation of `MediaGenerationPipeline.fromPretrained`, `.local(directory:)`, `UIImage` inputs, `.mask()`, the listed configuration fields, and generated results back to `UIImage`. No Player integration, model download, generation, device install, memory test or speed measurement was performed by this probe.

Full initial and incremental logs are at `.build/compile-ios-simulator.log` and `.build/compile-ios-simulator-current.log` (ignored build artifacts).

```sh
xcodebuild -scheme IOSOutpaintProbe -destination 'generic/platform=iOS Simulator' -derivedDataPath .build/DerivedData -clonedSourcePackagesDirPath .build/SourcePackages -jobs 2 build CODE_SIGNING_ALLOWED=NO
```

The function is a compile probe only. Klein's mask behavior, rectangular generation, and exact preservation must be validated in real inference.

Important mask contract found by reading the pinned upstream source: `maskDataToTensor` converts the supplied image to grayscale UInt8 and passes it to `LocalImageGenerator`, which interprets the low three bits as native categories. `3` means preserve/skip, `1` means absent content to generate, and `2`/`4` mean inpaint. A standard Diffusers black/white `0/255` mask is not interchangeable: `255 & 7 == 7`, which enters the fallback category. Supply the native encoding (and verify pixel values after the SDK conversion), or use the runtime's alpha-derived automatic mask with an RGBA canvas. Mask blur/compositing can change exact source preservation; always composite the original artwork independently.

Sources: [mask decoder](https://github.com/drawthingsai/draw-things-community/blob/d473a2f148b3e7dc9b90d0b7cfccc5cda999eb66/Libraries/MediaGenerationKit/Sources/MediaGenerationExecutionUtilities.swift), [native mask interpretation](https://github.com/drawthingsai/draw-things-community/blob/d473a2f148b3e7dc9b90d0b7cfccc5cda999eb66/Libraries/LocalImageGenerator/Sources/LocalImageGenerator.swift).

The probe also accepts `nativeEncodedMask: nil` with a transparent RGBA canvas. The SDK's alpha-derived automatic mask encodes an opaque source pixel as `3 | 248 == 251` and a completely transparent expansion pixel as `1`. It treats high five bits as alpha and low three bits as role; keep PNG/RGBA transparency through the input codec. See [ImageConverter.tensor](https://github.com/drawthingsai/draw-things-community/blob/d473a2f148b3e7dc9b90d0b7cfccc5cda999eb66/Libraries/LocalImageGenerator/Sources/ImageConverter.swift).
