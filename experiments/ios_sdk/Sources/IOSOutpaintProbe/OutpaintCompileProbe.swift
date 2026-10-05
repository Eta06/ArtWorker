import UIKit
import MediaGenerationKit

/// Compile-only API probe. No model downloads or inference run on import.
public enum OutpaintCompileProbe {
    public static func generate(
        canvas: UIImage,
        nativeEncodedMask: UIImage? = nil,
        modelsDirectory: String
    ) async throws -> UIImage? {
        var pipeline = try await MediaGenerationPipeline.fromPretrained(
            "flux_2_klein_4b_q6p.ckpt",
            backend: .local(directory: modelsDirectory)
        )
        pipeline.configuration.width = 512
        pipeline.configuration.height = 1024
        pipeline.configuration.steps = 4
        pipeline.configuration.strength = 1
        pipeline.configuration.guidanceScale = 1
        pipeline.configuration.batchSize = 1
        pipeline.configuration.batchCount = 1
        pipeline.configuration.preserveOriginalAfterInpaint = true
        var inputs: [MediaGenerationPipeline.Input] = [canvas]
        if let nativeEncodedMask {
            inputs.append(nativeEncodedMask.mask())
        }
        let results = try await pipeline.generate(
            prompt: "Extend the album cover beyond its borders, continuing the same scene, colors, lighting and visual style. No new text.",
            // This SDK forwards grayscale bytes to Draw Things' native mask encoding:
            // low three bits 3 = preserve/skip, 1 = absent content to generate.
            // A Diffusers-style 0/255 binary mask is not that encoding.
            // With no explicit mask, preserve RGBA transparency in `canvas`.
            // ImageConverter derives native 251 (opaque source) / 1 (empty) bytes.
            inputs: inputs
        )
        return results.first.map(UIImage.init)
    }
}
