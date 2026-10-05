// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "IOSOutpaintProbe",
    platforms: [.iOS(.v17)],
    products: [.library(name: "IOSOutpaintProbe", targets: ["IOSOutpaintProbe"])],
    dependencies: [
        .package(
            url: "https://github.com/drawthingsai/media-generation-kit.git",
            revision: "8868a9685d9c299816f43ef53efd455ffca437f0"
        )
    ],
    targets: [
        .target(
            name: "IOSOutpaintProbe",
            dependencies: [.product(name: "MediaGenerationKit", package: "media-generation-kit")]
        )
    ]
)
