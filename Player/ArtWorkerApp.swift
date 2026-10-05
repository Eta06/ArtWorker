import SwiftUI

@main
struct ArtWorkerApp: App {
    @StateObject private var player = MusicPlayer()
    var body: some Scene {
        WindowGroup { PlayerView().environmentObject(player).preferredColorScheme(.dark) }
    }
}
