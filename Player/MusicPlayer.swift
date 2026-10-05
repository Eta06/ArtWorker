import AVFoundation
import Combine
import UIKit

struct Track: Decodable, Identifiable {
    let id, title, artist, album: String
    let duration: Double
    private static let artworkCache = NSCache<NSString, UIImage>()
    var artwork: UIImage? {
        if let cached = Self.artworkCache.object(forKey: id as NSString) { return cached }
        guard let url = Bundle.main.url(forResource: id, withExtension: "jpg"),
              let image = UIImage(contentsOfFile: url.path) else { return nil }
        // These local MP3 covers contain a square cover centered in a 16:9 image.
        // Remove only the export's side padding; keep the embedded source unchanged.
        guard let cg = image.cgImage, cg.width * 9 == cg.height * 16 else {
            Self.artworkCache.setObject(image, forKey: id as NSString)
            return image
        }
        let side = cg.height
        let rect = CGRect(x: (cg.width - side) / 2, y: 0, width: side, height: side)
        guard let square = cg.cropping(to: rect) else { return image }
        let artwork = UIImage(cgImage: square)
        Self.artworkCache.setObject(artwork, forKey: id as NSString)
        return artwork
    }
}

@MainActor
final class MusicPlayer: NSObject, ObservableObject, AVAudioPlayerDelegate {
    @Published private(set) var tracks: [Track] = []
    @Published private(set) var index = 0
    @Published private(set) var isPlaying = false
    @Published private(set) var position: Double = 0
    @Published var shuffle = false
    @Published var repeatTrack = false
    @Published var favorites: Set<String> = []
    @Published var volume: Float = 0.7 { didSet { audio?.volume = volume } }
    @Published var error: String?
    private var audio: AVAudioPlayer?
    private var timer: AnyCancellable?
    var current: Track? { tracks.indices.contains(index) ? tracks[index] : nil }
    var duration: Double { audio?.duration ?? current?.duration ?? 1 }

    override init() {
        super.init()
        do {
            guard let url = Bundle.main.url(forResource: "tracks", withExtension: "json") else { return }
            tracks = try JSONDecoder().decode([Track].self, from: Data(contentsOf: url))
            try AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
            select(0, autoplay: false)
        } catch { self.error = error.localizedDescription }
        timer = Timer.publish(every: 0.25, on: .main, in: .common).autoconnect().sink { [weak self] _ in
            guard let self else { return }
            self.position = self.audio?.currentTime ?? 0
        }
    }
    func select(_ newIndex: Int, autoplay: Bool = true) {
        guard tracks.indices.contains(newIndex) else { return }
        audio?.stop()
        isPlaying = false
        index = newIndex
        position = 0
        do {
            guard let url = Bundle.main.url(forResource: tracks[newIndex].id, withExtension: "mp3") else { return }
            audio = try AVAudioPlayer(contentsOf: url)
            audio?.delegate = self
            audio?.volume = volume
            audio?.prepareToPlay()
            if autoplay { play() }
        } catch { self.error = error.localizedDescription }
    }
    private func play() {
        do {
            try AVAudioSession.sharedInstance().setActive(true)
            isPlaying = audio?.play() ?? false
        } catch { self.error = error.localizedDescription }
    }
    func togglePlayback() {
        if isPlaying { audio?.pause(); isPlaying = false } else { play() }
    }
    func seek(_ value: Double) {
        position = min(max(value, 0), duration)
        audio?.currentTime = position
    }
    func next() {
        guard !tracks.isEmpty else { return }
        let candidates = tracks.indices.filter { $0 != index }
        select(shuffle ? (candidates.randomElement() ?? index) : (index + 1) % tracks.count)
    }
    func previous() {
        if position > 3 { seek(0) }
        else if !tracks.isEmpty { select((index + tracks.count - 1) % tracks.count) }
    }
    func toggleFavorite() {
        guard let id = current?.id else { return }
        if favorites.contains(id) { favorites.remove(id) } else { favorites.insert(id) }
    }
    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        Task { @MainActor [weak self] in
            guard let self else { return }
            if self.repeatTrack { self.seek(0); self.play() } else { self.next() }
        }
    }
}
