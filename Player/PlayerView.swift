import SwiftUI

private let canvas = Color(red: 0.045, green: 0.04, blue: 0.055)
private let pearl = Color(red: 0.98, green: 0.965, blue: 0.95)

struct PlayerView: View {
    @EnvironmentObject var player: MusicPlayer
    @State private var showQueue = false
    @State private var seeking = false
    @State private var seekPosition = 0.0
    private var track: Track? { player.current }
    private var position: Double { seeking ? seekPosition : player.position }

    var body: some View {
        ZStack {
            atmosphere
                .ignoresSafeArea()
            GeometryReader { geometry in
                let coverSide = min(geometry.size.width - 56, max(180, geometry.size.height - 365))
                VStack(spacing: 0) {
                    header
                    Spacer(minLength: 16)
                    artwork(size: coverSide)
                    title.padding(.top, 30)
                    timeline.padding(.top, 14)
                    controls.padding(.top, 14)
                    Spacer(minLength: 24)
                    volume
                    queueShortcut.padding(.top, 5)
                }
                .padding(.horizontal, 28)
                .padding(.top, 4)
                .padding(.bottom, 6)
                .frame(width: geometry.size.width, height: geometry.size.height)
            }
        }
        .foregroundStyle(pearl)
        .sheet(isPresented: $showQueue) {
            queue
                .presentationDetents([.medium, .large])
                .presentationDragIndicator(.visible)
                .presentationCornerRadius(30)
        }
        .alert("Playback error", isPresented: Binding(get: { player.error != nil }, set: { if !$0 { player.error = nil } })) {
            Button("OK") { player.error = nil }
        } message: { Text(player.error ?? "") }
    }

    // This geometry is outside the safe area: status bar and player share one canvas.
    private var atmosphere: some View {
        GeometryReader { geometry in
            ZStack {
                canvas
                if let image = track?.artwork {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFill()
                        .frame(width: geometry.size.width + 160, height: geometry.size.height + 160)
                        .blur(radius: 65, opaque: true)
                        .saturation(0.85)
                        .position(x: geometry.size.width / 2, y: geometry.size.height * 0.32)
                        .opacity(0.65)
                }
                LinearGradient(
                    stops: [.init(color: .black.opacity(0.25), location: 0),
                            .init(color: canvas.opacity(0.18), location: 0.35),
                            .init(color: canvas.opacity(0.88), location: 0.76),
                            .init(color: canvas, location: 1)],
                    startPoint: .top, endPoint: .bottom
                )
            }
            .frame(width: geometry.size.width, height: geometry.size.height)
            .clipped()
        }
        .animation(.easeInOut(duration: 0.65), value: track?.id)
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }

    private var header: some View {
        HStack {
            Image(systemName: "waveform")
                .font(.system(size: 17, weight: .regular))
                .foregroundStyle(pearl.opacity(0.65))
                .frame(width: 44, height: 44, alignment: .leading)
            Spacer()
            Text("artworker")
                .font(.system(size: 16, weight: .medium, design: .rounded))
                .tracking(0.3)
                .foregroundStyle(pearl.opacity(0.85))
            Spacer()
            Button { showQueue = true } label: {
                Image(systemName: "ellipsis")
                    .font(.system(size: 20, weight: .medium))
                    .frame(width: 44, height: 44, alignment: .trailing)
                    .contentShape(Rectangle())
            }
            .accessibilityLabel("Open queue")
        }
        .frame(height: 44)
    }

    private func artwork(size: CGFloat) -> some View {
        ZStack {
            if let image = track?.artwork {
                Image(uiImage: image).resizable().scaledToFit()
            } else {
                Rectangle().fill(.white.opacity(0.06))
                Image(systemName: "music.note").font(.system(size: 64)).foregroundStyle(pearl.opacity(0.6))
            }
        }
        .frame(width: size, height: size)
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        .shadow(color: .black.opacity(0.3), radius: 24, x: 0, y: 20)
        .frame(maxWidth: .infinity)
        .accessibilityLabel("Album artwork")
    }

    private var title: some View {
        HStack(alignment: .center, spacing: 12) {
            VStack(alignment: .leading, spacing: 7) {
                Text(track?.title ?? "No tracks")
                    .font(.system(size: 28, weight: .semibold))
                    .tracking(-0.65)
                    .lineLimit(1)
                    .minimumScaleFactor(0.7)
                Text(track?.artist ?? "")
                    .font(.system(size: 14, weight: .regular))
                    .foregroundStyle(pearl.opacity(0.56))
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            Button { player.toggleFavorite() } label: {
                Image(systemName: player.favorites.contains(track?.id ?? "") ? "heart.fill" : "heart")
                    .font(.system(size: 20, weight: .regular))
                    .foregroundStyle(pearl.opacity(player.favorites.contains(track?.id ?? "") ? 1 : 0.65))
                    .frame(width: 44, height: 44, alignment: .trailing)
                    .contentShape(Rectangle())
            }
            .accessibilityLabel("Favorite")
            .accessibilityValue(player.favorites.contains(track?.id ?? "") ? "Saved" : "Not saved")
        }
    }

    private var timeline: some View {
        VStack(spacing: -4) {
            SlimSlider(
                value: Binding(get: { position }, set: { seekPosition = $0 }),
                range: 0...max(player.duration, 1),
                tint: pearl,
                thumbSize: 6
            ) { editing in
                if editing { seekPosition = player.position; seeking = true }
                else { player.seek(seekPosition); seeking = false }
            }
            .accessibilityLabel("Playback position")
            .accessibilityValue(time(position) + " of " + time(player.duration))
            HStack {
                Text(time(position))
                Spacer()
                Text("−" + time(max(0, player.duration - position)))
            }
            .font(.system(size: 11, weight: .regular))
            .monospacedDigit()
            .foregroundStyle(pearl.opacity(0.45))
            .accessibilityHidden(true)
        }
    }

    private var controls: some View {
        HStack(spacing: 0) {
            iconButton("shuffle", label: "Shuffle", active: player.shuffle) { player.shuffle.toggle() }
            Spacer(minLength: 4)
            iconButton("backward.end.fill", label: "Previous track", size: 26) { player.previous() }
            Spacer(minLength: 8)
            Button { player.togglePlayback() } label: {
                Image(systemName: player.isPlaying ? "pause.fill" : "play.fill")
                    .font(.system(size: 27, weight: .semibold))
                    .offset(x: player.isPlaying ? 0 : 2)
                    .foregroundStyle(canvas)
                    .frame(width: 74, height: 74)
                    .background(pearl, in: Circle())
                    .shadow(color: .black.opacity(0.12), radius: 12, y: 8)
            }
            .accessibilityLabel(player.isPlaying ? "Pause" : "Play")
            Spacer(minLength: 8)
            iconButton("forward.end.fill", label: "Next track", size: 26) { player.next() }
            Spacer(minLength: 4)
            iconButton(player.repeatTrack ? "repeat.1" : "repeat", label: "Repeat track", active: player.repeatTrack) { player.repeatTrack.toggle() }
        }
        .frame(height: 78)
    }

    private var volume: some View {
        HStack(spacing: 12) {
            Image(systemName: "speaker.fill")
            SlimSlider(
                value: Binding(get: { Double(player.volume) }, set: { player.volume = Float($0) }),
                range: 0...1,
                tint: pearl.opacity(0.55),
                trackOpacity: 0.1,
                thumbSize: 5
            )
            .accessibilityLabel("Volume")
            .accessibilityValue("\(Int(player.volume * 100)) percent")
            Image(systemName: "speaker.wave.2.fill")
        }
        .font(.system(size: 10, weight: .regular))
        .foregroundStyle(pearl.opacity(0.4))
        .frame(maxWidth: 250)
        .frame(maxWidth: .infinity)
    }

    private var queueShortcut: some View {
        Button { showQueue = true } label: {
            HStack(spacing: 8) {
                Image(systemName: "list.bullet")
                    .font(.system(size: 13, weight: .medium))
                Text("Up next")
                    .font(.system(size: 12, weight: .medium))
                Image(systemName: "chevron.up")
                    .font(.system(size: 8, weight: .semibold))
                    .padding(.leading, 2)
            }
            .foregroundStyle(pearl.opacity(0.6))
            .padding(.horizontal, 18)
            .frame(height: 38)
            .background(.white.opacity(0.045), in: Capsule())
            .frame(height: 44)
        }
        .accessibilityLabel("Open queue")
    }

    private var queue: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 8) {
                    ForEach(Array(player.tracks.enumerated()), id: \.element.id) { index, track in
                        Button {
                            player.select(index)
                            showQueue = false
                        } label: {
                            HStack(spacing: 15) {
                                if let artwork = track.artwork {
                                    Image(uiImage: artwork).resizable().scaledToFit()
                                        .frame(width: 56, height: 56)
                                        .clipShape(RoundedRectangle(cornerRadius: 8))
                                }
                                VStack(alignment: .leading, spacing: 6) {
                                    Text(track.title).font(.system(size: 15, weight: .semibold))
                                    Text(track.artist).font(.system(size: 12))
                                        .foregroundStyle(pearl.opacity(0.5)).lineLimit(1)
                                }
                                .frame(maxWidth: .infinity, alignment: .leading)
                                if index == player.index {
                                    Image(systemName: "waveform").font(.system(size: 17)).foregroundStyle(pearl)
                                }
                            }
                            .foregroundStyle(pearl)
                            .padding(12)
                            .background(.white.opacity(index == player.index ? 0.07 : 0.025), in: RoundedRectangle(cornerRadius: 16))
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                    }
                }
                .padding(.horizontal, 20)
                .padding(.top, 12)
            }
            .background(canvas)
            .navigationTitle("Up next")
            .navigationBarTitleDisplayMode(.inline)
            .toolbarBackground(canvas, for: .navigationBar)
            .toolbarBackground(.visible, for: .navigationBar)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button { showQueue = false } label: {
                        Image(systemName: "xmark").font(.system(size: 12, weight: .semibold))
                    }
                    .tint(pearl)
                    .accessibilityLabel("Close queue")
                }
            }
        }
        .preferredColorScheme(.dark)
    }

    private func iconButton(_ symbol: String, label: String, size: CGFloat = 17, active: Bool = false, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            VStack(spacing: 4) {
                Image(systemName: symbol)
                    .font(.system(size: size, weight: .regular))
                    .foregroundStyle(pearl.opacity(active || size > 17 ? 0.95 : 0.4))
                if size <= 17 {
                    Circle().fill(active ? pearl : .clear).frame(width: 3, height: 3)
                }
            }
            .frame(width: 44, height: 44)
            .contentShape(Rectangle())
        }
        .accessibilityLabel(label)
        .accessibilityValue(size <= 17 ? (active ? "On" : "Off") : "")
    }

    private func time(_ value: Double) -> String {
        String(format: "%d:%02d", Int(value) / 60, Int(value) % 60)
    }
}
