import SwiftUI

/// A fine slider rail with a full-size touch target for tap and drag scrubbing.
struct SlimSlider: View {
    @Binding var value: Double
    let range: ClosedRange<Double>
    var tint: Color = .white
    var trackOpacity: Double = 0.16
    var thumbSize: CGFloat = 7
    var onEditingChanged: (Bool) -> Void = { _ in }

    @Environment(\.layoutDirection) private var layoutDirection
    @Environment(\.isEnabled) private var isEnabled
    @State private var dragValue: Double?

    private let touchHeight: CGFloat = 44
    private let railHeight: CGFloat = 2

    init(
        value: Binding<Double>,
        range: ClosedRange<Double>,
        tint: Color = .white,
        trackOpacity: Double = 0.16,
        thumbSize: CGFloat = 7,
        onEditingChanged: @escaping (Bool) -> Void = { _ in }
    ) {
        self._value = value
        self.range = range
        self.tint = tint
        self.trackOpacity = trackOpacity
        self.thumbSize = thumbSize
        self.onEditingChanged = onEditingChanged
    }

    private var isEditing: Bool { dragValue != nil }
    private var restingThumbSize: CGFloat { max(thumbSize, 0) }
    private var activeThumbSize: CGFloat { max(restingThumbSize + 4, restingThumbSize * 1.5) }
    private var displayedValue: Double { clamped(dragValue ?? value) }

    var body: some View {
        GeometryReader { geometry in
            let inset = activeThumbSize / 2
            let railWidth = max(geometry.size.width - inset * 2, 0)
            let progress = fraction(for: displayedValue)
            let filledWidth = railWidth * progress
            let isRTL = layoutDirection == .rightToLeft
            let thumbX = isRTL
                ? geometry.size.width - inset - filledWidth
                : inset + filledWidth

            ZStack {
                Capsule()
                    .fill(tint.opacity(min(max(trackOpacity, 0), 1)))
                    .frame(width: railWidth, height: railHeight)
                    .position(x: geometry.size.width / 2, y: touchHeight / 2)

                Capsule()
                    .fill(tint)
                    .frame(width: filledWidth, height: railHeight)
                    .position(
                        x: isRTL ? geometry.size.width - inset - filledWidth / 2 : inset + filledWidth / 2,
                        y: touchHeight / 2
                    )

                Circle()
                    .fill(tint)
                    .frame(
                        width: isEditing ? activeThumbSize : restingThumbSize,
                        height: isEditing ? activeThumbSize : restingThumbSize
                    )
                    .position(x: thumbX, y: touchHeight / 2)
                    .animation(.easeOut(duration: 0.15), value: isEditing)
            }
            .frame(width: geometry.size.width, height: touchHeight)
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { gesture in
                        guard isEnabled else { return }
                        beginEditing()
                        updateValue(at: gesture.location.x, width: geometry.size.width)
                    }
                    .onEnded { gesture in
                        guard isEnabled else {
                            finishEditing()
                            return
                        }
                        beginEditing()
                        updateValue(at: gesture.location.x, width: geometry.size.width)
                        finishEditing()
                    }
            )
        }
        .frame(height: touchHeight)
        .opacity(isEnabled ? 1 : 0.45)
        .accessibilityElement(children: .ignore)
        .accessibilityValue(Text("\(Int((fraction(for: displayedValue) * 100).rounded())) percent"))
        .accessibilityAdjustableAction { direction in
            guard isEnabled else { return }
            let step = (range.upperBound - range.lowerBound) / 100
            guard step.isFinite, step > 0 else { return }
            let adjustment: Double
            switch direction {
            case .increment: adjustment = step
            case .decrement: adjustment = -step
            @unknown default: return
            }
            let nextValue = clamped(displayedValue + adjustment)
            beginEditing()
            dragValue = nextValue
            value = nextValue
            finishEditing()
        }
        .onDisappear { finishEditing() }
    }

    private func beginEditing() {
        guard dragValue == nil else { return }
        dragValue = clamped(value)
        // The owner may freeze its playback clock here. Notify it before writing
        // the first scrub value so that its initialization cannot overwrite it.
        onEditingChanged(true)
    }

    private func updateValue(at location: CGFloat, width: CGFloat) {
        let inset = activeThumbSize / 2
        let railWidth = width - inset * 2
        guard railWidth > 0 else { return }
        let position = layoutDirection == .rightToLeft ? width - location : location
        let progress = min(max(Double((position - inset) / railWidth), 0), 1)
        let nextValue = clamped(range.lowerBound + progress * (range.upperBound - range.lowerBound))
        dragValue = nextValue
        value = nextValue
    }

    private func finishEditing() {
        guard dragValue != nil else { return }
        // Keep the local scrub value visible until the owner has committed it.
        onEditingChanged(false)
        dragValue = nil
    }

    private func clamped(_ candidate: Double) -> Double {
        guard candidate.isFinite else { return range.lowerBound }
        return min(max(candidate, range.lowerBound), range.upperBound)
    }

    private func fraction(for candidate: Double) -> CGFloat {
        let span = range.upperBound - range.lowerBound
        guard span.isFinite, span > 0 else { return 0 }
        return CGFloat((clamped(candidate) - range.lowerBound) / span)
    }
}
