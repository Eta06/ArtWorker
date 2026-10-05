"""Manual, condition-only logo removal for one matched lower Fill experiment.

The ROI and same-row asphalt donor are explicitly hand selected on track3.
This is a hypothesis test, not generic artwork/logo detection or removal.
The displayed source is never replaced by this edited conditioning image.
"""
from __future__ import annotations

import hashlib

SOURCE_ROI = (238, 470, 275, 509)
DONOR_ROI = (191, 470, 228, 509)
PATCH_ROI = (238, 86, 275, 125)


def remove_footer_for_condition(np, Image, source, baseline_canvas):
    if source.mode != "RGB" or source.size != (512, 512):
        raise ValueError("Expected original RGB512 track3 source")
    if baseline_canvas.mode != "RGB" or baseline_canvas.size != (512, 448):
        raise ValueError("Expected original lower RGB512x448 input")
    original = np.asarray(source).copy()
    before = np.asarray(baseline_canvas).copy()
    if not np.array_equal(before[:128], original[384:]):
        raise AssertionError("Baseline lower128 context is not the original source")
    x0, y0, x1, y1 = SOURCE_ROI
    dx0, dy0, dx1, dy1 = DONOR_ROI
    px0, py0, px1, py1 = PATCH_ROI
    donor = original[dy0:dy1, dx0:dx1].copy()
    after = before.copy()
    after[py0:py1, px0:px1] = donor
    mask = np.zeros(before.shape[:2], dtype=np.uint8)
    mask[py0:py1, px0:px1] = 255
    changed = np.any(before != after, axis=2)
    if np.any(changed & (mask == 0)):
        raise AssertionError("Condition pixels changed beyond the manual ROI")
    if not np.array_equal(before[128:], after[128:]):
        raise AssertionError("Unknown canvas changed")
    if not np.array_equal(np.asarray(source), original):
        raise AssertionError("Original source changed")
    if not np.array_equal(after[py0:py1, px0:px1], donor):
        raise AssertionError("Condition replacement is not the declared donor")
    info = {
        "manual_roi": True, "generic_logo_detector": False,
        "source_roi_xyxy": list(SOURCE_ROI), "patch_roi_xyxy": list(PATCH_ROI),
        "donor_source_roi_xyxy": list(DONOR_ROI),
        "method": "byte-exact clone of nearby same-row asphalt, 47 pixels left",
        "roi_pixels": int(np.count_nonzero(mask)), "actual_changed_pixels": int(changed.sum()),
        "condition_before_uint8_sha256": hashlib.sha256(before.tobytes()).hexdigest(),
        "condition_after_uint8_sha256": hashlib.sha256(after.tobytes()).hexdigest(),
        "roi_mask_uint8_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
        "outside_roi_byte_exact": True, "unknown_canvas_byte_exact": True,
        "original_source_unmodified": True, "display_uses_original_source": True,
        "donor_mean_rgb": donor.mean(axis=(0, 1)).tolist(),
        "donor_std_rgb": donor.std(axis=(0, 1)).tolist(),
        "baseline_roi_mean_rgb": before[py0:py1, px0:px1].mean(axis=(0, 1)).tolist(),
        "baseline_roi_std_rgb": before[py0:py1, px0:px1].std(axis=(0, 1)).tolist(),
    }
    return Image.fromarray(after), Image.fromarray(mask), info


def assemble_lower_only(np, Image, source, baseline_upper, lower_raw):
    """Full display composite; upper320 is reused, only lower was inferred."""
    if source.size != (512, 512) or baseline_upper.size != (512, 448) or lower_raw.size != (512, 448):
        raise ValueError("Unexpected source/raw geometry")
    source_before = np.asarray(source).copy()
    upper_before, lower_before = np.asarray(baseline_upper).copy(), np.asarray(lower_raw).copy()
    result = Image.new("RGB", (512, 1152))
    result.paste(baseline_upper.crop((0, 0, 512, 320)), (0, 0))
    result.paste(source, (0, 320))
    result.paste(lower_raw.crop((0, 128, 512, 448)), (0, 832))
    pixels = np.asarray(result)
    if not np.array_equal(pixels[320:832], source_before):
        raise AssertionError("Display source not exact")
    if not np.array_equal(pixels[:320], upper_before[:320]):
        raise AssertionError("Reused upper pixels changed")
    if not np.array_equal(pixels[832:], lower_before[128:]):
        raise AssertionError("New generated lower pixels changed")
    if not np.array_equal(np.asarray(lower_raw), lower_before):
        raise AssertionError("Actual lower model raw changed during assembly")
    return result
