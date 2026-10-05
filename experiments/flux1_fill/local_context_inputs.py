"""Exact adjacent-source strips for two independent local FLUX Fill calls."""
from __future__ import annotations

import hashlib

PATCH_SIZE = (512, 448)
FINAL_SIZE = (512, 1152)
SOURCE_RECT = [0, 320, 512, 832]
PATCHES = {
    "upper": {"source_crop_xyxy": [0, 0, 512, 128], "known_patch_rect_xyxy": [0, 320, 512, 448],
              "generated_patch_rect_xyxy": [0, 0, 512, 320], "generated_final_rect_xyxy": [0, 0, 512, 320]},
    "lower": {"source_crop_xyxy": [0, 384, 512, 512], "known_patch_rect_xyxy": [0, 0, 512, 128],
              "generated_patch_rect_xyxy": [0, 128, 512, 448], "generated_final_rect_xyxy": [0, 832, 512, 1152]},
}


def prepare_patches(np, Image, source):
    if source.mode != "RGB" or source.size != (512, 512):
        raise ValueError("Local context requires the unchanged RGB512 source square")
    original = np.asarray(source).copy()
    results = {}
    for name, spec in PATCHES.items():
        crop = source.crop(spec["source_crop_xyxy"])
        canvas = Image.new("RGB", PATCH_SIZE, (128, 128, 128))
        rect = spec["known_patch_rect_xyxy"]
        canvas.paste(crop, (rect[0], rect[1]))
        mask = Image.new("L", PATCH_SIZE, 255)
        mask.paste(0, tuple(rect))
        canvas_array, mask_array = np.asarray(canvas), np.asarray(mask)
        if not np.array_equal(canvas_array[rect[1]:rect[3], rect[0]:rect[2]], np.asarray(crop)):
            raise AssertionError("Source strip changed while placing adjacent context")
        if int((mask_array == 0).sum()) != 512 * 128 or int((mask_array == 255).sum()) != 512 * 320:
            raise AssertionError("Local patch mask polarity/geometry changed")
        info = {**spec, "patch_size": list(PATCH_SIZE), "target_tokens": 896,
                "known_pixels": 512 * 128, "unknown_pixels": 512 * 320,
                "known_tokens": 256, "unknown_tokens": 640,
                "mask_convention": "white255 regenerates; black0 preserves",
                "source_strip_uint8_sha256": hashlib.sha256(np.asarray(crop).tobytes()).hexdigest(),
                "canvas_uint8_sha256": hashlib.sha256(canvas_array.tobytes()).hexdigest(),
                "mask_uint8_sha256": hashlib.sha256(mask_array.tobytes()).hexdigest(),
                "source_strip_byte_exact": True, "no_rotation_flip_or_resize": True}
        results[name] = {"canvas": canvas, "mask": mask, "source_strip": crop, "metadata": info}
    if not np.array_equal(np.asarray(source), original):
        raise AssertionError("Preparing local contexts mutated the source")
    return results


def assemble_composite(np, Image, source, raw_upper, raw_lower):
    if source.mode != "RGB" or source.size != (512, 512):
        raise ValueError("Expected unchanged RGB512 source")
    if any(image.mode != "RGB" or image.size != PATCH_SIZE for image in (raw_upper, raw_lower)):
        raise ValueError("Both raw calls must retain their original RGB512x448 dimensions")
    upper_before, lower_before = np.asarray(raw_upper).copy(), np.asarray(raw_lower).copy()
    composite = Image.new("RGB", FINAL_SIZE)
    composite.paste(raw_upper.crop((0, 0, 512, 320)), (0, 0))
    composite.paste(source, (0, 320))
    composite.paste(raw_lower.crop((0, 128, 512, 448)), (0, 832))
    pixels = np.asarray(composite)
    if not np.array_equal(pixels[320:832], np.asarray(source)):
        raise AssertionError("Assembled source is not byte-exact")
    if not np.array_equal(pixels[:320], upper_before[:320]) or not np.array_equal(pixels[832:], lower_before[128:]):
        raise AssertionError("Assembly changed generated external pixels")
    if not np.array_equal(np.asarray(raw_upper), upper_before) or not np.array_equal(np.asarray(raw_lower), lower_before):
        raise AssertionError("Assembly changed the retained raw crops")
    return composite


def assemble_raw_context_diagnostic(Image, source, raw_upper, raw_lower):
    """Retain learned128 context edges; central256 is original, not model output."""
    diagnostic = Image.new("RGB", FINAL_SIZE)
    diagnostic.paste(raw_upper, (0, 0))
    diagnostic.paste(source.crop((0, 128, 512, 384)), (0, 448))
    diagnostic.paste(raw_lower, (0, 704))
    return diagnostic
