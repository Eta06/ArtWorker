"""Preserve the source square and harmonize only the generated border colors.

This CPU finishing step restores the exterior color collar used in the Cinar
preview. It does not repair invented objects or improve model semantics.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/qwen"))
from run_boundary_compositor import compose, harmonic_transfer, metrics


def finish(raw, source, *, mode="color", strip=96):
    if raw.shape != (1152, 512, 3) or source.shape != (512, 512, 3):
        raise ValueError("Finishing requires the shared 512x1152 canvas and 512x512 source")
    if mode not in ("exact", "color") or not 1 <= strip <= 320:
        raise ValueError("Invalid finishing mode or exterior strip width")
    start = time.monotonic()
    exact = compose(raw, source)
    # Leave already coherent near-black borders intact, including their texture.
    source_edges = np.concatenate((source[:4], source[-4:]), axis=0)
    decoded_edges = np.concatenate((raw[320:324], raw[828:832]), axis=0)
    near_black = bool(np.percentile(source_edges, 99) <= 8 and
                      np.percentile(decoded_edges, 99) <= 8)
    applied = mode == "color" and not near_black
    if applied:
        corrected, residuals = harmonic_transfer(raw, source, strip=strip,
                                                sigma=6, residual_cap=24)
        output = compose(corrected, source)
    else:
        output, residuals = exact.copy(), None
    checks = metrics(output, exact, source, raw, strip)
    if not checks["source_pixels_exact"] or not checks["untouched_outer_pixels_exact"]:
        raise AssertionError("Finishing changed protected source or distant exterior")
    return output, {
        "mode": mode, "applied": applied, "near_black_bypass": near_black,
        "strip_pixels": strip, "sigma_x": 6, "residual_cap_255": 24,
        "color_residual": residuals, "checks": checks,
        "seconds": time.monotonic() - start,
        "limitation": "Color correction only; incorrect objects, clothing and geometry remain."
    }
