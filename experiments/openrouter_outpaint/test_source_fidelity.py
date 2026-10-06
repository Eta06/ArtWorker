"""Known-answer fidelity controls, independent of any provider output."""
import unittest

import cv2
import numpy as np
from PIL import Image, ImageDraw

from source_fidelity import SIDE, expected_rect, metrics, register, square_at


def reference():
    rng = np.random.default_rng(531)
    image = Image.fromarray(rng.integers(40, 180, (SIDE, SIDE, 3), dtype=np.uint8))
    # Deterministic broad features in all four quadrants, not a repetitive texture.
    draw = ImageDraw.Draw(image)
    for i in range(80):
        x, y = rng.integers(15, 450, 2)
        r = int(rng.integers(8, 40))
        color = tuple(map(int, rng.integers(0, 255, 3)))
        draw.rectangle((int(x), int(y), int(x+r), int(y+r)), fill=color)
    return np.asarray(image)


class FidelityControls(unittest.TestCase):
    def test_identity(self):
        a = reference()
        m = metrics(a, a)
        self.assertAlmostEqual(m['ssim_luminance'], 1, places=6)
        self.assertEqual(m['mae_rgb_0_255'], 0)
        self.assertEqual(m['delta_e_76_mean'], 0)
        self.assertEqual(m['pixel_exact_on_analysis_grid_fraction'], 1)
        self.assertEqual(m['pixels_within_8_rgb_fraction'], 1)
        self.assertIsNone(m['psnr_db'])  # JSON-safe representation of infinity.

    def test_known_color_change(self):
        a = np.full((SIDE, SIDE, 3), 80, np.uint8)
        b = a.copy()
        b[:, :, 0] += 24
        m = metrics(a, b)
        self.assertEqual(m['mean_signed_rgb_delta'], [24, 0, 0])
        self.assertEqual(m['mae_rgb_0_255'], 8)
        self.assertEqual(m['pixels_over_8_rgb_fraction'], 1)
        self.assertEqual(m['pixels_within_8_rgb_fraction'], 0)
        self.assertGreater(m['delta_e_76_mean'], 0)

    def test_local_redraw_is_not_hidden_by_mean(self):
        a = reference()
        b = a.copy()
        b[128:256, 128:256] = 255
        m = metrics(a, b)
        self.assertLess(m['worst_tile_ssim'], m['ssim_luminance'])
        self.assertGreater(m['mae_rgb_0_255'], 0)

    def test_known_shift_scale_and_rotation(self):
        a = reference()
        rect = (0, 199, 512, 711)
        transform = cv2.getRotationMatrix2D((256, 256), 2, .94)
        transform[:, 2] += np.array([0, 199+14])
        output = cv2.warpAffine(a, transform, (512, 910))
        d, aligned, valid = register(a, output, rect)
        self.assertEqual(d['status'], 'measured', d)
        self.assertAlmostEqual(d['scale_relative_to_expected'], .94, delta=.008)
        self.assertAlmostEqual(d['center_shift_fraction_of_cover'][1], 14/512, delta=.008)
        self.assertAlmostEqual(abs(d['rotation_degrees']), 2, delta=.4)
        fixed = square_at(Image.fromarray(output), rect)
        # Resampling cannot reproduce the original high-frequency pixels exactly,
        # but registration must improve the known geometric perturbation.
        self.assertLess(metrics(a, aligned, valid)['mae_rgb_0_255'], metrics(a, fixed)['mae_rgb_0_255'])

    def test_alignment_failure_is_not_success(self):
        a = np.full((512, 512, 3), 90, np.uint8)
        d, aligned, valid = register(a, np.full((910, 512, 3), 90, np.uint8), (0, 199, 512, 711))
        self.assertEqual(d['status'], 'unavailable')
        self.assertIsNone(aligned)
        self.assertIsNone(valid)

    def test_asymmetric_placement_without_center_penalty(self):
        a = reference()
        transform = np.float32([[.92,0,20],[0,.92,80]])
        output = cv2.warpAffine(a, transform, (512,910))
        rect = (0,199,512,711)
        bounded, _, _ = register(a, output, rect)
        self.assertEqual(bounded['status'], 'unavailable')
        free, aligned, valid = register(a, output, rect, layout_free=True)
        self.assertEqual(free['status'], 'measured', free)
        self.assertAlmostEqual(free['scale_relative_to_expected'], .92, delta=.01)
        self.assertAlmostEqual(free['matrix_source_to_output'][0][2], 20, delta=1)
        self.assertAlmostEqual(free['matrix_source_to_output'][1][2], 80, delta=1)
        self.assertGreater(valid.mean(), .99)
        fixed = square_at(Image.fromarray(output), rect)
        self.assertLess(metrics(a, aligned, valid)['mae_rgb_0_255'], metrics(a, fixed)['mae_rgb_0_255'])

    def test_geometry_uses_receipt_and_actual_dimensions(self):
        g = {'canvas_dimensions':[720, 1280], 'source_rect_xyxy':[0,280,720,1000]}
        self.assertEqual(expected_rect(g, (864,1536)), (0,336,864,1200))
        rect = expected_rect(g, (768,1376))
        self.assertEqual(rect[1], 301)
        self.assertEqual(rect[3], 1075)

    def test_overlap_excludes_hidden_source(self):
        a = reference()
        b = a.copy()
        b[:, :20] = 0
        mask = np.ones((512,512), bool)
        mask[:, :20] = False
        m = metrics(a,b,mask)
        self.assertAlmostEqual(m['evaluated_fraction'], 492/512)
        self.assertEqual(m['mae_rgb_0_255'], 0)
        self.assertAlmostEqual(m['ssim_luminance'], 1, places=5)


if __name__ == '__main__':
    unittest.main()
