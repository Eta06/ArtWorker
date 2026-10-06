"""Check source protection and color-drift repair without loading model weights."""
import unittest

import numpy as np

from finish_output import finish


class FinishTests(unittest.TestCase):
    def test_decoder_color_drift_is_repaired_without_touching_source_or_far_exterior(self):
        raw = np.full((1152, 512, 3), 100, dtype=np.uint8)
        source = np.full((512, 512, 3), 110, dtype=np.uint8)
        out, record = finish(raw, source)
        self.assertTrue(record["applied"])
        np.testing.assert_array_equal(out[320:832], source)
        np.testing.assert_array_equal(out[:224], raw[:224])
        np.testing.assert_array_equal(out[928:], raw[928:])
        self.assertLess(abs(float(out[319].mean()) - 110), 2)
        self.assertLess(abs(float(out[832].mean()) - 110), 2)

    def test_additive_color_repair_keeps_horizontal_texture(self):
        pattern = np.tile(np.array([60, 90], dtype=np.uint8), 256)
        raw = np.broadcast_to(pattern[None, :, None], (1152, 512, 3)).copy()
        source = raw[320:832].copy() + 10
        out, _ = finish(raw, source)
        np.testing.assert_array_equal(out[300, 1::2].astype(int) - out[300, ::2], 30)

    def test_coherent_black_surround_is_unchanged(self):
        raw = np.zeros((1152, 512, 3), dtype=np.uint8)
        source = np.zeros((512, 512, 3), dtype=np.uint8)
        out, record = finish(raw, source)
        self.assertFalse(record["applied"])
        np.testing.assert_array_equal(out, raw)

    def test_exact_mode_reproduces_hard_paste(self):
        raw = np.full((1152, 512, 3), 100, dtype=np.uint8)
        source = np.full((512, 512, 3), 110, dtype=np.uint8)
        expected = raw.copy()
        expected[320:832] = source
        out, record = finish(raw, source, mode="exact")
        np.testing.assert_array_equal(out, expected)
        self.assertFalse(record["applied"])

    def test_wrong_geometry_is_rejected_instead_of_pasting_the_wrong_rectangle(self):
        with self.assertRaises(ValueError):
            finish(np.zeros((1000, 1000, 3)), np.zeros((512, 512, 3)))


if __name__ == "__main__":
    unittest.main()
