"""Offline recovery controls: plausible features alone never certify a region."""
import unittest
import cv2
import numpy as np
from recover_registration import assess


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        # Sparse artwork with strong interior texture and a known global transform.
        self.reference = np.full((512, 512, 3), 110, np.uint8)
        rng = np.random.default_rng(8)
        self.reference[180:330, 180:330] = rng.integers(0, 256, (150, 150, 3), dtype=np.uint8)
        self.diagnostic = dict(matrix_source_to_output=[[1, 0, 0], [0, 1, 200]],
                               inliers=60, inlier_fraction=.9, median_inlier_residual_px=.2,
                               source_feature_hull_fraction=.05, quadrants=2)
        self.output = np.full((900, 512, 3), 110, np.uint8)
        self.output[200:712] = self.reference

    def test_sparse_but_identical_informative_content_supports_proposal(self):
        result = assess(self.reference, self.output, self.diagnostic)
        self.assertEqual(result['evidence']['classification'], 'dense_verified')
        self.assertGreater(result['evidence']['informative_ssim'], .99)
        self.assertEqual(self.diagnostic['source_feature_hull_fraction'], .05)

    def test_unrelated_pixels_fail_despite_strong_feature_claims(self):
        unrelated = np.random.default_rng(9).integers(0, 256, self.output.shape, dtype=np.uint8)
        self.assertIsNone(assess(self.reference, unrelated, self.diagnostic, self.diagnostic))

    def test_flat_art_has_no_independent_informative_evidence(self):
        flat = np.full_like(self.reference, 110)
        self.assertIsNone(assess(flat, self.output, self.diagnostic, self.diagnostic))

    def test_clipped_source_is_explicitly_partial(self):
        diagnostic = {**self.diagnostic, 'matrix_source_to_output': [[1, 0, -90], [0, 1, 200]]}
        output = cv2.warpAffine(self.reference, np.float32(diagnostic['matrix_source_to_output']), (512, 900))
        result = assess(self.reference, output, diagnostic)
        self.assertEqual(result['evidence']['classification'], 'partial_reference')
        self.assertAlmostEqual(result['evidence']['visible_source_fraction'], 422/512, places=3)

    def test_invalid_and_severely_clipped_matrices_are_rejected(self):
        for matrix in ([[1, 0, -200], [0, 1, 200]], [[float('nan'), 0, 0], [0, 1, 0]], []):
            self.assertIsNone(assess(self.reference, self.output, {**self.diagnostic, 'matrix_source_to_output': matrix}))


if __name__ == '__main__':
    unittest.main()
