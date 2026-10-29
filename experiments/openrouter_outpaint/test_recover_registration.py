"""Offline recovery controls: plausible features alone never certify a region."""
import unittest
import tempfile
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from recover_registration import assess, refine_assessment, needs_recovery, center_reference
from source_fidelity import sha


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

    def test_skipped_scope_includes_missing_notes_but_not_stale_or_rejected(self):
        row={'audit_status':'unavailable','raw_sha256':'image'}
        review={'raw_sha256':'image','decision':'pending','reason':''}
        self.assertTrue(needs_recovery(row,review,'skipped'))
        self.assertFalse(needs_recovery(row,review,'noted'))
        self.assertFalse(needs_recovery(row,{**review,'decision':'rejected'},'skipped'))
        self.assertFalse(needs_recovery(row,{**review,'raw_sha256':'old'},'skipped'))

    def test_pixel_refinement_corrects_a_nearby_feature_seed(self):
        seed={**self.diagnostic,'method':'orb','matrix_source_to_output':[[1,0,2],[0,1,201]]}
        result=refine_assessment(self.reference,self.output,[seed])
        self.assertIsNotNone(result)
        self.assertEqual(result['evidence']['classification'],'dense_verified')
        self.assertLess(abs(result['diagnostic']['matrix_source_to_output'][0][2]),.2)
        self.assertGreater(result['evidence']['informative_luminance_correlation'],.99)

    def test_refinement_does_not_certify_unrelated_pixels_or_flat_art(self):
        seed={**self.diagnostic,'method':'orb'}
        unrelated=np.random.default_rng(29).integers(0,256,self.output.shape,dtype=np.uint8)
        self.assertIsNone(refine_assessment(self.reference,unrelated,[seed,{**seed,'method':'sift'}]))
        self.assertIsNone(refine_assessment(np.full_like(self.reference,110),self.output,[seed]))

    def test_manual_center_is_square_and_has_no_fake_matching_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);asset=root/'asset';asset.mkdir();out=root/'out';out.mkdir()
            Image.new('RGB',(64,96),'orange').save(asset/'raw.png')
            Image.new('RGB',(64,64),'blue').save(asset/'source.png')
            row={'dimensions':[64,96],'raw_sha256':sha(asset/'raw.png')}
            audit={'raw_sha256':row['raw_sha256'],'source_png_sha256':sha(asset/'source.png')}
            result=center_reference(row,audit,asset,out)
            self.assertEqual(result['region']['polygon'],[[0,16],[64,16],[64,80],[0,80]])
            self.assertEqual(result['status'],'review_pending')
            self.assertFalse(result['evidence']['is_source_match'])
            self.assertIsNone(result['metrics'])
            self.assertFalse((Path(result['directory'])/'aligned.png').exists())
            self.assertEqual(sha(asset/'raw.png'),row['raw_sha256'])


if __name__ == '__main__':
    unittest.main()
