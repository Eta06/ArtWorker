"""Known-answer fidelity controls, independent of any provider output."""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image, ImageDraw

from source_fidelity import SIDE, expected_rect, metrics, register, square_at
from benchmark import ROOT, prepare, request_json


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
    def test_complete_json_does_not_wait_for_transport_eof(self):
        class Response:
            headers = {'x-generation-id':'known-id'}
            chunks = iter([b'{"data": {"nested": {}}',b', "usage": {"cost": 0.04}}'])
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def read1(self,size):
                try: return next(self.chunks)
                except StopIteration: raise TimeoutError('relay never closes')
        with patch('benchmark.urllib.request.urlopen',return_value=Response()):
            data,headers = request_json('images','placeholder',{'model':'test'})
        self.assertEqual(data['usage']['cost'],.04)
        self.assertEqual(headers['x-generation-id'],'known-id')

    def test_partial_response_timeout_retains_generation_id(self):
        class Response:
            headers = {'x-generation-id':'paid-request-id'}
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def read1(self,size): raise TimeoutError('incomplete response')
        with patch('benchmark.urllib.request.urlopen',return_value=Response()):
            with self.assertRaises(TimeoutError) as raised:
                request_json('images','placeholder',{'model':'test'})
        self.assertEqual(raised.exception.response_headers['x-generation-id'],'paid-request-id')

    def test_custom_cover_reproducible_crop_and_provenance(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.build') as folder:
            folder = Path(folder)
            source = Image.fromarray(np.random.default_rng(92).integers(0,256,(29,37,3),dtype=np.uint8))
            path = folder/'new-cover.png'
            source.save(path)
            geometry = prepare('cover01', folder, path)
            self.assertEqual(geometry['original_file'],str(path.relative_to(ROOT)))
            self.assertEqual(geometry['crop_xyxy'],[4,0,33,29])
            expected = source.crop((4,0,33,29)).resize((720,720),Image.Resampling.LANCZOS)
            with Image.open(folder/'source.png') as actual:
                self.assertTrue(np.array_equal(np.asarray(actual),np.asarray(expected)))
            with Image.open(folder/'input.png') as canvas:
                self.assertTrue(np.array_equal(np.asarray(canvas.crop((0,280,720,1000))),np.asarray(expected)))

    def test_reject_external_source_and_unsafe_identifier(self):
        with tempfile.TemporaryDirectory() as external:
            with self.assertRaisesRegex(ValueError,'inside repository'):
                prepare('cover01',ROOT/'.build',Path(external)/'cover.png')
        with self.assertRaisesRegex(ValueError,'Unsafe'):
            prepare('../cover',ROOT/'.build')

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
