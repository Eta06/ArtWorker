"""Offline duplicate decisions: encoding/scale equivalence versus different art."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from collect_unique_covers import CoverIndex, signature


class DuplicateDecisions(unittest.TestCase):
    def test_reencoded_resize_matches_but_new_art_does_not(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            y, x = np.mgrid[:512, :512]
            pixels = np.stack(((x//3+y//4)%256, (x//5+60)%256, (y//3+30)%256), axis=-1).astype('uint8')
            source = Image.fromarray(pixels)
            source.save(root/'source.png')
            source.resize((1024,1024), Image.Resampling.LANCZOS).save(root/'same.jpg',quality=92)
            index = CoverIndex()
            index.add(root/'source.png','archive:original')
            self.assertEqual(index.match(signature(root/'same.jpg'))[0], 'archive:original')
            altered = np.clip(pixels.astype('int16')+np.array([0,35,-25]),0,255).astype('uint8')
            Image.fromarray(altered).save(root/'different.png')
            self.assertEqual(index.match(signature(root/'different.png')), (None,None))
            source.transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(root/'flipped.png')
            self.assertEqual(index.match(signature(root/'flipped.png')), (None,None))

    def test_exact_pixels_match(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'source.png'
            Image.fromarray(np.random.default_rng(5).integers(0,256,(256,256,3),dtype='uint8')).save(path)
            index=CoverIndex()
            index.add(path,'same')
            self.assertEqual(index.match(signature(path)), ('same','normalized_pixels'))


if __name__ == '__main__':
    unittest.main()
