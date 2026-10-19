"""Synthetic feature images validate the actual registration, not model quality."""
import hashlib
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image
from deformation_audit import measure


class ImageTests(unittest.TestCase):
    def run_fixture(self, sx, sy, blank=False, bend=False, tone=False):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);rng=np.random.default_rng(819)
            image=np.full((720,720),100,np.uint8)
            if not blank:
                for _ in range(450):
                    x,y=rng.integers(15,705,2);radius=int(rng.integers(2,16))
                    cv2.circle(image,(int(x),int(y)),radius,int(rng.integers(0,256)),-1)
                for y in range(35,700,90):cv2.putText(image,str(y),(45,y),cv2.FONT_HERSHEY_SIMPLEX,.9,230,2)
            Image.fromarray(image).save(root/'source.png')
            angle=.07;c,s=np.cos(angle),np.sin(angle)
            matrix=np.float32([[sx*c,-sy*s,130],[sx*s,sy*c,230]])
            target=image
            if tone:
                target=np.uint8(np.clip(image.astype(float)*.8+30,0,255))
            if bend:
                y,x=np.indices(image.shape,dtype=np.float32)
                dx=20*np.exp(-((x-480)**2+(y-380)**2)/(2*130**2))
                target=cv2.remap(image,x+dx,y,cv2.INTER_LINEAR)
            raw=cv2.warpAffine(target,matrix,(1100,1600),borderValue=40)
            Image.fromarray(raw).save(root/'raw.png')
            return measure({'source_file':str(root/'source.png'),'raw_file':str(root/'raw.png'),
                            'raw_sha256':hashlib.sha256((root/'raw.png').read_bytes()).hexdigest()})

    def test_uniform_scale_rotation_and_shift_allowed(self):
        r=self.run_fixture(1,1)
        self.assertEqual(r['status'],'no_global_deformation_detected',r)

    def test_actual_unequal_scaling_is_flagged(self):
        r=self.run_fixture(1,.8)
        self.assertEqual(r['status'],'global_deformation_suspected',r)
        self.assertAlmostEqual(r['shape']['anisotropy_ratio'],1.25,delta=.02)

    def test_blank_image_is_not_certified_clean(self):
        self.assertEqual(self.run_fixture(1,1,True)['status'],'inconclusive')

    def test_local_bend_does_not_pass_as_globally_preserved(self):
        result=self.run_fixture(1,1,bend=True)
        self.assertIn(result['status'],['local_structure_change_suspected','global_deformation_suspected'],result)

    def test_tone_change_is_not_called_a_bend(self):
        result=self.run_fixture(1,1,tone=True)
        self.assertEqual(result['status'],'no_global_deformation_detected',result)


if __name__=='__main__':unittest.main()
