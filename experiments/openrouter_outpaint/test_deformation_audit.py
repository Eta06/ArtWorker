"""Shape classifier must allow placement changes but detect stretch and shear."""
import math
import unittest
from deformation_audit import shape_parameters, classify


class ShapeTests(unittest.TestCase):
    def test_translation_rotation_and_uniform_scale_are_not_distortion(self):
        angle = .3; a,b = 1.4*math.cos(angle),1.4*math.sin(angle)
        p = shape_parameters([[a,-b,130],[b,a,520]])
        self.assertAlmostEqual(p['anisotropy_ratio'],1)
        self.assertEqual(classify(p,True,.5,.6),'no_global_deformation_detected')

    def test_unequal_scaling_and_shear_are_flagged(self):
        for matrix in [[[1,0,0],[0,.8,0]], [[1,.2,0],[0,1,0]]]:
            self.assertEqual(classify(shape_parameters(matrix),True,.5,7),'global_deformation_suspected')

    def test_weak_fit_cannot_certify_or_condemn(self):
        self.assertEqual(classify(shape_parameters([[1,0,0],[0,.5,0]]),False,.3,5),'inconclusive')
        self.assertIsNone(shape_parameters([[-1,0,0],[0,1,0]]))
        self.assertIsNone(shape_parameters([[0,0,0],[0,0,0]]))

    def test_small_noise_and_marginal_change(self):
        p=shape_parameters([[1,0,0],[0,1.003,0]])
        self.assertEqual(classify(p,True,.5,.6),'no_global_deformation_detected')
        p=shape_parameters([[1,0,0],[0,1.035,0]])
        self.assertEqual(classify(p,True,.5,1),'borderline')


if __name__=='__main__':unittest.main()
