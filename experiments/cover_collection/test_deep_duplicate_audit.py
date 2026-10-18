from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase,main
import cv2
import numpy as np
from PIL import Image,ImageDraw
from deep_duplicate_audit import describe,distances,compare,features


class DeepDuplicateTests(TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        rgb=np.random.default_rng(42).integers(0,256,(512,512,3),dtype=np.uint8)
        self.art=Image.fromarray(cv2.GaussianBlur(rgb,(5,5),.8))
        draw=ImageDraw.Draw(self.art)
        for i in range(15):
            x,y=20+i*29,40+(i*73)%400
            draw.ellipse((x,y,x+18,y+18),fill=(i*13,250-i*9,50+i*7))
        self.art.save(self.root/'source.png');self.addCleanup(features.cache_clear)

    def test_reencoded_resized_art_is_retrieved_and_flagged(self):
        self.art.resize((1024,1024)).save(self.root/'encoded.jpg',quality=87)
        a,b=describe(self.root/'source.png'),describe(self.root/'encoded.jpg')
        self.assertLessEqual(distances(np.array([b['hashes']],dtype=np.uint64),a['hashes'])[0],20)
        self.assertIn('proposal',compare(self.root/'source.png',self.root/'encoded.jpg'))

    def test_recrop_is_retrieved_and_matches_by_global_geometry(self):
        self.art.crop((36,36,476,476)).resize((512,512)).save(self.root/'crop.png')
        a,b=describe(self.root/'source.png'),describe(self.root/'crop.png')
        self.assertLessEqual(distances(np.array([b['hashes']],dtype=np.uint64),a['hashes'])[0],20)
        result=compare(self.root/'source.png',self.root/'crop.png')
        self.assertEqual(result.get('proposal'),'registered_artwork_match')

    def test_same_palette_with_different_content_is_not_flagged(self):
        rgb=np.random.default_rng(91).integers(0,256,(512,512,3),dtype=np.uint8)
        Image.fromarray(cv2.GaussianBlur(rgb,(5,5),.8)).save(self.root/'other.png')
        self.assertNotIn('proposal',compare(self.root/'source.png',self.root/'other.png'))

    def test_same_art_with_large_color_shift_remains_a_review_proposal(self):
        rgb=np.asarray(self.art,dtype=np.int16)+np.array([35,0,-25])
        Image.fromarray(np.clip(rgb,0,255).astype(np.uint8)).save(self.root/'recolor.png')
        result=compare(self.root/'source.png',self.root/'recolor.png')
        self.assertIn('proposal',result)
        self.assertGreater(result['rgb_mae'],10)


if __name__=='__main__':main()
