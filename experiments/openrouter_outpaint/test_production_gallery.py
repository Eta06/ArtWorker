"""Offline geometry and local report integrity checks."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image
import production_gallery as g


class GalleryTests(unittest.TestCase):
    def audit(self,matrix):
        return {'analysis_side':512,'raw_dimensions':[512,1152],'aligned':{'ssim_luminance':.95},
                'alignment':{'status':'measured','matrix_source_to_output':matrix,'rotation_degrees':0}}

    def test_centered_and_asymmetric_expansion(self):
        centered=g.region(self.audit([[1,0,0],[0,1,320]]))
        self.assertEqual(centered['margins_output_px'],{'left':0,'right':0,'top':320,'bottom':320})
        asymmetric=g.region(self.audit([[.75,0,20],[0,.75,100]]))
        self.assertEqual(asymmetric['margins_output_px'],{'left':20,'right':108,'top':100,'bottom':668})

    def test_unreliable_alignment_has_no_fake_region(self):
        audit=self.audit([[1,0,0],[0,1,320]]);audit['alignment']['status']='unavailable'
        self.assertIsNone(g.region(audit))
        audit['alignment']['status']='measured';audit['aligned']=None
        self.assertIsNone(g.region(audit))

    def test_clipped_source_stays_negative_not_fake_positive_fill(self):
        result=g.region(self.audit([[1,0,-20],[0,1,320]]))
        self.assertEqual(result['margins_output_px']['left'],-20)

    def test_snapshot_preserves_raw_marks_missing_audit_and_excludes_private_fields(self):
        with TemporaryDirectory(dir=g.ROOT/'.build') as temp:
            root=Path(temp);directory=root/'source';directory.mkdir()
            for filename,size in [('raw.png',(64,128)),('source.png',(64,64)),('input.png',(64,128))]:
                Image.new('RGB',size,'orange').save(directory/filename)
            original=root/'original.png';Image.new('RGB',(64,64),'orange').save(original)
            raw=directory/'raw.png';fingerprint=hashlib.sha256(raw.read_bytes()).hexdigest()
            receipt={'images':[{'sha256':fingerprint,'dimensions':[64,128]}],
                     'request':{'prompt':'Preserve source. Extend around it.','quality':'high'},'reported_cost_usd':.05}
            (directory/'receipt.json').write_text(json.dumps(receipt))
            source={'id':'cover-test','status':'sunburst_completed_review_pending','directory':str(directory.relative_to(g.ROOT)),
                    'file':str(original.relative_to(g.ROOT)),'artists':['Artist'],'album':'Album'}
            state={'sources':[source],'status':'teacher_pipeline_sunburst','google_jobs':[],
                   'key':'should-not-export','initial_openrouter_credit_usd':'999'}
            path=root/'queue.json';path.write_text(json.dumps(state));out=root/'gallery';out.mkdir()
            result=g.snapshot(path,out)
            self.assertEqual(result['completed'],1);self.assertEqual(result['audit_pending'],1)
            self.assertIsNone(result['rows'][0]['metrics']);self.assertIsNone(result['rows'][0]['region'])
            g.snapshot(path,out)
            self.assertEqual(hashlib.sha256(raw.read_bytes()).hexdigest(),fingerprint)
            self.assertNotIn('should-not-export',(out/'data.json').read_text())
            self.assertNotIn('initial_openrouter_credit_usd',(out/'data.json').read_text())
            self.assertEqual((out/'assets/cover-test-sunburst/raw.png').resolve(),raw)
            # A proposal is displayed separately, never promoted to an old audit.
            fidelity=root/'fidelity';fidelity.mkdir()
            baseline=fidelity/'cover-test-sunburst.json'
            baseline.write_text(json.dumps({'id':'old-audit','raw_sha256':fingerprint,
                'alignment':{'status':'unavailable'},'aligned':None}))
            old_audit=baseline.read_bytes()
            proposal=root/'proposal';proposal.mkdir()
            for filename in ('source.png','aligned.png','aligned-overlay.png','aligned-difference.png','placement.png'):
                Image.new('RGB',(64,64),'orange').save(proposal/filename)
            candidate={'id':'new-proposal','status':'review_pending','raw_sha256':fingerprint,
                'original_audit_sha256':hashlib.sha256(old_audit).hexdigest(), 'directory':str(proposal),
                'evidence':{'classification':'dense_verified'},'metrics':{'ssim_luminance':.99},
                'algorithm':'fixture','region':{'polygon':[[0,32],[64,32],[64,96],[0,96]]}}
            (out/'registration-recovery.json').write_text(json.dumps({'candidates':{'cover-test-sunburst':candidate}}))
            row=g.snapshot(path,out)['rows'][0]
            self.assertEqual(row['audit_status'],'unavailable')
            self.assertIsNone(row['metrics'])
            self.assertEqual(row['registration_recovery']['id'],'new-proposal')
            self.assertEqual(row['region'],candidate['region'])
            self.assertEqual(baseline.read_bytes(),old_audit)
            self.assertEqual(hashlib.sha256(raw.read_bytes()).hexdigest(),fingerprint)


if __name__=='__main__':unittest.main()
