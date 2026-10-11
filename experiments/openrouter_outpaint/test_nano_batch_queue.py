"""Offline direct-trial budget, pause and duplicate-spend checks."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import budget_queue as q
import nano_batch_queue as n


class NanoTrialTests(unittest.TestCase):
    def state(self):
        return {'prompt_sha256':q.PROMPT_HASH,'google_jobs':[], 'sources':[
            {'id':'old','file':'old.png','sha256':'old','status':'sunburst_completed_review_pending'},
            {'id':'a','file':'a.png','sha256':'a','status':'pending'},
            {'id':'same','file':'same.png','sha256':'a','status':'pending'},
            {'id':'b','file':'b.png','sha256':'b','status':'fallback_pending'}]}

    def test_plan_preserves_completed_and_deduplicates(self):
        state=self.state()
        with patch.object(q,'sha',side_effect=lambda p:p.stem): plan=n.ensure_plan(state,2)
        self.assertEqual(plan['ids'],['a','b'])
        self.assertEqual(state['sources'][0]['status'],'sunburst_completed_review_pending')
        self.assertIs(n.ensure_plan(state,2),plan)
        with self.assertRaises(ValueError):n.ensure_plan(state,1)
        with self.assertRaises(ValueError):n.ensure_plan({},101)

    def test_explicit_submission_rejects_completed_before_api(self):
        with TemporaryDirectory() as temp:
            key=Path(temp)/'key';key.write_text('dummy')
            args=SimpleNamespace(google_key_file=key)
            with patch.object(q,'google') as api:
                for ids in (['old'],['a','a'],['a','b','same']):
                    with self.assertRaises(ValueError):q.submit_google(args,self.state(),selected_ids=ids)
                api.assert_not_called()

    def test_sunburst_must_be_paused_before_service_access(self):
        with patch.object(n.SUN_PAUSE.__class__,'exists',return_value=False),patch.object(q,'collect_google') as collect:
            with self.assertRaises(ValueError):n.run(SimpleNamespace(),self.state())
            collect.assert_not_called()

    def test_two_image_trial_finishes_without_sunburst_or_duplicate_submit(self):
        with TemporaryDirectory() as temp:
            args=SimpleNamespace(state=Path(temp)/'queue.json',limit=2,max_active_batches=1,poll_seconds=60)
            state=self.state()
            def collect(args,state):
                for j in state['google_jobs']:
                    if j['status']=='submitted':
                        j['status']='collected'
                        for r in state['sources']:
                            if r['id'] in j['ids']:r['status']='nano_completed_review_pending'
            def submit(args,state,selected_ids):
                state['google_jobs'].append({'status':'submitted','ids':selected_ids})
                for r in state['sources']:
                    if r['id'] in selected_ids:r['status']='fallback_batch_pending'
            with patch.object(n,'SUN_PAUSE',Path(temp)),patch.object(n,'NANO_PAUSE',Path(temp)/'absent'),patch.object(q,'sha',side_effect=lambda p:p.stem),patch.object(q,'collect_google',side_effect=collect),patch.object(q,'submit_google',side_effect=submit) as posts,patch.object(q,'run_parallel_sun') as sun,patch.object(n.time,'sleep'):
                n.run(args,state)
                self.assertEqual(posts.call_count,1);sun.assert_not_called()
                self.assertEqual(state['nano_direct_trial']['status'],'attempts_finished_review_pending')
                n.run(args,state)
                self.assertEqual(posts.call_count,1)

    def test_reservation_exhaustion_stops_without_more_post(self):
        with TemporaryDirectory() as temp:
            args=SimpleNamespace(state=Path(temp)/'queue.json',limit=2,max_active_batches=1,poll_seconds=60)
            state=self.state()
            with patch.object(n,'SUN_PAUSE',Path(temp)),patch.object(n,'NANO_PAUSE',Path(temp)/'absent'),patch.object(q,'sha',side_effect=lambda p:p.stem),patch.object(q,'collect_google'),patch.object(q,'submit_google',side_effect=ValueError('Google reservations exceed user budget; no POST')) as posts:
                n.run(args,state)
                self.assertEqual(posts.call_count,1)
                self.assertEqual(state['status'],'google_budget_stop')


if __name__=='__main__':unittest.main()
