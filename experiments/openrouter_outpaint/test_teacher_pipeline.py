"""Offline routing/order checks; no paid service calls."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import budget_queue as q
import teacher_pipeline as p


class PipelineTests(unittest.TestCase):
    def test_credit_rejected_sources_go_to_remaining_nano_before_untouched(self):
        state=self.state();state['sources'].append({'id':'credit','status':'primary_budget_rejected'})
        args=SimpleNamespace(max_active_batches=1)
        def submit(args,state,selected_ids):
            self.assertEqual(selected_ids,['refused','credit'])
            state['google_jobs'].append({'status':'submitted'})
        with patch.object(p,'paused',return_value=False),patch.object(q,'submit_google',side_effect=submit):
            p.pump_google(args,state,remaining=True)

    def state(self):
        return {'prompt_sha256':q.PROMPT_HASH,'google_jobs':[],'sources':[
            {'id':'old','status':'nano_completed_review_pending'},
            {'id':'new','status':'pending'}, {'id':'refused','status':'fallback_pending'}]}

    def test_only_fallback_before_primary_budget_stop(self):
        state=self.state();args=SimpleNamespace(max_active_batches=1)
        def submit(args,state,selected_ids):
            self.assertEqual(selected_ids,['refused'])
            state['google_jobs'].append({'status':'submitted'})
        with patch.object(p,'paused',return_value=False),patch.object(q,'submit_google',side_effect=submit) as post:
            p.pump_google(args,state)
            self.assertEqual(post.call_count,1)

    def test_remaining_phase_prioritizes_refusals_and_skips_completed(self):
        state=self.state();args=SimpleNamespace(max_active_batches=1)
        def submit(args,state,selected_ids):
            self.assertEqual(selected_ids,['refused','new'])
            state['google_jobs'].append({'status':'submitted'})
        with patch.object(p,'paused',return_value=False),patch.object(q,'submit_google',side_effect=submit):
            p.pump_google(args,state,remaining=True)

    def test_automatic_budget_handoff_and_resume_never_restarts_primary(self):
        with TemporaryDirectory() as temp:
            args=SimpleNamespace(state=Path(temp)/'state.json',max_active_batches=1)
            state=self.state()
            def sun(args,state):state['status']='openrouter_budget_stop'
            with patch.object(p,'paused',return_value=False),patch.object(q,'collect_google'),patch.object(q,'run_parallel_sun',side_effect=sun) as primary,patch.object(q,'submit_google',side_effect=ValueError('Google reservations exceed user budget; no POST')):
                p.run(args,state)
                self.assertEqual(state['teacher_pipeline']['phase'],'nano_remaining')
                self.assertEqual(state['teacher_pipeline']['status'],'budget_stop')
                p.run(args,state)
                self.assertEqual(primary.call_count,1)

    def test_uncertain_submission_prevents_all_service_access(self):
        state=self.state();state['google_jobs']=[{'status':'submit_uncertain'}]
        with patch.object(q,'collect_google') as read,patch.object(q,'run_parallel_sun') as primary:
            with self.assertRaises(ValueError):p.run(SimpleNamespace(),state)
            read.assert_not_called();primary.assert_not_called()

    def test_pause_prevents_posts_and_preserves_remote_jobs(self):
        with TemporaryDirectory() as temp:
            state=self.state();state['google_jobs']=[{'status':'submitted'}]
            with patch.object(p,'paused',return_value=True),patch.object(q,'collect_google') as read:
                p.run(SimpleNamespace(state=Path(temp)/'state.json'),state)
                read.assert_not_called()
                self.assertEqual(state['google_jobs'][0]['status'],'submitted')
                self.assertEqual(state['status'],'teacher_pipeline_paused')


if __name__=='__main__':unittest.main()
