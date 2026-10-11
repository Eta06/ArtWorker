"""Offline spending and route regression checks; never access paid services."""
import unittest
import json
import threading
import time
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import budget_queue as q


class QueueTests(unittest.TestCase):
    def test_credit_rejection_requires_explicit_metadata_without_output(self):
        receipt={'status':'http_error','http_status':402,'error':json.dumps({'error':{
            'code':402,'message':'Insufficient credits','metadata':{'limit_source':'openrouter_credits'}}})}
        self.assertTrue(q.credit_rejection(receipt))
        for change in ({'http_status':500},{'images':[{}]},{'generation_id':'gen-test'},
                       {'reported_cost_usd':0.01},{'error':'timeout'},
                       {'error':json.dumps({'error':{'code':402,'message':'Insufficient credits'}})}):
            self.assertFalse(q.credit_rejection(dict(receipt,**change)))

    def test_credit_402_drains_wave_and_stops_primary_without_retry(self):
        with TemporaryDirectory(dir=q.ROOT/'.build') as temp:
            root=Path(temp);key=root/'key';key.write_text('dummy')
            cover=root/'source.png';cover.write_bytes(b'cover')
            state={'initial_openrouter_credit_usd':'1','openrouter_budget_usd':'1','sources':[
                {'id':'a','status':'pending','sha256':q.sha(cover),'file':str(cover.relative_to(q.ROOT))}]}
            args=type('Args',(),{'state':root/'queue.json','key_file':key,'workers':3,'limit':9})()
            def run(command,**kw):
                out=Path(command[command.index('--output')+1]);out.mkdir(parents=True)
                q.save_json(out/'receipt.json',{'status':'http_error','http_status':402,
                    'request':{'prompt':q.PROMPT,'input_references':[{},{}]},
                    'error':json.dumps({'error':{'code':402,'message':'Insufficient credits',
                        'metadata':{'limit_source':'openrouter_credits'}}})})
            with patch.object(q,'credit',return_value=Decimal('1')),patch.object(q,'read_account',return_value=({'data':{'limit_remaining':1}},{})),patch.object(q,'lightweight_row',return_value={'actual_cost_usd':None}),patch.object(q.subprocess,'run',side_effect=run) as posts,patch.object(q.Path,'exists',autospec=True,side_effect=lambda p:False if str(p).endswith('teacher-queue.pause') else original_exists(p)):
                q.run_parallel_sun(args,state)
                self.assertEqual(posts.call_count,1)
                self.assertEqual(state['status'],'openrouter_budget_stop')
                self.assertEqual(state['sources'][0]['status'],'primary_budget_rejected')
                self.assertEqual(state['inflight_reserved_usd'],'0')

    def test_spend_uses_larger_of_receipts_and_live_account_movement(self):
        state={'initial_openrouter_credit_usd':'16.07', 'sources':[{'actual_cost_usd':0.2}]}
        self.assertEqual(q.committed_spend(state,Decimal('16')),Decimal('0.2'))
        self.assertEqual(q.committed_spend(state,Decimal('15.77')),Decimal('0.30'))

    def test_technical_errors_do_not_enter_safety_fallback(self):
        self.assertFalse(q.refusal({'http_status':429,'error':'safety quota exceeded'}))
        self.assertFalse(q.refusal({'http_status':400,'error':'invalid image dimensions'}))
        self.assertTrue(q.refusal({'http_status':400,'error':'content_policy_violation'}))

    def test_requested_artists_are_interleaved_without_duplicate_cover_hashes(self):
        def row(i,a): return {'id':i,'sha256':i,'artists':[a], 'origin':'turkish_focus_discography'}
        rows=[row('u1','UZI'),row('u2','UZI'),row('m1','Motive'),row('g1','Güneş'),row('r1','Murda')]
        rows.append(dict(rows[0]))
        self.assertEqual([r['id'] for r in q.select({'covers':rows})],['u1','m1','g1','r1','u2'])

    def test_unknown_batch_submission_blocks_another_post(self):
        with patch.object(q,'google',side_effect=AssertionError('No request allowed')):
            with TemporaryDirectory() as temp:
                key=Path(temp)/'key'; key.write_text('dummy')
                args=type('Args',(),{'google_key_file':key})()
                with self.assertRaisesRegex(ValueError,'Unknown batch'):
                    q.submit_google(args,{'google_jobs':[{'status':'submit_uncertain'}]})

    def test_batch_preserves_two_reference_order_and_prompt(self):
        with TemporaryDirectory() as temp:
            out=Path(temp)
            (out/'input.png').write_bytes(b'canvas'); (out/'source.png').write_bytes(b'source')
            request=q.nano_request(out)
            self.assertEqual(request['contents'][0]['parts'][0]['text'],q.PROMPT)
            self.assertEqual(len(request['contents'][0]['parts']),3)
            self.assertEqual(request['generationConfig']['imageConfig'],{'aspectRatio':'9:16','imageSize':'1K'})

    def test_rest_nested_batch_response_and_state(self):
        response={'metadata':{'state':'BATCH_STATE_SUCCEEDED'},'response':{'inlinedResponses':{'inlinedResponses':[{'metadata':{'key':'a'}}]}}}
        self.assertEqual(q.batch_outputs(response),('BATCH_STATE_SUCCEEDED',[{'metadata':{'key':'a'}}]))

    def test_missing_usage_cannot_be_estimated_as_free(self):
        self.assertIsNone(q.google_estimate({},False))
        self.assertIsNone(q.google_estimate({'promptTokenCount':100,'candidatesTokenCount':1120},True))
        usage={'promptTokenCount':1000,'candidatesTokenCount':1120,'thoughtsTokenCount':500,
               'candidatesTokensDetails':[{'modality':'IMAGE','tokenCount':1120}]}
        self.assertEqual(q.google_estimate(usage,True),Decimal('0.019425'))

    def test_concurrent_wave_reserves_all_slots_before_dispatch(self):
        state={'initial_openrouter_credit_usd':'10','openrouter_budget_usd':'0.30',
               'sources':[{'actual_cost_usd':None},{'actual_cost_usd':None}]}
        self.assertEqual(q.slots_available(state,Decimal('10'),10,3),3)
        state['sources'][0]['actual_cost_usd']='0.15'
        self.assertEqual(q.slots_available(state,Decimal('10'),10,3),1)
        self.assertEqual(q.slots_available(state,Decimal('0.05'),10,3),0)
        self.assertEqual(q.slots_available(state,Decimal('10'),0.05,3),0)

    def test_pause_marker_exits_before_any_api_call(self):
        import benchmark
        with TemporaryDirectory() as temp:
            pause=Path(temp)/'pause';pause.touch()
            argv=['benchmark.py','--key-file',str(Path(temp)/'absent-key'),'--model',q.SUN,
                  '--track','test','--output',str(Path(temp)/'out'),'--stop-file',str(pause)]
            with patch('sys.argv',argv),patch.object(benchmark,'request_json',side_effect=AssertionError('No API access')):
                with self.assertRaises(SystemExit) as raised: benchmark.main()
            self.assertEqual(raised.exception.code,75)
            self.assertFalse((Path(temp)/'out').exists())

    def test_account_timeout_retries_only_read_requests(self):
        error=q.urllib.error.HTTPError('https://example.invalid',408,'timeout',{},None)
        with patch.object(q,'request_json',side_effect=[error,({'data':{}}, {})]) as read,patch.object(q.time,'sleep'):
            self.assertEqual(q.read_account('credits','dummy'),({'data':{}},{}))
            self.assertEqual(read.call_count,2)
            self.assertEqual(read.call_args_list[0].args,('credits','dummy'))
        with patch.object(q,'request_json') as request:
            with self.assertRaises(ValueError):q.read_account('images','dummy')
            request.assert_not_called()

    def test_account_auth_error_is_not_retried(self):
        error=q.urllib.error.HTTPError('https://example.invalid',401,'unauthorized',{},None)
        with patch.object(q,'request_json',side_effect=error) as read,patch.object(q.time,'sleep') as sleep:
            with self.assertRaises(q.urllib.error.HTTPError):q.read_account('credits','dummy')
            self.assertEqual(read.call_count,1);sleep.assert_not_called()

    def test_parallel_dispatch_overlaps_and_stops_at_budget(self):
        with TemporaryDirectory(dir=q.ROOT/'.build') as temp:
            root=Path(temp);key=root/'key';key.write_text('dummy')
            state={'initial_openrouter_credit_usd':'10','openrouter_budget_usd':'0.30','sources':[]}
            for i in range(7):
                source=root/f'{i}.png';source.write_bytes(b'cover'+bytes([i]))
                state['sources'].append({'id':f'test{i}','sha256':q.sha(source),
                                        'file':str(source.relative_to(q.ROOT)),'status':'pending'})
            args=type('Args',(),{'state':root/'queue.json','key_file':key,'workers':3,'limit':7})()
            lock=threading.Lock();active=0;peak=0
            def run(command,**kw):
                nonlocal active,peak
                out=Path(command[command.index('--output')+1]);out.mkdir(parents=True)
                with lock: active+=1;peak=max(peak,active)
                time.sleep(.03)
                q.save_json(out/'receipt.json',{'status':'completed','request':{'prompt':q.PROMPT,'input_references':[{},{}]}})
                with lock: active-=1
            def row(source,key):return {'actual_cost_usd':'0.05'}
            with patch.object(q,'credit',return_value=Decimal('10')),patch.object(q,'request_json',return_value=({'data':{'limit_remaining':10}},{})),patch.object(q,'lightweight_row',side_effect=row),patch.object(q,'isolated_audit'),patch.object(q.subprocess,'run',side_effect=run),patch.object(q.Path,'exists',autospec=True,side_effect=lambda p:False if str(p).endswith('teacher-queue.pause') else original_exists(p)):
                q.run_parallel_sun(args,state)
            self.assertEqual(peak,3)
            self.assertEqual(sum(r['status']=='sunburst_completed_review_pending' for r in state['sources']),5)
            self.assertEqual(state['status'],'openrouter_budget_stop')


original_exists=Path.exists


if __name__=='__main__': unittest.main()
