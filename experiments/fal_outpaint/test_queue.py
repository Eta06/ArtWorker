from argparse import Namespace
import json
import io
import http.client
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import urllib.error

import importlib.util

spec=importlib.util.spec_from_file_location('fal_outpaint_queue',Path(__file__).with_name('fal_queue.py'))
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)


class QueueTests(unittest.TestCase):
    def test_every_ratio_meets_live_documented_size_constraints(self):
        self.assertEqual(len({r for r,_,_ in q.SIZES}),12)
        for ratio,w,h in q.SIZES:
            q.validate_size(w,h);a,b=map(int,ratio.split(':'))
            self.assertEqual(w*b,h*a)
            x,y,x2,y2=q.layout(w,h)
            self.assertEqual((x2-x,y2-y),(720,720))
            self.assertTrue(0<=x<x2<=w and 0<=y<y2<=h)
            prompt=q.prompt_for(ratio,w,h)
            self.assertIn(f'aspect ratio {ratio}',prompt)
            self.assertIn(f'{w}x{h}',prompt)
            if (w,h)!=(720,1280):self.assertNotIn('720x1280',prompt)

    def test_credentials_are_not_sent_to_arbitrary_queue_hosts(self):
        for url in ['https://evil.test/status','http://queue.fal.run/status','https://queue.fal.run@evil.test/status']:
            with self.assertRaises(ValueError):q.safe_queue_url(url)

    def test_refusal_skips_but_credit_or_auth_failure_stops(self):
        self.assertEqual(q.classification(422,'content_policy_violation'),'refused')
        for code in (401,402,403,429):self.assertEqual(q.classification(code,'permission or account limit'),'account_stop')
        self.assertEqual(q.classification(422,'image_size is invalid'),'failed')

    def test_missing_billing_is_unknown_not_zero(self):
        receipt={'actual_cost_usd':None};q.apply_cost(receipt,{})
        self.assertIsNone(receipt['actual_cost_usd'])
        q.apply_cost(receipt,{'x-fal-billable-units':'0.0412'})
        self.assertEqual(receipt['actual_cost_usd'],'0.0412')

    def test_uncertain_submission_prevents_all_new_paid_requests(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);manifest=root/'manifest.json';manifest.write_text('{}')
            state=root/'state.json';key=root/'key';key.write_text('test-only')
            state.write_text(json.dumps({'endpoint_id':q.ENDPOINT,'source_manifest':'manifest.json',
                'source_manifest_sha256':q.sha(manifest),'sources':[{'status':'submitting'}]}))
            with patch.object(q,'ROOT',root),patch.object(q,'request') as request:
                with self.assertRaisesRegex(ValueError,'Uncertain submission'):
                    q.run(Namespace(state=state,limit=1000,key_file=key))
                request.assert_not_called()

    def test_resume_polls_known_id_without_resubmitting(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);manifest=root/'manifest.json';manifest.write_text('{}')
            state=root/'state.json';key=root/'key';key.write_text('test-only')
            folder=root/'outputs/one';folder.mkdir(parents=True)
            (folder/'receipt.json').write_text('{"actual_cost_usd":null}')
            row={'id':'one','status':'submitted','submitted_unix':0,'width':864,'height':1152,
                'aspect_ratio':'3:4','status_url':'https://queue.fal.run/known/status',
                'response_url':'https://queue.fal.run/known/result'}
            state.write_text(json.dumps({'endpoint_id':q.ENDPOINT,'source_manifest':'manifest.json',
                'source_manifest_sha256':q.sha(manifest),'sources':[row]}))
            with (patch.object(q,'ROOT',root),patch.object(q,'request',side_effect=[
                ({'status':'COMPLETED'},{}),({'images':[{'url':'https://v3b.fal.media/test.png'}]}, {})]) as request,
                patch.object(q,'download',return_value={'sha256':'test','dimensions':[864,1152]})):
                q.run(Namespace(state=state,limit=1000,key_file=key,concurrency=4,poll_seconds=2))
                self.assertEqual(request.call_count,2)
                self.assertTrue(all(len(c.args)==2 for c in request.call_args_list))
            self.assertEqual(json.loads(state.read_text())['sources'][0]['status'],'completed')

    def test_lost_submit_response_is_durable_and_stops_the_next_cover(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);manifest=root/'manifest.json';manifest.write_text('{}')
            state=root/'state.json';key=root/'key';key.write_text('test-only')
            (root/'outputs/one').mkdir(parents=True)
            rows=[{'id':name,'status':'pending'} for name in ('one','two')]
            state.write_text(json.dumps({'endpoint_id':q.ENDPOINT,'source_manifest':'manifest.json',
                'source_manifest_sha256':q.sha(manifest),'sources':rows}))
            with (patch.object(q,'ROOT',root),patch.object(q,'prepare',return_value=({'prompt':'test'},{})),
                  patch.object(q,'request',side_effect=urllib.error.URLError('response lost')) as request):
                q.run(Namespace(state=state,limit=1000,key_file=key,concurrency=4,poll_seconds=2))
                self.assertEqual(request.call_count,1)
            saved=json.loads(state.read_text())
            self.assertEqual([r['status'] for r in saved['sources']],['uncertain','pending'])
            self.assertEqual(saved['status'],'stopped')
            self.assertEqual(json.loads((root/'outputs/one/receipt.json').read_text())['status'],'uncertain')

    def test_explicit_hold_never_repeats_uncertain_cover_and_continues_next(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);manifest=root/'manifest.json';manifest.write_text('{}')
            state=root/'state.json';key=root/'key';key.write_text('test-only')
            for name in ('one','two'):(root/'outputs'/name).mkdir(parents=True)
            receipt=root/'outputs/one/receipt.json';receipt.write_text('{"status":"uncertain"}')
            rows=[{'id':'one','status':'uncertain'},{'id':'two','status':'pending'}]
            state.write_text(json.dumps({'endpoint_id':q.ENDPOINT,'source_manifest':'manifest.json',
                'source_manifest_sha256':q.sha(manifest),'sources':rows}))
            error=urllib.error.HTTPError('https://queue.fal.run/test',422,'refused',{},io.BytesIO(b'content_policy_violation'))
            with (patch.object(q,'ROOT',root),patch.object(q,'prepare',return_value=({'prompt':'test'},{})) as prepare,
                  patch.object(q,'request',side_effect=error) as request):
                q.run(Namespace(state=state,limit=1000,key_file=key,concurrency=4,poll_seconds=2,hold_uncertain=True))
                self.assertEqual(request.call_count,1)
                self.assertEqual(prepare.call_args.args[0]['id'],'two')
            saved=json.loads(state.read_text())
            self.assertEqual([r['status'] for r in saved['sources']],['held_uncertain','refused'])
            self.assertEqual(json.loads(receipt.read_text())['status'],'held_uncertain')

    def test_receipt_with_known_request_id_is_recovered_instead_of_held(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);manifest=root/'manifest.json';manifest.write_text('{}')
            state=root/'state.json';key=root/'key';key.write_text('test-only')
            folder=root/'outputs/one';folder.mkdir(parents=True)
            queue={'request_id':'saved-id','status_url':'https://queue.fal.run/saved/status',
                'response_url':'https://queue.fal.run/saved/result'}
            (folder/'receipt.json').write_text(json.dumps({'queue':queue}))
            state.write_text(json.dumps({'endpoint_id':q.ENDPOINT,'source_manifest':'manifest.json',
                'source_manifest_sha256':q.sha(manifest),'sources':[
                    {'id':'one','status':'submitting','submitted_unix':0,'aspect_ratio':'3:4'}]}))
            with (patch.object(q,'ROOT',root),patch.object(q,'request',side_effect=[
                    ({'status':'COMPLETED'},{}),({'error':'content_policy_violation'}, {})]) as request):
                q.run(Namespace(state=state,limit=1000,key_file=key,concurrency=4,poll_seconds=2,hold_uncertain=True))
                self.assertEqual(request.call_count,2)
                self.assertTrue(all(len(c.args)==2 for c in request.call_args_list))
            row=json.loads(state.read_text())['sources'][0]
            self.assertEqual(row['request_id'],'saved-id')
            self.assertEqual(row['status'],'refused')

    def test_interrupted_download_retries_only_reads_of_same_generation(self):
        for error in (ConnectionResetError(54,'reset'),http.client.IncompleteRead(b'partial',10)):
            with self.subTest(error=type(error).__name__),TemporaryDirectory() as temp:
                root=Path(temp);manifest=root/'manifest.json';manifest.write_text('{}')
                state=root/'state.json';key=root/'key';key.write_text('test-only')
                folder=root/'outputs/one';folder.mkdir(parents=True)
                (folder/'receipt.json').write_text('{"actual_cost_usd":null}')
                row={'id':'one','status':'submitted','request_id':'original-id','submitted_unix':0,
                    'width':864,'height':1152,'aspect_ratio':'3:4',
                    'status_url':'https://queue.fal.run/original/status',
                    'response_url':'https://queue.fal.run/original/result'}
                state.write_text(json.dumps({'endpoint_id':q.ENDPOINT,'source_manifest':'manifest.json',
                    'source_manifest_sha256':q.sha(manifest),'sources':[row]}))
                responses=[({'status':'COMPLETED'},{}),({'images':[{'url':'https://v3b.fal.media/test.png'}]},
                    {'x-fal-billable-units':'0.0559'})]*2
                with (patch.object(q,'ROOT',root),patch.object(q,'request',side_effect=responses) as request,
                      patch.object(q,'download',side_effect=[error,{'sha256':'test','dimensions':[864,1152]}]) as download,
                      patch.object(q.time,'sleep')):
                    q.run(Namespace(state=state,limit=1000,key_file=key,concurrency=4,poll_seconds=2))
                self.assertEqual(request.call_count,4)
                self.assertTrue(all(len(call.args)==2 for call in request.call_args_list))
                self.assertEqual(download.call_count,2)
                saved=json.loads(state.read_text())['sources'][0]
                self.assertEqual(saved['request_id'],'original-id')
                self.assertEqual(saved['status'],'completed')
                receipt=json.loads((folder/'receipt.json').read_text())
                self.assertEqual(receipt['actual_cost_usd'],'0.0559')
                self.assertEqual(receipt['last_read_error']['type'],type(error).__name__)


if __name__=='__main__':unittest.main()
