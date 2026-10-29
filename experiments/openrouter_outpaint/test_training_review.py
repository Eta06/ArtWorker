"""Offline admission/persistence tests; all writes stay in temporary fixtures."""
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from http.server import HTTPServer
import json
from pathlib import Path
import tempfile
import threading
import socket
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from review_server import Handler, ReviewServer, ReviewStore, training_crop


class TrainingReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = [dict(id=key, cover_id=key, model='nano', raw_sha256=key*64,
                         asset=key+'/', audit_status=status, dimensions=[1000, 2000],
                         region={'polygon': [[0, 500], [1000, 500], [1000, 1500], [0, 1500]]})
                     for key, status in [('a', 'reliable'), ('b', 'unavailable'), ('c', 'reliable')]]
        self.publish()
        self.store = ReviewStore(self.root)

    def publish(self):
        (self.root/'data.json').write_text(json.dumps({'rows': self.rows}))

    def decision(self, key, value, revision=0):
        return dict(id=key, raw_sha256=key*64, decision=value, revision=revision, reason='test')

    def test_only_explicit_acceptance_with_registration_is_candidate(self):
        self.assertEqual(self.store.snapshot()['manifest']['candidates'], [])
        self.store.decide(self.decision('a', 'rejected'))
        self.store.decide(self.decision('b', 'accepted'))
        snapshot = self.store.decide(self.decision('c', 'accepted'))
        manifest = snapshot['manifest']
        self.assertEqual([r['id'] for r in manifest['candidates']], ['c'])
        self.assertEqual([r['id'] for r in manifest['excluded']], ['a'])
        self.assertEqual([r['id'] for r in manifest['blocked']], ['b'])

    def test_recovery_needs_explicit_confirmation_and_keeps_quality_vote(self):
        self.rows[1]['registration_recovery'] = {'id': 'proposal-1', 'evidence': {'classification': 'dense_verified'}}
        self.publish()
        first = self.store.decide(self.decision('b', 'accepted'))
        self.assertEqual(first['manifest']['blocked'][0]['reason'], 'registration_recovery_confirmation_required')
        confirmed = self.store.decide({**self.decision('b', 'accepted', 1), 'registration_confirmation': 'proposal-1'})
        self.assertEqual(confirmed['manifest']['candidates'][0]['id'], 'b')
        self.assertEqual(self.rows[1]['audit_status'], 'unavailable')
        # A note edit from an older client retains the confirmation.
        saved = self.store.decide(self.decision('b', 'accepted', 2))
        self.assertEqual(saved['decisions']['b']['registration_confirmation'], 'proposal-1')
        self.rows[1]['registration_recovery']['id'] = 'proposal-2'
        self.publish()
        self.assertEqual(self.store.snapshot()['manifest']['candidates'], [])
        with self.assertRaises(ValueError):
            self.store.decide({**self.decision('b', 'accepted', 3), 'registration_confirmation': 'proposal-1'})

    def test_boundary_confirmation_cannot_admit_rejected_output(self):
        self.rows[1]['registration_recovery'] = {'id': 'proposal-1', 'evidence': {'classification': 'partial_reference'}}
        self.publish()
        result = self.store.decide({**self.decision('b', 'rejected'), 'registration_confirmation': 'proposal-1'})
        self.assertEqual(result['manifest']['candidates'], [])
        self.assertEqual(result['decisions']['b']['decision'], 'rejected')
        self.assertEqual(result['manifest']['excluded'][1]['reason'], 'rejected')

    def test_restart_reset_and_no_transfer_to_replaced_image(self):
        self.store.decide(self.decision('a', 'accepted'))
        reopened = ReviewStore(self.root)
        self.assertEqual(reopened.snapshot()['decisions']['a']['decision'], 'accepted')
        reopened.decide(self.decision('a', 'pending', revision=1))
        self.assertEqual(reopened.snapshot()['manifest']['candidates'], [])
        self.rows[0]['raw_sha256'] = 'new-output'
        self.publish()
        self.assertNotIn('a', reopened.snapshot()['decisions'])
        with self.assertRaises(ValueError):
            reopened.decide(self.decision('a', 'accepted'))

    def test_conflict_and_invalid_requests_never_overwrite(self):
        self.store.decide(self.decision('a', 'rejected'))
        original = self.store.path.read_bytes()
        with self.assertRaises(FileExistsError):
            self.store.decide(self.decision('a', 'accepted'))
        with self.assertRaises(ValueError):
            self.store.decide(self.decision('a', 'invented', revision=1))
        self.assertEqual(self.store.path.read_bytes(), original)

    def test_lost_response_can_be_retried_after_restart_without_changing_revision(self):
        payload = {**self.decision('a', 'accepted'), 'request_id': 'one-operation'}
        self.store.decide(payload)
        restarted = ReviewStore(self.root)
        retried = restarted.decide(payload)
        self.assertEqual(retried['decisions']['a']['revision'], 1)
        self.assertEqual(retried['decisions']['a']['request_id'], 'one-operation')
        with self.assertRaises(FileExistsError):
            restarted.decide({**payload, 'decision': 'rejected'})

    def test_bottom_crop_persists_without_rewriting_registration_or_resetting_on_old_client(self):
        polygon = json.dumps(self.rows[0]['region'])
        payload = {**self.decision('a', 'accepted'), 'crop_extensions': {'top': False, 'bottom': True}}
        result = self.store.decide(payload)
        self.assertEqual(result['manifest']['candidates'][0]['conditioning']['box_px'], [0, 500, 1000, 2000])
        self.assertEqual(result['manifest']['candidates'][0]['conditioning']['remaining_target_pixels'], 500000)
        self.assertEqual(json.dumps(self.rows[0]['region']), polygon)
        self.store.decide(self.decision('a', 'accepted', revision=1))
        reopened = ReviewStore(self.root).snapshot()
        self.assertTrue(reopened['decisions']['a']['crop_extensions']['bottom'])
        self.assertEqual(reopened['manifest']['candidates'][0]['conditioning']['box_px'], [0, 500, 1000, 2000])

    def test_top_and_bottom_covering_full_target_is_blocked_and_can_be_reset(self):
        top = {**self.decision('a', 'accepted'), 'crop_extensions': {'top': True, 'bottom': False}}
        self.assertEqual(self.store.decide(top)['manifest']['candidates'][0]['conditioning']['box_px'], [0, 0, 1000, 1500])
        both = {**top, 'revision': 1, 'crop_extensions': {'top': True, 'bottom': True}}
        result = self.store.decide(both)
        self.assertEqual(result['manifest']['candidates'], [])
        self.assertEqual(result['manifest']['blocked'][0]['reason'], 'no_region_left_to_outpaint')
        reset = {**top, 'revision': 2, 'crop_extensions': {'top': False, 'bottom': False}}
        self.assertEqual(self.store.decide(reset)['manifest']['candidates'][0]['conditioning']['box_px'], [0, 500, 1000, 1500])

    def test_crop_rejects_invalid_and_unreliable_extensions_and_clamps_rotated_bounds(self):
        for extensions in ({'top': 'yes', 'bottom': False}, {'top': True}, {'top': True, 'bottom': False, 'x': 2}):
            with self.assertRaises(ValueError):
                self.store.decide({**self.decision('a', 'accepted'), 'crop_extensions': extensions})
        with self.assertRaises(ValueError):
            self.store.decide({**self.decision('b', 'accepted'), 'crop_extensions': {'top': False, 'bottom': True}})
        row = {**self.rows[0], 'region': {'polygon': [[-2.5, 400.8], [900.4, 420], [905, 1400.1], [-3, 1390]]}}
        crop = training_crop(row, {'top': False, 'bottom': True})
        self.assertEqual(crop['box_px'], [0, 400, 905, 2000])

    def test_target_trim_coordinates_and_mask_are_relative_to_retained_target(self):
        payload = {**self.decision('a', 'accepted'), 'target_trim': {'top': True, 'bottom': False}}
        result = self.store.decide(payload)
        crop = result['manifest']['candidates'][0]['conditioning']
        self.assertEqual(crop['target_box_px'], [0, 500, 1000, 2000])
        self.assertEqual(crop['target_dimensions'], [1000, 1500])
        self.assertEqual(crop['input_box_in_target_px'], [0, 0, 1000, 1000])
        self.assertEqual(crop['remaining_target_pixels'], 500000)
        self.store.decide(self.decision('a', 'accepted', revision=1))
        self.assertTrue(ReviewStore(self.root).snapshot()['decisions']['a']['target_trim']['top'])
        bottom = {**payload, 'revision': 2, 'target_trim': {'top': False, 'bottom': True}}
        crop = self.store.decide(bottom)['manifest']['candidates'][0]['conditioning']
        self.assertEqual(crop['target_box_px'], [0, 0, 1000, 1500])
        self.assertEqual(crop['input_box_in_target_px'], [0, 500, 1000, 1500])

    def test_trimming_both_sides_leaves_no_outpainting_and_can_be_undone(self):
        payload = {**self.decision('a', 'accepted'), 'target_trim': {'top': True, 'bottom': True}, 'request_id': 'trim-test'}
        result = self.store.decide(payload)
        self.assertEqual(result['manifest']['candidates'], [])
        self.assertEqual(result['manifest']['blocked'][0]['reason'], 'no_region_left_to_outpaint')
        with self.assertRaises(FileExistsError):
            self.store.decide({**payload, 'target_trim': {'top': True, 'bottom': False}})
        undone = self.store.decide({**payload, 'request_id': 'undo-trim', 'revision': 1, 'target_trim': {'top': False, 'bottom': False}})
        self.assertEqual(len(undone['manifest']['candidates']), 1)

    def test_trim_validation_and_asymmetric_source_keep_only_vertical_completion(self):
        with self.assertRaises(ValueError):
            self.store.decide({**self.decision('a', 'accepted'), 'target_trim': {'top': 'yes', 'bottom': False}})
        with self.assertRaises(ValueError):
            self.store.decide({**self.decision('b', 'accepted'), 'target_trim': {'top': True, 'bottom': False}})
        row = {**self.rows[0], 'region': {'polygon': [[100, 400], [900, 400], [900, 1400], [100, 1400]]}}
        crop = training_crop(row, {'top': False, 'bottom': False}, {'top': True, 'bottom': False})
        self.assertEqual(crop['target_box_px'], [100, 400, 900, 2000])
        self.assertEqual(crop['input_box_in_target_px'], [0, 0, 800, 1000])

    def test_landscape_left_and_right_geometry_and_persistence(self):
        self.rows[0].update(dimensions=[2000,1000],region={'polygon':[[500,100],[1500,100],[1500,900],[500,900]]})
        self.publish()
        empty={'top':False,'bottom':False,'left':False,'right':False}
        result=self.store.decide({**self.decision('a','accepted'),'crop_extensions':{**empty,'left':True}})
        self.assertEqual(result['manifest']['candidates'][0]['conditioning']['box_px'],[0,100,1500,900])
        result=self.store.decide({**self.decision('a','accepted',1),'crop_extensions':empty,'target_trim':{**empty,'left':True}})
        crop=result['manifest']['candidates'][0]['conditioning']
        self.assertEqual(crop['target_box_px'],[500,100,2000,900])
        self.assertEqual(crop['input_box_in_target_px'],[0,0,1000,800])
        self.assertEqual(crop['remaining_target_pixels'],400000)
        reopened=ReviewStore(self.root).snapshot()
        self.assertTrue(reopened['decisions']['a']['target_trim']['left'])
        result=self.store.decide({**self.decision('a','accepted',2),'target_trim':{**empty,'right':True}})
        self.assertEqual(result['manifest']['candidates'][0]['conditioning']['target_box_px'],[0,100,1500,900])

    def test_separate_gallery_route_uses_same_durable_store(self):
        server=HTTPServer(('127.0.0.1',0),partial(Handler,store=self.store,directory=str(self.root),route='/nilufer'))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            origin=f'http://127.0.0.1:{server.server_address[1]}'
            with urlopen(origin+'/nilufer/review-api') as response:self.assertEqual(response.status,200)
            request=Request(origin+'/nilufer/review-api',data=json.dumps(self.decision('a','accepted')).encode(),
                            headers={'Origin':origin,'Content-Type':'application/json'})
            with urlopen(request) as response:self.assertEqual(json.load(response)['decisions']['a']['decision'],'accepted')
        finally:server.shutdown();server.server_close();thread.join()

    def test_idle_browser_connection_does_not_block_page_or_api(self):
        (self.root/'index.html').write_text('review fixture')
        server=ReviewServer(('127.0.0.1',0),partial(Handler,store=self.store,directory=str(self.root),route='/nilufer'))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        idle=socket.create_connection(server.server_address,timeout=1)
        try:
            origin=f'http://127.0.0.1:{server.server_port}'
            with urlopen(origin+'/',timeout=1) as response:
                self.assertEqual(response.read(),b'review fixture')
            with urlopen(origin+'/nilufer/review-api',timeout=1) as response:
                self.assertEqual(json.load(response)['decisions'],{})
        finally:
            idle.close();server.shutdown();server.server_close();thread.join()

    def parallel_decisions(self, payloads):
        barrier=threading.Barrier(2)
        records=self.store.records
        def slow_records():
            saved=records();time.sleep(.02);return saved
        self.store.records=slow_records
        def save(payload):
            barrier.wait(timeout=2)
            try:self.store.decide(payload);return 'saved'
            except FileExistsError:return 'conflict'
        with ThreadPoolExecutor(max_workers=2) as pool:
            return list(pool.map(save,payloads))

    def test_parallel_different_decisions_both_persist(self):
        result=self.parallel_decisions([self.decision('a','accepted'),self.decision('c','rejected')])
        self.assertEqual(result,['saved','saved'])
        saved=ReviewStore(self.root).snapshot()['decisions']
        self.assertEqual(saved['a']['decision'],'accepted')
        self.assertEqual(saved['c']['decision'],'rejected')

    def test_parallel_same_revision_has_one_conflict(self):
        result=self.parallel_decisions([self.decision('a','accepted'),self.decision('a','rejected')])
        self.assertCountEqual(result,['saved','conflict'])
        self.assertEqual(ReviewStore(self.root).snapshot()['decisions']['a']['revision'],1)

    def test_square_fit_fills_short_edge_before_bottom_or_side_extension(self):
        self.rows[0]['region']['polygon']=[[200,600],[800,600],[800,1200],[200,1200]]
        row=self.rows[0]
        crop=training_crop(row,{'bottom':True},square_fit=True)
        self.assertEqual(crop['reference_box_px'],[0,400,1000,1400])
        self.assertEqual(crop['box_px'],[0,400,1000,2000])
        self.assertEqual(crop['remaining_target_pixels'],400000)
        landscape={**row,'dimensions':[2000,1000],'region':{'polygon':[[600,200],[1200,200],[1200,800],[600,800]]}}
        crop=training_crop(landscape,{'right':True},square_fit=True)
        self.assertEqual(crop['reference_box_px'],[400,0,1400,1000])
        self.assertEqual(crop['box_px'],[400,0,2000,1000])
        self.assertEqual(crop['remaining_target_pixels'],400000)

    def test_square_choice_persists_export_matches_and_old_note_client_keeps_it(self):
        self.rows[0]['region']['polygon']=[[200,600],[800,600],[800,1200],[200,1200]]
        self.publish()
        original=json.dumps(self.rows[0]['region'])
        payload={**self.decision('a','accepted'),'square_fit':True,'request_id':'square-operation',
                 'crop_extensions':{'top':False,'bottom':True}}
        result=self.store.decide(payload)
        self.assertEqual(result['manifest']['candidates'][0]['conditioning']['box_px'],[0,400,1000,2000])
        self.assertTrue(ReviewStore(self.root).snapshot()['decisions']['a']['square_fit'])
        self.assertEqual(self.store.decide(payload)['decisions']['a']['revision'],1)
        with self.assertRaises(FileExistsError):self.store.decide({**payload,'square_fit':False})
        result=self.store.decide(self.decision('a','accepted',1))
        self.assertTrue(result['decisions']['a']['square_fit'])
        result=self.store.decide({**self.decision('a','accepted',2),'square_fit':False})
        self.assertEqual(result['manifest']['candidates'][0]['conditioning']['box_px'],[200,600,800,2000])
        self.assertEqual(json.dumps(self.rows[0]['region']),original)

    def test_square_fit_contains_reference_at_canvas_edge_and_refuses_clipping(self):
        row={**self.rows[0],'region':{'polygon':[[200,1500],[800,1500],[800,1900],[200,1900]]}}
        self.assertEqual(training_crop(row,{},square_fit=True)['reference_box_px'],[0,1000,1000,2000])
        self.rows[0]['region']['polygon']=[[100,100],[900,100],[900,1500],[100,1500]]
        self.publish()
        self.assertIsNone(training_crop(self.rows[0],{},square_fit=True))
        with self.assertRaises(ValueError):self.store.decide({**self.decision('a','accepted'),'square_fit':True})
        with self.assertRaises(ValueError):self.store.decide({**self.decision('a','accepted'),'square_fit':'yes'})
        self.assertFalse(self.store.path.exists())

    def test_http_persistence_and_foreign_origin_block(self):
        server = HTTPServer(('127.0.0.1', 0), partial(Handler, store=self.store, directory=str(self.root)))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = 'http://127.0.0.1:'+str(server.server_port)
        payload = json.dumps(self.decision('a', 'rejected')).encode()
        request = Request(base+'/sumbul/review-api', data=payload,
                          headers={'Content-Type': 'application/json', 'Origin': 'https://elsewhere.invalid'})
        with self.assertRaises(HTTPError) as caught:
            urlopen(request)
        self.assertEqual(caught.exception.code, 403)
        caught.exception.close()
        self.assertFalse(self.store.path.exists())
        request.add_header('Origin', base)
        with urlopen(request) as response:
            self.assertEqual(json.load(response)['decisions']['a']['decision'], 'rejected')
        with urlopen(base+'/sumbul/training-manifest') as response:
            self.assertEqual(json.load(response)['candidates'], [])


if __name__ == '__main__':
    unittest.main()
