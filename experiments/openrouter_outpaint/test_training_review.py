"""Offline admission/persistence tests; all writes stay in temporary fixtures."""
from functools import partial
from http.server import HTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from review_server import Handler, ReviewStore, training_crop


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
