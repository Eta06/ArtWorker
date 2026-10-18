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

from review_server import Handler, ReviewStore


class TrainingReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = [dict(id=key, cover_id=key, model='nano', raw_sha256=key*64,
                         asset=key+'/', audit_status=status)
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
