"""Local gallery plus durable, fingerprint-bound human training decisions.

No generation, model loading, or training. Only explicitly accepted, reliably
registered outputs enter the candidate manifest; unreviewed items stay out.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
from pathlib import Path
from urllib.parse import urlsplit


class ReviewStore:
    def __init__(self, gallery: Path):
        self.gallery = gallery
        self.path = gallery / 'training-reviews.json'

    def rows(self):
        return {r['id']: r for r in json.loads((self.gallery / 'data.json').read_text())['rows']}

    def records(self):
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text())['decisions']

    def snapshot(self):
        rows, saved = self.rows(), self.records()
        # A decision never carries over to a different output with the same id.
        decisions = {k: v for k, v in saved.items()
                     if k in rows and v['raw_sha256'] == rows[k]['raw_sha256']}
        candidates, excluded, blocked = [], [], []
        for key, row in rows.items():
            decision = decisions.get(key, {}).get('decision', 'pending')
            item = {k: row[k] for k in ('id', 'cover_id', 'model', 'raw_sha256')}
            item['raw_file'] = str((self.gallery / row['asset'] / 'raw.png').resolve())
            if decision == 'accepted' and row['audit_status'] == 'reliable':
                candidates.append(item)
            elif decision == 'accepted':
                blocked.append({**item, 'reason': 'reliable_source_registration_required'})
            else:
                excluded.append({**item, 'reason': decision})
        manifest = {'schema_version': 1, 'policy': 'explicit_accept_and_reliable_registration',
                    'training_started': False, 'candidates': candidates,
                    'excluded': excluded, 'blocked': blocked}
        return {'schema_version': 1, 'decisions': decisions, 'manifest': manifest}

    @staticmethod
    def write(path, value):
        temp = path.with_suffix('.tmp')
        with temp.open('w') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            import os
            os.fsync(stream.fileno())
        temp.replace(path)

    def decide(self, payload):
        if payload.get('decision') not in ('accepted', 'rejected', 'pending'):
            raise ValueError('Unknown decision')
        row = self.rows().get(payload.get('id'))
        if not row or payload.get('raw_sha256') != row['raw_sha256']:
            raise ValueError('Output fingerprint changed; reload the gallery')
        reason = payload.get('reason', '')
        if not isinstance(reason, str) or len(reason) > 500:
            raise ValueError('Reason must be at most 500 characters')
        records = self.records()
        old = records.get(row['id'], {})
        revision = old.get('revision', 0) if old.get('raw_sha256') == row['raw_sha256'] else 0
        if payload.get('revision', 0) != revision:
            raise FileExistsError('Decision changed in another tab; reload before editing')
        records[row['id']] = {'raw_sha256': row['raw_sha256'],
                              'decision': payload['decision'], 'reason': reason,
                              'revision': revision + 1,
                              'updated_utc': datetime.now(timezone.utc).isoformat()}
        self.write(self.path, {'schema_version': 1, 'decisions': records})
        snapshot = self.snapshot()
        self.write(self.gallery / 'training-admission.json', snapshot['manifest'])
        return snapshot


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, store, **kwargs):
        self.store = store
        super().__init__(*args, **kwargs)

    def send_json(self, value, status=200):
        data = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == '/sumbul/review-api':
            try:
                self.send_json(self.store.snapshot())
            except (OSError, ValueError, KeyError):
                self.send_json({'error': 'Review data could not be read; nothing overwritten'}, 500)
        elif path == '/sumbul/training-manifest':
            self.send_json(self.store.snapshot()['manifest'])
        else:
            super().do_GET()

    def do_POST(self):
        if urlsplit(self.path).path != '/sumbul/review-api':
            return self.send_json({'error': 'Not found'}, 404)
        # Browser writes only from this exact local origin, never foreign pages.
        if self.headers.get('Origin') != 'http://' + self.headers.get('Host', ''):
            return self.send_json({'error': 'Same-origin request required'}, 403)
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 4096:
                raise ValueError('Invalid body size')
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError('Expected decision object')
            result = self.store.decide(payload)
        except FileExistsError as e:
            return self.send_json({'error': str(e)}, 409)
        except (ValueError, KeyError, TypeError) as e:
            return self.send_json({'error': str(e)}, 400)
        except OSError:
            return self.send_json({'error': 'Save failed; please retry'}, 500)
        self.send_json(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--gallery', type=Path, required=True)
    parser.add_argument('--port', type=int, default=54146)
    args = parser.parse_args()
    store = ReviewStore(args.gallery.resolve())
    # Regenerate the derived manifest from authoritative decisions on restart.
    store.write(store.gallery / 'training-admission.json', store.snapshot()['manifest'])
    handler = partial(Handler, directory=str(args.directory.resolve()), store=store)
    server = HTTPServer(('127.0.0.1', args.port), handler)
    print(f'Gallery and training reviews: http://127.0.0.1:{args.port}/sumbul/', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
