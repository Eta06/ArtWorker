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
import math
from pathlib import Path
from urllib.parse import urlsplit


def training_crop(row, extensions, trim=None):
    """Axis-aligned crop of the teacher output, distinct from source registration."""
    if row.get('audit_status') != 'reliable' or not row.get('region'):
        return None
    points = row['region'].get('polygon', [])
    dimensions = row.get('dimensions', [])
    if len(points) != 4 or len(dimensions) != 2 or any(len(p) != 2 for p in points):
        return None
    w, h = dimensions
    if w <= 0 or h <= 0 or not all(math.isfinite(v) for p in points for v in p):
        return None
    xs, ys = zip(*points)
    trim = trim or {'top': False, 'bottom': False}
    base = [max(0, min(w, math.floor(min(xs)))), max(0, min(h, math.floor(min(ys)))),
            max(0, min(w, math.ceil(max(xs)))), max(0, min(h, math.ceil(max(ys))))]
    target = [base[0] if any(trim.values()) else 0, base[1] if trim['top'] else 0,
              base[2] if any(trim.values()) else w, base[3] if trim['bottom'] else h]
    box = [max(0, min(w, math.floor(min(xs)))),
           0 if extensions['top'] else max(0, min(h, math.floor(min(ys)))),
           max(0, min(w, math.ceil(max(xs)))),
           h if extensions['bottom'] else max(0, min(h, math.ceil(max(ys))))]
    box = [max(box[0], target[0]), max(box[1], target[1]),
           min(box[2], target[2]), min(box[3], target[3])]
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return {'source': 'teacher_output_crop', 'box_px': box,
            'raw_dimensions': [w, h], 'target_box_px': target,
            'target_dimensions': [target[2]-target[0], target[3]-target[1]],
            'input_box_in_target_px': [box[0]-target[0], box[1]-target[1], box[2]-target[0], box[3]-target[1]],
            'extensions': extensions, 'target_trim': trim,
            'remaining_target_pixels': (target[2]-target[0])*(target[3]-target[1])-(box[2]-box[0])*(box[3]-box[1])}


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
            extensions = decisions.get(key, {}).get('crop_extensions', {'top': False, 'bottom': False})
            trim = decisions.get(key, {}).get('target_trim', {'top': False, 'bottom': False})
            crop = training_crop(row, extensions, trim)
            if crop:
                item['conditioning'] = crop
            if decision == 'accepted' and row['audit_status'] == 'reliable':
                if not crop:
                    blocked.append({**item, 'reason': 'valid_training_crop_required'})
                elif crop['remaining_target_pixels'] == 0:
                    blocked.append({**item, 'reason': 'no_region_left_to_outpaint'})
                else:
                    candidates.append(item)
            elif decision == 'accepted':
                blocked.append({**item, 'reason': 'reliable_source_registration_required'})
            else:
                excluded.append({**item, 'reason': decision})
        manifest = {'schema_version': 3, 'policy': 'explicit_accept_and_reliable_registration',
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
        extensions = payload.get('crop_extensions', old.get('crop_extensions', {'top': False, 'bottom': False})
                                 if old.get('raw_sha256') == row['raw_sha256'] else {'top': False, 'bottom': False})
        if (not isinstance(extensions, dict) or set(extensions) != {'top', 'bottom'}
                or any(type(value) is not bool for value in extensions.values())):
            raise ValueError('Invalid crop extensions')
        if any(extensions.values()) and not training_crop(row, extensions):
            raise ValueError('Reliable registration required for extending the training crop')
        trim = payload.get('target_trim', old.get('target_trim', {'top': False, 'bottom': False})
                           if old.get('raw_sha256') == row['raw_sha256'] else {'top': False, 'bottom': False})
        if (not isinstance(trim, dict) or set(trim) != {'top', 'bottom'}
                or any(type(value) is not bool for value in trim.values())):
            raise ValueError('Invalid target trim')
        if any(trim.values()) and not training_crop(row, extensions, trim):
            raise ValueError('Reliable registration required for trimming the training target')
        revision = old.get('revision', 0) if old.get('raw_sha256') == row['raw_sha256'] else 0
        request_id = payload.get('request_id')
        if request_id is not None and (not isinstance(request_id, str) or not 1 <= len(request_id) <= 100):
            raise ValueError('Invalid request id')
        if request_id and old.get('request_id') == request_id:
            if any(old.get(k) != payload.get(k, '') for k in ('raw_sha256', 'decision', 'reason')):
                raise FileExistsError('Request id reused for a different decision')
            if 'crop_extensions' in payload and old.get('crop_extensions') != extensions:
                raise FileExistsError('Request id reused for a different training crop')
            if 'target_trim' in payload and old.get('target_trim') != trim:
                raise FileExistsError('Request id reused for a different training target')
            snapshot = self.snapshot()
            self.write(self.gallery / 'training-admission.json', snapshot['manifest'])
            return snapshot
        if payload.get('revision', 0) != revision:
            raise FileExistsError('Decision changed in another tab; reload before editing')
        records[row['id']] = {'raw_sha256': row['raw_sha256'],
                              'decision': payload['decision'], 'reason': reason,
                              'crop_extensions': extensions,
                              'target_trim': trim,
                              'revision': revision + 1,
                              'updated_utc': datetime.now(timezone.utc).isoformat()}
        if request_id:
            records[row['id']]['request_id'] = request_id
        self.write(self.path, {'schema_version': 1, 'decisions': records})
        snapshot = self.snapshot()
        self.write(self.gallery / 'training-admission.json', snapshot['manifest'])
        return snapshot


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, store, **kwargs):
        self.store = store
        super().__init__(*args, **kwargs)

    def end_headers(self):
        # Avoid running an old keyboard/save implementation after a refresh.
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def send_json(self, value, status=200):
        data = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
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
