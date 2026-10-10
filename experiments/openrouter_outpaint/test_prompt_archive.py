"""Offline archive coverage and preservation checks with small local fixtures."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from prompt_archive import ROOT, build, collect, prompt_groups


class PromptArchiveTests(unittest.TestCase):
    def test_exact_wording_not_only_candidate_labels(self):
        rows = [{'prompt_sha256': hashlib.sha256(p.encode()).hexdigest(), 'prompt': p}
                for p in ['Keep square.', 'Keep square.', 'Keep square. ']]
        self.assertEqual(len(prompt_groups(rows)), 2)

    def test_same_model_two_routes_and_symlink_do_not_inflate_counts(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.build') as tmp:
            root = Path(tmp)
            for tag, model, route in [('first', 'google/gemini-nano-banana-2.1', None),
                                      ('second', 'gemini-nano-banana-2.1', 'direct_google_gemini_api')]:
                d = root / '2026-10-11' / 'stage' / tag
                d.mkdir(parents=True)
                (d / 'receipt.json').write_text(json.dumps({'model': model, 'route': route,
                    'status': 'http_error', 'request': {'prompt': 'Keep square.'}}))
            (root / 'duplicate').symlink_to(root / '2026-10-11', target_is_directory=True)
            rows = collect(root)
            self.assertEqual(len(rows), 2)
            self.assertEqual(len({r['model'] for r in rows}), 1)

    def test_raw_receipt_preservation_and_fresh_report_required(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.build') as tmp:
            root = Path(tmp)
            d = root / 'results' / '2026-10-11' / 'stage' / 'attempt'
            d.mkdir(parents=True)
            for name in ['source.png', 'raw.png']:
                Image.new('RGB', (8, 8), 'blue').save(d / name)
            hashes = {name: hashlib.sha256((d / name).read_bytes()).hexdigest()
                      for name in ['source.png', 'raw.png']}
            receipt = {'model': 'test/model', 'track': 'test', 'status': 'completed',
                       'request': {'prompt': 'Keep square.', 'input_references':
                                   [{'local_file': 'source.png', 'sha256': hashes['source.png']}]},
                       'images': [{'file': 'raw.png', 'sha256': hashes['raw.png'], 'dimensions': [8, 8]}]}
            path = d / 'receipt.json'
            path.write_text(json.dumps(receipt))
            before = path.read_bytes()
            report = root / 'report'
            summary = build(root / 'results', report, None)
            self.assertEqual(summary['existing_image_count'], 1)
            self.assertEqual(summary['new_provider_requests'], 0)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(hashlib.sha256((d / 'raw.png').read_bytes()).hexdigest(), hashes['raw.png'])
            with self.assertRaisesRegex(ValueError, 'Fresh output'):
                build(root / 'results', report, None)


if __name__ == '__main__':
    unittest.main()
