"""Offline checks for direct Google routing, source payload and cost separation."""
import base64
import unittest
import tempfile
from pathlib import Path
from google_trial import API, MODEL, payload_for, output_images, cost_estimate
from review_gallery import render_gallery


class GoogleTrialTests(unittest.TestCase):
    def test_direct_square_payload(self):
        p = payload_for(b'original square')
        self.assertEqual(API, 'https://generativelanguage.googleapis.com/v1beta/interactions')
        self.assertEqual(p['model'], MODEL)
        self.assertFalse(p['store'])
        self.assertEqual(len(p['input']), 2)
        self.assertEqual(base64.b64decode(p['input'][1]['data']), b'original square')
        self.assertEqual(p['response_format']['aspect_ratio'], '9:16')
        self.assertEqual(p['response_format']['mime_type'], 'image/jpeg')

    def test_only_final_model_images(self):
        image = {'type': 'image', 'data': 'YWJj'}
        response = {'steps': [{'type': 'user_input', 'content': [image]},
                              {'type': 'model_output', 'content': [{'type': 'text'}, image]}]}
        self.assertEqual(output_images(response), [image])
        self.assertEqual(output_images({'steps': []}), [])

    def test_internal_text_and_thoughts_are_not_free(self):
        usage = {'total_input_tokens': 1216, 'total_output_tokens': 1540,
                 'total_thought_tokens': 628,
                 'output_tokens_by_modality': [{'modality': 'image', 'tokens': 1120}]}
        result = cost_estimate(usage)
        self.assertEqual(result['estimated_cost_usd'], .043284)
        self.assertIsNone(result['actual_cost_usd'])

    def test_missing_or_ambiguous_usage_is_not_zero_cost(self):
        for usage in ({}, {'total_input_tokens': 10},
                      {'total_input_tokens': 10, 'total_output_tokens': 1120,
                       'total_cached_tokens': 10,
                       'output_tokens_by_modality': [{'modality': 'image', 'tokens': 1120}]}):
            result = cost_estimate(usage)
            self.assertIsNone(result['estimated_cost_usd'])
            self.assertIsNone(result['actual_cost_usd'])

    def test_gallery_does_not_present_unknown_billing_as_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            row = dict(model=MODEL, profile='1K', track='track2', status='no_image', reused=False,
                       directory='.build/offline-gallery-fixture')
            render_gallery([row], dest, {'stage': 'rezene', 'new_charge_usd': None},
                           'Tamamlandı', {'track2': 'Kapak'})
            html = (dest / 'index.html').read_text()
            self.assertIn('Yeni API ücreti: Doğrulanmadı', html)
            self.assertNotIn('Yeni API ücreti: $0.', html)


if __name__ == '__main__':
    unittest.main()
