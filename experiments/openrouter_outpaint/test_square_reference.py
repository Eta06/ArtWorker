"""Offline checks: actual square-mode request contains no placeholder canvas."""
import base64
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import benchmark


class SquareReferenceTests(unittest.TestCase):
    def test_square_wire_payload_and_receipt(self):
        self.check_wire()

    def test_custom_prompt_and_automatic_routing_wire_payload(self):
        self.check_wire(custom=True)

    def check_wire(self, custom=False):
        build = benchmark.ROOT/'.build'
        build.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=build) as directory:
            root = Path(directory)
            source = root/'original.png'
            Image.new('RGB', (720,720), (30,80,120)).save(source)
            key = root/'fake-key'
            key.write_text('offline-test-key')
            blob = io.BytesIO()
            Image.new('RGB',(72,128),(50,60,70)).save(blob, format='PNG')
            posted = []
            def fake_request(path, key_value, payload=None, timeout=240):
                if payload is None:
                    return {'endpoints':[{'provider_tag':'test', 'supported_parameters':{
                        'input_references':{'max':2},'quality':{'values':['high']},'n':{}}}]}, {}
                posted.append(payload)
                return {'data':[{'b64_json':base64.b64encode(blob.getvalue()).decode()}],
                        'usage':{'cost':0}}, {}
            args = ['benchmark.py','--key-file',str(key),'--model','openai/gpt-image-2.5-sunburst',
                    '--track','test','--source-image',str(source),'--output',str(root/'output'),
                    '--input-mode','square','--quality','high']
            if custom:
                prompt=root/'prompt.txt'
                prompt.write_text('Keep source unchanged; extend its surroundings.\n')
                args.extend(['--prompt-file',str(prompt),'--automatic-routing'])
            with patch.object(sys,'argv',args), patch.object(benchmark,'request_json',fake_request), \
                 patch.object(benchmark,'key_usage',return_value={'usage':0}):
                benchmark.main()
            self.assertEqual(len(posted),1)
            request = posted[0]
            self.assertEqual(request['prompt'], 'Keep source unchanged; extend its surroundings.' if custom else benchmark.SQUARE_PROMPT)
            if custom:self.assertNotIn('provider',request)
            else:self.assertEqual(request['provider'],{'only':['test'],'allow_fallbacks':False})
            self.assertEqual(request['aspect_ratio'],'9:16')
            self.assertEqual(request['quality'],'high')
            self.assertEqual(len(request['input_references']),1)
            data = request['input_references'][0]['image_url']['url'].split(',',1)[1]
            transmitted = Image.open(io.BytesIO(base64.b64decode(data))).convert('RGB')
            self.assertEqual(transmitted.size,(720,720))
            self.assertEqual(transmitted.tobytes(),Image.open(source).tobytes())
            receipt = json.loads((root/'output/receipt.json').read_text())
            self.assertFalse(receipt['diagnostic_canvas_sent'])
            self.assertEqual(receipt['request']['input_references'][0]['local_file'],'source.png')
            self.assertEqual(receipt['status'],'completed')

    def test_anchored_mode_sends_only_square_reference(self):
        names,prompt=benchmark.reference_spec('square-anchored',2)
        self.assertEqual(names,('source.png',))
        self.assertEqual(prompt,benchmark.ANCHORED_SQUARE_PROMPT)
        self.assertNotIn('green',prompt.lower())
        with self.assertRaises(ValueError): benchmark.reference_spec('square-anchored',0)

    def test_existing_canvas_protocol_unchanged(self):
        names,prompt=benchmark.reference_spec('canvas',2)
        self.assertEqual(names,('input.png','source.png'))
        self.assertEqual(prompt,benchmark.PROMPT)
        names,prompt=benchmark.reference_spec('canvas',1)
        self.assertEqual(names,('input.png',))
        self.assertNotIn('Reference 2',prompt)

    def test_text_only_endpoint_rejected(self):
        with self.assertRaises(ValueError): benchmark.reference_spec('square',0)


if __name__ == '__main__':
    unittest.main()
