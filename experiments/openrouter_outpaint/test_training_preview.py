"""Previews follow saved decisions and crops without writing archive assets."""
from functools import partial
from http.server import HTTPServer
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen

from PIL import Image
from review_server import Handler, ReviewStore
from training_preview import preview


class PreviewTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);asset=self.root/'asset';asset.mkdir()
        image=Image.new('RGB',(100,300),'blue')
        image.paste((220,30,20),(0,0,100,100))
        image.paste((20,210,40),(0,100,100,200))
        self.raw=asset/'raw.png';image.save(self.raw);self.original=self.raw.read_bytes()
        row=dict(id='one',cover_id='one',model='sunburst',raw_sha256='output-sha',asset='asset/',
                 dimensions=[100,300],audit_status='reliable',region={'polygon':[[0,100],[100,100],[100,200],[0,200]]})
        (self.root/'data.json').write_text(json.dumps({'rows':[row]}))
        self.store=ReviewStore(self.root)
        self.payload=dict(id='one',raw_sha256='output-sha',decision='accepted',reason='',revision=0)

    def test_input_uses_teacher_pixels_and_target_respects_discarded_top(self):
        self.store.decide({**self.payload,'target_trim':{'top':True,'bottom':False}})
        source=Image.open(BytesIO(preview(self.store,'one','input','output-sha','1')))
        target=Image.open(BytesIO(preview(self.store,'one','target','output-sha','1')))
        self.assertEqual(source.size,(100,100));self.assertEqual(target.size,(100,200))
        self.assertLess(sum(abs(a-b) for a,b in zip(source.getpixel((50,50)),(20,210,40))),10)
        self.assertLess(sum(abs(a-b) for a,b in zip(target.getpixel((50,50)),source.getpixel((50,50)))),10)
        self.assertEqual(self.raw.read_bytes(),self.original)

    def test_including_bottom_changes_input_but_does_not_escalate_admission(self):
        self.store.decide({**self.payload,'crop_extensions':{'top':False,'bottom':True}})
        self.assertEqual(Image.open(BytesIO(preview(self.store,'one','input','output-sha','1'))).size,(100,200))
        self.store.decide({**self.payload,'decision':'rejected','revision':1})
        with self.assertRaises(PermissionError):preview(self.store,'one','input','output-sha','2')
        with self.assertRaises(FileExistsError):preview(self.store,'one','input','output-sha','1')

    def test_fingerprint_and_no_remaining_region_block_preview(self):
        self.store.decide(self.payload)
        with self.assertRaises(FileExistsError):preview(self.store,'one','target','other-output','1')
        self.store.decide({**self.payload,'revision':1,'target_trim':{'top':True,'bottom':True}})
        with self.assertRaises(PermissionError):preview(self.store,'one','input','output-sha','2')

    def test_http_returns_bounded_jpeg_and_rejects_stale_revision(self):
        self.store.decide(self.payload)
        server=HTTPServer(('127.0.0.1',0),partial(Handler,store=self.store,directory=str(self.root)))
        threading.Thread(target=server.serve_forever,daemon=True).start()
        self.addCleanup(server.server_close);self.addCleanup(server.shutdown)
        base='http://127.0.0.1:'+str(server.server_port)+'/sumbul/training-preview/one/'
        with urlopen(base+'input.jpg?sha=output-sha&revision=1') as response:
            self.assertEqual(response.headers['Content-Type'],'image/jpeg')
            self.assertEqual(response.headers['Cache-Control'],'no-store')
            self.assertLessEqual(max(Image.open(BytesIO(response.read())).size),780)
        with self.assertRaises(HTTPError) as caught:urlopen(base+'input.jpg?sha=output-sha&revision=0')
        self.assertEqual(caught.exception.code,409);caught.exception.close()


if __name__=='__main__':unittest.main()
