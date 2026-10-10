"""Offline controls for safe dispatch; no provider calls."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from benchmark import ROOT
import prompt_search
from source_fidelity import sha


class DispatchControls(unittest.TestCase):
    def scenario(self, directory):
        root=Path(directory)
        source=root/'source';source.write_bytes(b'private source fixture')
        prompt=root/'prompt';prompt.write_text('extend the source')
        key=root/'key';key.write_text('offline-test-key')
        cell=dict(directory=str((root/'output').relative_to(ROOT)),source_file=str(source.relative_to(ROOT)),
                  prompt_file=str(prompt.relative_to(ROOT)),source_sha256=sha(source),prompt_sha256=sha(prompt),
                  model='openai/gpt-image-2.5-sunburst',track='fixture',status='pending',quality='high')
        plan=root/'plan.json';plan.write_text(json.dumps(dict(stage='fixture',cells=[cell])))
        args=['prompt_search.py','--plan',str(plan),'--results',str(root/'results'),
              '--key-file',str(key),'--run','--budget-usd','1']
        return root,cell,args

    def test_low_remaining_credit_stops_before_post(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.build') as directory:
            root,cell,args=self.scenario(directory)
            with patch.object(sys,'argv',args),patch('prompt_search.key_usage',return_value={'usage':0,'limit_remaining':.49}),patch('prompt_search.subprocess.run') as post:
                with self.assertRaises(SystemExit):prompt_search.main()
                post.assert_not_called()
                self.assertEqual(json.loads((root/'results/batch.json').read_text())['status'],'paused_budget')

    def test_uncertain_receipt_never_reposts(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.build') as directory:
            root,cell,args=self.scenario(directory)
            out=ROOT/cell['directory'];out.mkdir()
            (out/'receipt.json').write_text(json.dumps({'status':'request_started'}))
            with patch.object(sys,'argv',args),patch('prompt_search.key_usage',return_value={'usage':0,'limit_remaining':10}),patch('prompt_search.subprocess.run') as post:
                with self.assertRaises(SystemExit):prompt_search.main()
                post.assert_not_called()

    def test_changed_source_rejected_before_network(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.build') as directory:
            root,cell,args=self.scenario(directory)
            (ROOT/cell['source_file']).write_bytes(b'changed source')
            with patch.object(sys,'argv',args),patch('prompt_search.key_usage') as auth,patch('prompt_search.subprocess.run') as post:
                with self.assertRaises(SystemExit):prompt_search.main()
                auth.assert_not_called();post.assert_not_called()

    def test_lagging_account_counter_does_not_erase_reported_cost(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.build') as directory:
            root,cell,args=self.scenario(directory)
            out=ROOT/cell['directory'];out.mkdir()
            (out/'receipt.json').write_text(json.dumps({'status':'completed','reported_cost_usd':.6}))
            second=dict(cell,directory=str((root/'second').relative_to(ROOT)))
            (root/'plan.json').write_text(json.dumps(dict(stage='fixture',cells=[cell,second])))
            with patch.object(sys,'argv',args),patch('prompt_search.key_usage',return_value={'usage':0,'limit_remaining':10}),patch('prompt_search.subprocess.run') as post:
                with self.assertRaises(SystemExit):prompt_search.main()
                post.assert_not_called()
                self.assertEqual(json.loads((root/'results/batch.json').read_text())['status'],'paused_budget')


if __name__=='__main__':unittest.main()
