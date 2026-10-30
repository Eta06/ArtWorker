"""CPU-only safety/measurement tests; no checkpoint is imported or loaded."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('benchmark',Path(__file__).with_name('benchmark.py'))
bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)


class BenchmarkTests(unittest.TestCase):
    def fixture(self,folder,arms=None):
        arms=arms or [{'id':'a','kind':'klein'}]
        bench.write(folder/'inputs.json',{'arms':arms,'tracks':[{'id':'c1'},{'id':'c2'}]})
        bench.write(folder/'state.json',{'status':'prepared','jobs':[
            {'id':a['id']+'/'+c,'arm':a['id'],'cover':c,'status':'pending'}
            for c in ['c1','c2'] for a in arms]})

    def test_warning_pressure_never_launches_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);self.fixture(folder)
            with patch.object(bench,'pressure_and_swap',return_value=(2,0)),patch.object(bench.time,'monotonic',side_effect=[0,10]),patch.object(bench,'run') as run:
                bench.execute(folder,1)
                run.assert_not_called()
            state=bench.read(folder/'state.json')
            self.assertEqual(state['status'],'paused_memory_pressure')
            self.assertTrue(all(j['status']=='pending' for j in state['jobs']))

    def test_interrupted_worker_is_not_automatically_replayed(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);self.fixture(folder)
            state=bench.read(folder/'state.json');state['jobs'][0]['status']='running';bench.write(folder/'state.json',state)
            with patch.object(bench,'run') as run,self.assertRaisesRegex(RuntimeError,'Interrupted'):
                bench.execute(folder,1)
            run.assert_not_called()

    def test_oversized_arm_does_not_repeat_on_remaining_covers(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);self.fixture(folder,[{'id':'a','kind':'klein'},{'id':'b','kind':'klein'}])
            calls=[]
            def fake_run(command,guard,limit,timeout,**kwargs):
                calls.append(command)
                if '--arm' in command and command[command.index('--arm')+1]=='a':
                    return {'status':'stopped_or_failed','stop_reason':'memory_limit'}
                out=Path(command[-1]);bench.write(out/'result.json',{'seconds_including_load':1})
                return {'status':'completed','stop_reason':'complete'}
            with patch.object(bench,'pressure_and_swap',return_value=(1,0)),patch.object(bench.time,'sleep'),patch.object(bench,'run',side_effect=fake_run),patch.object(bench,'render_report'):
                bench.execute(folder,10)
            jobs=bench.read(folder/'state.json')['jobs']
            self.assertEqual([j['status'] for j in jobs],['failed','completed','blocked_by_first_failure','completed'])
            self.assertEqual(len(calls),3)

    def test_pressure_stop_pauses_other_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);self.fixture(folder)
            with patch.object(bench,'pressure_and_swap',return_value=(1,0)),patch.object(bench.time,'sleep'),patch.object(bench,'run',return_value={'status':'stopped_or_failed','stop_reason':'system_memory_pressure'}):
                bench.execute(folder,10)
            state=bench.read(folder/'state.json')
            self.assertEqual(state['status'],'paused_resource_guard')
            self.assertEqual(state['jobs'][1]['status'],'pending')

    def test_explicit_warning_permission_does_not_allow_critical_pressure(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);self.fixture(folder)
            with patch.object(bench,'pressure_and_swap',return_value=(4,0)),patch.object(bench.time,'monotonic',side_effect=[0,10]),patch.object(bench,'run') as run:
                bench.execute(folder,1,True)
                run.assert_not_called()
            self.assertEqual(bench.read(folder/'state.json')['allowed_pressure_levels'],[1,2])

    def test_explicit_warning_permission_keeps_process_and_swap_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);self.fixture(folder)
            receipt={'status':'stopped_or_failed','stop_reason':'swap_growth_limit'}
            with patch.object(bench,'pressure_and_swap',return_value=(2,0)),patch.object(bench.time,'sleep'),patch.object(bench,'run',return_value=receipt) as run:
                bench.execute(folder,10,True)
                self.assertEqual(run.call_args.kwargs['allowed_pressure_levels'],(1,2))
                self.assertEqual(run.call_args.args[2],6.0)
            self.assertEqual(bench.read(folder/'state.json')['status'],'paused_resource_guard')

    def test_resource_retry_preserves_original_attempt_and_never_retries_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);self.fixture(folder)
            old=folder/'outputs/a/c1';old.mkdir(parents=True)
            (old/'execution.log').write_text('preserved first attempt')
            state=bench.read(folder/'state.json')
            state['jobs'][0].update(status='failed',output=str(old),execution={'stop_reason':'memory_limit'})
            state['jobs'][1]['status']='completed';bench.write(folder/'state.json',state)
            bench.retry_resource_failures(folder,['a'])
            updated=bench.read(folder/'state.json')['jobs']
            self.assertEqual(updated[0]['status'],'pending')
            self.assertEqual(updated[0]['previous_attempts'][0]['output'],str(old))
            self.assertEqual(updated[1]['status'],'completed')
            self.assertEqual((old/'execution.log').read_text(),'preserved first attempt')

    def test_source_paste_does_not_hide_raw_error_or_change_exterior(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);out=folder/'out';out.mkdir()
            Image.new('RGB',(512,512),'red').save(folder/'source.png')
            Image.new('RGB',(512,1152),'blue').save(out/'raw.png')
            inputs=folder/'one.json';bench.write(inputs,{'source_rect_xyxy':[0,320,512,832],'tracks':[{'source_512':str(folder/'source.png')}]})
            bench.write(folder/'inputs.json',{'arms':[{'id':'a','kind':'klein'}],'tracks':[{'id':'c1','title':'fixture','inputs_file':str(inputs),'source_sha256':bench.sha(folder/'source.png')}]})
            with patch.object(bench,'klein',return_value=(out/'raw.png',{})):
                bench.worker(folder,'a','c1',out)
            result=bench.read(out/'result.json')
            self.assertGreater(result['raw_source_mae_255'],0)
            self.assertTrue(result['source_exact_after_composite'])
            self.assertFalse(result['visual_quality_accepted'])
            with Image.open(out/'composite.png') as image:
                self.assertEqual(image.getpixel((0,0)),(0,0,255))
                self.assertEqual(image.getpixel((0,320)),(255,0,0))


if __name__=='__main__':unittest.main()
