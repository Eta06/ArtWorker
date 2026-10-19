"""Offline, incremental source-region audits and a separate Nilüfer review gallery.

No provider calls, credentials or training. One bounded audit subprocess at a
time; decisions remain authoritative in the gallery and are never rewritten.
"""
import argparse
import fcntl
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/openrouter_outpaint'))
sys.path.insert(0,str(ROOT))
from benchmark import save_json
from production_gallery import snapshot
from scripts.commit_stage import guarded_write, memory_gate


def wait_for_memory(output):
    while True:
        try:memory_gate();return
        except RuntimeError:
            save_json(output/'pipeline-status.json',{'status':'waiting_for_normal_memory','updated_unix':time.time()})
            time.sleep(15)


def static_files(output):
    source=ROOT/'experiments/openrouter_outpaint'
    for name in ('production-gallery.html','training-review.js','training-crop.js','fast-review.js','aspect-review.js'):
        target=output/('index.html' if name.endswith('.html') else name)
        if not target.exists() or target.read_bytes()!=(source/name).read_bytes():
            temp=target.with_suffix('.tmp');temp.write_bytes((source/name).read_bytes());temp.replace(target)
    approved=output/'approved';approved.mkdir(exist_ok=True)
    for src,dst in [('approved-training.html','index.html'),('approved-training.js','approved-training.js')]:
        (approved/dst).write_bytes((source/src).read_bytes())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--interval',type=int,default=30)
    args=parser.parse_args();args.state=args.state.resolve();args.output=args.output.resolve()
    private=ROOT/'experiments/fal_outpaint/results'
    if not all(p.is_relative_to(private) for p in (args.state,args.output)):parser.error('Use private fal results')
    if args.interval<30:parser.error('Minimum 30 seconds')
    args.output.mkdir(parents=True,exist_ok=True)
    lock=(args.output/'pipeline.lock').open('a+')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    wait_for_memory(args.output);static_files(args.output)
    fidelity=args.state.parent/'fidelity';fidelity.mkdir(exist_ok=True)
    pending_row=args.output/'audit-row.json'
    order=['3:4','4:3','9:16','16:9','2:3','3:2','4:5','5:4','1:2','2:1','1:3','3:1']
    code=('import json,sys;from pathlib import Path;'
          'sys.path.insert(0,"experiments/openrouter_outpaint");'
          'from source_fidelity import audit;from benchmark import save_json;'
          'row=json.loads(Path(sys.argv[1]).read_text());'
          'save_json(Path(sys.argv[3]),audit(row,Path(sys.argv[2]),placement_mode="free"))')
    while True:
        wait_for_memory(args.output)
        static_files(args.output)
        data=snapshot(args.state,args.output)
        print(json.dumps({'gallery_rows':data['completed'],'reliable':data['reliable'],'pending':data['audit_pending']}),flush=True)
        state=json.loads(args.state.read_text())
        sources=[r for r in state['sources'] if r['status']=='completed']
        sources.sort(key=lambda r:(order.index(r['aspect_ratio']),r['id']))
        done=0
        for source in sources:
            target=fidelity/(source['id']+'-sunburst.json')
            if target.exists():
                if json.loads(target.read_text()).get('raw_sha256')!=source['raw_sha256']:
                    raise ValueError('Audit fingerprint mismatch')
                continue
            row={'model':state['endpoint_id'],'profile':'high','track':source['id'],
                'directory':source['directory'],'raw_sha256':source['raw_sha256'],
                'raw_dimensions':source['raw_dimensions'],'source_sha256':source['sha256'],
                'actual_cost_usd':source.get('actual_cost_usd')}
            save_json(pending_row,row)
            while True:
                wait_for_memory(args.output)
                try:
                    guarded_write([sys.executable,'-c',code,str(pending_row),str(fidelity),str(target)],384,
                                  args.output/'audit-memory.jsonl')
                    break
                except RuntimeError as error:
                    if 'macOS memory pressure' not in str(error):raise
                    save_json(args.output/'pipeline-status.json',{'status':'waiting_for_normal_memory','updated_unix':time.time()})
                    time.sleep(15)
            save_json(args.output/'pipeline-status.json',{'status':'auditing','last_id':source['id'],'updated_unix':time.time()})
            done+=1
            if done%5==0:snapshot(args.state,args.output)
            print(json.dumps({'audited':source['id'],'aspect_ratio':source['aspect_ratio']}),flush=True)
        data=snapshot(args.state,args.output)
        current=json.loads(args.state.read_text())
        if not args.watch or (current['status']=='completed_target_reached' and not data['audit_pending']):
            save_json(args.output/'pipeline-status.json',{'status':'finished','rows':data['completed'],'updated_unix':time.time()})
            print(json.dumps({'status':'snapshot_finished','rows':data['completed'],'reliable':data['reliable']}),flush=True);return
        time.sleep(args.interval)


if __name__=='__main__':main()
