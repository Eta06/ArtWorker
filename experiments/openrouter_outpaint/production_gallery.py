"""Incremental local gallery of existing production outputs and source audits.

No provider calls or secret reads. A lightweight watcher publishes snapshots
atomically while the producer continues; it never reruns paid generation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

from PIL import Image

ROOT=Path(__file__).resolve().parents[2]


def region(audit):
    a=audit.get('alignment',{})
    if a.get('status')!='measured' or not audit.get('aligned'): return None
    side=audit['analysis_side'];w,h=audit['raw_dimensions']
    matrix=a['matrix_source_to_output']
    working_h=round(h*side/w)
    points=[[(matrix[0][0]*x+matrix[0][1]*y+matrix[0][2])*w/side,
             (matrix[1][0]*x+matrix[1][1]*y+matrix[1][2])*h/working_h]
            for x,y in ((0,0),(side,0),(side,side),(0,side))]
    if not all(math.isfinite(v) for p in points for v in p): return None
    xs,ys=zip(*points)
    return {'polygon':points,'margins_output_px':{'left':min(xs),'right':w-max(xs),
            'top':min(ys),'bottom':h-max(ys)},'rotation_degrees':a['rotation_degrees'],
            'note':'Approximate output-pixel distances to the registered polygon bounding box; negative means clipped source.'}


def alias(source,target):
    source=source.resolve()
    if not source.is_relative_to(ROOT) or not source.is_file(): raise ValueError('Asset outside workspace or missing')
    if target.is_symlink():
        if target.resolve()!=source: raise ValueError('Existing alias has different source')
    elif target.exists(): raise ValueError('Refuse overwriting existing asset')
    else:target.symlink_to(source)


def thumbnail(source,target):
    if target.exists(): return
    with Image.open(source) as im:
        im.thumbnail((420,700),Image.Resampling.LANCZOS)
        tmp=target.with_suffix('.tmp')
        im.convert('RGB').save(tmp,format='JPEG',quality=86)
        tmp.replace(target)


def snapshot(state_path,output):
    state=json.loads(state_path.read_text());rows=[];fidelity=state_path.parent/'fidelity'
    for source in state['sources']:
        kind='sunburst' if source['status']=='sunburst_completed_review_pending' else 'nano' if source['status']=='nano_completed_review_pending' else None
        if not kind:continue
        directory=ROOT/source['directory' if kind=='sunburst' else 'fallback_directory']
        receipt=json.loads((directory/'receipt.json').read_text())
        images=receipt.get('images',[])
        if len(images)!=1 or not (directory/'raw.png').is_file():continue
        audit_path=fidelity/(source['id']+'-'+kind+'.json')
        audit=json.loads(audit_path.read_text()) if audit_path.exists() else None
        if audit and audit['raw_sha256']!=images[0]['sha256']:raise ValueError('Audit/output fingerprint mismatch')
        name=source['id']+'-'+kind;asset=output/'assets'/name;asset.mkdir(parents=True,exist_ok=True)
        for filename in ('raw.png','source.png','input.png'):alias(directory/filename,asset/filename)
        original=ROOT/source['file'];alias(original,asset/('original'+original.suffix))
        thumbnail(directory/'raw.png',asset/'preview.jpg')
        thumbnail(original,asset/'original-preview.jpg')
        if audit:
            audit_assets=fidelity/audit['id']
            for filename in ('source.png','aligned.png','aligned-overlay.png','aligned-difference.png','placement.png'):
                if (audit_assets/filename).exists():alias(audit_assets/filename,asset/('audit-'+filename))
        names=[a if isinstance(a,str) else a.get('name','') for a in source['artists']]
        album=source['album'];album=album if isinstance(album,str) else album.get('name',source['id'])
        aligned=audit.get('aligned') if audit else None
        matched=bool(audit and audit.get('alignment',{}).get('status')=='measured' and aligned)
        rows.append({'id':name,'cover_id':source['id'],'model':kind,'title':album,'artists':names,
            'raw_sha256':images[0]['sha256'],'dimensions':images[0]['dimensions'],
            'asset':'assets/'+name+'/','original_file':'original'+original.suffix,
            'audit_status':'reliable' if matched else 'unavailable' if audit else 'pending',
            'alignment':audit.get('alignment') if audit else None,'metrics':aligned if matched else None,
            'region':region(audit) if audit else None,
            'actual_cost_usd':receipt.get('reported_cost_usd'),
            'estimated_cost_usd':receipt.get('estimated_cost_usd'),
            'prompt':receipt['request']['prompt'],
            'settings':{k:receipt['request'][k] for k in ('quality','resolution','aspect_ratio','response_format','generationConfig') if k in receipt['request']}})
    counts=dict(Counter(r['status'] for r in state['sources']))
    data={'updated_utc':datetime.now(timezone.utc).isoformat(),'queue_status':state['status'],
          'phase':state.get('teacher_pipeline',{}).get('phase'),'status_counts':counts,
          'active_batches':sum(j['status']=='submitted' for j in state['google_jobs']),
          'completed':len(rows),'reliable':sum(r['audit_status']=='reliable' for r in rows),
          'audit_unavailable':sum(r['audit_status']=='unavailable' for r in rows),
          'audit_pending':sum(r['audit_status']=='pending' for r in rows),'rows':rows}
    temp=output/'data.tmp';temp.write_text(json.dumps(data,ensure_ascii=False,allow_nan=False));temp.replace(output/'data.json')
    return data


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--watch',action='store_true');p.add_argument('--interval',type=int,default=30)
    args=p.parse_args();args.state=args.state.resolve();args.output=args.output.resolve()
    private=ROOT/'experiments/openrouter_outpaint/results'
    if not args.state.is_relative_to(private) or not args.output.is_relative_to(private):p.error('Keep gallery and state inside ignored results')
    if args.interval<30:p.error('Refresh interval must be at least 30 seconds')
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'index.html').write_text(Path(__file__).with_name('production-gallery.html').read_text())
    if args.watch:
        # Separate short snapshots have a 256 MiB/pressure guard. They do not
        # acquire the heavy-model lock held by the running teacher coordinator.
        sys.path.insert(0,str(ROOT))
        from scripts.commit_stage import guarded_write
        previous=None
        while True:
            current=args.state.stat().st_mtime_ns
            if current!=previous:
                guarded_write([sys.executable,str(Path(__file__).resolve()),'--state',str(args.state),
                    '--output',str(args.output)],256,args.output/'snapshot-memory.jsonl')
                previous=current
                data=json.loads((args.output/'data.json').read_text())
                terminal=('budget_stop' in data['queue_status'] or 'attempts_finished' in data['queue_status'])
                if terminal and not data['active_batches'] and not data['audit_pending']:
                    print('Final snapshot published; all completed images have audit results.',flush=True)
                    return
            time.sleep(args.interval)
        return
    previous=None
    while True:
        current=args.state.stat().st_mtime_ns
        if current!=previous:
            data=snapshot(args.state,args.output);previous=current
            print(json.dumps({k:data[k] for k in ('completed','reliable','audit_unavailable','audit_pending','queue_status')}),flush=True)
        if not args.watch:return
        time.sleep(args.interval)


if __name__=='__main__':main()
