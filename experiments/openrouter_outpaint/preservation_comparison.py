"""Offline comparison of new ledgers and preserved baseline outputs. No API calls."""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import time

from benchmark import ROOT, save_json
from review_gallery import render_gallery
from source_fidelity import SIDE, audit, render


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-ledger',type=Path,action='append',default=[])
    parser.add_argument('--ledger',type=Path,action='append',required=True)
    parser.add_argument('--sources',type=Path,required=True)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--stage',required=True)
    args=parser.parse_args()
    results=args.results.resolve()
    if not results.is_relative_to(ROOT): parser.error('Results must be inside repository')
    if (results/'index.html').exists(): parser.error('Choose a fresh report directory')
    sources=json.loads(args.sources.read_text())
    labels={s['id']:s['title'] for s in sources}
    rows=[]
    charges=Decimal(0)
    reconciled=True
    for baseline,paths in [(True,args.baseline_ledger),(False,args.ledger)]:
        for path in paths:
            ledger=json.loads(path.read_text())
            rows.extend(dict(r,reused=baseline) for r in ledger['rows']
                        if not baseline or r['status']=='completed')
            if not baseline:
                charges+=Decimal(str(ledger['new_charge_usd']))
                reconciled=reconciled and ledger.get('costs_reconcile') is True
    results.mkdir(parents=True,exist_ok=True)
    # Keep browser URLs inside this report even when served through a shallow
    # stage symlink. These private aliases reuse media without copying it.
    media=results/'media';media.mkdir()
    for row in rows:
        original=(ROOT/row['directory']).resolve()
        if not original.is_relative_to(ROOT) or not original.is_dir():
            raise ValueError('Media directory must exist inside repository')
        alias=media/hashlib.sha256(row['directory'].encode()).hexdigest()[:20]
        if not alias.exists(): alias.symlink_to(original,target_is_directory=True)
        row['original_directory']=row['directory']
        row['directory']=str(alias.relative_to(ROOT))
    fidelity=results/'fidelity';fidelity.mkdir()
    scores=[]
    for row in rows:
        if row['status']!='completed':continue
        score=audit(row,fidelity,'free')
        score['raw_url']='../media/'+Path(row['directory']).name+'/raw.png'
        scores.append(score)
        print(json.dumps({'cover':row['track'],'profile':row['profile'],'model':row['model'],
                          'alignment':score['alignment']['status'],'method':score['alignment'].get('method')}),flush=True)
    metadata=dict(schema_version=2,stage=args.stage,created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                  cover_labels=labels,analysis_side=SIDE,placement_mode='free',rows=scores,
                  notes=['ORB then SIFT, unchanged reliability gates; no local warping or color correction.',
                         'Unavailable alignment may show an unverified candidate outline, never a preservation score.',
                         'Baseline media and votes preserved; no new model requests from this report.'])
    save_json(fidelity/'metrics.json',metadata)
    fields=['model','profile','track','actual_cost_usd','alignment_status','alignment_method']
    fields += [frame+'_'+name for frame in ('fixed','aligned') for name in
               ('ssim_luminance','mae_rgb_0_255','pixels_within_8_rgb_fraction','pixels_within_16_rgb_fraction','worst_tile_ssim','pixel_exact_on_analysis_grid_fraction')]
    with (fidelity/'metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in scores:
            data={k:r.get(k) for k in fields[:4]}
            data.update(alignment_status=r['alignment']['status'],alignment_method=r['alignment'].get('method'))
            for frame in ('fixed','aligned'):
                for k in fields:
                    if k.startswith(frame+'_'):data[k]=(r.get(frame) or {}).get(k[len(frame)+1:])
            writer.writerow(data)
    render(scores,fidelity,metadata)
    ledger=dict(stage=args.stage,rows=rows,new_charge_usd=float(charges),costs_reconcile=reconciled)
    save_json(results/'ledger.json',ledger)
    render_gallery(rows,results,ledger,'completed_with_provider_errors',labels)


if __name__=='__main__':main()
