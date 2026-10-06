"""Serial four-teacher comparison on a local cover manifest; never retry a POST.

Run under scripts/run_bounded_model.py. Sources, receipts, raw outputs, review
votes and content scores remain local. --run sends new requests; the default
only reconciles existing receipts and builds the comparison pages.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from decimal import Decimal
import json
import re
from pathlib import Path
import subprocess
import sys
import time
import urllib.parse

from benchmark import ROOT, key_usage, request_json, save_json
from review_gallery import render_gallery
from source_fidelity import audit, render, sha, SIDE

PROFILES = [
    ('openai/gpt-image-2.5-sunburst', 'high', None),
    ('openai/gpt-image-2.5-flare', 'high', None),
    ('openai/gpt-image-2', 'high', None),
    ('google/gemini-nano-banana-2.1', None, '1K'),
]


def row_for(cell: dict, key: str) -> dict:
    row = dict(cell, reused=False)
    directory = ROOT/cell['directory']
    path = directory/'receipt.json'
    if not path.exists():
        return row
    receipt = json.loads(path.read_text())
    row.update(status=receipt['status'], elapsed_seconds=receipt.get('elapsed_seconds'),
               actual_cost_usd=receipt.get('reported_cost_usd'),
               cost_source=receipt.get('cost_source'), http_status=receipt.get('http_status'),
               error=receipt.get('error'), source_sha256=receipt['geometry']['original_sha256'],
               reference_count=len(receipt['request']['input_references']))
    generation = receipt.get('response_headers', {}).get('x-generation-id') or receipt.get('reconciled_generation_id')
    if generation:
        try:
            data, _ = request_json('generation?id='+urllib.parse.quote(generation), key)
            row.update(generation_id=generation, actual_cost_usd=data['data']['total_cost'],
                       cost_source='generation.total_cost')
            row['generation_billing'] = {k:data['data'].get(k) for k in
                ('id','model','provider_name','total_cost','created_at','finish_reason')}
        except Exception as exc:
            row['billing_lookup_error'] = type(exc).__name__
    if receipt.get('images'):
        image = receipt['images'][0]
        row.update(raw_sha256=image['sha256'], raw_dimensions=image['dimensions'])
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', required=True, type=Path)
    parser.add_argument('--results', required=True, type=Path)
    parser.add_argument('--key-file', required=True, type=Path)
    parser.add_argument('--stage', default='ardic')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--budget-usd', type=float, default=5)
    args = parser.parse_args()
    sources = json.loads(args.sources.read_text())
    if not re.fullmatch(r'[a-z][a-z0-9-]*',args.stage):
        parser.error('Stage must be a safe single-word identifier')
    if len(sources) != 5 or len({s['id'] for s in sources}) != 5:
        parser.error('Exactly five distinct cover identifiers required')
    for source in sources:
        if not re.fullmatch(r'[a-zA-Z0-9_-]+',source['id']):
            parser.error('Unsafe cover identifier')
        path = (ROOT/source['source_file']).resolve()
        if not path.is_relative_to(ROOT) or sha(path) != source['sha256']:
            parser.error('Source provenance does not match manifest')
    key = args.key_file.read_text().strip()
    results = args.results.resolve()
    if not results.is_relative_to(ROOT):
        parser.error('Results must be inside repository')
    results.mkdir(parents=True, exist_ok=True)
    batch_path = results/'batch.json'
    if batch_path.exists():
        state = json.loads(batch_path.read_text())
        if state['sources_sha256'] != sha(args.sources) or state['stage'] != args.stage:
            parser.error('Existing batch uses different sources/stage')
    else:
        state = {'stage':args.stage, 'sources_sha256':sha(args.sources),
                 'initial_key_usage':key_usage(key), 'status':'prepared', 'cells':[]}
        for source in sources:
            for model, quality, resolution in PROFILES:
                profile = quality or resolution
                directory = results/'outputs'/source['id']/model.split('/')[-1]/profile
                state['cells'].append(dict(model=model, profile=profile, quality=quality,
                    resolution=resolution, track=source['id'], status='pending',
                    directory=str(directory.relative_to(ROOT))))
        save_json(batch_path, state)
    if args.run:
        state['status'] = 'running'
        save_json(batch_path, state)
        for cell in state['cells']:
            directory = ROOT/cell['directory']
            receipt = directory/'receipt.json'
            if receipt.exists():
                prior = json.loads(receipt.read_text())['status']
                if prior not in {'completed', 'http_error', 'no_image', 'completed_output_unavailable'}:
                    state['status'] = 'paused_uncertain_request'
                    save_json(batch_path, state)
                    raise SystemExit('Uncertain request exists; reconcile before continuing')
                cell['status'] = prior
                continue
            usage = key_usage(key)
            spent = Decimal(str(usage['usage']))-Decimal(str(state['initial_key_usage']['usage']))
            # Reserve $0.50 before each request. Actual output charges are variable;
            # this preflight is not a provider-enforced spending cap.
            if spent+Decimal('.50') > Decimal(str(args.budget_usd)) or (usage.get('limit_remaining') or 0) < .50:
                state['status'] = 'paused_budget'
                save_json(batch_path, state)
                raise SystemExit('Budget preflight stopped further requests')
            source = next(s for s in sources if s['id'] == cell['track'])
            command = [sys.executable, str(Path(__file__).with_name('benchmark.py')),
                '--key-file', str(args.key_file), '--model', cell['model'], '--track', cell['track'],
                '--source-image', str(ROOT/source['source_file']), '--output', str(directory),
                '--request-timeout', '600']
            for name in ('quality','resolution'):
                if cell[name]: command += ['--'+name, cell[name]]
            cell['status'] = 'launching'
            save_json(batch_path, state)
            completed = subprocess.run(command)
            cell['status'] = json.loads(receipt.read_text())['status'] if receipt.exists() else 'launch_failed'
            save_json(batch_path, state)
            if completed.returncode or cell['status'] not in {'completed','http_error','no_image'}:
                state['status'] = 'paused_request_failure'
                save_json(batch_path, state)
                raise SystemExit('Request failure stopped batch; no POST retry')
        state['status'] = 'requests_finished'
        save_json(batch_path, state)
    rows = [row_for(cell, key) for cell in state.get('superseded_attempts', [])+state['cells']]
    charges = [Decimal(str(r['actual_cost_usd'])) for r in rows if r.get('actual_cost_usd') is not None]
    delta = Decimal(str(key_usage(key)['usage']))-Decimal(str(state['initial_key_usage']['usage']))
    known = sum(charges, Decimal(0))
    if delta == known:
        for r in rows:
            if r['status'] == 'http_error' and r.get('actual_cost_usd') is None:
                r.update(actual_cost_usd=0, cost_source='dedicated_key_reconciled_http_error')
    ledger = dict(stage=args.stage, currency='USD', rows=rows, new_charge_usd=float(known),
                  account_delta_usd=float(delta), costs_reconcile=(known == delta),
                  unknown_cost_attempts=[r['directory'] for r in rows if r['status'] != 'pending' and r.get('actual_cost_usd') is None],
                  status_counts=dict(Counter(r['status'] for r in rows)),
                  per_model_usd={m:float(sum((Decimal(str(r['actual_cost_usd'])) for r in rows
                      if r['model']==m and r.get('actual_cost_usd') is not None),Decimal(0))) for m,_,_ in PROFILES})
    save_json(results/'ledger.json', ledger)
    labels = {s['id']:s['title'] for s in sources}
    fidelity = results/'fidelity'
    fidelity.mkdir(exist_ok=True)
    scores = []
    for row in rows:
        if row['status'] != 'completed': continue
        try:
            score = audit(row, fidelity, 'free')
        except Exception as exc:
            score = dict(model=row['model'], profile=row['profile'], track=row['track'],
                         status='failed', error=type(exc).__name__+': '+str(exc), id='failed-'+str(len(scores)))
        scores.append(score)
        print(json.dumps({'event':'fidelity', 'model':row['model'], 'cover':row['track'],
                          'status':score['status'], 'alignment':score.get('alignment',{}).get('status')}),flush=True)
    metadata = dict(schema_version=2,stage=args.stage,created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                    cover_labels=labels,analysis_side=SIDE,placement_mode='free',rows=scores,
                    notes=['One sample per cover/model; five covers, not a general benchmark.',
                           'Source position/scale/rotation compensated; no local warp, color correction or source paste.',
                           'Native quality tiers are not equivalent resolutions. Common 512-pixel analysis.',
                           'Registration failure is unavailable, not poor or perfect preservation.',
                           'Source covers are 544-pixel YouTube Music album images; no native-detail or semantic guarantee.'])
    save_json(fidelity/'metrics.json', metadata)
    fields = ['model','profile','track','status','actual_cost_usd','alignment_status']
    fields += [frame+'_'+name for frame in ('fixed','aligned') for name in
               ('ssim_luminance','mae_rgb_0_255','delta_e_76_mean','pixels_within_8_rgb_fraction','evaluated_fraction')]
    with (fidelity/'metrics.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for score in scores:
            record = {k:score.get(k) for k in fields[:5]}
            record['alignment_status'] = score.get('alignment',{}).get('status')
            for frame in ('fixed','aligned'):
                for name,value in (score.get(frame) or {}).items():
                    if frame+'_'+name in fields: record[frame+'_'+name] = value
            writer.writerow(record)
    render(scores, fidelity, metadata)
    render_gallery(rows, results, ledger, state['status'], labels)
    public = {k:ledger[k] for k in ('stage','currency','new_charge_usd','account_delta_usd','costs_reconcile','status_counts','per_model_usd')}
    public['unknown_cost_count'] = len(ledger['unknown_cost_attempts'])
    public['attempts'] = [{k:r.get(k) for k in ('model','profile','track','status','http_status','actual_cost_usd','cost_source','raw_dimensions')} for r in rows]
    save_json(results/'public-costs.json', public)
    print(json.dumps(public, ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
