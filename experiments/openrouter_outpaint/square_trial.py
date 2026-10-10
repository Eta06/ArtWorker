"""Three-cover, single-square-reference trial. No automatic paid POST retries.

Default prepares inputs and a local gallery without network access. --run uses
the private key file; execute it under scripts/run_bounded_model.py.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from decimal import Decimal
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from benchmark import ROOT, SQUARE_PROMPT, key_usage, prepare, save_json
from cover_consistency import row_for
from review_gallery import render_gallery
from source_fidelity import SIDE, audit, render, sha

PROFILES = [
    ('google/gemini-nano-banana-2.1', None, '1K'),
    ('openai/gpt-image-2.5-sunburst', 'high', None),
    ('openai/gpt-image-2.5-flare', 'high', None),
]


def existing_row(cell: dict, key: str | None) -> dict:
    path = ROOT/cell['directory']/'receipt.json'
    if not path.exists():
        return dict(cell)
    if key:
        return row_for(cell, key)
    receipt = json.loads(path.read_text())
    row = dict(cell, status=receipt['status'], elapsed_seconds=receipt.get('elapsed_seconds'),
               actual_cost_usd=receipt.get('reported_cost_usd'), cost_source=receipt.get('cost_source'),
               http_status=receipt.get('http_status'), source_sha256=receipt['geometry']['original_sha256'])
    if receipt.get('images'):
        row.update(raw_sha256=receipt['images'][0]['sha256'], raw_dimensions=receipt['images'][0]['dimensions'])
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', required=True, type=Path)
    parser.add_argument('--results', required=True, type=Path)
    parser.add_argument('--stage', default='filiz')
    parser.add_argument('--key-file', type=Path)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--budget-usd', type=Decimal, default=Decimal('2'))
    parser.add_argument('--models', nargs='+', choices=[m for m,_,_ in PROFILES],
                        help='Run only these models in a fresh trial; existing receipts remain untouched')
    args = parser.parse_args()
    profiles = [p for p in PROFILES if args.models is None or p[0] in args.models]
    sources = json.loads(args.sources.read_text())
    if not re.fullmatch(r'[a-z][a-z0-9-]*', args.stage):
        parser.error('Unsafe stage name')
    if len(sources) != 3 or len({s['id'] for s in sources}) != 3:
        parser.error('Exactly three distinct covers required')
    if args.budget_usd <= 0:
        parser.error('Positive budget required')
    for source in sources:
        path = (ROOT/source['source_file']).resolve()
        if not re.fullmatch(r'[a-zA-Z0-9_-]+', source['id']) or not path.is_relative_to(ROOT) or sha(path) != source['sha256']:
            parser.error('Unsafe source or changed provenance')
    results = args.results.resolve()
    if not results.is_relative_to(ROOT):
        parser.error('Results must be inside repository')
    results.mkdir(parents=True, exist_ok=True)
    batch_path = results/'batch.json'
    protocol = dict(stage=args.stage, sources_sha256=sha(args.sources),
                    input_mode='square', prompt=SQUARE_PROMPT,
                    profiles=profiles, aspect_ratio='9:16')
    # Round-trip tuples to JSON lists before protocol comparison.
    protocol = json.loads(json.dumps(protocol))
    if batch_path.exists():
        state = json.loads(batch_path.read_text())
        if state['protocol'] != protocol:
            parser.error('Existing trial has a different protocol; use a fresh directory')
    else:
        state = dict(protocol=protocol, status='prepared_no_requests', cells=[])
        for source in sources:
            for model, quality, resolution in profiles:
                directory = results/'outputs'/source['id']/model.split('/')[-1]/(quality or resolution)
                directory.mkdir(parents=True)
                prepare(source['id'], directory, ROOT/source['source_file'])
                state['cells'].append(dict(model=model, profile=quality or resolution,
                    quality=quality, resolution=resolution, track=source['id'],
                    directory=str(directory.relative_to(ROOT)), status='pending', reused=False))
        save_json(batch_path, state)
    key = None
    if args.run:
        if not args.key_file:
            parser.error('--run requires --key-file')
        key = args.key_file.read_text().strip()
        usage = key_usage(key)  # Authentication preflight, before any POST.
        state.setdefault('initial_key_usage', usage)
        state['status'] = 'running'
        save_json(batch_path, state)
        for cell in state['cells']:
            receipt = ROOT/cell['directory']/'receipt.json'
            if receipt.exists():
                status = json.loads(receipt.read_text())['status']
                if status not in {'completed', 'http_error', 'no_image', 'completed_output_unavailable'}:
                    raise SystemExit('Uncertain prior request; reconcile it before proceeding')
                cell['status'] = status
                continue
            usage = key_usage(key)
            spent = Decimal(str(usage['usage']))-Decimal(str(state['initial_key_usage']['usage']))
            if spent+Decimal('.50') > args.budget_usd or Decimal(str(usage.get('limit_remaining') or 0)) < Decimal('.50'):
                state['status'] = 'paused_budget'
                save_json(batch_path, state)
                raise SystemExit('Budget preflight stopped new requests')
            source = next(s for s in sources if s['id'] == cell['track'])
            command = [sys.executable, str(Path(__file__).with_name('benchmark.py')),
                       '--key-file', str(args.key_file), '--model', cell['model'],
                       '--track', cell['track'], '--source-image', str(ROOT/source['source_file']),
                       '--output', str(ROOT/cell['directory']), '--input-mode', 'square',
                       '--request-timeout', '600']
            for name in ('quality', 'resolution'):
                if cell[name]: command += ['--'+name, cell[name]]
            completed = subprocess.run(command)
            cell['status'] = json.loads(receipt.read_text())['status'] if receipt.exists() else 'launch_failed'
            save_json(batch_path, state)
            if completed.returncode or cell['status'] not in {'completed', 'http_error', 'no_image'}:
                state['status'] = 'paused_request_failure'
                save_json(batch_path, state)
                raise SystemExit('Request outcome uncertain; no automatic POST retry')
        state['status'] = 'requests_finished'
        save_json(batch_path, state)
    rows = [existing_row(c, key) for c in state['cells']]
    known = sum((Decimal(str(r['actual_cost_usd'])) for r in rows if r.get('actual_cost_usd') is not None), Decimal(0))
    delta = Decimal(str(key_usage(key)['usage']))-Decimal(str(state['initial_key_usage']['usage'])) if key else None
    # The account counter can lag behind completed generation receipts. GET only;
    # never resend a generation to resolve accounting.
    deadline = time.monotonic()+30
    while key and delta != known and time.monotonic() < deadline:
        time.sleep(3)
        delta = Decimal(str(key_usage(key)['usage']))-Decimal(str(state['initial_key_usage']['usage']))
    if delta is not None and delta == known:
        for row in rows:
            if row['status'] == 'http_error' and row.get('actual_cost_usd') is None:
                row.update(actual_cost_usd=0, cost_source='key_delta_reconciled_http_error')
    ledger = dict(stage=args.stage, currency='USD', rows=rows, new_charge_usd=float(known),
                  account_delta_usd=float(delta) if delta is not None else None,
                  costs_reconcile=delta == known if delta is not None else None,
                  status_counts=dict(Counter(r['status'] for r in rows)),
                  per_model_usd={m:float(sum((Decimal(str(r['actual_cost_usd'])) for r in rows
                    if r['model']==m and r.get('actual_cost_usd') is not None),Decimal(0))) for m,_,_ in profiles},
                  unknown_cost_count=sum(r['status'] != 'pending' and r.get('actual_cost_usd') is None for r in rows))
    save_json(results/'ledger.json', ledger)
    labels = {s['id']:s['title'] for s in sources}
    scores = []
    if any(r['status'] == 'completed' for r in rows):
        fidelity = results/'fidelity'
        fidelity.mkdir(exist_ok=True)
        for row in rows:
            if row['status'] != 'completed': continue
            try:
                scores.append(audit(row, fidelity, 'free'))
            except Exception as exc:
                scores.append(dict(model=row['model'], profile=row['profile'], track=row['track'],
                                   status='failed', id='failed-'+str(len(scores)), error=type(exc).__name__+': '+str(exc)))
        metadata = dict(schema_version=2, stage=args.stage, cover_labels=labels,
                        created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                        analysis_side=SIDE, placement_mode='free', rows=scores,
                        notes=['Single square reference and short prompt; no layout canvas transmitted.',
                               'Fixed-location metrics use a diagnostic centered canvas, not a model constraint.',
                               'One sample per cell; prompt and reference count changed together.',
                               'Aligned scores are numerical diagnostics, not automatic visual acceptance.'])
        save_json(fidelity/'metrics.json', metadata)
        fields = ['model','profile','track','status','actual_cost_usd','alignment_status']
        fields += [frame+'_'+name for frame in ('fixed','aligned') for name in
                   ('ssim_luminance','mae_rgb_0_255','delta_e_76_mean',
                    'pixels_within_8_rgb_fraction','evaluated_fraction')]
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
    public = {k:v for k,v in ledger.items() if k != 'rows'}
    public['attempts'] = [{k:r.get(k) for k in ('model','profile','track','status','http_status','actual_cost_usd','cost_source','raw_dimensions')} for r in rows]
    save_json(results/'public-costs.json', public)
    print(json.dumps(public, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
