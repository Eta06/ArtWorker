"""Serial, resumable prompt experiments with receipts and an account-budget gate.

A private plan defines exact source/model/prompt cells. No automatic POST retry;
an uncertain outcome stops the run. Reporting is offline and separate.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import json
from pathlib import Path
import subprocess
import sys
import time

from benchmark import ROOT, key_usage, save_json
from cover_consistency import row_for
from source_fidelity import sha


def reconcile(cells: list[dict], key: str, initial: dict, stage: str) -> dict:
    rows=[row_for(cell,key) for cell in cells]
    known=sum((Decimal(str(r['actual_cost_usd'])) for r in rows
               if r.get('actual_cost_usd') is not None),Decimal(0))
    delta=Decimal(str(key_usage(key)['usage']))-Decimal(str(initial['usage']))
    if known == delta:
        for row in rows:
            if row['status']=='http_error' and row.get('actual_cost_usd') is None:
                row.update(actual_cost_usd=0,cost_source='settled_key_delta_http_error')
    return dict(stage=stage,rows=rows,new_charge_usd=float(known),account_delta_usd=float(delta),
                costs_reconcile=known==delta,unknown_cost_count=sum(r['status']!='pending' and
                r.get('actual_cost_usd') is None for r in rows))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--key-file',type=Path,required=True)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--budget-usd',type=Decimal,default=Decimal('4'))
    args=parser.parse_args()
    results=args.results.resolve()
    if not results.is_relative_to(ROOT) or args.budget_usd<=0:parser.error('Invalid results/budget')
    plan=json.loads(args.plan.read_text())
    for cell in plan['cells']:
        for field in ('directory','source_file','prompt_file'):
            if not (ROOT/cell[field]).resolve().is_relative_to(ROOT):parser.error('Plan path outside repository')
        if sha(ROOT/cell['source_file'])!=cell['source_sha256']:parser.error('Source changed')
        if sha(ROOT/cell['prompt_file'])!=cell['prompt_sha256']:parser.error('Prompt changed')
    key=args.key_file.read_text().strip()
    results.mkdir(parents=True,exist_ok=True)
    state_path=results/'batch.json'
    if state_path.exists():
        state=json.loads(state_path.read_text())
        if state['plan_sha256']!=sha(args.plan):parser.error('Existing run has a different plan')
    else:
        state=dict(stage=plan['stage'],plan_sha256=sha(args.plan),initial_key_usage=key_usage(key),
                   cells=plan['cells'],status='prepared')
        save_json(state_path,state)
    if args.run:
        state['status']='running';save_json(state_path,state)
        for cell in state['cells']:
            receipt=ROOT/cell['directory']/'receipt.json'
            if receipt.exists():
                prior=json.loads(receipt.read_text())['status']
                if prior not in {'completed','http_error','no_image'}:
                    raise SystemExit('Uncertain receipt; reconcile before continuing')
                continue
            usage=key_usage(key)
            spent=Decimal(str(usage['usage']))-Decimal(str(state['initial_key_usage']['usage']))
            # Account counters can lag. Never discard already reported charges
            # when deciding whether another paid request fits this batch.
            reported=Decimal(0)
            for prior_cell in state['cells']:
                prior_path=ROOT/prior_cell['directory']/'receipt.json'
                if prior_path.exists():
                    cost=json.loads(prior_path.read_text()).get('reported_cost_usd')
                    if cost is not None:reported+=Decimal(str(cost))
            spent=max(spent,reported)
            if spent+Decimal('.50')>args.budget_usd or Decimal(str(usage.get('limit_remaining') or 0))<Decimal('.50'):
                state['status']='paused_budget';save_json(state_path,state)
                raise SystemExit('Account budget gate stopped dispatch')
            cmd=[sys.executable,str(Path(__file__).with_name('benchmark.py')),
                 '--key-file',str(args.key_file),'--model',cell['model'],'--track',cell['track'],
                 '--source-image',str(ROOT/cell['source_file']),'--output',str(ROOT/cell['directory']),
                 '--input-mode','square','--prompt-file',str(ROOT/cell['prompt_file']),
                 '--request-timeout','600']
            for field in ('quality','resolution'):
                if cell.get(field):cmd.extend(['--'+field,cell[field]])
            completed=subprocess.run(cmd)
            status=json.loads(receipt.read_text())['status'] if receipt.exists() else 'launch_failed'
            cell['status']=status;save_json(state_path,state)
            save_json(results/'ledger.json',reconcile(state['cells'],key,state['initial_key_usage'],plan['stage']))
            if completed.returncode or status not in {'completed','http_error','no_image'}:
                raise SystemExit('Uncertain request stopped dispatch; no retry')
        state['status']='requests_finished';save_json(state_path,state)
    ledger=reconcile(state['cells'],key,state['initial_key_usage'],plan['stage'])
    save_json(results/'ledger.json',ledger)
    print(json.dumps({k:v for k,v in ledger.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
