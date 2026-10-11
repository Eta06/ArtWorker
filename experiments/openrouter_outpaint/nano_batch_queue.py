"""Bounded direct Nano trial; reuse frozen references, receipts and batch audits.

Sunburst must remain paused. Submit at most 100 previously unattempted covers,
in small inline batches, retaining worst-case reserves until usage is known.
Never retry a generation POST or count a no-image response as a usable example.
"""
from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal
import fcntl
import json
from pathlib import Path
import time

import budget_queue as q

SUN_PAUSE=q.ROOT/'.build/teacher-queue.pause'
NANO_PAUSE=q.ROOT/'.build/nano-batch.pause'


def ensure_plan(state, limit):
    if not 1<=limit<=100:
        raise ValueError('Direct Nano trial must contain 1 to 100 covers')
    if 'nano_direct_trial' in state:
        plan=state['nano_direct_trial']
        if plan['limit']!=limit:
            raise ValueError('Existing trial has a different limit; preserve its plan')
        return plan
    candidates=[r for r in state['sources'] if r['status'] in ('pending','fallback_pending')]
    selected=[];seen=set()
    for source in candidates:
        if source['sha256'] in seen: continue
        out=q.ROOT/source['file']
        if q.sha(out)!=source['sha256']: raise ValueError('Trial source changed')
        seen.add(source['sha256']);selected.append(source['id'])
        if len(selected)==limit: break
    if len(selected)!=limit: raise ValueError('Not enough distinct unattempted covers')
    plan={'stage':'susam','limit':limit,'ids':selected,'model':q.NANO,
          'prompt_sha256':q.PROMPT_HASH,'image_size':'1K','status':'planned',
          'selection':'existing artist priority; no completed cover is regenerated',
          'training_admission':'manual review pending'}
    state['nano_direct_trial']=plan
    return plan


def run(args,state):
    if not SUN_PAUSE.exists(): raise ValueError('Sunburst pause must remain present')
    if state['prompt_sha256']!=q.PROMPT_HASH: raise ValueError('Frozen prompt mismatch')
    if any(r['status'] in ('request_started','uncertain','unresolved_error','fallback_batch_submitting') for r in state['sources']):
        raise ValueError('Unresolved request must be reconciled before trial')
    plan=ensure_plan(state,args.limit)
    q.save_json(args.state,state)
    by_id={r['id']:r for r in state['sources']}
    while True:
        if not SUN_PAUSE.exists() or NANO_PAUSE.exists():
            plan['status']='paused';q.save_json(args.state,state);return
        # All already-submitted fallback jobs also remain recoverable here.
        q.collect_google(args,state)
        pending=[i for i in plan['ids'] if by_id[i]['status'] in ('pending','fallback_pending')]
        active=sum(j['status']=='submitted' for j in state['google_jobs'])
        trial_active=any(by_id[i]['status'] in ('fallback_batch_pending','fallback_batch_submitting') for i in plan['ids'])
        if not pending and not trial_active:
            plan['status']='attempts_finished_review_pending'
            q.save_json(args.state,state);return
        capacity_wait=False
        while pending and active<args.max_active_batches:
            if not SUN_PAUSE.exists() or NANO_PAUSE.exists(): break
            chosen=pending[:2]
            try:
                q.submit_google(args,state,selected_ids=chosen)
            except ValueError as exc:
                if str(exc)!='Google reservations exceed user budget; no POST': raise
                capacity_wait=True;break
            pending=pending[len(chosen):];active+=1
            plan['status']='running';state['status']='nano_direct_batch_running'
            q.save_json(args.state,state)
        if capacity_wait and not active:
            plan['status']='budget_stop';state['status']='google_budget_stop'
            q.save_json(args.state,state);return
        print(json.dumps({'trial_counts':dict(Counter(by_id[i]['status'] for i in plan['ids'])),
                         'active_batches':active,'reservation_wait':capacity_wait}),flush=True)
        # No service requests while waiting. Small inline jobs bound response RAM.
        time.sleep(args.poll_seconds)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state',type=Path,required=True)
    parser.add_argument('--google-key-file',type=Path,default=Path('/tmp/artworker-google-session.key'))
    parser.add_argument('--limit',type=int,default=100)
    parser.add_argument('--max-active-batches',type=int,default=8)
    parser.add_argument('--poll-seconds',type=int,default=60)
    args=parser.parse_args();args.state=args.state.resolve()
    if not args.state.is_relative_to(q.ROOT/'experiments/openrouter_outpaint/results'):
        parser.error('State must remain private and ignored')
    if not 1<=args.max_active_batches<=8 or args.poll_seconds<30:
        parser.error('At most eight two-image batches; polls at least 30 seconds apart')
    with (args.state.parent/'queue.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=json.loads(args.state.read_text())
        try: run(args,state)
        except Exception as exc:
            state['nano_direct_trial_error']=type(exc).__name__
            state['status']='nano_direct_reconciliation_required'
            q.save_json(args.state,state);raise


if __name__=='__main__': main()
