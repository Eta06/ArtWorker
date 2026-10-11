"""Sunburst until its existing budget stops, then Nano for remaining covers.

Reuses durable receipts, native Google batch reservations and isolated audits.
Neither ambiguous paid requests nor completed covers are automatically repeated.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
import time

import budget_queue as q


def paused():
    return any((q.ROOT/'.build'/name).exists() for name in ('teacher-queue.pause','nano-batch.pause'))


def pump_google(args,state,remaining=False):
    """Fallbacks first; untouched covers only after the primary budget stops."""
    blocked=False
    while sum(j['status']=='submitted' for j in state['google_jobs'])<args.max_active_batches:
        if paused(): break
        ids=[r['id'] for r in state['sources'] if r['status']=='fallback_pending']
        if remaining:
            ids.extend(r['id'] for r in state['sources'] if r['status']=='pending')
        if not ids: break
        try: q.submit_google(args,state,selected_ids=ids[:2])
        except ValueError as exc:
            if str(exc)!='Google reservations exceed user budget; no POST': raise
            blocked=True;break
    return blocked


def run(args,state):
    if state['prompt_sha256']!=q.PROMPT_HASH: raise ValueError('Frozen prompt mismatch')
    if any(r['status'] in ('request_started','uncertain','unresolved_error','fallback_batch_submitting') for r in state['sources']):
        raise ValueError('Unresolved earlier request; reconcile before more POSTs')
    if any(j['status']=='submit_uncertain' for j in state['google_jobs']):
        raise ValueError('Uncertain batch submission; reconcile before more POSTs')
    plan=state.setdefault('teacher_pipeline',{'stage':'reyhan','phase':'sunburst',
        'policy':'Sunburst primary; Nano refusals during primary, then remaining within existing budgets'})
    if 'nano_direct_trial' in state:
        state['nano_direct_trial']['status']='superseded_by_teacher_pipeline'
    q.save_json(args.state,state)
    while True:
        if paused():
            state['status']='teacher_pipeline_paused';q.save_json(args.state,state);return
        q.collect_google(args,state)
        if plan['phase']=='sunburst':
            if any(r['status']=='pending' for r in state['sources']):
                args.limit=9
                q.run_parallel_sun(args,state)
                if state['status']=='paused_between_waves': return
            else:
                state['status']='primary_sources_exhausted'
            if state['status'] in ('openrouter_budget_stop','primary_sources_exhausted'):
                plan['phase']='nano_remaining'
                plan['primary_stop_reason']=state['status']
                q.save_json(args.state,state)
                print(json.dumps({'phase':'nano_remaining','reason':plan['primary_stop_reason']}),flush=True)
            else:
                pump_google(args,state)
                state['status']='teacher_pipeline_sunburst'
                q.save_json(args.state,state)
                continue
        blocked=pump_google(args,state,remaining=True)
        active=sum(j['status']=='submitted' for j in state['google_jobs'])
        eligible=any(r['status'] in ('pending','fallback_pending') for r in state['sources'])
        if not active and (blocked or not eligible):
            plan['status']='budget_stop' if blocked else 'attempts_finished_review_pending'
            state['status']='teacher_pipeline_'+plan['status']
            q.save_json(args.state,state);return
        state['status']='teacher_pipeline_nano_remaining'
        q.save_json(args.state,state)
        print(json.dumps({'phase':plan['phase'],'counts':dict(Counter(r['status'] for r in state['sources'])),
                         'active_google_batches':active,'reservation_wait':blocked}),flush=True)
        time.sleep(args.poll_seconds)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state',type=Path,required=True)
    parser.add_argument('--key-file',type=Path,default=Path('/tmp/artworker-openrouter-session.key'))
    parser.add_argument('--google-key-file',type=Path,default=Path('/tmp/artworker-google-session.key'))
    parser.add_argument('--workers',type=int,choices=[3],default=3)
    parser.add_argument('--max-active-batches',type=int,default=8)
    parser.add_argument('--poll-seconds',type=int,default=60)
    args=parser.parse_args();args.state=args.state.resolve()
    if not args.state.is_relative_to(q.ROOT/'experiments/openrouter_outpaint/results'):
        parser.error('State must remain private and ignored')
    if not 1<=args.max_active_batches<=8 or args.poll_seconds<30:
        parser.error('At most eight batches; polls at least 30 seconds apart')
    with (args.state.parent/'queue.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=json.loads(args.state.read_text())
        try: run(args,state)
        except Exception as exc:
            state.update(status='teacher_pipeline_reconciliation_required',pipeline_error=type(exc).__name__)
            q.save_json(args.state,state);raise


if __name__=='__main__': main()
