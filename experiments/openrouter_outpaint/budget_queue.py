"""Serial Sunburst queue and separately submitted Google batch fallbacks.

Run under run_bounded_model.py. Durable receipts forbid blind POST retries.
All plans, balances, source art and outputs must live in ignored results/.
Google token estimates are never reported as settled billing charges.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
import fcntl
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from PIL import Image
from benchmark import ROOT, PROMPT, prepare, request_json, save_json

SUN = 'openai/gpt-image-2.5-sunburst'
NANO = 'gemini-nano-banana-2.1'
GOOGLE = 'https://generativelanguage.googleapis.com/v1beta/'
PROMPT_HASH = '736716dd0146c1cbda160d21c6615dbe78e494a6e98dc779cb94701bd006efc8'
FX = Decimal('49.27')  # Budget conversion assumption, not Google's invoice FX.
MAX_OUTPUT = 8192


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def credit(key):
    d, _ = request_json('credits', key)
    return Decimal(str(d['data']['total_credits'])) - Decimal(str(d['data']['total_usage']))


def google(path, key, payload=None):
    request = urllib.request.Request(GOOGLE + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={'x-goog-api-key': key, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=120 if payload else 60) as response:
        raw = response.read(64 * 1024 * 1024 + 1)
        if len(raw) > 64 * 1024 * 1024:
            raise ValueError('Google response exceeds 64 MiB cap; do not repeat POST')
        return json.loads(raw)


def select(manifest):
    """Round robin the requested Turkish artists, then other Turkish art, then rest."""
    rows = manifest['covers']
    def names(row):
        return [a if isinstance(a, str) else a.get('name', '') for a in row['artists']]
    groups = [[r for r in rows if artist in names(r)]
              for artist in ('UZI', 'Motive', 'Güneş', 'Murda')]
    result, seen = [], set()
    for i in range(max(map(len, groups))):
        for group in groups:
            if i < len(group) and group[i]['sha256'] not in seen:
                result.append(group[i]); seen.add(group[i]['sha256'])
    for row in sorted(rows, key=lambda r: r['origin'] != 'turkish_focus_discography'):
        if row['sha256'] not in seen:
            result.append(row); seen.add(row['sha256'])
    return result


def init(args):
    if args.budget_usd is None or args.google_budget_tl is None or args.budget_usd <= 0 or args.google_budget_tl < 0:
        raise ValueError('Explicit positive primary budget and nonnegative fallback budget required')
    if args.state.exists():
        raise ValueError('State exists; refusing to reset spending baseline')
    manifest = json.loads(args.manifest.read_text())
    base = args.manifest.resolve().parent
    sources = []
    for row in select(manifest):
        path = (base / row['file']).resolve()
        if not path.is_relative_to(ROOT) or sha(path) != row['sha256']:
            raise ValueError('Cover provenance mismatch')
        sources.append({'id': row['id'], 'sha256': row['sha256'],
                        'file': str(path.relative_to(ROOT)), 'artists': row['artists'],
                        'album': row['album'], 'status': 'pending'})
    args.state.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.state, {'stage': 'mahlep', 'schema': 1, 'sources': sources,
        'prompt_sha256': PROMPT_HASH, 'initial_openrouter_credit_usd': str(credit(args.key_file.read_text().strip())),
        'openrouter_budget_usd': str(args.budget_usd),
        'google_budget_tl': str(args.google_budget_tl), 'google_fx_assumption': str(FX),
        'google_balance_source': 'user supplied; not independently verified',
        'google_jobs': [], 'status': 'prepared', 'training_admission': 'manual review required'})


def committed_spend(state, current):
    known = sum((Decimal(str(r.get('actual_cost_usd') or 0)) for r in state['sources']), Decimal(0))
    account_delta = max(Decimal(0), Decimal(state['initial_openrouter_credit_usd']) - current)
    return max(known, account_delta)


def refusal(receipt):
    if receipt.get('http_status') not in (400, 403, 422):
        return False
    text = receipt.get('error', '').lower()
    return any(s in text for s in ('safety', 'content_policy', 'content policy', 'moderation', 'refusal'))


def run_sun(args, state):
    key = args.key_file.read_text().strip()
    catalog, _ = request_json('images/models/' + SUN + '/endpoints', key)
    if catalog['endpoints'][0]['supported_parameters'].get('input_references', {}).get('max', 0) < 2:
        raise ValueError('Two-reference baseline unsupported; no request sent')
    from cover_consistency import row_for
    from source_fidelity import audit
    count = 0
    for source in state['sources']:
        if count >= args.limit:
            break
        if source['status'] != 'pending':
            if source['status'] in ('request_started', 'uncertain', 'unresolved_error'):
                raise ValueError('Unresolved earlier request; reconcile before continuing')
            continue
        current = credit(key)
        spent = committed_spend(state, current)
        key_data, _ = request_json('key', key)
        remaining_key = key_data['data'].get('limit_remaining')
        # A conservative preflight reserve, not a provider-enforced price cap.
        reserve = Decimal('0.10')
        if min(Decimal(state['openrouter_budget_usd']) - spent, current,
               Decimal(str(remaining_key)) if remaining_key is not None else current) < reserve:
            state.update(status='openrouter_budget_stop', openrouter_spend_usd=str(spent))
            save_json(args.state, state); break
        path = ROOT / source['file']
        if sha(path) != source['sha256']:
            raise ValueError('Source changed before request')
        out = args.state.parent / 'outputs' / source['id'] / 'gpt-image-2.5-sunburst' / 'high'
        if (out / 'receipt.json').exists():
            raise ValueError('Existing receipt for pending item; do not repeat request')
        source.update(status='request_started', directory=str(out.relative_to(ROOT)))
        state['status'] = 'running_sunburst'; save_json(args.state, state)
        subprocess.run([sys.executable, str(Path(__file__).with_name('benchmark.py')),
            '--key-file', str(args.key_file), '--model', SUN, '--track', source['id'],
            '--source-image', str(path), '--output', str(out), '--quality', 'high',
            '--input-mode', 'canvas', '--request-timeout', '600',
            '--require-prompt-sha256', PROMPT_HASH, '--require-reference-count', '2'], check=True)
        receipt = json.loads((out / 'receipt.json').read_text())
        if hashlib.sha256(receipt['request']['prompt'].encode()).hexdigest() != PROMPT_HASH or len(receipt['request']['input_references']) != 2:
            raise ValueError('Frozen baseline drift; stop')
        row = row_for({'directory': source['directory'], 'track': source['id'],
                       'model': SUN, 'profile': 'high'}, key)
        source['measurement_row'] = row
        source['actual_cost_usd'] = row.get('actual_cost_usd')
        if receipt['status'] == 'completed' and row.get('actual_cost_usd') is not None:
            source['status'] = 'sunburst_completed_review_pending'
        elif refusal(receipt):
            source['status'] = 'fallback_pending'
        elif receipt['status'] == 'no_image' and row.get('actual_cost_usd') is not None:
            source['status'] = 'fallback_pending'
        else:
            source['status'] = 'uncertain' if receipt['status'] in ('request_started', 'uncertain_or_decode_error') else 'unresolved_error'
        current = credit(key)
        state['openrouter_spend_usd'] = str(committed_spend(state, current))
        if row.get('actual_cost_usd') is not None and Decimal(str(row['actual_cost_usd'])) > reserve:
            state['status'] = 'unexpected_price_stop'
            save_json(args.state, state)
            raise ValueError('Actual call price exceeds preflight reserve; re-estimate before more POSTs')
        save_json(args.state, state)
        if source['status'] == 'sunburst_completed_review_pending':
            destination = args.state.parent/'fidelity'
            destination.mkdir(exist_ok=True)
            measurement = audit(row, destination, placement_mode='free')
            save_json(destination/(source['id']+'-sunburst.json'), measurement)
            source['fidelity_file'] = str((destination/(source['id']+'-sunburst.json')).relative_to(ROOT))
            save_json(args.state,state)
        print(json.dumps({'id': source['id'], 'status': source['status'],
                          'spend_usd': state['openrouter_spend_usd']}, ensure_ascii=False), flush=True)
        count += 1
        if source['status'] in ('uncertain', 'unresolved_error'):
            raise ValueError('Ambiguous/technical result; no blind retry or fallback')
    if state['status'] == 'running_sunburst':
        state['status'] = 'sunburst_group_finished'
        save_json(args.state, state)


def nano_request(out):
    return {'contents': [{'role': 'user', 'parts': [{'text': PROMPT}] + [
        {'inlineData': {'mimeType': 'image/png', 'data': base64.b64encode((out/n).read_bytes()).decode()}}
        for n in ('input.png', 'source.png')]}],
        'generationConfig': {'responseModalities': ['IMAGE'], 'candidateCount': 1, 'maxOutputTokens': MAX_OUTPUT,
                             'imageConfig': {'aspectRatio': '9:16', 'imageSize': '1K'}}}


def slots_available(state,current,key_remaining,workers):
    available=min(Decimal(state['openrouter_budget_usd'])-committed_spend(state,current),current,
                  Decimal(str(key_remaining)) if key_remaining is not None else current)
    return max(0,min(workers,int(available//Decimal('0.10'))))


def lightweight_row(source,key):
    receipt=json.loads((ROOT/source['directory']/'receipt.json').read_text())
    row={'directory':source['directory'],'track':source['id'],'model':SUN,'profile':'high',
         'source_sha256':source['sha256'],'status':receipt['status'],
         'elapsed_seconds':receipt.get('elapsed_seconds'),
         'actual_cost_usd':receipt.get('reported_cost_usd'),'cost_source':receipt.get('cost_source')}
    generation=receipt.get('response_headers',{}).get('x-generation-id')
    if generation:
        try:
            data,_=request_json('generation?id='+urllib.parse.quote(generation),key)
            row.update(generation_id=generation,actual_cost_usd=data['data']['total_cost'],cost_source='generation.total_cost')
        except Exception as exc: row['billing_lookup_error']=type(exc).__name__
    if receipt.get('images'):
        row.update(raw_sha256=receipt['images'][0]['sha256'],raw_dimensions=receipt['images'][0]['dimensions'])
    return row


def isolated_audit(row,destination,path):
    """Release OpenCV/SIFT allocations when each serial audit finishes."""
    destination.mkdir(exist_ok=True)
    row_path=path.with_suffix('.row.json');save_json(row_path,row)
    code=('import json,sys; from pathlib import Path; from source_fidelity import audit; '
          'from benchmark import save_json; row=json.loads(Path(sys.argv[1]).read_text()); '
          'save_json(Path(sys.argv[3]),audit(row,Path(sys.argv[2]),placement_mode="free"))')
    subprocess.run([sys.executable,'-c',code,str(row_path),str(destination),str(path)],
                   cwd=Path(__file__).resolve().parent,check=True)


def run_parallel_sun(args,state):
    """One coordinator reserves each whole wave; only network workers overlap.

    Drain all launched calls before evaluating errors or starting another wave.
    Registration and state updates run serially after the network workers exit.
    """
    key=args.key_file.read_text().strip()
    if any(r['status'] in ('request_started','uncertain','unresolved_error') for r in state['sources']):
        raise ValueError('Unresolved earlier request; reconcile before parallel work')
    count=0
    while count<args.limit:
        if (ROOT/'.build/teacher-queue.pause').exists():
            state['status']='paused_between_waves'; save_json(args.state,state); return
        pending=[r for r in state['sources'] if r['status']=='pending']
        if not pending: break
        current=credit(key); key_data,_=request_json('key',key)
        slots=slots_available(state,current,key_data['data'].get('limit_remaining'),args.workers)
        wave=pending[:min(slots,args.limit-count)]
        if not wave:
            state['status']='openrouter_budget_stop'; save_json(args.state,state); return
        commands=[]
        for source in wave:
            path=ROOT/source['file']
            out=args.state.parent/'outputs'/source['id']/'gpt-image-2.5-sunburst'/'high'
            if sha(path)!=source['sha256'] or (out/'receipt.json').exists():
                raise ValueError('Source changed or receipt already exists; no wave sent')
            command=[sys.executable,str(Path(__file__).with_name('benchmark.py')),
                '--key-file',str(args.key_file),'--model',SUN,'--track',source['id'],
                '--source-image',str(path),'--output',str(out),'--quality','high',
                '--input-mode','canvas','--request-timeout','600',
                '--require-prompt-sha256',PROMPT_HASH,'--require-reference-count','2']
            commands.append(command)
        for source in wave:
            out=args.state.parent/'outputs'/source['id']/'gpt-image-2.5-sunburst'/'high'
            source.update(status='request_started',directory=str(out.relative_to(ROOT)))
        state.update(status='running_parallel_sunburst',inflight_reserved_usd=str(Decimal('0.10')*len(wave)),workers=args.workers)
        save_json(args.state,state)
        print(json.dumps({'wave_started':len(wave),'ids':[r['id'] for r in wave]}),flush=True)
        failures={}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(subprocess.run,command,check=True):source for source,command in zip(wave,commands)}
            for future in as_completed(futures):
                source=futures[future]
                try: future.result()
                except Exception as exc: failures[source['id']]={'type':type(exc).__name__,'returncode':getattr(exc,'returncode',None)}
        stop=False
        for source in wave:
            out=ROOT/source['directory']; receipt_path=out/'receipt.json'
            if not receipt_path.exists():
                if failures.get(source['id'],{}).get('returncode')==75:
                    source.update(status='pending',deferred_before_post={'reason':'pause marker; child exit 75 before receipt or POST'})
                    continue
                # Child may have been stopped before POST; retain ambiguity until
                # its exit receipt is inspected rather than automatically redoing it.
                source.update(status='uncertain',worker_error=failures.get(source['id']))
                stop=True; continue
            receipt=json.loads(receipt_path.read_text())
            if hashlib.sha256(receipt['request']['prompt'].encode()).hexdigest()!=PROMPT_HASH or len(receipt['request']['input_references'])!=2:
                source['status']='uncertain';stop=True;continue
            row=lightweight_row(source,key)
            source.update(measurement_row=row,actual_cost_usd=row.get('actual_cost_usd'))
            if receipt['status']=='completed' and row.get('actual_cost_usd') is not None:
                source['status']='sunburst_completed_review_pending'
            elif refusal(receipt) or (receipt['status']=='no_image' and row.get('actual_cost_usd') is not None):
                source['status']='fallback_pending'
            else:
                source['status']='uncertain' if receipt['status'] in ('request_started','uncertain_or_decode_error') else 'unresolved_error'
                stop=True
            if row.get('actual_cost_usd') is not None and Decimal(str(row['actual_cost_usd']))>Decimal('0.10'):
                source['unexpected_price']=True;stop=True
            save_json(args.state,state)
            if source['status']=='sunburst_completed_review_pending':
                destination=args.state.parent/'fidelity';destination.mkdir(exist_ok=True)
                path=destination/(source['id']+'-sunburst.json')
                isolated_audit(row,destination,path)
                source['fidelity_file']=str(path.relative_to(ROOT))
        state['openrouter_spend_usd']=str(committed_spend(state,credit(key)))
        state['inflight_reserved_usd']='0' if not stop else 'unresolved; reconcile receipts before more requests'
        state['status']='parallel_reconciliation_required' if stop else 'parallel_wave_finished'
        save_json(args.state,state);count+=len(wave)
        print(json.dumps({'wave_finished':len(wave),'spend_usd':state['openrouter_spend_usd'],'stop':stop}),flush=True)
        if stop: raise ValueError('Wave drained; reconcile error/billing before more POSTs')


def submit_google(args, state):
    key = args.google_key_file.read_text().strip()
    if any(j['status'] == 'submit_uncertain' for j in state['google_jobs']):
        raise ValueError('Unknown batch submission; reconcile before another POST')
    selected = [r for r in state['sources'] if r['status'] == 'fallback_pending'][:min(args.limit, 2)]
    if not selected:
        print('No definite Sunburst refusals/no-image fallbacks pending'); return
    requests, refs = [], []
    reserve = Decimal(0)
    for source in selected:
        path = ROOT/source['file']
        if sha(path) != source['sha256']:
            raise ValueError('Fallback source changed')
        out = args.state.parent/'outputs'/source['id']/NANO/'1K'
        out.mkdir(parents=True, exist_ok=True)
        if (out/'receipt.json').exists():
            raise ValueError('Fallback already attempted')
        geometry = prepare(source['id'], out, path)
        request = nano_request(out)
        tokens = google('models/'+NANO+':countTokens', key, {'generateContentRequest': {
            'model': 'models/'+NANO, **request}})['totalTokens']
        # Highest output-token price also covers text/thinking. Reserve unresolved
        # jobs at the configured output cap rather than expected image price.
        amount = (Decimal(tokens)*Decimal('0.75') + MAX_OUTPUT*Decimal('15'))/Decimal(1000000)*FX*Decimal('1.10')
        reserve += amount
        receipt = {'model': 'google/'+NANO, 'track': source['id'], 'geometry': geometry,
            'input_mode': 'canvas', 'route': 'direct_google_batch', 'status': 'prepared',
            'request': {'prompt': PROMPT, 'generationConfig': request['generationConfig'],
                        'input_references': [{'local_file': n, 'sha256': sha(out/n)} for n in ('input.png','source.png')]},
            'counted_input_tokens': tokens, 'reserved_upper_estimate_tl': str(amount),
            'actual_cost_usd': None, 'cost_source': 'invoice_not_available',
            'training_eligibility': 'not established; manual review required'}
        refs.append((source, out, receipt))
        requests.append({'request': request, 'metadata': {'key': source['id']}})
    allocated = sum((Decimal(j.get('reserved_tl','0')) for j in state['google_jobs']), Decimal(0))
    if allocated + reserve > Decimal(state['google_budget_tl']):
        raise ValueError('Google reservations exceed user budget; no POST')
    payload = {'batch': {'display_name': 'mahlep-fallback-'+str(len(state['google_jobs'])+1),
                        'input_config': {'requests': {'requests': requests}}}}
    if len(json.dumps(payload).encode()) >= 19*1024*1024:
        raise ValueError('Inline batch exceeds conservative size cap')
    job = {'status': 'submit_uncertain', 'ids': [r['id'] for r in selected], 'reserved_tl': str(reserve)}
    for source, out, receipt in refs:
        receipt['status'] = 'request_started'; save_json(out/'receipt.json', receipt)
        source['status'] = 'fallback_batch_submitting'
    state['google_jobs'].append(job); save_json(args.state,state)
    try:
        response = google('models/'+NANO+':batchGenerateContent', key, payload)
        name = response.get('name')
        if not isinstance(name,str) or not name.startswith('batches/'):
            raise ValueError('No batch name; unknown outcome, do not retry')
        job.update(status='submitted', name=name)
        save_json(args.state,state)  # Persist recoverable ID before further work.
        save_json(args.state.parent/('google-job-'+str(len(state['google_jobs']))+'.json'),response)
        for source,out,receipt in refs:
            source['status']='fallback_batch_pending'
            receipt.update(status='batch_pending', batch_name=name)
            save_json(out/'receipt.json',receipt)
        save_json(args.state,state)
        print(json.dumps({'batch_submitted':len(selected),'reserved_tl':str(reserve)}),flush=True)
    except Exception:
        # No POST retry even after HTTP error; job/intent remains reconcilable.
        save_json(args.state,state); raise


def batch_outputs(response):
    status=response.get('metadata',{}).get('state') or response.get('state')
    bodies=[response.get('response',{}),response.get('dest',{}),
            response.get('response',{}).get('output',{}),response.get('metadata',{}).get('output',{})]
    for body in bodies:
        inline=body.get('inlinedResponses')
        if isinstance(inline,dict): inline=inline.get('inlinedResponses')
        if isinstance(inline,list): return status,inline
    return status,None


def google_estimate(usage,has_image):
    if not usage or 'promptTokenCount' not in usage or 'candidatesTokenCount' not in usage:
        return None
    image_tokens=sum(d['tokenCount'] for d in usage.get('candidatesTokensDetails',[]) if d['modality']=='IMAGE')
    if has_image and not image_tokens:
        return None
    return (Decimal(usage['promptTokenCount'])*Decimal('0.75') + Decimal(image_tokens)*15 +
        Decimal(max(0,usage['candidatesTokenCount']-image_tokens)+usage.get('thoughtsTokenCount',0))*Decimal('3.75'))/1000000


def collect_google(args,state):
    key=args.google_key_file.read_text().strip()
    by_id={r['id']:r for r in state['sources']}
    for job in state['google_jobs']:
        if job['status'] != 'submitted': continue
        response=google(job['name'],key)
        status,inlined=batch_outputs(response)
        job['provider_state']=status
        if status not in ('JOB_STATE_SUCCEEDED','BATCH_STATE_SUCCEEDED'):
            if status and any(status.endswith(s) for s in ('FAILED','CANCELLED','EXPIRED')):
                job['status']='provider_failed_unknown_billing'
                for ident in job['ids']: by_id[ident]['status']='quarantine_nano_batch_failed'
            save_json(args.state,state); print(json.dumps({'google_state':status}),flush=True); continue
        if not isinstance(inlined,list) or len(inlined)!=len(job['ids']):
            raise ValueError('Unexpected batch output count; do not resubmit')
        ids=[i.get('metadata',{}).get('key') for i in inlined]
        if set(ids)!=set(job['ids']) or len(set(ids))!=len(ids):
            raise ValueError('Batch response IDs cannot be safely mapped')
        estimate=Decimal(0); billing_known=True
        for item,ident in zip(inlined,ids):
            source=by_id[ident]; out=args.state.parent/'outputs'/ident/NANO/'1K'
            receipt=json.loads((out/'receipt.json').read_text()); result=item.get('response',{})
            usage=result.get('usageMetadata',{})
            receipt['usage']=usage
            images=[]
            for candidate in result.get('candidates',[]):
                for part in candidate.get('content',{}).get('parts',[]):
                    data=part.get('inlineData')
                    if data:
                        blob=base64.b64decode(data['data'],validate=True)
                        image=Image.open(io.BytesIO(blob)); image.load()
                        image.save(out/'raw.png')
                        images.append({'file':'raw.png','sha256':sha(out/'raw.png'),'dimensions':list(image.size)})
            if len(images)>1: raise ValueError('Unexpected multiple images; retain reservation and inspect')
            value=google_estimate(usage,bool(images))
            receipt.update(estimated_cost_usd=str(value) if value is not None else None,
                           actual_cost_usd=None,cost_source='published_batch_rates_times_usage_not_invoice' if value is not None else 'usage_missing_actual_unknown')
            if value is None: billing_known=False
            else: estimate+=value*FX
            receipt.update(images=images,status='completed' if images else 'no_image',
                           response_metadata={'error':item.get('error'),'promptFeedback':result.get('promptFeedback')})
            save_json(out/'receipt.json',receipt)
            source['fallback_directory']=str(out.relative_to(ROOT))
            source['status']='nano_completed_review_pending' if images else 'quarantine_nano_no_image'
            save_json(args.state,state)
            if images:
                destination=args.state.parent/'fidelity'; destination.mkdir(exist_ok=True)
                row={'directory':source['fallback_directory'],'model':'google/'+NANO,
                     'profile':'1K','track':ident,'source_sha256':source['sha256'],
                     'raw_sha256':images[0]['sha256'],'raw_dimensions':images[0]['dimensions'],
                     'actual_cost_usd':None}
                isolated_audit(row,destination,destination/(ident+'-nano.json'))
        job.update(status='collected',estimated_cost_tl=str(estimate) if billing_known else None,
                   reserved_tl=str(estimate*Decimal('1.10')) if billing_known else job['reserved_tl'],
                   billing='usage estimate; invoice/actual TL remains unknown')
        save_json(args.state,state)
        print(json.dumps({'google_completed':len(inlined),'estimated_tl':str(estimate)}),flush=True)


def audit_existing(args,state):
    from source_fidelity import audit
    destination=args.state.parent/'fidelity'; destination.mkdir(exist_ok=True)
    for source in state['sources']:
        if source['status']=='sunburst_completed_review_pending' and not source.get('fidelity_file'):
            measurement=audit(source['measurement_row'],destination,placement_mode='free')
            path=destination/(source['id']+'-sunburst.json'); save_json(path,measurement)
            source['fidelity_file']=str(path.relative_to(ROOT)); save_json(args.state,state)


def production(args,state):
    """Continue primary work; submit only the first tiny Google protocol pilot.

    Larger Google spending requires inspection of the first returned images.
    Pending remote jobs remain recoverable when this local process exits.
    """
    limit=args.limit
    started=sum(r['status']!='pending' for r in state['sources'])
    group_size=3*args.workers if args.workers>1 else 10
    for _ in range((limit+group_size-1)//group_size):
        completed=sum(r['status']!='pending' for r in state['sources'])-started
        if completed>=limit: break
        args.limit=min(group_size,limit-completed)
        if args.workers>1: run_parallel_sun(args,state)
        else: run_sun(args,state)
        collect_google(args,state)
        fallback=any(r['status']=='fallback_pending' for r in state['sources'])
        if fallback and (not state['google_jobs'] or state.get('google_protocol_pilot_reviewed')):
            args.limit=2
            try: submit_google(args,state)
            except ValueError as exc:
                if str(exc)!='Google reservations exceed user budget; no POST': raise
                state['google_budget_status']='Reservations full; no further Google POST'
                save_json(args.state,state)
        if state['status'] in ('openrouter_budget_stop','paused_between_waves') or not any(r['status']=='pending' for r in state['sources']):
            break
    state['google_next_action']='Inspect first batch raw images and source audit before further batch submissions' if state['google_jobs'] else 'No fallback batch needed yet'
    save_json(args.state,state)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['init','sun','google-submit','google-collect','audit','production','status'])
    parser.add_argument('--state',type=Path,required=True)
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--key-file',type=Path,default=Path('/tmp/artworker-openrouter-session.key'))
    parser.add_argument('--google-key-file',type=Path,default=Path('/tmp/artworker-google-session.key'))
    parser.add_argument('--budget-usd',type=Decimal,help='Required for init; account budget stays in private state')
    parser.add_argument('--google-budget-tl',type=Decimal,help='Required for init; fallback budget stays in private state')
    parser.add_argument('--limit',type=int,default=4)
    parser.add_argument('--workers',type=int,choices=[1,2,3],default=1,
                        help='Network calls only; source audits and state writes remain serial')
    args=parser.parse_args(); args.state=args.state.resolve()
    if not args.state.is_relative_to(ROOT/'experiments/openrouter_outpaint/results'):
        parser.error('Private state must be under ignored results/')
    if hashlib.sha256(PROMPT.encode()).hexdigest()!=PROMPT_HASH:
        parser.error('Frozen prompt changed')
    if args.limit<1: parser.error('Positive count required')
    args.state.parent.mkdir(parents=True,exist_ok=True)
    if args.command=='status':
        from collections import Counter
        state=json.loads(args.state.read_text())  # Atomic replacement allows a live read without the mutation lock.
        print(json.dumps({'status':state['status'],'counts':dict(Counter(r['status'] for r in state['sources'])),
            'spend_usd':state.get('openrouter_spend_usd'),'workers':state.get('workers',1),
            'google_jobs':[{k:j.get(k) for k in ('status','provider_state','ids','estimated_cost_tl','reserved_tl')} for j in state['google_jobs']]},ensure_ascii=False))
        return
    with (args.state.parent/'queue.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.command=='init': init(args); return
        state=json.loads(args.state.read_text())
        if state['prompt_sha256']!=PROMPT_HASH: raise ValueError('Plan hash mismatch')
        if args.command=='sun': run_sun(args,state)
        elif args.command=='google-submit': submit_google(args,state)
        elif args.command=='google-collect': collect_google(args,state)
        elif args.command=='audit': audit_existing(args,state)
        elif args.command=='production': production(args,state)
        else:
            from collections import Counter
            print(json.dumps({'status':state['status'],'counts':dict(Counter(r['status'] for r in state['sources'])),
                'spend_usd':state.get('openrouter_spend_usd'),'google_jobs':[{
                k:j.get(k) for k in ('status','provider_state','ids','estimated_cost_tl','reserved_tl')} for j in state['google_jobs']]},ensure_ascii=False))


if __name__=='__main__': main()
