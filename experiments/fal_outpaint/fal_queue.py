"""Durable fal Sunburst outpaint queue. No fallback model or paid POST retries.

Private source art, receipts and state belong under ignored results/. A lost
submission response pauses new spending; explicitly held uncertain rows are
never resubmitted and need not block unrelated covers.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from decimal import Decimal
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/openrouter_outpaint'))
from benchmark import PROMPT, save_json

ENDPOINT = 'openai/gpt-image-2.5/sunburst/edit'
SIZES = [('3:4',864,1152),('4:3',1152,864),('9:16',720,1280),
         ('16:9',1280,720),('2:3',768,1152),('3:2',1152,768),
         ('4:5',896,1120),('5:4',1120,896),('1:2',720,1440),
         ('2:1',1440,720),('1:3',720,2160),('3:1',2160,720)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_size(width, height):
    if min(width,height) <= 0 or width%16 or height%16:
        raise ValueError('Dimensions must be positive multiples of 16')
    if max(width,height)>3840 or max(width,height)>3*min(width,height):
        raise ValueError('Edge/aspect constraint exceeded')
    if not 655360 <= width*height <= 8294400 or width==height:
        raise ValueError('Pixel budget violated or square output requested')


def layout(width,height):
    validate_size(width,height)
    if min(width,height)<720: raise ValueError('Source square must fit')
    x,y=(width-720)//2,(height-720)//2
    return [x,y,x+720,y+720]


def prompt_for(ratio,width,height):
    x,y,_,_=layout(width,height)
    prompt=PROMPT.replace('portrait image, aspect ratio 9:16',f'image, aspect ratio {ratio}')
    prompt=prompt.replace('720x1280',f'{width}x{height}').replace('x=0, y=280',f'x={x}, y={y}')
    prompt=prompt.replace('above and below','outside the source square')
    prompt=prompt.replace('centered, full width,','centered,')
    return prompt.replace('across both boundaries','across every source boundary')


def safe_queue_url(url):
    parsed=urllib.parse.urlsplit(url)
    if parsed.scheme!='https' or parsed.netloc!='queue.fal.run' or parsed.username:
        raise ValueError('Unexpected queue URL; do not send credentials')
    return url


def request(url,key,payload=None):
    safe_queue_url(url)
    headers={'Authorization':'Key '+key,'Content-Type':'application/json'}
    if payload is not None:
        headers.update({'X-Fal-No-Retry':'1',
            'X-Fal-Object-Lifecycle-Preference':json.dumps({'expiration_duration_seconds':86400})})
    req=urllib.request.Request(url,headers=headers,
        data=json.dumps(payload).encode() if payload is not None else None)
    with urllib.request.urlopen(req,timeout=90) as response:
        raw=response.read(2*1024*1024+1)
        if len(raw)>2*1024*1024: raise ValueError('JSON response too large')
        return json.loads(raw),{k.lower():v for k,v in response.headers.items()
            if k.lower() in {'x-fal-billable-units','x-request-id','x-fal-request-id'}}


def classification(code,message):
    text=message.lower()
    if any(word in text for word in ('content_policy','content policy','safety','moderation','refusal')):
        return 'refused'
    if code in (401,402,403,429): return 'account_stop'
    return 'failed'


def prepare(row,folder):
    original=ROOT/row['file']
    if sha(original)!=row['sha256']: raise ValueError('Source checksum mismatch')
    folder.mkdir(parents=True,exist_ok=True)
    with Image.open(original) as im:
        w,h=im.size;side=min(w,h);x,y=(w-side)//2,(h-side)//2
        source=im.convert('RGB').crop((x,y,x+side,y+side)).resize((720,720),Image.Resampling.LANCZOS)
    width,height=row['width'],row['height'];rect=layout(width,height)
    canvas=Image.new('RGB',(width,height),(0,255,0));canvas.paste(source,rect[:2])
    source.save(folder/'source.png');canvas.save(folder/'input.png')
    references=[{'local_file':name,'sha256':sha(folder/name)} for name in ('input.png','source.png')]
    geometry={'original_file':row['file'],'original_sha256':row['sha256'],
        'original_dimensions':[w,h],'crop_xyxy':[x,y,x+side,y+side],
        'canvas_dimensions':[width,height],'source_rect_xyxy':rect}
    payload={'prompt':prompt_for(row['aspect_ratio'],width,height),
        'image_urls':['data:image/png;base64,'+base64.b64encode((folder/r['local_file']).read_bytes()).decode() for r in references],
        'image_size':{'width':width,'height':height},'quality':'high','num_images':1,
        'output_format':'png','background':'opaque','enable_safety_checker':True}
    receipt={'provider':'fal','model':ENDPOINT,'status':'prepared','geometry':geometry,
        'input_mode':'canvas','request':{k:v for k,v in payload.items() if k!='image_urls'},
        'prompt_sha256':hashlib.sha256(payload['prompt'].encode()).hexdigest(),
        'actual_cost_usd':None,'cost_source':'pending provider billing; no invented charge'}
    receipt['request']['input_references']=references
    return payload,receipt


def init(args):
    if args.state.exists(): raise ValueError('Existing state must not be reset')
    manifest=json.loads(args.manifest.read_text());base=args.manifest.resolve().parent
    rows=[]
    for i,r in enumerate(manifest['covers']):
        if not re.fullmatch(r'cover[0-9]{4}',r['id']):raise ValueError('Unsafe source identifier')
        path=(base/r['file']).resolve()
        if not path.is_relative_to(ROOT) or sha(path)!=r['sha256']:raise ValueError('Source provenance mismatch')
        ratio,w,h=SIZES[i%len(SIZES)];validate_size(w,h)
        rows.append({'id':args.stage+'-'+r['id'],'source_id':manifest['stage']+':'+r['id'],
            'file':str(path.relative_to(ROOT)),'sha256':r['sha256'],'album':r['album'],
            'artists':r['artists'],'aspect_ratio':ratio,'width':w,'height':h,'status':'pending'})
    args.state.parent.mkdir(parents=True)
    save_json(args.state,{'stage':args.stage,'endpoint_id':ENDPOINT,'quality':'high',
        'source_manifest':str(args.manifest.resolve().relative_to(ROOT)),
        'source_manifest_sha256':sha(args.manifest),'sources':rows,'status':'prepared',
        'fallback_model':None,'training_input':'crop from teacher output, not original cover',
        'training_admission':'manual review required','created_unix':time.time()})


def download(url,destination):
    parsed=urllib.parse.urlsplit(url)
    if parsed.scheme!='https' or not parsed.hostname or not parsed.hostname.endswith('.fal.media'):
        raise ValueError('Unexpected output CDN')
    temp=destination.with_suffix('.part');size=0
    with urllib.request.urlopen(url,timeout=90) as response,temp.open('wb') as f:
        while chunk:=response.read(65536):
            size+=len(chunk)
            if size>32*1024*1024:raise ValueError('Output exceeds 32 MiB')
            f.write(chunk)
    with Image.open(temp) as im:
        dimensions=list(im.size);im.verify()
    temp.replace(destination)
    return {'sha256':sha(destination),'dimensions':dimensions,'file':'raw.png'}


def apply_cost(receipt,headers):
    receipt.setdefault('response_headers',{}).update(headers)
    units=receipt['response_headers'].get('x-fal-billable-units')
    if units is not None:
        value=Decimal(units)
        if not value.is_finite() or value<0:raise ValueError('Invalid billing units')
        receipt.update(reported_cost_usd=str(value),actual_cost_usd=str(value),
            cost_source='x-fal-billable-units * published USD 1/unit; invoice not reconciled')


def run(args):
    state=json.loads(args.state.read_text());root=args.state.parent
    if state['endpoint_id']!=ENDPOINT:raise ValueError('Wrong model state')
    if sha(ROOT/state['source_manifest'])!=state['source_manifest_sha256']:raise ValueError('Manifest changed')
    rows=state['sources'][:args.limit];key=args.key_file.read_text().strip()
    if getattr(args,'hold_uncertain',False):
        for row in rows:
            if row['status'] not in {'submitting','uncertain'}:continue
            receipt_path=root/'outputs'/row['id']/'receipt.json'
            receipt=json.loads(receipt_path.read_text())
            queue=receipt.get('queue')
            if queue and all(queue.get(k) for k in ('request_id','status_url','response_url')):
                safe_queue_url(queue['status_url']);safe_queue_url(queue['response_url'])
                row.update({k:queue[k] for k in ('request_id','status_url','response_url')})
                row['status']='submitted';receipt['status']='submitted'
            else:
                row['status']='held_uncertain'
                row['hold_reason']='User authorized continuing other covers without repeating this uncertain submission'
                row['held_unix']=time.time()
                receipt.update(status='held_uncertain',hold_reason=row['hold_reason'],held_unix=row['held_unix'])
            save_json(receipt_path,receipt);save_json(args.state,state)
    if any(r['status'] in {'submitting','uncertain'} for r in rows):
        raise ValueError('Uncertain submission exists; reconcile before new spending')
    state.setdefault('resumes',[]).append({'unix':time.time(),'previous_stop_reason':state.pop('stop_reason',None),
        'held_uncertain':sum(r['status']=='held_uncertain' for r in state['sources'])})
    stopping=False
    while True:
        active=[r for r in rows if r['status']=='submitted']
        if (root/'STOP').exists():stopping=True
        for row in active:
            folder=root/'outputs'/row['id'];receipt_path=folder/'receipt.json'
            receipt=json.loads(receipt_path.read_text())
            try:
                status,headers=request(row['status_url'],key);apply_cost(receipt,headers)
                row['remote_status']=status['status'];row['last_poll_unix']=time.time()
                if status['status']!='COMPLETED':continue
                result,headers=request(row['response_url'],key);apply_cost(receipt,headers)
                save_json(folder/'result.json',result)
                if result.get('error'):
                    row['status']=classification(422,json.dumps(result));receipt['error']=result['error']
                elif not result.get('images'):
                    row['status']='failed';receipt['error']='No output image returned'
                else:
                    receipt['images']=[download(result['images'][0]['url'],folder/'raw.png')]
                    row['status']='completed';row['raw_dimensions']=receipt['images'][0]['dimensions']
                    row['size_matches_request']=row['raw_dimensions']==[row['width'],row['height']]
                    if not row['size_matches_request']:stopping=True;state['stop_reason']='unexpected_output_size'
                    row['raw_sha256']=receipt['images'][0]['sha256']
            except urllib.error.HTTPError as exc:
                message=exc.read(4000).decode(errors='replace')
                if exc.code>=500 or exc.code in (408,429):
                    row['last_poll_error']=f'HTTP {exc.code}';continue
                row['status']=classification(exc.code,message)
                receipt.update(error=message,http_status=exc.code)
                if row['status']!='refused':stopping=True;state['stop_reason']=row['status']
            except (urllib.error.URLError,TimeoutError) as exc:
                row['last_poll_error']=type(exc).__name__;continue
            if row['status'] in {'failed','account_stop'}:
                stopping=True;state['stop_reason']=row['status']
            receipt['status']=row['status'];receipt['elapsed_seconds']=time.time()-row['submitted_unix']
            row['actual_cost_usd']=receipt.get('actual_cost_usd')
            save_json(receipt_path,receipt);save_json(args.state,state)
            print(json.dumps({'id':row['id'],'status':row['status'],'aspect_ratio':row['aspect_ratio'],'cost':row.get('actual_cost_usd')}),flush=True)
        active=[r for r in rows if r['status']=='submitted']
        if not stopping:
            for row in (r for r in rows if r['status']=='pending'):
                if len(active)>=args.concurrency:break
                folder=root/'outputs'/row['id'];receipt_path=folder/'receipt.json'
                if receipt_path.exists():raise ValueError('Pending row has receipt; manual reconciliation required')
                payload,receipt=prepare(row,folder)
                row.update(status='submitting',directory=str(folder.relative_to(ROOT)),submitted_unix=time.time())
                receipt['status']='submitting';save_json(receipt_path,receipt);save_json(args.state,state)
                try:
                    result,headers=request('https://queue.fal.run/'+ENDPOINT,key,payload)
                    for field in ('request_id','status_url','response_url'):
                        if field not in result:raise ValueError('Submission omitted '+field)
                    safe_queue_url(result['status_url']);safe_queue_url(result['response_url'])
                    row.update({k:result[k] for k in ('request_id','status_url','response_url')})
                    row['status']='submitted';receipt.update(queue=result,status='submitted');apply_cost(receipt,headers)
                    active.append(row)
                except urllib.error.HTTPError as exc:
                    message=exc.read(4000).decode(errors='replace');receipt.update(error=message,http_status=exc.code)
                    row['status']=classification(exc.code,message) if exc.code<500 else 'uncertain'
                    receipt['status']=row['status']
                    if row['status']!='refused':stopping=True;state['stop_reason']=row['status']
                except Exception as exc:
                    row['status']='uncertain';receipt.update(status='uncertain',error=type(exc).__name__)
                    stopping=True;state['stop_reason']='uncertain_submission'
                save_json(receipt_path,receipt);save_json(args.state,state)
                print(json.dumps({'id':row['id'],'status':row['status'],'request_id':row.get('request_id')}),flush=True)
                if stopping:break
        state['counts']=dict(Counter(r['status'] for r in state['sources']))
        state['updated_unix']=time.time();state['status']='draining' if stopping else 'running'
        save_json(args.state,state)
        if not any(r['status']=='submitted' for r in rows):
            if stopping or not any(r['status']=='pending' for r in rows):break
        time.sleep(args.poll_seconds)
    state['status']='stopped' if stopping else 'selected_slice_finished';save_json(args.state,state)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['init','run'])
    parser.add_argument('--state',type=Path,required=True)
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--stage',default='nilufer')
    parser.add_argument('--key-file',type=Path)
    parser.add_argument('--limit',type=int,default=1000)
    parser.add_argument('--concurrency',type=int,default=4)
    parser.add_argument('--poll-seconds',type=float,default=8)
    parser.add_argument('--hold-uncertain',action='store_true',
                        help='Hold unresolved submissions without repeating them; continue unrelated pending covers')
    args=parser.parse_args();args.state=args.state.resolve()
    if not args.state.is_relative_to(ROOT/'experiments/fal_outpaint/results'):parser.error('Use private ignored results')
    if not re.fullmatch(r'[a-z][a-z0-9]*',args.stage):parser.error('Use a safe single-word stage')
    if not 1<=args.concurrency<=4 or not 1<=args.limit<=1000 or args.poll_seconds<2:parser.error('Invalid bounded queue limits')
    args.state.parent.parent.mkdir(parents=True,exist_ok=True)
    with (args.state.parent.parent/(args.state.parent.name+'.lock')).open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.action=='init':
            if not args.manifest:parser.error('Manifest required')
            init(args)
        else:
            if not args.key_file:parser.error('Key file required')
            run(args)


if __name__=='__main__':main()
