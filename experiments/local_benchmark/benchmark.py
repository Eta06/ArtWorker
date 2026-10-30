"""Serial, offline ten-cover benchmark. Media/state are private under results/.

Heavy workers use the existing inference guard (6 GiB maximum), never the API
queue's budget. Preparation/reporting load no model. No human votes are changed.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from run_bounded_model import pressure_and_swap, run

FILL_PROMPT = ('Extend this album artwork above and below the central square. '
               'Continue the existing scene, colors, lighting, perspective and texture seamlessly. '
               'Keep the central artwork in place. Do not add new text, logos, borders, '
               'duplicate people, mirrored artwork or blurred background copies.')
ARMS = [
    {'id':'migan', 'kind':'migan', 'steps':1, 'limit_gib':1.0},
    {'id':'dreamlite-mobile', 'kind':'dreamlite', 'variant':'mobile', 'steps':4},
    {'id':'dreamlite-base', 'kind':'dreamlite', 'variant':'base', 'steps':28},
    {'id':'mobile-o-latent-lock', 'kind':'legacy', 'script':'mobile_o/run_trial.py', 'steps':20},
    {'id':'flux1-fill-q4', 'kind':'legacy', 'script':'flux1_fill/run_fill_trial.py', 'steps':50},
    {'id':'sdxl-inpaint', 'kind':'legacy', 'script':'sdxl_inpaint/run_sdxl_trial.py', 'steps':20},
    {'id':'sdxl-promax', 'kind':'legacy', 'script':'sdxl_promax/run_promax_trial.py', 'steps':30},
    {'id':'klein-distilled-q4', 'kind':'klein', 'model':'klein-4b', 'steps':4, 'bits':4},
    {'id':'klein-base-q4', 'kind':'klein', 'model':'klein-4b-base', 'steps':28, 'bits':4},
    {'id':'klein-distilled-q4-lora', 'kind':'klein', 'model':'klein-4b', 'steps':4, 'bits':4, 'lora':True},
    {'id':'klein-base-outpaint-q4', 'kind':'klein', 'model':'klein-4b-base', 'steps':12, 'bits':4, 'baked':True},
    {'id':'klein-base-outpaint-q4-cached', 'kind':'klein', 'model':'klein-4b-base', 'steps':12, 'bits':4, 'baked':True, 'cached':True},
    {'id':'klein-base-outpaint-q3-cached', 'kind':'klein', 'model':'klein-4b-base', 'steps':12, 'bits':3, 'baked':True, 'cached':True},
    {'id':'klein-base-outpaint-q2-cached', 'kind':'klein', 'model':'klein-4b-base', 'steps':12, 'bits':2, 'baked':True, 'cached':True},
    {'id':'klein-base-outpaint-mlp3-cached', 'kind':'klein', 'model':'klein-4b-base', 'steps':12, 'bits':4, 'baked':True, 'cached':True, 'profile':'mlp3'},
]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    with tmp.open('w') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n')
        stream.flush();os.fsync(stream.fileno())
    tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def prepare(folder):
    from PIL import Image, ImageOps, ImageDraw
    if (folder/'inputs.json').exists():raise FileExistsError('Use a new benchmark folder')
    old=read(ROOT/'experiments/evaluation/inputs/inputs.json')
    picks=[(r['title'],r.get('artist',''),Path(r['source_square'])) for r in old['tracks']]
    old_gallery=ROOT/'experiments/openrouter_outpaint/results/2026-10-11/sumbul'
    new_gallery=ROOT/'experiments/fal_outpaint/results/2026-10-18/nilufer/gallery'
    for gallery,key in [(old_gallery,'cover0781-sunburst'),(new_gallery,'nilufer-cover0264-sunburst'),
                        (new_gallery,'nilufer-cover0837-sunburst'),(new_gallery,'nilufer-cover0149-sunburst'),
                        (new_gallery,'nilufer-cover0591-sunburst'),(new_gallery,'nilufer-cover0052-sunburst'),
                        (new_gallery,'nilufer-cover0505-sunburst')]:
        row=next(r for r in read(gallery/'data.json')['rows'] if r['id']==key)
        picks.append((row['title'],', '.join(row.get('artists',[])),gallery/row['asset']/'source.png'))
    tracks=[]
    sheet=Image.new('RGB',(5*220,2*255),'#151515');draw=ImageDraw.Draw(sheet)
    for i,(title,artist,original) in enumerate(picks,1):
        key=f'cover{i:02d}';dest=folder/'inputs'/key;dest.mkdir(parents=True)
        with Image.open(original) as source:
            if source.width!=source.height:raise ValueError(f'Not a square source: {original}')
            source=source.convert('RGB').resize((512,512),Image.Resampling.LANCZOS)
        source.save(dest/'source.png')
        canvas=Image.new('RGB',(512,1152),'black');canvas.paste(source,(0,320));canvas.save(dest/'canvas.png')
        green=Image.new('RGB',canvas.size,(0,255,0));green.paste(source,(0,320));green.save(dest/'green.png')
        mask=Image.new('L',canvas.size,255);mask.paste(0,(0,320,512,832));mask.save(dest/'mask.png')
        # Legacy adapters keep their frozen internal track1 label. The external ID
        # and exact per-cover manifest bind that alias to this unique source.
        track={'id':'track1','benchmark_id':key,'title':title,'artist':artist,
               'source_square':str((dest/'source.png').resolve()),'source_512':str((dest/'source.png').resolve()),
               'canvas':str((dest/'canvas.png').resolve()),'mask':str((dest/'mask.png').resolve())}
        manifest={'canvas_size':[512,1152],'source_rect_xyxy':[0,320,512,832],'tracks':[track]}
        write(dest/'inputs.json',manifest)
        tracks.append({'id':key,'title':title,'artist':artist,'source_file':str(original.resolve()),
                       'original_sha256':sha(original),'source_sha256':sha(dest/'source.png'),
                       'inputs_file':str((dest/'inputs.json').resolve())})
        tile=ImageOps.contain(source,(204,204));x=((i-1)%5)*220+8;y=((i-1)//5)*255+8
        sheet.paste(tile,(x,y));draw.text((x,y+208),f'{key} {title[:25]}',fill='white')
    sheet.save(folder/'selected-covers.jpg')
    write(folder/'inputs.json',{'canvas_size':[512,1152],'source_rect_xyxy':[0,320,512,832],
                              'tracks':tracks,'arms':ARMS,'prompt':FILL_PROMPT,'seed':42,
                              'quality_votes_modified':False,'same_seed_is_not_same_noise_across_models':True})
    jobs=[{'id':a['id']+'/'+t['id'],'arm':a['id'],'cover':t['id'],'status':'pending'} for t in tracks for a in ARMS]
    write(folder/'state.json',{'status':'prepared','jobs':jobs,'created':datetime.datetime.now().isoformat()})
    print(json.dumps({'covers':len(tracks),'arms':len(ARMS),'jobs':len(jobs),'folder':str(folder)}))


def legacy(arm,inputs,out):
    script=arm['script'];python=ROOT/('.build/mobile-o-venv/bin/python' if script.startswith('mobile_o/') else
                                   'experiments/qwen/.venv/bin/python' if script.startswith('flux1_') else '.build/dreamlite-venv/bin/python')
    command=[str(python),str(ROOT/'experiments'/script),'--inputs-file',str(inputs),'--steps',str(arm['steps'])]
    if script.startswith('mobile_o/'):
        command+=['--tracks','track1','--latent-lock','--output',str(out/'native')]
        raw=out/'native/track1-seed42-raw.png'
    else:
        command+=['--track','track1','--prompt',FILL_PROMPT,'--output',str(out/'native')]
        raw=out/'native/raw.png'
    subprocess.run(command,check=True,cwd=ROOT)
    return raw,{'command':command,'method':'existing offline runner; native metrics in native/'}


def klein(arm,inputs,out):
    models=ROOT/('.build/models/flux2-outpaint-baked'+('-mlp3' if arm.get('profile')=='mlp3' else '') if arm.get('baked') else '.build/models/flux2')
    prompt='Fill the green spaces according to the image' if arm.get('baked') or arm.get('lora') else 'Fill the green spaces according to the image. '+FILL_PROMPT
    cmd=[str(ROOT/'.build/flux2-native/Build/Products/Release/Flux2CLI'),'i2i',prompt,
         '--images',str(inputs.parent/'green.png'),'--model',arm['model'],'--models-dir',str(models),
         '--text-quant','4bit','--transformer-quant',f"int{arm['bits']}",'--steps',str(arm['steps']),
         '--guidance','1','--seed','42','--width','512','--height','1152',
         '--vae-variant','standard' if arm.get('baked') else 'small-decoder',
         '--memory-profile','conservative','--profile','--verbose','--output',str(out/'raw.png')]
    env=os.environ.copy()
    if arm.get('baked'):
        env.update(ARTWORKER_REQUIRE_PREQUANTIZED='1',ARTWORKER_RELEASE_TRANSFORMER_BEFORE_DECODE='1')
    if arm.get('cached'):
        emb=ROOT/'.build/models/artworker-fixed-fill/embedding.safetensors'
        expected=read(ROOT/'experiments/klein_compression/fixed_embedding_manifest.json')['sha256']
        if sha(emb)!=expected:raise ValueError('Fixed embedding fingerprint mismatch')
        env.update(ARTWORKER_FIXED_EMBEDDING=str(emb),ARTWORKER_FIXED_EMBEDDING_SHA256=expected)
    if arm.get('profile'):env['ARTWORKER_QUANT_PROFILE']=arm['profile']
    if arm.get('low_memory'):env['ARTWORKER_LOW_MEMORY']='1'
    if arm.get('vae_tiling'):
        if arm['vae_tiling']!='portrait80':raise ValueError('Unsupported benchmark VAE tiling recipe')
        env['ARTWORKER_VAE_TILING']=arm['vae_tiling']
    if arm.get('lora'):cmd+=['--lora',str(ROOT/'.build/models/flux2-outpaint-lora/flux-outpaint-lora.safetensors'),'--lora-scale','1.1']
    with (out/'native.log').open('w') as log:subprocess.run(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    text=(out/'native.log').read_text()
    if arm.get('baked') and ('LoRA weights BAKED IN' not in text or 'Loaded pre-quantized' not in text):raise ValueError('Baked/prequantized load unconfirmed')
    if arm.get('cached') and ('ArtWorker fixed embedding validated; text encoder not loaded' not in text or 'Loading text encoder for' in text):raise ValueError('Encoder bypass unconfirmed')
    if arm.get('profile') and f"ArtWorker quantization profile validated: {arm['profile']}" not in text:raise ValueError('Mixed precision load unconfirmed')
    if arm.get('low_memory') and 'ArtWorker: low-memory execution, evalEvery=1' not in text:raise ValueError('Low-memory execution was not confirmed')
    if arm.get('vae_tiling') and 'ArtWorker: portrait80 tiled VAE decode enabled (2 vertical tiles)' not in text:raise ValueError('Tiled VAE decode was not confirmed')
    if arm.get('lora') and '[LoRA] Merged 88 layers (0 not found)' not in text:raise ValueError('All 88 adapter layers were not confirmed')
    return out/'raw.png',{'command':cmd,'prompt':prompt,'binary_sha256':sha(cmd[0]),'vae_tiling':arm.get('vae_tiling'),'low_memory':arm.get('low_memory',False),'method':'native image-conditioned edit; no color harmonization'}


def worker(folder,arm_id,cover_id,out):
    from PIL import Image
    import numpy as np
    config=read(folder/'inputs.json');arm=next(a for a in config['arms'] if a['id']==arm_id)
    cover=next(t for t in config['tracks'] if t['id']==cover_id);inputs=Path(cover['inputs_file'])
    data=read(inputs);track=data['tracks'][0];rect=data['source_rect_xyxy']
    source=Image.open(track['source_512']).convert('RGB')
    if sha(track['source_512'])!=cover['source_sha256']:raise ValueError('Benchmark source changed')
    out.mkdir(parents=True,exist_ok=True);started=time.perf_counter();extra={}
    if arm['kind']=='legacy':raw_path,extra=legacy(arm,inputs,out)
    elif arm['kind']=='klein':raw_path,extra=klein(arm,inputs,out)
    elif arm['kind']=='migan':
        import onnxruntime as ort
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        options.enable_mem_pattern=False;options.enable_cpu_mem_arena=False
        model=ROOT/'.build/models/migan/migan_pipeline_v2.onnx'
        manifest=read(ROOT/'experiments/migan/download_manifest.json')
        if sha(model)!=manifest['sha256']:raise ValueError('MI-GAN fingerprint mismatch')
        session=ort.InferenceSession(str(model),options,providers=['CPUExecutionProvider'])
        canvas=Image.new('RGB',(512,1152),(128,128,128));canvas.paste(source,(0,320))
        mask=np.zeros((1,1,1152,512),dtype=np.uint8);mask[:,:,320:832]=255
        pred=session.run(None,{'image':np.asarray(canvas).transpose(2,0,1)[None],'mask':mask})[0]
        Image.fromarray(pred[0].transpose(1,2,0)).save(out/'raw.png');raw_path=out/'raw.png'
        extra={'method':'upstream complete ONNX pipeline; already blends known pixels','model_sha256':sha(model)}
    elif arm['kind']=='dreamlite':
        import torch
        from diffusers import DreamLiteMobilePipeline,DreamLitePipeline
        torch.set_num_threads(2)
        cls=DreamLiteMobilePipeline if arm['variant']=='mobile' else DreamLitePipeline
        load=time.perf_counter()
        pipe=cls.from_pretrained(ROOT/f".build/models/dreamlite/{arm['variant']}",torch_dtype=torch.bfloat16,local_files_only=True).to('mps')
        torch.mps.synchronize();extra['load_seconds']=time.perf_counter()-load
        begin=time.perf_counter()
        image=Image.open(track['canvas']).convert('RGB')
        raw=pipe(prompt=FILL_PROMPT,image=image,width=512,height=1152,num_inference_steps=arm['steps'],generator=torch.Generator('cpu').manual_seed(42)).images[0].convert('RGB')
        torch.mps.synchronize();extra.update(generation_seconds=time.perf_counter()-begin,
             mps_driver_bytes=torch.mps.driver_allocated_memory(),mps_current_bytes=torch.mps.current_allocated_memory(),
             method='native instruction editor, no spatial mask; no color harmonization')
        raw.save(out/'raw.png');raw_path=out/'raw.png'
    else:raise ValueError('Unknown model family')
    raw=Image.open(raw_path).convert('RGB')
    if raw.size!=(512,1152):raise ValueError(f'Wrong output dimensions {raw.size}')
    if raw_path!=out/'raw.png':raw.save(out/'raw.png')
    pixels=np.asarray(raw);src=np.asarray(source)
    raw_mae=float(np.abs(pixels[320:832].astype(np.float32)-src.astype(np.float32)).mean())
    composite=raw.copy();composite.paste(source,(0,320));composite.save(out/'composite.png')
    if not np.array_equal(np.asarray(composite)[320:832],src):raise ValueError('Composite changed source')
    unknown=np.ones((1152,512),bool);unknown[320:832]=False
    if not np.array_equal(np.asarray(composite)[unknown],pixels[unknown]):raise ValueError('Composite changed exterior')
    report={'status':'completed','arm':arm_id,'cover':cover_id,'title':cover['title'],
            'source_sha256':cover['source_sha256'],'inputs_sha256':sha(inputs),'recipe':arm,
            'raw_sha256':sha(out/'raw.png'),'composite_sha256':sha(out/'composite.png'),
            'benchmark_runner_sha256':sha(Path(__file__)),
            'raw_source_mae_255':raw_mae,'source_exact_after_composite':True,
            'source_paste_is_not_a_model_quality_score':True,'visual_quality_accepted':False,
            'seconds_including_load':time.perf_counter()-started,'details':extra}
    write(out/'result.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)


def worker_python(arm):
    if arm['kind']=='migan':return ROOT/'.build/migan-venv/bin/python'
    if arm['kind']=='klein':return ROOT/'experiments/qwen/.venv/bin/python'
    return ROOT/'.build/dreamlite-venv/bin/python'


def retry_resource_failures(folder,arm_ids):
    state=read(folder/'state.json')
    if any(j['status']=='running' for j in state['jobs']):raise RuntimeError('A worker is still marked running')
    retried=set()
    for job in state['jobs']:
        if job['arm'] not in arm_ids or job['status']!='failed':continue
        if job.get('execution',{}).get('stop_reason') not in ('memory_limit','swap_growth_limit','system_memory_pressure'):continue
        previous={k:v for k,v in job.items() if k!='previous_attempts'}
        job.setdefault('previous_attempts',[]).append(previous)
        for key in ('execution','output','started','finished','error'):job.pop(key,None)
        job['status']='pending';retried.add(job['id'])
    for job in state['jobs']:
        if job['status']=='blocked_by_first_failure' and job.get('reason') in retried:
            job['status']='pending';job.pop('reason',None)
    state['status']='prepared_resource_retry';write(folder/'state.json',state)


def execute(folder,max_wait_seconds,allow_warning_pressure=False,low_memory_first=False,retry_arms=(),only_arms=()):
    with (folder/'queue.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if retry_arms:retry_resource_failures(folder,retry_arms)
        execute_locked(folder,max_wait_seconds,allow_warning_pressure,low_memory_first,only_arms)


def execute_locked(folder,max_wait_seconds,allow_warning_pressure=False,low_memory_first=False,only_arms=()):
    config=read(folder/'inputs.json');state=read(folder/'state.json')
    if any(j['status']=='running' for j in state['jobs']):raise RuntimeError('Interrupted running job needs inspection; refusing automatic rerun')
    arms={a['id']:a for a in config['arms']};deadline=time.monotonic()+max_wait_seconds
    allowed=(1,2) if allow_warning_pressure else (1,)
    state['allowed_pressure_levels']=list(allowed)
    if low_memory_first:
        priority=['migan','klein-base-outpaint-q2-cached','klein-base-outpaint-mlp3-cached',
                  'klein-base-outpaint-q4-cached','klein-base-outpaint-q3-cached']
        rank={name:i for i,name in enumerate(priority)}
        state['jobs'].sort(key=lambda j:(rank.get(j['arm'],len(rank)),j['arm'],j['cover']))
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
    os.environ.update(env)
    stable=0
    for job in state['jobs']:
        if job['status']!='pending':continue
        if only_arms and job['arm'] not in only_arms:continue
        arm=arms[job['arm']]
        while True:
            pressure,swap=pressure_and_swap()
            stable=stable+1 if pressure in allowed else 0
            if stable>=3:break
            state.update(status='waiting_for_normal_memory',pressure_level=pressure,updated=time.time())
            write(folder/'state.json',state)
            if time.monotonic()>deadline:
                state['status']='paused_memory_pressure';write(folder/'state.json',state);return
            time.sleep(1 if pressure in allowed else 5)
        out=folder/'outputs'/job['arm']/job['cover']
        if job.get('previous_attempts'):out=out/f"attempt{len(job['previous_attempts'])+1:02d}"
        if (out/'result.json').exists():raise RuntimeError('A completed artifact already exists; refusing to overwrite it')
        out.mkdir(parents=True,exist_ok=True)
        guard=out/('guard-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        command=[str(worker_python(arm)),str(Path(__file__).resolve()),'worker','--folder',str(folder),'--arm',job['arm'],'--cover',job['cover'],'--output',str(out)]
        job.update(status='running',output=str(out),started=time.time());state.update(status='running',current=job['id']);write(folder/'state.json',state)
        try:
            receipt=run(command,guard,arm.get('limit_gib',6.0),900,allowed_pressure_levels=allowed,
                        max_swap_mib=arm.get('max_swap_mib',256),min_free_percent=25)
            job['execution']=receipt
            success=receipt['status']=='completed' and (out/'result.json').is_file()
            job['status']='completed' if success else 'failed'
            reason=receipt['stop_reason']
            if not success and reason in ('system_memory_pressure','swap_growth_limit','system_memory_headroom'):
                state['status']='paused_resource_guard';write(folder/'state.json',state);return
            # A failed model is not a bad-quality result. Avoid repeatedly loading
            # a broken/incompatible/oversized runtime on nine more covers.
            if not success:
                for other in state['jobs']:
                    if other['arm']==job['arm'] and other['status']=='pending':other.update(status='blocked_by_first_failure',reason=job['id'])
        except Exception as exc:
            job.update(status='not_started',error=str(exc));state['status']='paused_launch_error';write(folder/'state.json',state);return
        job['finished']=time.time();write(folder/'state.json',state)
        print(job['id'],job['status'],flush=True)
        stable=0
        render_report(folder)
    state['status']='paused_subset_complete' if any(j['status']=='pending' for j in state['jobs']) else 'finished_with_failures' if any(j['status']!='completed' for j in state['jobs']) else 'completed'
    state['finished']=time.time();write(folder/'state.json',state);render_report(folder)


def render_report(folder):
    from PIL import Image,ImageOps,ImageDraw
    config=read(folder/'inputs.json');state=read(folder/'state.json');summary=[]
    for arm in config['arms']:
        jobs=[j for j in state['jobs'] if j['arm']==arm['id']];done=[j for j in jobs if j['status']=='completed']
        if done:
            sheet=Image.new('RGB',(5*220,2*535),'#151515');draw=ImageDraw.Draw(sheet)
            for i,cover in enumerate(config['tracks']):
                job=next(j for j in jobs if j['cover']==cover['id'])
                path=Path(job.get('output',folder/'outputs'/arm['id']/cover['id']))/'composite.png'
                x=(i%5)*220+8;y=(i//5)*535+8
                if path.is_file():
                    with Image.open(path) as im:sheet.paste(ImageOps.contain(im,(204,470)),(x,y))
                draw.text((x,y+474),cover['id']+' '+cover['title'][:24],fill='white')
            dest=folder/'comparisons';dest.mkdir(exist_ok=True);sheet.save(dest/(arm['id']+'.jpg'))
        seconds=[read(Path(j['output'])/'result.json')['seconds_including_load'] for j in done]
        peaks=[max(j['execution']['peak_sampled_rss_bytes'],j['execution']['peak_sampled_phys_footprint_bytes']) for j in done]
        summary.append({'arm':arm['id'],'statuses':dict(collections.Counter(j['status'] for j in jobs)),
                        'mean_cold_seconds':sum(seconds)/len(seconds) if seconds else None,
                        'max_sampled_memory_gib':max(peaks)/1024**3 if peaks else None,
                        'visual_quality_rank':None})
    write(folder/'summary.json',{'status':state['status'],'arms':summary,'quality_not_inferred_from_execution_or_source_paste':True})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','worker','report'])
    parser.add_argument('--folder',type=Path,required=True)
    parser.add_argument('--arm');parser.add_argument('--cover');parser.add_argument('--output',type=Path)
    parser.add_argument('--max-wait-seconds',type=int,default=7200)
    parser.add_argument('--allow-warning-pressure',action='store_true',help='Explicit user-authorized benchmark only: allow pressure 2; critical pressure, process and swap limits still stop the job')
    parser.add_argument('--low-memory-first',action='store_true')
    parser.add_argument('--retry-resource-arm',action='append',default=[],help='Explicit retry in a fresh attempt folder; previous receipts are retained')
    parser.add_argument('--arms',nargs='+',default=[],help='Controlled subset; other queued arms remain pending')
    args=parser.parse_args();folder=args.folder.resolve();folder.mkdir(parents=True,exist_ok=True)
    if args.action=='prepare':prepare(folder)
    elif args.action=='run':execute(folder,args.max_wait_seconds,args.allow_warning_pressure,args.low_memory_first,args.retry_resource_arm,args.arms)
    elif args.action=='worker':worker(folder,args.arm,args.cover,args.output)
    else:render_report(folder)


if __name__=='__main__':main()
