"""Offline review proposals for recrops, recolors and changed-text cover reuse.

Never delete, reject, download or regenerate artwork. Hash retrieval proposes
neighbors; dense comparison and SIFT geometry narrow them for human review.
"""
from collections import Counter
from functools import lru_cache
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from collect_covers import ROOT, atomic_json
from collect_unique_covers import baseline_files
sys.path.insert(0,str(ROOT/'experiments/openrouter_outpaint'))
from source_fidelity import ssim_map

cv2.setNumThreads(1)
SCALES=(1.,.94,.85,.75)
BITS=np.array([int(i).bit_count() for i in range(256)],dtype=np.uint8)


def phash(image):
    gray=np.asarray(image.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=np.float32)
    values=cv2.dct(gray)[:8,:8].flatten()[1:]
    return np.uint64(sum(int(v>np.median(values))<<i for i,v in enumerate(values)))


def describe(path):
    with Image.open(path) as source:
        image=ImageOps.exif_transpose(source).convert('RGB')
        hashes=[]
        for scale in SCALES:
            w,h=image.size;dx=(1-scale)*w/2;dy=(1-scale)*h/2
            hashes.append(phash(image.crop((round(dx),round(dy),round(w-dx),round(h-dy)))))
        pixels=np.asarray(image.resize((256,256),Image.Resampling.LANCZOS))
        return {'pixel_sha':hashlib.sha256(pixels.tobytes()).hexdigest(),
                'hashes':hashes,'small':np.asarray(image.resize((64,64),Image.Resampling.LANCZOS))}


def distances(hashes,query):
    best=np.full(len(hashes),64,dtype=np.uint8)
    for value in query:
        xor=np.ascontiguousarray(hashes^value)
        count=BITS[xor.view(np.uint8).reshape(len(hashes),len(SCALES),8)].sum(axis=2).min(axis=1)
        best=np.minimum(best,count)
    return best


def correlation(a,b,valid=None):
    a=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY).astype(np.float32)
    b=cv2.cvtColor(b,cv2.COLOR_RGB2GRAY).astype(np.float32)
    if valid is not None:a,b=a[valid],b[valid]
    else:a,b=a.ravel(),b.ravel()
    if min(a.std(),b.std())<3:return 0.
    return float(np.corrcoef(a,b)[0,1])


@lru_cache(maxsize=24)
def features(path):
    with Image.open(path) as source:
        rgb=np.asarray(ImageOps.exif_transpose(source).convert('RGB').resize((512,512),Image.Resampling.LANCZOS))
    points,descriptors=cv2.SIFT_create(nfeatures=900).detectAndCompute(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),None)
    return rgb,np.array([p.pt for p in points],np.float32),descriptors


def compare(path_a,path_b):
    a,pa,da=features(str(path_a));b,pb,db=features(str(path_b))
    direct_ncc=correlation(a,b)
    direct_ssim=float(ssim_map(a,b)[5:-5,5:-5].mean())
    result={'direct_luma_ssim':direct_ssim,'direct_luma_ncc':direct_ncc,
            'rgb_mae':float(np.abs(a.astype(np.float32)-b).mean())}
    if direct_ncc>=.975 and direct_ssim>=.94:
        result['proposal']='dense_visual_match'
    if da is None or db is None or min(len(da),len(db))<12:return result
    pairs=cv2.BFMatcher().knnMatch(da,db,k=2)
    good=[m for pair in pairs if len(pair)==2 for m,n in [pair] if m.distance<.72*n.distance]
    if len(good)<12:return result
    src=np.float32([pa[m.queryIdx] for m in good]);dst=np.float32([pb[m.trainIdx] for m in good])
    matrix,inliers=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=2.5,maxIters=2000)
    if matrix is None or inliers is None:return result
    mask=inliers.ravel().astype(bool);count=int(mask.sum())
    scale=float(np.hypot(matrix[0,0],matrix[1,0]));angle=math.degrees(math.atan2(matrix[1,0],matrix[0,0]))
    hull=float(cv2.contourArea(cv2.convexHull(src[mask]))/(512**2)) if count>=3 else 0
    result.update(inliers=count,inlier_fraction=count/len(good),feature_hull_fraction=hull,scale=scale,rotation=angle)
    if count<12 or count/len(good)<.5 or hull<.08 or not .6<=scale<=1.65 or abs(angle)>4:return result
    aligned=cv2.warpAffine(b,matrix,(512,512),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
    valid=cv2.warpAffine(np.ones((512,512),np.uint8),matrix,(512,512),flags=cv2.INTER_NEAREST|cv2.WARP_INVERSE_MAP)
    valid=cv2.erode(valid,np.ones((11,11),np.uint8)).astype(bool)
    if valid.mean()<.55:return result
    ncc=correlation(a,aligned,valid);ssim=float(ssim_map(a,aligned)[valid].mean())
    result.update(aligned_luma_ncc=ncc,aligned_luma_ssim=ssim,visible_fraction=float(valid.mean()),matrix=matrix.tolist())
    if (ssim>=.8 and ncc>=.88) or (ncc>=.9 and count>=40 and hull>=.18):
        result['proposal']='registered_artwork_match'
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('collection',type=Path)
    parser.add_argument('--baseline-manifest',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if not out.is_relative_to(ROOT/'experiments/cover_collection/results'):parser.error('Use ignored cover results')
    if out.exists():parser.error('Keep prior audits; use a fresh output folder')
    out.mkdir(parents=True)
    collection=args.collection.resolve();manifest=json.loads((collection/'manifest.json').read_text())
    known,baseline=baseline_files(args.baseline_manifest)
    labels={};items=[];signatures=[];seen={}
    for path in args.baseline_manifest:
        data=json.loads(path.read_text())
        for row in data['covers']:labels[str((path.parent/row['file']).resolve())]={'title':row.get('album',''),'artists':row.get('artists',[])}
    for path,label in baseline:
        sig=describe(path)
        if sig['pixel_sha'] in seen:continue
        seen[sig['pixel_sha']]=label
        items.append({'id':label,'path':str(path),'new':False,**labels.get(str(path),{})});signatures.append(sig)
    query_indices=[]
    for row in manifest['covers']:
        path=(collection/row['file']).resolve()
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Collection image changed')
        sig=describe(path)
        query_indices.append(len(items));items.append({'id':manifest['stage']+':'+row['id'],'path':str(path),'new':True,
            'title':row.get('album',''),'artists':row.get('artists',[])});signatures.append(sig)
    atomic_json(out/'inventory.json',items)
    hashes=np.array([s['hashes'] for s in signatures],dtype=np.uint64)
    proposals=[];tested=[]
    print(json.dumps({'indexed_unique_reference_images':len(items)-len(query_indices),'new_images':len(query_indices)}),flush=True)
    for position,index in enumerate(query_indices):
        distance=distances(hashes,hashes[index]);distance[index:]=64
        neighbors=[j for j in np.argsort(distance)[:12] if distance[j]<=20]
        for j in neighbors:
            score=compare(items[index]['path'],items[j]['path'])
            record={'a':items[index],'b':items[j],'hash_distance':int(distance[j]),**score}
            tested.append(record)
            if score.get('proposal'):proposals.append(record)
        if (position+1)%50==0 or position+1==len(query_indices):
            atomic_json(out/'proposals.json',{'status':'complete' if position+1==len(query_indices) else 'running',
                'query_images_checked':position+1,'pairs_tested':len(tested),'proposals':proposals})
            print(json.dumps({'checked':position+1,'pairs_tested':len(tested),'proposals':len(proposals)}),flush=True)
    atomic_json(out/'tested-pairs.json',tested)
    proposals.sort(key=lambda r:max(r.get('aligned_luma_ssim',0),r['direct_luma_ssim']),reverse=True)
    for start in range(0,len(proposals),16):
        subset=proposals[start:start+16];sheet=Image.new('RGB',(1000,210*len(subset)),'#111310');draw=ImageDraw.Draw(sheet)
        for i,row in enumerate(subset):
            draw.text((5,210*i+5),f"#{start+i+1} {row['a']['id']} / {row['b']['id']}",fill='white')
            for column,key in enumerate(('a','b')):
                with Image.open(row[key]['path']) as source:
                    image=source.convert('RGB');image.thumbnail((190,180));sheet.paste(image,(column*500+5,210*i+25))
                draw.text((column*500+205,210*i+45),row[key].get('title','')[:38],fill='white')
        sheet.save(out/f'pairs-{start//16+1}.jpg',quality=90)
    atomic_json(out/'sorted-proposals.json',proposals)


if __name__=='__main__':main()
