"""Reopen accepted artwork and verify integrity/duplicates against old archives."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from collect_covers import atomic_json
from collect_unique_covers import CoverIndex, baseline_files, signature


def audit(root, baselines):
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest['status'] != 'target_reached' or len(manifest['covers']) != manifest['target']:
        raise ValueError('Collection has not reached its target')
    known, files=baseline_files(baselines)
    index=CoverIndex()
    for path,label in files:
        index.add(path,label)
    hashes, pixels, albums=set(),set(),set()
    dimensions=Counter()
    for row in manifest['covers']:
        path=(root/row['file']).resolve()
        if not path.is_relative_to(root.resolve()/'covers'):
            raise ValueError('Unsafe image path')
        content=path.read_bytes()
        digest=hashlib.sha256(content).hexdigest()
        sig=signature(path)
        if digest != row['sha256'] or sig['pixel'] != row['normalized_rgb_sha256']:
            raise ValueError('Image changed after download')
        if digest in hashes or sig['pixel'] in pixels or row['album_id'] in albums:
            raise ValueError('Duplicate accepted row')
        if row['album_id'] in known['albums'] or row['asset_identity'] in known['assets']:
            raise ValueError('Old release or image asset accepted again')
        match,method=index.match(sig)
        if match:
            raise ValueError(f'Cross-archive duplicate {row["id"]} -> {match} ({method})')
        width,height=sig['dimensions']
        if [width,height] != row['dimensions'] or width != height or width < 512:
            raise ValueError('Image dimensions changed')
        hashes.add(digest);pixels.add(sig['pixel']);albums.add(row['album_id'])
        dimensions[f'{width}x{height}']+=1
        index.add(path,row['id'])
    summary=dict(stage=manifest['stage'],verified_covers=len(hashes),
        all_files_reopened=True,hashes_verified=True,cross_archive_duplicate_checks_passed=True,
        duplicate_method='Asset/release IDs, normalized RGB hash, pHash+dHash plus RGB MAE/p95; conservative heuristic, not semantic proof',
        baseline_indexed_files=len(files),dimensions=dict(dimensions),
        total_bytes=sum(r['bytes'] for r in manifest['covers']),
        origins=dict(Counter(r.get('origin','unknown') for r in manifest['covers'])),
        attempt_status_counts=dict(Counter(r['status'] for r in manifest['attempts'])),
        new_teacher_requests=0,new_generation_cost_usd=0,audio_downloads=0,
        rights_status='Local source candidates only; artwork and preferences not published',
        training_status='No paid generations or training; every source/output pair requires subsequent review')
    atomic_json(root/'audit.json',summary)
    # Representative samples across the accepted order, not a complete visual audit.
    covers=manifest['covers']
    sample=[covers[round(i*(len(covers)-1)/23)] for i in range(24)]
    sheet=Image.new('RGB',(6*180,4*205),'#161b24')
    draw=ImageDraw.Draw(sheet)
    for i,row in enumerate(sample):
        with Image.open(root/row['file']) as image:
            thumb=ImageOps.exif_transpose(image).convert('RGB').resize((170,170),Image.Resampling.LANCZOS)
            x,y=(i%6)*180+5,(i//6)*205+5
            sheet.paste(thumb,(x,y));draw.text((x,y+175),row['id'],fill='white')
    sheet.save(root/'overview.jpg',quality=90)
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--baseline-manifest',action='append',required=True,type=Path)
    args=parser.parse_args()
    audit(args.root.resolve(),args.baseline_manifest)
