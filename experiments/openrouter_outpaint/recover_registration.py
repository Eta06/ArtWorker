"""Propose source regions for human review without changing original audits.

This is an offline recovery pass, not a relaxed replacement for register().
Independent pixel evidence checks rejected ORB/SIFT hypotheses; every proposal
requires an explicit human boundary confirmation before training admission.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from source_fidelity import SIDE, metrics, sha, ssim_map, heatmap
from production_gallery import region

cv2.setNumThreads(1)


def needs_recovery(row, review, scope):
    return (row.get('audit_status') == 'unavailable'
            and review.get('raw_sha256') == row['raw_sha256']
            and (review.get('decision') == 'pending' if scope == 'skipped'
                 else bool(review.get('reason', '').strip())))

def assess(reference, output, diagnostic, peer=None):
    matrix = np.asarray(diagnostic.get('matrix_source_to_output', []), dtype=np.float32)
    if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        return None
    scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    rotation = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
    if not .25 <= scale <= 2 or abs(rotation) > 10:
        return None
    if (diagnostic.get('inliers', 0) < 20 or diagnostic.get('inlier_fraction', 0) < .30
            or diagnostic.get('median_inlier_residual_px', 100) > 1.5):
        return None
    aligned = cv2.warpAffine(output, matrix, (SIDE, SIDE), flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
    valid = cv2.warpAffine(np.ones(output.shape[:2], np.uint8), matrix, (SIDE, SIDE),
                           flags=cv2.INTER_NEAREST|cv2.WARP_INVERSE_MAP).astype(bool)
    if valid.mean() < .75:
        return None
    gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gradient = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    informative = cv2.dilate((gradient > 12).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    usable = informative & cv2.erode(valid.astype(np.uint8), np.ones((11, 11), np.uint8)).astype(bool)
    if usable.mean() < .01:
        return None
    a = gray[usable]
    b = cv2.cvtColor(aligned, cv2.COLOR_RGB2GRAY).astype(np.float32)[usable]
    if min(a.std(), b.std()) < 1:
        return None
    correlation = float(np.corrcoef(a, b)[0, 1])
    informative_ssim = float(ssim_map(reference, aligned)[usable].mean())
    scores = metrics(reference, aligned, valid)
    corners = np.float32([[0, 0], [SIDE, 0], [SIDE, SIDE], [0, SIDE]])
    agreement = None
    if peer and peer.get('matrix_source_to_output'):
        other = np.asarray(peer['matrix_source_to_output'], dtype=np.float32)
        if other.shape == (2, 3) and np.isfinite(other).all():
            agreement = float(np.linalg.norm(corners @ matrix[:, :2].T + matrix[:, 2]
                                            - corners @ other[:, :2].T - other[:, 2], axis=1).max())
    agreed = agreement is not None and agreement <= 8
    strong_single = (diagnostic.get('inliers', 0) >= 35 and diagnostic.get('inlier_fraction', 0) >= .7
                     and informative_ssim >= .85 and correlation >= .9)
    dense = informative_ssim >= .65 and correlation >= .85 and (agreed or strong_single)
    edited = (agreed and valid.mean() >= .95 and
              ((diagnostic.get('source_feature_hull_fraction', 0) >= .25 and
                diagnostic.get('quadrants', 0) == 4 and diagnostic.get('inliers', 0) >= 200 and correlation >= .65)
               or (scores['ssim_luminance'] >= .8 and correlation >= .5)))
    if not (dense or edited):
        return None
    classification = 'partial_reference' if dense and valid.mean() < .95 else 'dense_verified' if dense else 'edited_reference'
    evidence = {'classification': classification, 'visible_source_fraction': float(valid.mean()),
                'informative_fraction': float(usable.mean()), 'informative_ssim': informative_ssim,
                'informative_luminance_correlation': correlation,
                'orb_sift_max_corner_disagreement_px': agreement,
                'note': 'Geometry proposal only. Similarity is not unchanged-pixel percentage; manual confirmation required.'}
    return {'diagnostic': diagnostic, 'evidence': evidence, 'metrics': scores,
            'aligned': aligned, 'valid': valid}


def refine_assessment(reference, output, diagnostics):
    """Opt-in affine ECC proposals; content preservation is not inferred."""
    gray=cv2.cvtColor(reference,cv2.COLOR_RGB2GRAY)
    other=cv2.cvtColor(output,cv2.COLOR_RGB2GRAY)
    gradient=cv2.magnitude(cv2.Sobel(gray,cv2.CV_32F,1,0),cv2.Sobel(gray,cv2.CV_32F,0,1))
    informative=cv2.dilate((gradient>12).astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
    center=np.zeros((SIDE,SIDE),bool);center[SIDE//10:-SIDE//10,SIDE//10:-SIDE//10]=True
    if informative.mean()<.01:return None
    corners=np.float32([[0,0],[SIDE,0],[SIDE,SIDE],[0,SIDE]])
    candidates=[]
    for seed in diagnostics:
        matrix=np.asarray(seed.get('matrix_source_to_output',[]),np.float32)
        if (matrix.shape!=(2,3) or not np.isfinite(matrix).all()
                or seed.get('inliers',0)<20 or seed.get('inlier_fraction',0)<.3
                or seed.get('median_inlier_residual_px',100)>1.5):continue
        try:
            _,matrix=cv2.findTransformECC(gray,other,matrix.copy(),cv2.MOTION_AFFINE,
                (cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,100,1e-5),None,5)
        except cv2.error:continue
        singular=np.linalg.svd(matrix[:,:2],compute_uv=False)
        rotation=math.degrees(math.atan2(matrix[1,0],matrix[0,0]))
        if (not np.isfinite(matrix).all() or np.linalg.det(matrix[:,:2])<=0
                or not .2<=singular.min()<=singular.max()<=2.5
                or singular.max()/singular.min()>1.6 or abs(rotation)>10):continue
        aligned=cv2.warpAffine(output,matrix,(SIDE,SIDE),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
        valid=cv2.warpAffine(np.ones(output.shape[:2],np.uint8),matrix,(SIDE,SIDE),
                            flags=cv2.INTER_NEAREST|cv2.WARP_INVERSE_MAP).astype(bool)
        if valid.mean()<.75:continue
        usable=informative&cv2.erode(valid.astype(np.uint8),np.ones((11,11),np.uint8)).astype(bool)
        values=cv2.cvtColor(aligned,cv2.COLOR_RGB2GRAY);sm=ssim_map(reference,aligned)
        def evidence(mask):
            if mask.mean()<.01 or min(gray[mask].std(),values[mask].std())<1:return None
            return {'fraction':float(mask.mean()),'correlation':float(np.corrcoef(gray[mask],values[mask])[0,1]),
                    'ssim':float(sm[mask].mean())}
        full,inner=evidence(usable),evidence(usable&center)
        if not full or not inner:continue
        candidates.append({'matrix':matrix,'seed':seed,'aligned':aligned,'valid':valid,
                           'full':full,'center':inner,'rotation':rotation})
    accepted=[]
    broad=any(d.get('inliers',0)>=50 and d.get('inlier_fraction',0)>=.5
              and d.get('source_feature_hull_fraction',0)>=.25 and d.get('quadrants',0)>=3
              and d.get('median_inlier_residual_px',100)<=1.5 for d in diagnostics)
    for candidate in candidates:
        matrix=candidate['matrix'];peers=[]
        for other in candidates:
            distance=float(np.linalg.norm(corners@matrix[:,:2].T+matrix[:,2]
                          -corners@other['matrix'][:,:2].T-other['matrix'][:,2],axis=1).max())
            if distance<=3:peers.append((other,distance))
        methods={p['seed'].get('method') or 'feature_seed' for p,_ in peers}
        consensus=len(methods)>=2
        full,inner=candidate['full'],candidate['center']
        dense=full['correlation']>=.9 and full['ssim']>=.85
        central=consensus and inner['correlation']>=.9 and inner['ssim']>=.8
        edited=consensus and broad and inner['correlation']>=.6 and full['correlation']>=.45
        if not (dense or central or edited):continue
        proof={'classification':'dense_verified' if dense else 'edited_reference',
               'verification_scope':'informative_source_pixels' if dense else 'central_artwork_anchor' if central else 'edited_structure_anchors',
               'visible_source_fraction':float(candidate['valid'].mean()),
               'informative_fraction':full['fraction'],'informative_ssim':full['ssim'],
               'informative_luminance_correlation':full['correlation'],
               'center_informative_ssim':inner['ssim'],'center_luminance_correlation':inner['correlation'],
               'seed_methods':sorted(methods),'max_consensus_corner_difference_px':max(d for _,d in peers),
               'note':'Geometry proposal, not unchanged-content proof. Edited borders/text may fall outside it; human confirmation required.'}
        accepted.append({'diagnostic':{'status':'measured','method':'ecc_affine_from_feature_anchors',
                         'matrix_source_to_output':matrix.tolist(),'rotation_degrees':candidate['rotation']},
                         'evidence':proof,'metrics':metrics(reference,candidate['aligned'],candidate['valid']),
                         'aligned':candidate['aligned'],'valid':candidate['valid'],
                         'algorithm':'feature-anchors-pixel-ecc-v2'})
    return max(accepted,key=lambda c:(c['evidence']['classification']=='dense_verified',
                                     c['evidence']['center_luminance_correlation'])) if accepted else None


def recover(row, audit, asset, output, refine=False, shape_seed=None):
    raw, source = asset/'raw.png', asset/'source.png'
    if sha(raw) != row['raw_sha256'] or audit['raw_sha256'] != row['raw_sha256']:
        raise ValueError('Raw fingerprint mismatch')
    if sha(source) != audit['source_png_sha256']:
        raise ValueError('Source fingerprint mismatch')
    with Image.open(source) as im:
        reference = np.asarray(im.convert('RGB').resize((SIDE, SIDE), Image.Resampling.LANCZOS))
    with Image.open(raw) as im:
        working = np.asarray(im.convert('RGB').resize((SIDE, round(im.height*SIDE/im.width)), Image.Resampling.LANCZOS))
    diagnostics = [audit['alignment'], audit['alignment'].get('fallback_diagnostic', {})]
    choices = [candidate for i, d in enumerate(diagnostics)
               if (candidate := assess(reference, working, d, diagnostics[1-i]))]
    if not choices and refine:
        seeds=diagnostics+([shape_seed] if shape_seed else [])
        candidate=refine_assessment(reference,working,seeds)
        if candidate:choices.append(candidate)
    if not choices:
        return {'status': 'unresolved', 'raw_sha256': row['raw_sha256']}
    chosen = max(choices, key=lambda c: c['evidence']['informative_ssim'])
    diagnostic = {**chosen['diagnostic'], 'status': 'measured'}
    # Only for drawing this versioned proposal, never written into the old audit.
    proposal_audit = {'alignment': diagnostic, 'aligned': chosen['metrics'],
                      'analysis_side': SIDE, 'raw_dimensions': row['dimensions']}
    proposal = {'status': 'review_pending', 'raw_sha256': row['raw_sha256'],
                'source_png_sha256': sha(source), 'region': region(proposal_audit),
                'alignment': chosen['diagnostic'], 'metrics': chosen['metrics'],
                'evidence': chosen['evidence'], 'algorithm': chosen.get('algorithm','rejected-features-plus-dense-evidence-v1')}
    token = hashlib.sha256(json.dumps(proposal, sort_keys=True, allow_nan=False).encode()).hexdigest()
    proposal['id'] = token
    folder = output/token
    folder.mkdir()
    Image.fromarray(reference).save(folder/'source.png')
    Image.fromarray(chosen['aligned']).save(folder/'aligned.png')
    heatmap(reference, chosen['aligned']).save(folder/'aligned-difference.png')
    overlay = np.uint8((reference.astype(np.float32)+chosen['aligned'])/2)
    overlay[~chosen['valid']] = (100, 100, 100)
    Image.fromarray(overlay).save(folder/'aligned-overlay.png')
    marked = Image.fromarray(working)
    matrix = np.asarray(diagnostic['matrix_source_to_output'])
    points = np.float32([[0, 0], [SIDE, 0], [SIDE, SIDE], [0, SIDE]]) @ matrix[:, :2].T + matrix[:, 2]
    ImageDraw.Draw(marked).line([tuple(p) for p in points]+[tuple(points[0])], fill='#ffad45', width=3)
    marked.save(folder/'placement.png')
    proposal['directory'] = str(folder.resolve())
    return proposal


def center_reference(row, audit, asset, output):
    """User-requested manual square, explicitly not a recovered source match."""
    raw,source=asset/'raw.png',asset/'source.png'
    if sha(raw)!=row['raw_sha256'] or audit['raw_sha256']!=row['raw_sha256']:
        raise ValueError('Raw fingerprint mismatch')
    if sha(source)!=audit['source_png_sha256']:raise ValueError('Source fingerprint mismatch')
    w,h=row['dimensions'];side=min(w,h);x,y=(w-side)//2,(h-side)//2
    proposal={'status':'review_pending','raw_sha256':row['raw_sha256'],'source_png_sha256':sha(source),
        'region':{'polygon':[[x,y],[x+side,y],[x+side,y+side],[x,y+side]],
                  'margins_output_px':{'left':x,'right':w-x-side,'top':y,'bottom':h-y-side},
                  'rotation_degrees':0,'note':'User-requested centered square, not source registration.'},
        'alignment':{'status':'manual','method':'user_requested_center_square'},'metrics':None,
        'evidence':{'classification':'manual_center_reference','is_source_match':False,
                    'note':'Centered full-short-edge square requested by the user. Human boundary and quality decisions remain required.'},
        'algorithm':'manual-center-square-user-request-v1'}
    token=hashlib.sha256(json.dumps(proposal,sort_keys=True,allow_nan=False).encode()).hexdigest()
    proposal['id']=token;folder=output/token;folder.mkdir()
    with Image.open(source) as image:image.convert('RGB').resize((SIDE,SIDE),Image.Resampling.LANCZOS).save(folder/'source.png')
    with Image.open(raw) as image:
        if image.size!=(w,h):raise ValueError('Raw dimensions mismatch')
        marked=image.convert('RGB').resize((SIDE,round(h*SIDE/w)),Image.Resampling.LANCZOS)
    points=[(px*marked.width/w,py*marked.height/h) for px,py in proposal['region']['polygon']]
    ImageDraw.Draw(marked).line(points+[points[0]],fill='#ffad45',width=3);marked.save(folder/'placement.png')
    proposal['directory']=str(folder.resolve())
    return proposal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gallery', type=Path, required=True)
    parser.add_argument('--audits', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scope', choices=['noted','skipped'], default='noted',
                        help='Skipped includes pending reviews without a written note')
    parser.add_argument('--refine',action='store_true',help='Try independently checked affine ECC proposals')
    parser.add_argument('--shape-audits',type=Path,help='Optional existing full-affine seed measurements')
    parser.add_argument('--center-fallback',action='store_true',help='Manual centered square for unresolved or weak edited structure matches')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output folder; preserve earlier attempts')
    args.output.mkdir(parents=True)
    gallery = json.loads((args.gallery/'data.json').read_text())
    reviews = json.loads((args.gallery/'training-reviews.json').read_text())['decisions']
    candidates = {}
    for row in gallery['rows']:
        review = reviews.get(row['id'], {})
        if not needs_recovery(row, review, args.scope):
            continue
        path = args.audits/(row['id']+'.json')
        audit = json.loads(path.read_text())
        seed=None
        if args.shape_audits:
            shape_paths=list(args.shape_audits.glob('*--'+row['id']+'.json'))
            if len(shape_paths)>1:raise ValueError('Ambiguous shape seed files')
            if shape_paths:
                shape_path=shape_paths[0]
                shape=json.loads(shape_path.read_text())
                if shape['raw_sha256']!=row['raw_sha256']:raise ValueError('Shape/output fingerprint mismatch')
                w,h=row['dimensions'];expected=[SIDE,round(h*SIDE/w)]
                if shape.get('analysis_dimensions')==expected and shape.get('matrix_source_to_analysis'):
                    seed={**shape,'method':'affine_sift_audit','matrix_source_to_output':shape['matrix_source_to_analysis'],
                          'median_inlier_residual_px':shape['median_affine_residual_px']}
        proposal = recover(row, audit, args.gallery/row['asset'], args.output,args.refine,seed)
        if args.center_fallback and (proposal['status']=='unresolved' or
                proposal.get('evidence',{}).get('verification_scope')=='edited_structure_anchors'):
            proposal=center_reference(row,audit,args.gallery/row['asset'],args.output)
        proposal['original_audit_sha256'] = sha(path)
        candidates[row['id']] = proposal
        (args.output/'registration-recovery.json').write_text(json.dumps({'schema_version': 1, 'candidates': candidates}, ensure_ascii=False, indent=2, allow_nan=False))
        print(json.dumps({'id': row['id'], 'status': proposal['status'],
                          'classification': proposal.get('evidence', {}).get('classification')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
