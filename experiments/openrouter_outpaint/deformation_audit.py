"""Read-only global shape-change screening of existing teacher outputs.

Translation, rotation and uniform scale are allowed. A well-supported full
affine fit tests unequal scaling and shear, which the older similarity-only
source locator could not measure. This is not an anatomy/semantic guarantee.
No review decisions, source images or existing fidelity files are modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def shape_parameters(matrix):
    """Scale-invariant singular-value ratio and transformed axis angle."""
    a, b = matrix[0][:2]
    c, d = matrix[1][:2]
    u, v, cross = a*a+c*c, b*b+d*d, a*b+c*d
    trace = u+v
    disc = math.sqrt(max(0, (u-v)**2+4*cross*cross))
    low, high = (trace-disc)/2, (trace+disc)/2
    if low <= 1e-12 or a*d-b*c <= 0:
        return None
    return {'axis_scale_ratio': math.sqrt(u/v),
            'anisotropy_ratio': math.sqrt(high/low),
            'axis_angle_degrees': math.degrees(math.acos(max(-1, min(1, cross/math.sqrt(u*v))))),
            'shear_deviation_degrees': abs(90-math.degrees(math.acos(max(-1, min(1, cross/math.sqrt(u*v))))))}


def classify(parameters, reliable, affine_error, similarity_error):
    if not reliable or parameters is None:
        return 'inconclusive'
    # Lower residual is evidence that an extra affine degree of freedom was
    # needed. Avoid flagging subpixel estimator noise on almost identical art.
    improvement = similarity_error-affine_error
    if ((parameters['anisotropy_ratio'] >= 1.05 or parameters['shear_deviation_degrees'] >= 3)
            and improvement >= .4 and similarity_error >= affine_error*1.2):
        return 'global_deformation_suspected'
    if parameters['anisotropy_ratio'] >= 1.025 or parameters['shear_deviation_degrees'] >= 1.5:
        return 'borderline'
    return 'no_global_deformation_detected'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(131072), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def local_structure(reference, working, matrix):
    """Bidirectionally consistent dense displacement after a global affine fit.

    Kept as a review warning: local edits and texture redraws can also move
    image structures. Never infer that a face is anatomically correct.
    """
    import cv2
    import numpy as np
    aligned = cv2.warpAffine(working, matrix, (512, 512),
                             flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    visible = cv2.warpAffine(np.ones(working.shape, np.uint8), matrix, (512, 512),
                             flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
    visible = cv2.erode(visible, np.ones((11,11), np.uint8)).astype(bool)
    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    flow.setUseMeanNormalization(True)
    forward = flow.calc(reference, aligned, None)
    backward = flow.calc(aligned, reference, None)
    y,x = np.indices(reference.shape, dtype=np.float32)
    mx,my = x+forward[:,:,0], y+forward[:,:,1]
    reverse = cv2.remap(backward, mx, my, cv2.INTER_LINEAR)
    consistency = np.linalg.norm(forward+reverse, axis=2)
    texture = np.hypot(cv2.Sobel(reference,cv2.CV_32F,1,0),cv2.Sobel(reference,cv2.CV_32F,0,1)) > 20
    valid = visible & texture & (mx>5) & (mx<506) & (my>5) & (my<506) & (consistency<1)
    if valid.mean()<.1:
        return {'status':'inconclusive','valid_textured_fraction':float(valid.mean())}
    magnitude = np.linalg.norm(forward, axis=2)[valid]
    p50,p90 = map(float,np.percentile(magnitude,[50,90]))
    moved = float((magnitude>3).mean())
    # A local bend can occupy less than half of the source; requiring a large
    # median would silently accept an altered arm against an unchanged scene.
    return {'status':'local_structure_change_suspected' if p90>=4 and moved>=.12 else 'no_large_local_displacement_detected',
            'valid_textured_fraction':float(valid.mean()),'median_displacement_px':p50,
            'p90_displacement_px':p90,'fraction_over_3px':moved,
            'note':'At 512px reference scale; local redraw and shape changes both require visual review.'}


def build_manifest(old, new, output):
    rows = []
    provenance = []
    for label, gallery, accepted_only in [('sumbul_approved', old, True), ('nilufer_all', new, False)]:
        data_path = gallery/'data.json'
        data = json.loads(data_path.read_text())
        decision_path = gallery/'training-reviews.json'
        decisions = json.loads(decision_path.read_text())['decisions'] if decision_path.exists() else {}
        provenance.append({'dataset': label, 'data_file': str(data_path.resolve()), 'data_sha256': sha(data_path),
                           'decisions_sha256': sha(decision_path) if decision_path.exists() else None})
        for row in data['rows']:
            vote = decisions.get(row['id'], {})
            if accepted_only and not (vote.get('decision') == 'accepted' and vote.get('raw_sha256') == row['raw_sha256']):
                continue
            asset = gallery/row['asset']
            rows.append({'dataset': label, 'id': row['id'], 'title': row['title'], 'artists': row['artists'],
                         'model': row['model'], 'aspect_ratio': row.get('aspect_ratio', row.get('settings', {}).get('aspect_ratio')),
                         'decision': vote.get('decision', 'pending'), 'raw_sha256': row['raw_sha256'],
                         'source_file': str((asset/'source.png').resolve()), 'raw_file': str((asset/'raw.png').resolve()),
                         'original_file': str((asset/row['original_file']).resolve()),
                         'old_alignment_status': row.get('audit_status'), 'old_metrics': row.get('metrics')})
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'created_unix': time.time(), 'provenance': provenance, 'rows': rows,
                'scope': 'all hash-matched old human approvals and all locally published new outputs at snapshot time'}
    atomic(output/'manifest.json', manifest)
    return manifest


def measure(row):
    import cv2
    import numpy as np
    from PIL import Image
    cv2.setNumThreads(1)
    cv2.setRNGSeed(42)
    raw = Path(row['raw_file'])
    if sha(raw) != row['raw_sha256']:
        raise ValueError('Output hash differs from snapshot')
    with Image.open(row['source_file']) as source:
        source_size = list(source.size)
        reference = np.asarray(source.convert('L').resize((512, 512), Image.Resampling.LANCZOS))
    if source_size[0] != source_size[1]:
        raise ValueError('Expected original square conditioning image')
    with Image.open(raw) as image:
        native_size = image.size
        ratio = min(512/min(image.size), 960/max(image.size))
        working_size = tuple(round(d*ratio) for d in image.size)
        if max(working_size) > 1600:
            raise ValueError('Unexpected analysis dimensions')
        working = np.asarray(image.convert('L').resize(working_size, Image.Resampling.LANCZOS))
    detector = cv2.SIFT_create(nfeatures=3000)
    ka, da = detector.detectAndCompute(reference, None)
    kb, db = detector.detectAndCompute(working, None)
    base = {'source_dimensions': source_size, 'native_dimensions': list(native_size), 'analysis_dimensions': list(working_size)}
    if da is None or db is None:
        return {**base, 'status': 'inconclusive', 'reason': 'no reliable descriptors'}
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    reverse = {m.queryIdx: m.trainIdx for m in matcher.match(db, da)}
    pairs = matcher.knnMatch(da, db, k=2)
    matches = [a for pair in pairs if len(pair) == 2 for a,b in [pair]
               if a.distance < .72*b.distance and reverse.get(a.trainIdx) == a.queryIdx]
    if len(matches) < 20:
        return {**base, 'status': 'inconclusive', 'reason': 'too few mutual matches', 'matches': len(matches)}
    src = np.float32([ka[m.queryIdx].pt for m in matches])
    dst = np.float32([kb[m.trainIdx].pt for m in matches])
    matrix, mask = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2,
                                      maxIters=4000, confidence=.999, refineIters=20)
    if matrix is None or mask is None:
        return {**base, 'status': 'inconclusive', 'reason': 'affine fit failed', 'matches': len(matches)}
    ok = mask.ravel().astype(bool)
    points = src[ok]
    coverage = cv2.contourArea(cv2.convexHull(points))/512**2
    quadrants = len({(int(x >= 256), int(y >= 256)) for x,y in points})
    residual = np.linalg.norm(src @ matrix[:,:2].T+matrix[:,2]-dst, axis=1)
    affine_error = float(np.median(residual[ok]))
    # Best similarity fit to the SAME affine inliers; do not let a similarity
    # RANSAC retain a small unchanged fragment and hide stretching elsewhere.
    x, y = src[ok].astype(float), dst[ok].astype(float)
    mx, my = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x-mx, y-my
    u,s,vt = np.linalg.svd(xc.T @ yc)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1; rotation = vt.T @ u.T
    scale = float(np.sum((xc @ rotation.T)*yc)/np.sum(xc*xc))
    translation = my-scale*rotation @ mx
    similarity_error = float(np.median(np.linalg.norm(x @ (scale*rotation).T+translation-y, axis=1)))
    params = shape_parameters(matrix.tolist())
    reliable = int(ok.sum()) >= 20 and float(ok.mean()) >= .35 and coverage >= .2 and quadrants >= 3 and affine_error <= 2
    global_status = classify(params, reliable, affine_error, similarity_error)
    local = local_structure(reference, working, matrix) if reliable else {'status':'inconclusive'}
    status = global_status
    if status not in ('inconclusive','global_deformation_suspected') and local['status']=='local_structure_change_suspected':
        status = 'local_structure_change_suspected'
    return {**base, 'status': status, 'global_status':global_status, 'local_structure':local,
            'method': 'mutual_SIFT_full_affine_RANSAC_vs_similarity_same_inliers',
            'matches': len(matches), 'inliers': int(ok.sum()), 'inlier_fraction': float(ok.mean()),
            'source_feature_hull_fraction': coverage, 'quadrants': quadrants,
            'median_affine_residual_px': affine_error, 'median_similarity_residual_px': similarity_error,
            'matrix_source_to_analysis': matrix.tolist(), 'shape': params,
            'reason': None if reliable else 'insufficient spatial support for whole-cover judgement'}


def summary(output, rows):
    measured = []
    for row in rows:
        path = output/'measurements'/(row['dataset']+'--'+row['id']+'.json')
        if path.exists():
            measured.append(json.loads(path.read_text()))
    counts = {label: dict(Counter(r['status'] for r in measured if r['dataset'] == label))
              for label in {r['dataset'] for r in rows}}
    atomic(output/'summary.json', {'updated_unix': time.time(), 'total': len(rows), 'measured': len(measured),
                                  'remaining': len(rows)-len(measured), 'counts': counts,
                                  'notes': ['Source placement/rotation/uniform scale do not count as distortion.',
                                            'No-global-deformation result does not rule out local changes, faces, text or anatomy errors.',
                                            'Flagged results need visual review; human decisions were not modified.'], 'rows': measured})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    make = sub.add_parser('manifest')
    make.add_argument('--old', type=Path, required=True)
    make.add_argument('--new', type=Path, required=True)
    make.add_argument('--output', type=Path, required=True)
    run = sub.add_parser('run')
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--only', help='Optional exact row ID for a pilot')
    run.add_argument('--max-new', type=int, default=0, help='Bound a subprocess batch; zero means all')
    args = parser.parse_args()
    if args.command == 'manifest':
        manifest = build_manifest(args.old, args.new, args.output)
        print(json.dumps(dict(Counter(r['dataset'] for r in manifest['rows']))));return
    # Check before importing OpenCV/NumPy or decoding any image. Outer guard
    # additionally samples memory and swap during work.
    sys.path.insert(0, str(ROOT/'scripts'))
    from commit_stage import memory_gate
    memory_gate()
    rows = json.loads((args.output/'manifest.json').read_text())['rows']
    (args.output/'measurements').mkdir(exist_ok=True)
    done = 0
    for row in rows:
        if args.only and row['id'] != args.only:continue
        path = args.output/'measurements'/(row['dataset']+'--'+row['id']+'.json')
        if path.exists():continue
        memory_gate()
        result = measure(row)
        atomic(path, {**row, **result})
        print(json.dumps({'dataset': row['dataset'], 'id': row['id'], 'status': result['status']}), flush=True)
        done += 1
        if done % 20 == 0:
            summary(args.output, rows)
        if args.max_new and done >= args.max_new:
            break
    summary(args.output, rows)


if __name__ == '__main__':
    main()
