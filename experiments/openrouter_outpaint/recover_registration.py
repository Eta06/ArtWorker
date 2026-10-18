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


def recover(row, audit, asset, output):
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
                'evidence': chosen['evidence'], 'algorithm': 'rejected-features-plus-dense-evidence-v1'}
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gallery', type=Path, required=True)
    parser.add_argument('--audits', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output folder; preserve earlier attempts')
    args.output.mkdir(parents=True)
    gallery = json.loads((args.gallery/'data.json').read_text())
    reviews = json.loads((args.gallery/'training-reviews.json').read_text())['decisions']
    candidates = {}
    for row in gallery['rows']:
        review = reviews.get(row['id'], {})
        if (row['audit_status'] != 'unavailable' or not review.get('reason', '').strip()
                or review.get('raw_sha256') != row['raw_sha256']):
            continue
        path = args.audits/(row['id']+'.json')
        audit = json.loads(path.read_text())
        proposal = recover(row, audit, args.gallery/row['asset'], args.output)
        proposal['original_audit_sha256'] = sha(path)
        candidates[row['id']] = proposal
        (args.output/'registration-recovery.json').write_text(json.dumps({'schema_version': 1, 'candidates': candidates}, ensure_ascii=False, indent=2, allow_nan=False))
        print(json.dumps({'id': row['id'], 'status': proposal['status'],
                          'classification': proposal.get('evidence', {}).get('classification')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
