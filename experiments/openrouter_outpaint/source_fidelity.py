"""Audit source-square fidelity of existing raw outpaints. No network or model calls.

Dependencies: Pillow, NumPy, OpenCV. All metrics use a common 512px source grid.
Fixed geometry is authoritative; similarity registration is diagnostic only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import time
import urllib.parse

import cv2
import numpy as np
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[2]
SIDE = 512
COVERS = {'track2': 'Karambol', 'track3': 'Aklın Hep Bende'}
cv2.setNumThreads(1)
cv2.setRNGSeed(42)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(row: dict) -> str:
    return hashlib.sha256(json.dumps([row['model'], row['profile'], row['track'],
                                     row.get('raw_sha256')]).encode()).hexdigest()


def ssim_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Luminance SSIM: 11x11 Gaussian, sigma=1.5, L=255, K1=.01, K2=.03.
    a = cv2.cvtColor(a.astype(np.float32), cv2.COLOR_RGB2GRAY)
    b = cv2.cvtColor(b.astype(np.float32), cv2.COLOR_RGB2GRAY)
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    ma, mb = blur(a), blur(b)
    va, vb, cov = blur(a*a)-ma*ma, blur(b*b)-mb*mb, blur(a*b)-ma*mb
    return ((2*ma*mb+2.55**2)*(2*cov+7.65**2) /
            ((ma*ma+mb*mb+2.55**2)*(va+vb+7.65**2)))


def metrics(reference: np.ndarray, candidate: np.ndarray,
            mask: np.ndarray | None = None) -> dict:
    if reference.shape != candidate.shape:
        raise ValueError('Metric images must have the same dimensions')
    valid = np.ones(reference.shape[:2], bool) if mask is None else mask.astype(bool)
    # Only evaluate SSIM windows entirely inside the image and valid overlap.
    inner = cv2.erode(valid.astype(np.uint8), np.ones((11, 11), np.uint8),
                      borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    if not inner.any() or not valid.any():
        raise ValueError('Insufficient common pixels')
    a, b = reference.astype(np.float32), candidate.astype(np.float32)
    delta = b-a
    absolute = np.abs(delta)
    mse = float(np.mean(delta[valid]**2))
    la = cv2.cvtColor(a/255, cv2.COLOR_RGB2LAB)
    lb = cv2.cvtColor(b/255, cv2.COLOR_RGB2LAB)
    de = np.linalg.norm(lb-la, axis=2)[valid]
    sm = ssim_map(reference, candidate)
    tiles = []
    for y in range(0, SIDE, SIDE//4):
        for x in range(0, SIDE, SIDE//4):
            m = inner[y:y+SIDE//4, x:x+SIDE//4]
            if m.mean() > .5:
                tiles.append(float(sm[y:y+SIDE//4, x:x+SIDE//4][m].mean()))
    return {'ssim_luminance': float(sm[inner].mean()),
            'mae_rgb_0_255': float(absolute[valid].mean()),
            'psnr_db': 10*math.log10(255**2/mse) if mse else None,
            'pixel_exact_on_analysis_grid_fraction': float(np.all(delta == 0, axis=2)[valid].mean()),
            'pixels_over_8_rgb_fraction': float((absolute.max(axis=2)[valid] > 8).mean()),
            'mean_signed_rgb_delta': delta[valid].mean(axis=0).tolist(),
            'delta_e_76_mean': float(de.mean()), 'delta_e_76_p95': float(np.percentile(de, 95)),
            'worst_tile_ssim': min(tiles) if tiles else None,
            'evaluated_fraction': float(valid.mean())}


def expected_rect(geometry: dict, size: tuple[int, int]) -> tuple[float, ...]:
    cw, ch = geometry['canvas_dimensions']
    x0, y0, x1, y1 = geometry['source_rect_xyxy']
    return x0/cw*size[0], y0/ch*size[1], x1/cw*size[0], y1/ch*size[1]


def square_at(image: Image.Image, rect: tuple[float, ...]) -> np.ndarray:
    return np.asarray(image.transform((SIDE, SIDE), Image.Transform.EXTENT, rect,
                                      resample=Image.Resampling.BICUBIC))


def register(reference: np.ndarray, output: np.ndarray, rect: tuple[float, ...]) -> tuple[dict, np.ndarray | None, np.ndarray | None]:
    """Estimate one global similarity transform; never locally warp/color-correct."""
    orb = cv2.ORB_create(nfeatures=4000, edgeThreshold=15, fastThreshold=8)
    ka, da = orb.detectAndCompute(cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY), None)
    kb, db = orb.detectAndCompute(cv2.cvtColor(output, cv2.COLOR_RGB2GRAY), None)
    failed = {'status': 'unavailable', 'reason': 'insufficient reliable features'}
    if da is None or db is None or len(da) < 12 or len(db) < 12:
        return failed, None, None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    reverse = {m.queryIdx: m.trainIdx for m in matcher.match(db, da)}
    matches = [a for pair in matcher.knnMatch(da, db, k=2) if len(pair) == 2
               for a, b in [pair] if a.distance < .75*b.distance and reverse.get(a.trainIdx) == a.queryIdx]
    if len(matches) < 12:
        return {**failed, 'matches': len(matches)}, None, None
    src = np.float32([ka[m.queryIdx].pt for m in matches])
    dst = np.float32([kb[m.trainIdx].pt for m in matches])
    transform, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                               ransacReprojThreshold=2, maxIters=3000, confidence=.995)
    if transform is None:
        return failed, None, None
    ok = mask.ravel().astype(bool)
    points = src[ok]
    coverage = cv2.contourArea(cv2.convexHull(points))/SIDE**2 if len(points) >= 3 else 0
    quadrants = len({(int(x >= SIDE/2), int(y >= SIDE/2)) for x, y in points})
    scale = float(np.hypot(transform[0, 0], transform[1, 0]))
    rotation = math.degrees(math.atan2(transform[1, 0], transform[0, 0]))
    center = transform @ np.array([SIDE/2, SIDE/2, 1])
    expected_center = np.array([(rect[0]+rect[2])/2, (rect[1]+rect[3])/2])
    shift = center-expected_center
    residual = np.linalg.norm((src @ transform[:, :2].T + transform[:, 2])-dst, axis=1)
    diagnostic = {'status': 'measured', 'matches': len(matches), 'inliers': int(ok.sum()),
                  'inlier_fraction': float(ok.mean()), 'source_feature_hull_fraction': coverage,
                  'quadrants': quadrants, 'median_inlier_residual_px': float(np.median(residual[ok])),
                  'scale_relative_to_expected': scale/((rect[2]-rect[0])/SIDE),
                  'rotation_degrees': rotation, 'center_shift_fraction_of_cover': (shift/SIDE).tolist(),
                  'matrix_source_to_output': transform.tolist()}
    if ok.sum() < 12 or ok.mean() < .4 or coverage < .15 or quadrants < 3:
        return {**diagnostic, 'status': 'unavailable', 'reason': 'weak or spatially concentrated matches'}, None, None
    if not .75 <= diagnostic['scale_relative_to_expected'] <= 1.25 or abs(rotation) > 5 or max(abs(shift)) > SIDE*.18:
        return {**diagnostic, 'status': 'unavailable', 'reason': 'transform outside diagnostic search bounds'}, None, None
    aligned = cv2.warpAffine(output, transform, (SIDE, SIDE),
                            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    valid = cv2.warpAffine(np.ones(output.shape[:2], np.uint8), transform, (SIDE, SIDE),
                          flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP).astype(bool)
    diagnostic['visible_source_fraction'] = float(valid.mean())
    if valid.mean() < .85:
        return {**diagnostic, 'status': 'unavailable', 'reason': 'too little source visible'}, None, None
    return diagnostic, aligned, valid


def heatmap(a: np.ndarray, b: np.ndarray) -> Image.Image:
    # Shared, fixed gain: every heatmap uses the same scale (0..64 RGB difference).
    difference = np.abs(a.astype(np.float32)-b).mean(axis=2)
    colors = cv2.applyColorMap(np.uint8(np.clip(difference/64, 0, 1)*255), cv2.COLORMAP_INFERNO)
    return Image.fromarray(cv2.cvtColor(colors, cv2.COLOR_BGR2RGB))


def audit(row: dict, destination: Path) -> dict:
    directory = (ROOT / row['directory']).resolve()
    if not directory.is_relative_to(ROOT):
        raise ValueError('Ledger directory outside repository')
    raw, source = directory/'raw.png', directory/'source.png'
    if sha(raw) != row['raw_sha256']:
        raise ValueError('Raw hash differs from reviewed output')
    receipt = json.loads((directory/'receipt.json').read_text())
    for reference in receipt['request']['input_references']:
        name = reference['local_file']
        if name not in {'source.png', 'input.png'} or sha(directory/name) != reference['sha256']:
            raise ValueError('Input reference checksum mismatch')
    geometry = receipt['geometry']
    with Image.open(source) as im:
        square = im.convert('RGB')
    with Image.open(directory/'input.png') as im:
        canvas = im.convert('RGB')
    if canvas.size != tuple(geometry['canvas_dimensions']) or square.size != (720, 720):
        raise ValueError('Unexpected source geometry')
    if ImageChops.difference(canvas.crop(geometry['source_rect_xyxy']), square).getbbox():
        raise ValueError('Source differs from preserved canvas square')
    original = ROOT/'Player/Resources'/f"{row['track']}.jpg"
    if sha(original) != geometry['original_sha256'] or row['source_sha256'] != geometry['original_sha256']:
        raise ValueError('Original cover changed since generation')
    with Image.open(original) as im:
        reconstructed = im.convert('RGB').crop(geometry['crop_xyxy']).resize(square.size, Image.Resampling.LANCZOS)
    if ImageChops.difference(reconstructed, square).getbbox():
        raise ValueError('Source cannot be reproduced from original cover')
    reference = np.asarray(square.resize((SIDE, SIDE), Image.Resampling.LANCZOS))
    with Image.open(raw) as im:
        native_size = im.size
        if list(native_size) != row['raw_dimensions']:
            raise ValueError('Output dimensions differ from ledger')
        # Decode only one output at a time. No float arrays at native 4K resolution.
        working = im.convert('RGB').resize((SIDE, round(native_size[1]*SIDE/native_size[0])), Image.Resampling.LANCZOS)
    rect = expected_rect(geometry, working.size)
    fixed = square_at(working, rect)
    # Identity outpaint, resized through the same native dimensions, establishes
    # the resampling-only floor. It is a diagnostic, not a subtracted score.
    control = canvas.resize(native_size, Image.Resampling.LANCZOS).resize(working.size, Image.Resampling.LANCZOS)
    control_crop = square_at(control, rect)
    alignment, aligned, valid = register(reference, np.asarray(working), rect)
    out = {k: row.get(k) for k in ('model', 'profile', 'track', 'raw_sha256', 'raw_dimensions', 'actual_cost_usd', 'elapsed_seconds')}
    out.update(id=identity(row), status='measured', source_png_sha256=sha(source),
               source_original_sha256=sha(original), geometry=geometry,
               analysis_side=SIDE, fixed=metrics(reference, fixed),
               resampling_control=metrics(reference, control_crop), alignment=alignment,
               aligned=metrics(reference, aligned, valid) if aligned is not None else None)
    item = destination/out['id']
    item.mkdir(exist_ok=True)
    Image.fromarray(reference).save(item/'source.png')
    Image.fromarray(fixed).save(item/'fixed.png')
    heatmap(reference, fixed).save(item/'difference.png')
    if aligned is not None:
        Image.fromarray(aligned).save(item/'aligned.png')
        heatmap(reference, aligned).save(item/'aligned-difference.png')
    marked = np.asarray(working).copy()
    corners = np.float32([[0,0], [SIDE-1,0], [SIDE-1,SIDE-1], [0,SIDE-1]])
    cv2.rectangle(marked, (round(rect[0]), round(rect[1])), (round(rect[2])-1, round(rect[3])-1), (93,230,180), 2)
    if aligned is not None:
        transform = np.array(alignment['matrix_source_to_output'])
        polygon = np.int32(corners @ transform[:, :2].T + transform[:, 2])
        cv2.polylines(marked, [polygon], True, (255,180,80), 2)
    Image.fromarray(marked).save(item/'placement.png')
    out['raw_url'] = urllib.parse.quote(os.path.relpath(raw, destination))
    return out


def render(rows: list[dict], destination: Path, metadata: dict) -> None:
    template = Path(__file__).with_name('source-fidelity.html').read_text()
    data = json.dumps(rows, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    template = template.replace('{{ROWS}}', data).replace('{{COUNT}}', str(len(rows)))
    template = template.replace('{{CREATED}}', html.escape(metadata['created_utc']))
    (destination/'index.html').write_text(template)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--selections', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scope', choices=['liked', 'liked-models', 'all'], default='liked',
                        help='Liked outputs (default), all settings of liked models, or all completed outputs')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Fresh output directory required; preserve prior audits')
    ledger = json.loads(args.ledger.read_text())
    selections = json.loads(args.selections.read_text())
    if selections.get('schema_version') != 1:
        parser.error('Unsupported selection schema')
    lookup = {identity(r): r for r in ledger['rows']}
    votes = selections['selections']
    if len({s['id'] for s in votes}) != len(votes):
        parser.error('Duplicate selections')
    for s in votes:
        r = lookup.get(s['id'])
        if r is None or s['id'] != identity(s) or s['decision'] not in {'liked', 'disliked'}:
            parser.error('Selection does not match ledger identity/decision')
        if r['status'] != 'completed' or not s.get('available'):
            parser.error('Selection points to unavailable image')
    liked = {s['id'] for s in votes if s['decision'] == 'liked'}
    models = {lookup[i]['model'] for i in liked}
    selected = [r for r in ledger['rows'] if r['status'] == 'completed' and
                (args.scope == 'all' or identity(r) in liked or
                 (args.scope == 'liked-models' and r['model'] in models))]
    if not selected:
        parser.error('No outputs selected')
    args.output.mkdir(parents=True)
    results = []
    started = time.monotonic()
    for r in selected:
        try:
            result = audit(r, args.output)
        except (OSError, ValueError, KeyError, cv2.error) as error:
            result = {k:r.get(k) for k in ('model','profile','track')}
            result.update(id=identity(r), status='failed', error=str(error))
        results.append(result)
        print(json.dumps({'done':len(results), 'total':len(selected), 'status':result['status'],
                          'model':r['model'], 'profile':r['profile'], 'track':r['track']}, ensure_ascii=False), flush=True)
    metadata = {'schema_version':1, 'stage':'iris', 'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'scope':args.scope, 'ledger_sha256':sha(args.ledger), 'selections_sha256':sha(args.selections),
                'analysis_side':SIDE, 'seconds':round(time.monotonic()-started, 3),
                'versions':{'numpy':np.__version__, 'opencv':cv2.__version__, 'pillow':Image.__version__},
                'notes':['Fixed location is the primary score; registered score never replaces it.',
                         'Metrics use resampled pixels, not a claim of native pixel equality.',
                         'No overall quality score, trained perceptual model, face/text-specific guarantee or automatic eligibility decision.',
                         'Alignment failures are unavailable, not perfect preservation; colored geometry is diagnostic.',
                         'No local warping, color correction, source pasting, provider calls or new billing.'],
                'rows':results}
    (args.output/'metrics.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    fields = ['model','profile','track','status','actual_cost_usd','ssim_luminance','mae_rgb_0_255','delta_e_76_mean',
              'pixels_over_8_rgb_fraction','worst_tile_ssim','aligned_ssim','alignment_status','scale','shift_x','shift_y']
    with (args.output/'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in results:
            record = {k:r.get(k) for k in fields[:5]}
            record.update({k:r.get('fixed',{}).get(k) for k in fields[5:10]})
            alignment = r.get('alignment', {})
            record.update(aligned_ssim=(r.get('aligned') or {}).get('ssim_luminance'), alignment_status=alignment.get('status'),
                          scale=alignment.get('scale_relative_to_expected'),
                          shift_x=alignment.get('center_shift_fraction_of_cover',[None,None])[0],
                          shift_y=alignment.get('center_shift_fraction_of_cover',[None,None])[1])
            writer.writerow(record)
    render(results, args.output, metadata)
    print(json.dumps({'completed':sum(r['status']=='measured' for r in results), 'failed':sum(r['status']=='failed' for r in results),
                      'seconds':metadata['seconds'], 'report':str(args.output/'index.html')}), flush=True)
    if any(r['status'] == 'failed' for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
