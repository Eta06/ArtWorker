"""Bounded yt-dlp artwork downloads with cross-archive duplicate exclusion.

Only observed Music album-image URLs enter the thumbnail-only download playlist.
Run under run_bounded_model.py. All manifests, cookies and images stay private.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageOps

from collect_covers import ROOT, atomic_json, fingerprint, image_request_url, safe_candidate

cv2.setNumThreads(1)


def vividness(path: Path) -> dict:
    """A transparent color prefilter, not a beauty or training-quality score."""
    with Image.open(path) as image:
        rgb=np.asarray(image.convert('RGB').resize((128,128)),dtype=np.float32)
    red,green,blue=rgb[:,:,0],rgb[:,:,1],rgb[:,:,2]
    rg=np.abs(red-green);yb=np.abs((red+green)/2-blue)
    colorfulness=float(np.hypot(rg.std(),yb.std())+.3*np.hypot(rg.mean(),yb.mean()))
    hsv=cv2.cvtColor(rgb.astype(np.uint8),cv2.COLOR_RGB2HSV)
    saturated=float(((hsv[:,:,1]>=70)&(hsv[:,:,2]>=60)).mean())
    return {'colorfulness':round(colorfulness,3),'saturated_pixel_fraction':round(saturated,4),
            'note':'Color prefilter only; no aesthetic or training approval implied'}


def signature(path: Path) -> dict:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert('RGB')
        pixel, dhash, variation = fingerprint(image)
        rgb = np.asarray(image.resize((64, 64), Image.Resampling.LANCZOS))
        gray = np.asarray(image.convert('L').resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
        dct = cv2.dct(gray)[:8, :8].flatten()[1:]
        bits = sum(int(v > np.median(dct)) << i for i, v in enumerate(dct))
        return dict(pixel=pixel, dhash=int(dhash, 16), phash=bits, rgb=rgb,
                    variation=variation, dimensions=list(image.size))


class CoverIndex:
    def __init__(self):
        self.pixels = {}
        self.rows = []

    def add(self, path: Path, label: str) -> dict:
        sig = signature(path)
        self.pixels[sig['pixel']] = label
        self.rows.append((label, sig))
        return sig

    def match(self, sig: dict) -> tuple[str | None, str | None]:
        exact = self.pixels.get(sig['pixel'])
        if exact:
            return exact, 'normalized_pixels'
        for label, prior in self.rows:
            if (sig['phash'] ^ prior['phash']).bit_count() > 8:
                continue
            if (sig['dhash'] ^ prior['dhash']).bit_count() > 8:
                continue
            difference = np.abs(sig['rgb'].astype(np.float32)-prior['rgb'].astype(np.float32))
            if float(difference.mean()) <= 3 and float(np.quantile(difference, .95)) <= 12:
                return label, 'perceptual_and_rgb'
        return None, None


def baseline_files(manifests: list[Path]) -> tuple[dict, list[tuple[Path, str]]]:
    metadata = {'albums': set(), 'assets': set(), 'files': set()}
    files = []
    for path in manifests:
        root = path.resolve().parent
        data = json.loads(path.read_text())
        for row in data['covers']:
            image = (root/row['file']).resolve()
            if not image.is_relative_to(root) or not image.exists():
                raise ValueError('Missing or unsafe baseline image')
            if hashlib.sha256(image.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('Baseline integrity mismatch')
            metadata['albums'].add(row['album_id'])
            metadata['assets'].add(row['asset_identity'])
            if image not in metadata['files']:
                metadata['files'].add(image)
                files.append((image, f"{data['stage']}:{row['id']}"))
    # Original references only; never index teacher output/heatmaps as covers.
    extras = list((ROOT/'Player/Resources').glob('*.jpg'))
    extras += list((ROOT/'experiments/openrouter_outpaint/results').rglob('source.png'))
    extras += list((ROOT/'experiments/openrouter_outpaint/results').rglob('original.png'))
    music = Path.home()/'test-player'
    if music.exists():
        extras += [p for p in music.rglob('*') if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'}]
    for path in extras:
        image = path.resolve()
        if image not in metadata['files']:
            with Image.open(image) as check:
                if check.width != check.height:
                    continue
            metadata['files'].add(image)
            files.append((image, f'original:{len(files)}'))
    return metadata, files


def run(args):
    output = args.output.resolve()
    if not output.is_relative_to(ROOT/'experiments/cover_collection/results'):
        raise ValueError('Use ignored cover results directory')
    output.mkdir(parents=True, exist_ok=True)
    (output/'covers').mkdir(exist_ok=True)
    staging = output/'staging'
    staging.mkdir(exist_ok=True)
    manifest_path = output/'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else dict(
        schema_version=2, stage=args.stage, target=args.target, covers=[], attempts=[],
        source=args.source_description,
        rights_status='Local reference candidates only; artwork is not published',
        audio_downloads=0, new_generation_cost_usd=0)
    if manifest['stage'] != args.stage:
        raise ValueError('Stage mismatch')
    manifest.update(status='running', target=args.target)
    atomic_json(manifest_path, manifest)
    index = CoverIndex()
    known, files = baseline_files(args.baseline_manifest)
    for path, label in files:
        index.add(path, label)
    for row in manifest['covers']:
        path = output/row['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Resume integrity mismatch')
        index.add(path, row['id'])
        known['albums'].add(row['album_id'])
        known['assets'].add(row['asset_identity'])
    manifest['baseline_indexed_files'] = len(files)
    artist_counts=Counter(r.get('catalog_artist') or str(r.get('artists',[{}])[0]) for r in manifest['covers'])
    manifest['selection_policy']={'minimum_colorfulness':args.min_colorfulness,
        'minimum_saturated_pixel_fraction':args.min_saturated_fraction,'maximum_per_artist':args.max_per_artist,
        'thumbnail_workers':args.workers,
        'note':'Color and diversity prefilters, not human aesthetic acceptance'}
    attempts = {(r['album_id'], r['asset_identity']) for r in manifest['attempts']
                if r['status'] not in {'duplicate_pending_album', 'duplicate_pending_asset'}
                and (r['status'] != 'error' or not args.retry_errors)}
    candidates = json.loads(args.candidates.read_text())
    pending, local_assets, local_albums = [], set(), set()
    for row in candidates:
        try:
            album = safe_candidate(row)
            url = image_request_url(row.get('observed_image_url', ''), 1024)
        except (ValueError, KeyError):
            continue
        asset = row['observed_image_url'].split('=', 1)[0]
        if (album, asset) in attempts:
            continue
        record = dict(album_id=album, asset_identity=asset)
        if album in known['albums']:
            record['status'] = 'duplicate_album'
        elif asset in known['assets']:
            record['status'] = 'duplicate_asset'
        elif album in local_albums:
            record['status'] = 'duplicate_pending_album'
        elif asset in local_assets:
            record['status'] = 'duplicate_pending_asset'
        else:
            local_assets.add(asset)
            local_albums.add(album)
            pending.append((row, record, url))
            continue
        manifest['attempts'].append(record)
        attempts.add((album, asset))
    # Batches are bounded to avoid downloading an entire backlog after target.
    start = 0
    while start < len(pending) and len(manifest['covers']) < args.target:
        batch=[];capacity=min(50,args.target-len(manifest['covers']))
        while start<len(pending) and len(batch)<capacity:
            item=pending[start];start+=1
            row,record,url=item
            artist=row.get('catalog_artist') or str(row.get('artists',[{}])[0])
            if args.max_per_artist and artist_counts[artist]>=args.max_per_artist:
                manifest['attempts'].append({**record,'status':'artist_quota','artist':artist})
            else:batch.append(item)
        if not batch:break
        entries = []
        for row, record, url in batch:
            entries.append(dict(id=record['album_id'], title=row.get('album') or record['album_id'],
                url=url, ext='jpg', webpage_url=row.get('song_url') or row['album_url'],
                extractor='artworker:observed-music-cover', extractor_key='ObservedMusicCover',
                thumbnails=[dict(url=url, id='music-cover', ext='jpg')]))
        info = output/'thumbnail-batch.json'
        # load-info-json's clean-infojson removes a playlist's nested entries.
        # A top-level list of individual info dictionaries survives sanitizing.
        atomic_json(info, entries)
        command = [args.yt_dlp, '--ignore-config', '--load-info-json', str(info),
            '--skip-download', '--write-thumbnail', '--no-progress', '--ignore-errors',
            '--socket-timeout', '20', '--retries', '1', '--max-filesize', '8M',
            '-o', str(staging/'%(id)s.%(ext)s')]
        if args.chrome_cookies:
            command += ['--cookies-from-browser', 'chrome:Default']
        if args.workers>1:
            command=[sys.executable,str(Path(__file__).with_name('download_thumbnails.py')),
                '--info',str(info),'--output',str(staging),'--workers',str(args.workers)]
        with (output/'yt-dlp.log').open('a') as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=900)
        for row, record, url in batch:
            path = staging/(record['album_id']+'.jpg')
            try:
                if not path.exists():
                    raise ValueError(f'yt-dlp image unavailable (exit {completed.returncode})')
                if path.stat().st_size > 8*1024**2:
                    raise ValueError('Image exceeded 8 MiB')
                sig = signature(path)
                width, height = sig['dimensions']
                if width != height or not 512 <= width <= 4096 or sig['variation'] < 2:
                    raise ValueError('Expected detailed square cover >=512 pixels')
                artist=row.get('catalog_artist') or str(row.get('artists',[{}])[0])
                colors=vividness(path)
                if (args.max_per_artist and artist_counts[artist]>=args.max_per_artist):
                    record.update(status='artist_quota',artist=artist)
                    path.unlink()
                    manifest['attempts'].append(record);atomic_json(manifest_path,manifest)
                    continue
                if (colors['colorfulness']<args.min_colorfulness or
                        colors['saturated_pixel_fraction']<args.min_saturated_fraction):
                    record.update(status='color_prefilter',color_metrics=colors)
                    path.unlink()
                    manifest['attempts'].append(record);atomic_json(manifest_path,manifest)
                    continue
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                record['sha256'] = digest
                duplicate, method = index.match(sig)
                if duplicate:
                    record.update(status='duplicate_pixels', duplicate_of=duplicate, match_method=method)
                    path.unlink()  # Newly downloaded excluded copy only; archive is untouched.
                else:
                    cover_id = f"cover{len(manifest['covers'])+1:04d}"
                    relative = f'covers/{cover_id}.jpg'
                    size = path.stat().st_size
                    path.replace(output/relative)
                    index.add(output/relative, cover_id)
                    cover = dict(row, **record, id=cover_id, file=relative, dimensions=sig['dimensions'],
                        bytes=size, normalized_rgb_sha256=sig['pixel'], dhash=f"{sig['dhash']:016x}",
                        color_metrics=colors,request_image_url=url, native_source_resolution='Unknown; CDN response dimensions only')
                    manifest['covers'].append(cover)
                    artist_counts[artist]+=1
                    record.update(status='downloaded', cover_id=cover_id)
            except Exception as error:
                record.update(status='error', error=f'{type(error).__name__}: {error}')
            manifest['attempts'].append(record)
            atomic_json(manifest_path, manifest)
        print(json.dumps({'downloaded':len(manifest['covers']), 'attempts':len(manifest['attempts'])}), flush=True)
    manifest.update(target=args.target, downloaded_count=len(manifest['covers']),
        status='target_reached' if len(manifest['covers']) >= args.target else 'candidates_exhausted',
        status_counts=dict(Counter(r['status'] for r in manifest['attempts'])))
    atomic_json(manifest_path, manifest)
    print(json.dumps({k:manifest[k] for k in ('status','downloaded_count','status_counts','baseline_indexed_files')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--baseline-manifest', action='append', required=True, type=Path)
    parser.add_argument('--target', type=int, default=1000)
    parser.add_argument('--stage', required=True)
    parser.add_argument('--yt-dlp', default='yt-dlp')
    parser.add_argument('--chrome-cookies', action='store_true')
    parser.add_argument('--retry-errors', action='store_true', help='Explicit retry of failed inbound image downloads')
    parser.add_argument('--source-description',default='Authorized Chrome Music DOM; yt-dlp thumbnail-only download')
    parser.add_argument('--min-colorfulness',type=float,default=0)
    parser.add_argument('--min-saturated-fraction',type=float,default=0)
    parser.add_argument('--max-per-artist',type=int,default=0)
    parser.add_argument('--workers',type=int,default=1,help='1: existing CLI; 2..4: yt-dlp Python thumbnail-only workers')
    args = parser.parse_args()
    if not 1 <= args.target <= 1000:
        parser.error('Target must be in 1..1000')
    if args.min_colorfulness<0 or not 0<=args.min_saturated_fraction<=1 or args.max_per_artist<0:
        parser.error('Color thresholds and artist limit must be nonnegative, saturation <=1')
    if not 1<=args.workers<=4 or (args.workers>1 and args.chrome_cookies):
        parser.error('Use 1..4 workers; authenticated CLI downloads require one worker')
    run(args)
