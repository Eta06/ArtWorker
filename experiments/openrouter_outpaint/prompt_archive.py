"""Offline archive of exact teacher prompts and every existing output.

No provider client, credentials, network or generation calls. Original receipts
and images are read-only; small previews and local media aliases are new files.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
import os
from pathlib import Path
import re

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
NAMES = {
    '736716dd0146': ('old-canvas', 'Eski kaliteli referans · canvas + kare'),
    '0d0954887ac4': ('old-canvas-single', 'Eski canvas · tek referans sürümü'),
    '57e23e2c6346': ('short', 'Kısa prompt · yalnız kare'),
    '0ea1042aa4d3': ('anchored', 'Anchored · ortada ve tam genişlikte'),
    'fcc477d3cb82': ('canvas-only', 'Canvas-only · kısa genişletme talimatı'),
    'bfbaac7acc25': ('free-layout', 'Free-layout · serbest konum ve ölçek'),
    'b77397721455': ('edge-continuity', 'Edge-continuity · kenar devamlılığı'),
    '72e541115182': ('verbatim', 'Verbatim · kaynak içeriğini aynen tut'),
    'c3ec80dee1f2': ('copy-square', 'Copy-square · kareyi kopyala'),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(results: Path) -> list[dict]:
    rows = []
    for base, dirs, files in os.walk(results, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not Path(base, d).is_symlink())
        if 'receipt.json' not in files:
            continue
        path = Path(base) / 'receipt.json'
        data = json.loads(path.read_text())
        request = data.get('request') or {}
        prompt = request.get('prompt') or data.get('prompt')
        if not isinstance(prompt, str) or not data.get('model'):
            raise ValueError(f'Unknown receipt schema: {path.relative_to(ROOT)}')
        model = data['model']
        direct = data.get('route') == 'direct_google_gemini_api'
        if direct and '/' not in model:
            model = 'google/' + model
        response_format = request.get('response_format') or {}
        settings = {k: request[k] for k in ('quality', 'resolution', 'aspect_ratio') if k in request}
        settings.update({k: response_format[k] for k in ('image_size', 'mime_type', 'aspect_ratio') if k in response_format})
        rows.append({'receipt_path': str(path.relative_to(ROOT)), 'receipt_sha256': sha(path),
                     'directory': str(path.parent.relative_to(ROOT)),
                     'prompt': prompt, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                     'model': model, 'route': 'Doğrudan Google' if direct else 'OpenRouter',
                     'track': data.get('track'), 'stage': path.relative_to(results).parts[1],
                     'status': data.get('status'), 'http_status': data.get('http_status'),
                     'settings': settings, 'references': request.get('input_references', []),
                     'images': data.get('images', []), 'actual_cost_usd': data.get('reported_cost_usd'),
                     'estimated_cost_usd': data.get('estimated_cost_usd')})
    return rows


def prompt_groups(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row['prompt_sha256']].append(row)
    ordered = sorted(groups, key=lambda h: list(NAMES).index(h[:12]) if h[:12] in NAMES else 99)
    return [{'sha256': h, 'id': NAMES.get(h[:12], (h[:12], 'Diğer prompt'))[0],
             'title': NAMES.get(h[:12], (h[:12], 'Diğer prompt'))[1],
             'prompt': groups[h][0]['prompt'], 'rows': groups[h]} for h in ordered]


def preview(source: Path, target: Path) -> None:
    if target.exists():
        return
    with Image.open(source) as im:
        im.thumbnail((320, 480), Image.Resampling.LANCZOS)
        im.convert('RGB').save(target, quality=88)


def build(results: Path, output: Path, selections: Path | None) -> dict:
    if not results.resolve().is_relative_to(ROOT) or not output.resolve().is_relative_to(ROOT):
        raise ValueError('Results and new archive must be inside repository')
    if output.exists():
        raise ValueError('Fresh output directory required; existing reports are preserved')
    rows = collect(results)
    groups = prompt_groups(rows)
    votes = {}
    if selections:
        data = json.loads(selections.read_text())
        if data.get('schema_version') != 1:
            raise ValueError('Unknown preference export schema')
        votes = {r['raw_sha256']: r['decision'] for r in data['selections'] if r.get('raw_sha256')}
    output.mkdir(parents=True)
    (output / 'media').mkdir()
    (output / 'previews').mkdir()
    cards = []
    preferred = None
    image_count = 0
    for group in groups:
        for row in group['rows']:
            directory = (ROOT / row['directory']).resolve()
            alias = hashlib.sha256(row['directory'].encode()).hexdigest()[:20]
            (output / 'media' / alias).symlink_to(directory, target_is_directory=True)
            refs = []
            for ref in row['references']:
                name = ref['local_file']
                if name not in {'source.png', 'input.png'} or sha(directory / name) != ref['sha256']:
                    raise ValueError('Source reference integrity failure')
                thumb = 'previews/' + ref['sha256'] + '.jpg'
                preview(directory / name, output / thumb)
                refs.append({'url': f'media/{alias}/{name}', 'thumb': thumb,
                             'label': 'Orijinal kare' if name == 'source.png' else 'Gönderilen canvas'})
            outputs = []
            for im in row['images']:
                name = im['file']
                if not re.fullmatch(r'raw(?:-\d+)?\.png', name) or sha(directory / name) != im['sha256']:
                    raise ValueError('Output checksum failure')
                thumb = 'previews/' + im['sha256'] + '.jpg'
                preview(directory / name, output / thumb)
                entry = {'url': f'media/{alias}/{name}', 'thumb': thumb, 'sha256': im['sha256'],
                         'dimensions': im['dimensions'], 'old_vote': votes.get(im['sha256']) if row['stage'] == 'zeytin' else None}
                outputs.append(entry)
                image_count += 1
                if (group['id'] == 'old-canvas' and row['model'] == 'openai/gpt-image-2.5-sunburst'
                        and row['stage'] == 'zeytin' and row['track'] == 'track2'
                        and row['settings'].get('quality') == 'high' and entry['old_vote'] == 'liked'):
                    preferred = dict(entry, prompt_id=group['id'], prompt=group['prompt'])
            card = {k: row[k] for k in ('model', 'route', 'track', 'stage', 'status', 'http_status', 'settings',
                                       'actual_cost_usd', 'estimated_cost_usd', 'receipt_sha256')}
            card.update(prompt_id=group['id'], refs=refs, outputs=outputs)
            cards.append(card)
    # Re-read receipt fingerprints after rendering: archiving must not change originals.
    for row in rows:
        if sha(ROOT / row['receipt_path']) != row['receipt_sha256']:
            raise ValueError('A source receipt changed during archive build')
    models = sorted(set(r['model'] for r in rows))
    summary = {'stage': 'mersin', 'scope': 'Local OpenRouter and direct Google teacher receipts',
               'new_provider_requests': 0, 'new_generation_cost_usd': 0,
               'receipt_count': len(rows), 'exact_prompt_count': len(groups), 'model_count': len(models),
               'existing_image_count': image_count, 'status_counts': dict(Counter(r['status'] for r in rows)),
               'prompts': [{'id': g['id'], 'title': g['title'], 'prompt': g['prompt'], 'sha256': g['sha256'],
                            'request_count': len(g['rows']), 'models': sorted(set(r['model'] for r in g['rows']))} for g in groups]}
    archive = {'summary': summary, 'models': models, 'cards': cards, 'preferred': preferred}
    (output / 'archive.json').write_text(json.dumps(archive, ensure_ascii=False, indent=2) + '\n')
    template = Path(__file__).with_name('prompt-archive.html').read_text()
    serialized = json.dumps(archive, ensure_ascii=False).replace('<', '\\u003c')
    (output / 'index.html').write_text(template.replace('{{ARCHIVE}}', serialized))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, default=ROOT / 'experiments/openrouter_outpaint/results')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--selections', type=Path)
    args = p.parse_args()
    summary = build(args.results.resolve(), args.output.resolve(), args.selections)
    print(json.dumps({k: v for k, v in summary.items() if k != 'prompts'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
