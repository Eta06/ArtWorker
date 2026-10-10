"""Render a local preference gallery from an existing ledger; no provider calls."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import urllib.parse

ROOT = Path(__file__).resolve().parents[2]
COVERS = {'track3': 'Aklın Hep Bende', 'track2': 'Karambol'}


def render_gallery(rows: list[dict], results: Path, ledger: dict, status: str,
                   covers: dict | None = None) -> None:
    cards, entries = [], []
    covers = covers or COVERS
    stage = ledger.get('stage', 'zeytin')
    results = results.resolve()
    for row in rows:
        image_path = ROOT / row.get('directory', '') / 'raw.png'
        available = row['status'] == 'completed' and image_path.is_file()
        identity = [row['model'], row['profile'], row['track'], row.get('raw_sha256')]
        review_id = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        entry = {k: row.get(k) for k in ('model', 'profile', 'track', 'resolution', 'quality',
                  'raw_dimensions', 'actual_cost_usd', 'raw_sha256')}
        entry.update(id=review_id, available=available, cover=covers[row['track']])
        entries.append(entry)
        image = '<div class="missing">Bu ayarda görsel yok</div>'
        if available:
            url = urllib.parse.quote(os.path.relpath(image_path, results))
            image = f'<a class="art" href="{url}" target="_blank" aria-label="{html.escape(entry["cover"])} tam boyut"><img loading="lazy" src="{url}" alt="{html.escape(entry["cover"])} · {html.escape(row["model"])} · {html.escape(row["profile"])}"></a>'
        cost = row.get('actual_cost_usd')
        info = f'${cost:.7f}' if cost is not None else 'Ücret doğrulanmadı'
        if row['status'] in ('blocked_from_prior_filter', 'pending'):
            info = 'Bu ayarda istek yapılmadı'
        elif row['status'] == 'http_error':
            info += f' · Sağlayıcı HTTP {row.get("http_status", "hatası")}; görsel döndürmedi'
        elif row['status'] == 'completed_output_unavailable':
            info += ' · Sağlayıcı tamamladı; görsel yanıtı alınamadı'
        if row.get('elapsed_seconds') is not None:
            info += f' · {row["elapsed_seconds"]:.1f}s'
        if row.get('raw_dimensions'):
            info += ' · ' + '×'.join(map(str, row['raw_dimensions']))
        if row['reused']:
            info += ' · Önceki sonuç'
        vote = (f'<div class="vote"><button data-vote="liked" aria-pressed="false">♡ Beğendim</button><button data-vote="disliked" aria-pressed="false">× Beğenmedim</button></div><p class="decision">Karar verilmedi</p>'
                if available else '<p class="decision">Görsel olmadığı için seçim yapılamaz</p>')
        cards.append(f'<article data-id="{review_id}" data-model="{html.escape(row["model"])}" data-track="{row["track"]}" data-available="{str(available).lower()}"><div class="card-head"><p class="cover">{html.escape(entry["cover"])}</p><h2>{html.escape(row["model"])}</h2><p class="profile">{html.escape(row["profile"])}</p></div>{image}<div class="card-foot"><p class="measure">{html.escape(info)}</p>{vote}</div></article>')
    originals = []
    for track, cover in covers.items():
        row = next(r for r in rows if r['track'] == track and r.get('directory'))
        url = urllib.parse.quote(os.path.relpath(ROOT / row['directory'] / 'source.png', results))
        originals.append(f'<a href="{url}" target="_blank"><img src="{url}" alt="{html.escape(cover)} orijinal kapak"><span>{html.escape(cover)}<small>Orijinal kare</small></span></a>')
    model_options = ''.join('<option>' + html.escape(m) + '</option>' for m in dict.fromkeys(r['model'] for r in rows))
    template = Path(__file__).with_name('review-gallery.html').read_text()
    # Script data is JSON, never raw model HTML. Escape closing-script sequences.
    data = json.dumps(entries, ensure_ascii=False).replace('<', '\\u003c')
    values = {'TITLE': 'ArtWorker · '+html.escape(stage.capitalize()), 'STATUS': html.escape(status),
              'STAGE_JSON': json.dumps(stage).replace('<', '\\u003c'),
              'TRACKS': ''.join('<option value="'+html.escape(t)+'">'+html.escape(c)+'</option>' for t,c in covers.items()),
        'COST': f'${ledger["new_charge_usd"]:.7f}' if ledger.get('new_charge_usd') is not None else 'Doğrulanmadı', 'MODELS': model_options,
              'CARDS': ''.join(cards), 'ORIGINALS': ''.join(originals), 'ENTRIES': data,
              'FIDELITY': ('<p style="max-width:1600px;margin:0 auto;padding:0 28px 14px"><a href="fidelity/">Ana kapağa sadakat raporu →</a></p>'
                           if (results / 'fidelity/index.html').is_file() else '')}
    for name, value in values.items():
        template = template.replace('{{' + name + '}}', value)
    (results / 'index.html').write_text(template)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    ledger = json.loads(args.ledger.read_text())
    batch = json.loads((args.results / 'batch.json').read_text())
    render_gallery(ledger['rows'], args.results, ledger, batch['status'])
    print('Rendered preference gallery from existing results; no API calls.')


if __name__ == '__main__':
    main()
