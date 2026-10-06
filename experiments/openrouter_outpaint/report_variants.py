"""Reconcile per-generation billing and build a gallery of unmodified outputs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from decimal import Decimal
import json
from pathlib import Path
import urllib.parse

from PIL import Image, ImageDraw, ImageFont
from benchmark import ROOT, key_usage, request_json, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--key-file', type=Path, required=True)
    parser.add_argument('--skip-billing', action='store_true', help='Live preview: use response costs without generation lookups')
    args = parser.parse_args()
    state = json.loads((args.results / 'batch.json').read_text())
    key = args.key_file.read_text().strip()
    rows = []
    for cell in state['cells']:
        row = {k: cell.get(k) for k in ('model', 'profile', 'track', 'resolution', 'quality', 'status')}
        directory = Path(cell['directory']) if cell.get('directory') else None
        if directory and not directory.is_absolute():
            directory = ROOT / directory
        receipt_path = directory / 'receipt.json' if directory else None
        row['reused'] = cell['status'] == 'reused'
        if receipt_path and receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            row.update(status=receipt['status'], elapsed_seconds=receipt.get('elapsed_seconds'),
                       actual_cost_usd=receipt.get('reported_cost_usd'),
                       cost_source=receipt.get('cost_source'), http_status=receipt.get('http_status'),
                       error=receipt.get('error'), directory=str(directory.relative_to(ROOT)),
                       reference_count=len(receipt['request']['input_references']),
                       source_sha256=receipt['geometry']['original_sha256'],
                       provider=receipt['request']['provider'], created_utc=receipt.get('created_utc'))
            headers = {k.lower(): v for k, v in receipt.get('response_headers', {}).items()}
            generation_id = headers.get('x-generation-id')
            if generation_id and not args.skip_billing:
                row['generation_id'] = generation_id
                try:
                    billing, _ = request_json('generation?id=' + urllib.parse.quote(generation_id), key)
                    row['generation_billing'] = {k: billing['data'].get(k) for k in (
                        'id', 'model', 'provider_name', 'total_cost', 'created_at', 'generation_time')}
                    row['actual_cost_usd'] = billing['data']['total_cost']
                    row['cost_source'] = 'generation.total_cost'
                    row['billing_matches_response'] = (Decimal(str(row['actual_cost_usd'])) ==
                        Decimal(str(receipt.get('reported_cost_usd')))) if receipt.get('reported_cost_usd') is not None else None
                except Exception as exc:
                    row['billing_lookup_error'] = type(exc).__name__
            if receipt.get('images'):
                row.update(raw_dimensions=receipt['images'][0]['dimensions'], raw_sha256=receipt['images'][0]['sha256'])
        rows.append(row)
    account = key_usage(key)
    new_rows = [r for r in rows if not r['reused'] and r.get('directory')]
    known = sum((Decimal(str(r['actual_cost_usd'])) for r in new_rows
                 if r.get('actual_cost_usd') is not None), Decimal(0))
    delta = Decimal(str(account['usage'])) - Decimal(str(state['initial_key_usage']['usage']))
    reconciled = known == delta
    if reconciled:
        for row in new_rows:
            if row['status'] == 'http_error' and row.get('actual_cost_usd') is None:
                row.update(actual_cost_usd=0, cost_source='definitive_http_error_and_dedicated_key_reconciliation')
    totals, new_totals = defaultdict(Decimal), defaultdict(Decimal)
    for row in rows:
        if row.get('actual_cost_usd') is not None:
            cost = Decimal(str(row['actual_cost_usd']))
            totals[row['model']] += cost
            if not row['reused']:
                new_totals[row['model']] += cost
    ledger = {'stage': 'zeytin', 'currency': 'USD', 'rows': rows,
              'status_counts': dict(Counter(r['status'] for r in rows)),
              'new_charge_usd': float(sum(new_totals.values(), Decimal(0))),
              'comparison_charge_including_reused_usd': float(sum(totals.values(), Decimal(0))),
              'per_model_new_usd': {m: float(c) for m, c in new_totals.items()},
              'per_model_including_reused_usd': {m: float(c) for m, c in totals.items()},
              'key_usage_usd': account['usage'], 'new_account_delta_usd': float(delta),
              'new_costs_reconcile': reconciled,
              'unknown_cost_attempts': [r['directory'] for r in new_rows if r.get('actual_cost_usd') is None],
              'quality_selection_owner': 'user',
              'notes': ['Raw output images, no source paste or color finishing.',
                        'Single sample per cover and setting; not a general quality benchmark.',
                        'Auto is provider-selected quality. Native resolution tiers can have different actual dimensions.',
                        'Reused outputs incur no additional charge in this stage. Prior filtered cells are not new attempts.',
                        'Credit purchase fees, tax and training costs excluded.',
                        'Training and redistribution eligibility not established.']}
    save_json(args.ledger, ledger)
    columns = ['model','profile','track','resolution','quality','status','reused','reference_count',
               'raw_dimensions','elapsed_seconds','actual_cost_usd','cost_source','http_status','directory']
    with args.ledger.with_suffix('.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    summary = ['# Zeytin — per-setting results', '',
               'Raw results only. The user selects visual quality. Charges marked reused were paid in Kehribar and incur no new charge here.', '',
               '| Model / native profile | Aklın: pixels / seconds / USD | Karambol: pixels / seconds / USD |',
               '|---|---|---|']
    def describe(row):
        if row['status'] != 'completed':
            text = row['status'] + (f' HTTP {row["http_status"]}' if row.get('http_status') else '')
            if row['status'] == 'blocked_from_prior_filter':
                return text + ' / not requested'
            if row.get('actual_cost_usd') is not None:
                text += f' / ${row["actual_cost_usd"]:.7f}'
            return text
        dimensions = '×'.join(map(str, row['raw_dimensions']))
        cost = row.get('actual_cost_usd')
        price = f'${cost:.7f}' if cost is not None else 'unknown charge'
        return f'{dimensions} / {row["elapsed_seconds"]:.3f}s / {price}' + (' (reused)' if row['reused'] else '')
    pairs = defaultdict(dict)
    for row in rows:
        pairs[(row['model'],row['profile'])][row['track']] = row
    for (model, profile), pair in pairs.items():
        summary.append(f'| {model} / {profile} | {describe(pair["track3"])} | {describe(pair["track2"])} |')
    summary += ['', f'New reported charges: ${ledger["new_charge_usd"]:.7f}. Dedicated-key reconciliation: {reconciled}.',
                '', 'Per-generation IDs, resolved model versions, provider routing, image hashes and billing sources are in the JSON ledger. Native tiers are not equivalent pixel counts across providers. HTTP execution success is not source-preserving outpaint acceptance.']
    args.ledger.with_suffix('.md').write_text('\n'.join(summary)+'\n')
    from review_gallery import render_gallery
    render_gallery(rows, args.results, ledger, state['status'])
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 15)
    for track in ('track3', 'track2'):
        # One default/first profile per model in the overview; every variant remains in the gallery.
        overview = list({r['model']: next(v for v in rows if v['model']==r['model'] and v['track']==track)
                         for r in rows}.values())
        sheet = Image.new('RGB', (5*244, 4*484), '#14171c'); draw=ImageDraw.Draw(sheet)
        for i, row in enumerate(overview):
            x,y=(i%5)*244+6,(i//5)*484+6
            draw.text((x,y), row['model'].split('/')[-1], font=font, fill='white')
            draw.text((x,y+22), row['profile'], font=font, fill='#b5c0d0')
            image_path = ROOT / row.get('directory','') / 'raw.png'
            if image_path.is_file():
                with Image.open(image_path) as im:
                    sheet.paste(im.convert('RGB').resize((230,409),Image.Resampling.LANCZOS),(x,y+46))
            else:
                draw.text((x,y+160),row['status'],font=font,fill='white')
            cost=row.get('actual_cost_usd')
            draw.text((x,y+459),f'${cost:.6f}' if cost is not None else 'Cost unconfirmed',font=font,fill='#b5c0d0')
        sheet.save(args.results / (track+'-overview.png'))
    print(json.dumps({k: v for k,v in ledger.items() if k not in ('rows','notes')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
