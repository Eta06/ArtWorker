"""Re-measure existing active teacher outputs, allowing free source placement.

Offline only. Originals are verified and preserved; diagnostic crops and orange
outlines are written into a fresh directory. No inference or provider client.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from prompt_archive import ROOT, collect, prompt_groups
from source_fidelity import audit, sha

MODELS = ('openai/gpt-image-2.5-sunburst', 'openai/gpt-image-2.5-flare',
          'google/gemini-nano-banana-2.1')


def eligible(row: dict) -> bool:
    return bool(row.get('aligned') and row['alignment']['status'] == 'measured'
                and row['alignment'].get('visible_source_fraction', 0) >= .995)


def matched_table(rows: list[dict], conditions: list[tuple[str, str]],
                  prompts: list[str]) -> list[dict]:
    """Compare identical model/cover conditions; missing measurements stay missing.

    OpenAI high settings only; repeats in a condition have equal median weight.
    Never substitute a fixed-location score when source detection fails.
    """
    groups = defaultdict(list)
    for row in rows:
        if eligible(row) and row['settings'].get('quality') == 'high':
            groups[(row['prompt_id'], row['model'], row['track'])].append(row)
    table = []
    for prompt in prompts:
        cells = []
        for model, track in conditions:
            found = groups[(prompt, model, track)]
            cells.append({'model': model, 'track': track, 'n': len(found),
                          **{metric: statistics.median(r['aligned'][metric] for r in found)
                             if found else None for metric in
                             ('ssim_luminance', 'pixels_within_8_rgb_fraction',
                              'mae_rgb_0_255', 'worst_tile_ssim')}})
        complete = all(c['n'] for c in cells)
        table.append({'prompt_id': prompt, 'complete': complete, 'cells': cells,
                      'mean_ssim': statistics.mean(c['ssim_luminance'] for c in cells)
                      if complete else None,
                      'mean_within_8': statistics.mean(c['pixels_within_8_rgb_fraction']
                                                      for c in cells) if complete else None})
    return sorted(table, key=lambda r: (r['mean_ssim'] is not None,
                                        r['mean_ssim'] or -1), reverse=True)


def comparison_image(rows: list[dict], output: Path) -> Path:
    """Static diagnostic sheet for chat, not a generated or repaired artwork."""
    prompts = ['old-canvas', 'anchored', 'edge-continuity', 'copy-square',
               'canvas-only', 'verbatim', 'short', 'free-layout']
    sheet = Image.new('RGB', (1220, 300+len(prompts)*210), '#15191c')
    draw = ImageDraw.Draw(sheet)
    font_path = Path('/System/Library/Fonts/Supplemental/Arial.ttf')
    def font(size):
        return ImageFont.truetype(str(font_path), size) if font_path.exists() else ImageFont.load_default()
    draw.text((24, 18), 'Kaynak koruma — Sunburst high', fill='white', font=font(30))
    draw.text((24, 65), 'Konum puanlanmadı. Turuncu: bulunan kare · Hizalanmış kırpım · Piksel farkı',
              fill='#b9c3cb', font=font(20))
    for column, (track, title) in enumerate([('track2', 'Karambol'), ('cover02', 'Gripin')]):
        x = 24+column*602
        reference = next(r for r in rows if r['track'] == track and r['model'] == MODELS[0])
        draw.text((x, 108), title+' — orijinal kare', fill='white', font=font(22))
        with Image.open(output/reference['id']/'source.png') as original:
            sheet.paste(original.resize((138, 138)), (x, 146))
        for index, prompt in enumerate(prompts):
            y = 300+index*210
            candidates = [r for r in rows if r['prompt_id'] == prompt and r['track'] == track
                          and r['model'] == MODELS[0] and r['settings'].get('quality') == 'high']
            if not candidates:
                draw.text((x, y), prompt+' — denenmedi', fill='white', font=font(20))
                continue
            row = candidates[0]
            score = row['aligned']
            if score is None:
                draw.text((x, y), prompt+' — güvenilir eşleşme yok', fill='white', font=font(20))
                continue
            suffix = '' if row['ranking_eligible'] else ' · kısmi görünürlük'
            draw.text((x, y), f"{prompt} · SSIM {100*score['ssim_luminance']:.1f}%"+suffix,
                      fill='#b9efc3' if prompt == 'old-canvas' else 'white', font=font(18))
            raw = (output/row['raw_url']).resolve()
            with Image.open(raw) as im:
                working = im.convert('RGB').resize((512, round(im.height*512/im.width)),
                                                    Image.Resampling.LANCZOS)
            transform = np.asarray(row['alignment']['matrix_source_to_output'])
            corners = np.float64([[0, 0], [511, 0], [511, 511], [0, 511]])
            polygon = corners@transform[:, :2].T+transform[:, 2]
            ImageDraw.Draw(working).line([tuple(v) for v in polygon]+[tuple(polygon[0])],
                                        fill=(255, 180, 80), width=5)
            working.thumbnail((105, 175))
            sheet.paste(working, (x, y+29))
            for dx, file in [(125, 'aligned.png'), (302, 'aligned-difference.png')]:
                with Image.open(output/row['id']/file) as im:
                    sheet.paste(im.resize((170, 170)), (x+dx, y+29))
    target = output/'source-preservation-comparison.png'
    sheet.save(target)
    return target


def export_csv(rows: list[dict], output: Path) -> Path:
    fields = ['prompt', 'model', 'stage', 'cover', 'settings', 'alignment_status',
              'whole_source_ranking_eligible', 'visible_source_percent',
              'aligned_ssim_percent', 'within_8_rgb_percent', 'within_16_rgb_percent',
              'exact_pixels_on_analysis_grid_percent', 'mean_absolute_rgb_error',
              'worst_tile_ssim_percent']
    path = output/'aligned-scores.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            score = row['aligned'] or {}
            def percent(key):
                return round(100*score[key], 4) if score.get(key) is not None else None
            visible = row['alignment'].get('visible_source_fraction')
            writer.writerow(dict(prompt=row['prompt_id'], model=row['model'], stage=row['stage'],
                cover=row['track'], settings=json.dumps(row['settings'], sort_keys=True),
                alignment_status=row['alignment']['status'],
                whole_source_ranking_eligible=row['ranking_eligible'],
                visible_source_percent=100*visible if visible is not None else None,
                aligned_ssim_percent=percent('ssim_luminance'),
                within_8_rgb_percent=percent('pixels_within_8_rgb_fraction'),
                within_16_rgb_percent=percent('pixels_within_16_rgb_fraction'),
                exact_pixels_on_analysis_grid_percent=percent('pixel_exact_on_analysis_grid_fraction'),
                mean_absolute_rgb_error=score.get('mae_rgb_0_255'),
                worst_tile_ssim_percent=percent('worst_tile_ssim')))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        parser.error('Fresh directory inside repository required')
    originals = [r for r in collect(ROOT/'experiments/openrouter_outpaint/results')
                 if r['model'] in MODELS and r['images']]
    groups = prompt_groups(originals)
    output.mkdir(parents=True)
    rows = []
    for group in groups:
        for original in group['rows']:
            receipt = json.loads((ROOT/original['receipt_path']).read_text())
            for image in original['images']:
                if image['file'] != 'raw.png':
                    raise ValueError('Multiple-output receipt needs explicit audit support')
                row = {'model': original['model'], 'track': original['track'],
                       'profile': group['id']+'-'+original['receipt_sha256'][:16],
                       'directory': original['directory'], 'raw_sha256': image['sha256'],
                       'raw_dimensions': image['dimensions'],
                       'source_sha256': receipt['geometry']['original_sha256']}
                score = audit(row, output, 'free')
                score.update(prompt_id=group['id'], stage=original['stage'],
                             settings=original['settings'], route=original['route'],
                             receipt_sha256=original['receipt_sha256'])
                score['ranking_eligible'] = eligible(score)
                rows.append(score)
                print(json.dumps({'done': len(rows), 'prompt': group['id'],
                                  'model': row['model'], 'track': row['track'],
                                  'aligned': score['alignment']['status'],
                                  'ssim': (score['aligned'] or {}).get('ssim_luminance')}),
                      flush=True)
    for original in originals:
        if sha(ROOT/original['receipt_path']) != original['receipt_sha256']:
            raise ValueError('Original receipt changed')
        for image in original['images']:
            if sha(ROOT/original['directory']/image['file']) != image['sha256']:
                raise ValueError('Original output changed')
    prompts = [g['id'] for g in groups]
    summary = {'stage': 'sahil', 'new_provider_requests': 0,
               'new_generation_cost_usd': 0, 'placement_mode': 'free',
               'analysis_side': 512, 'output_count': len(rows),
               'active_prompt_count': len(groups),
               'reliably_aligned_count': sum(r['alignment']['status'] == 'measured' for r in rows),
               'full_source_ranking_count': sum(eligible(r) for r in rows),
               'metric_note': 'SSIM is structural similarity, not unchanged pixel percentage. '
                              'Within-8 means every RGB channel differs by at most 8/255 '
                              'after one global alignment on a 512px analysis grid. '
                              'No local warping or color correction. Position is not penalized.',
               'sunburst_high_two_cover_comparison': matched_table(rows,
                   [(MODELS[0], 'track2'), (MODELS[0], 'cover02')], prompts),
               'openai_high_karambol_comparison': matched_table(rows,
                   [(MODELS[0], 'track2'), (MODELS[1], 'track2')], prompts)}
    (output/'metrics.json').write_text(json.dumps({'summary': summary, 'rows': rows},
                                                 ensure_ascii=False, indent=2)+'\n')
    (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    comparison_image(rows, output)
    export_csv(rows, output)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
