"""Compare matched compression arms outside the original album-cover rectangle.

Pixel similarity is descriptive evidence, not a semantic-quality acceptance gate.
Local contact sheets retain copyrighted evaluation media outside normal Git.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    before = json.loads((args.baseline / 'metrics.json').read_text())
    after = json.loads((args.candidate / 'metrics.json').read_text())
    for key in ('track', 'model', 'steps', 'guidance', 'seed', 'prompt', 'vae', 'source_rect', 'width', 'height', 'adapter_scale'):
        if before[key] != after[key]:
            raise ValueError(f'Unmatched recipe: {key}')
    if before['status'] != 'completed' or after['status'] != 'completed':
        raise ValueError('Both arms must have completed generation')
    a = np.asarray(Image.open(args.baseline / 'raw.png').convert('RGB'))
    b = np.asarray(Image.open(args.candidate / 'raw.png').convert('RGB'))
    if a.shape != b.shape:
        raise ValueError('Image shapes differ')
    x0, y0, x1, y1 = before['source_rect']
    mask = np.ones(a.shape[:2], dtype=bool)
    mask[y0:y1, x0:x1] = False
    delta = np.abs(a.astype(np.float32) - b.astype(np.float32))
    args.output.mkdir(parents=True, exist_ok=False)
    result = {'baseline': str(args.baseline), 'candidate': str(args.candidate),
              'track': before['track'], 'seed': before['seed'],
              'raw_pixel_exact': bool(np.array_equal(a, b)),
              'outside_mae_255': float(delta[mask].mean()),
              'outside_rmse_255': float(np.sqrt((delta[mask] ** 2).mean())),
              'outside_changed_pixel_fraction': float(np.any(delta > 0, axis=2)[mask].mean()),
              'outside_max_channel_difference': float(delta[mask].max()),
              'visual_quality_accepted': None,
              'limitation': 'Pixel differences do not measure semantic correctness; inspect both arms.'}
    (args.output / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    sheet = Image.new('RGB', (a.shape[1] * 2, a.shape[0] + 48), '#171717')
    draw = ImageDraw.Draw(sheet)
    candidate_label = f"INT{after['transformer_bits']} cached instruction"
    if after.get('quant_profile', 'uniform') != 'uniform':
        candidate_label = after['quant_profile'] + ' cached instruction'
    for index, (directory, label) in enumerate(((args.baseline, 'INT4 baseline'), (args.candidate, candidate_label))):
        sheet.paste(Image.open(directory / 'composite.png').convert('RGB'), (index * a.shape[1], 48))
        draw.text((index * a.shape[1] + 12, 16), label, fill='white')
    sheet.save(args.output / 'contact-sheet.png')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
