"""Single direct Gemini outpaint, with private credentials and no POST retries.

Uses Google's documented Interactions API. Actual billing is never inferred from
token usage; receipts retain usage and label published-rate estimates separately.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

from PIL import Image
from benchmark import ANCHORED_SQUARE_PROMPT, prepare, save_json

API = 'https://generativelanguage.googleapis.com/v1beta/interactions'
MODEL = 'gemini-nano-banana-2.1'
PRICE_SOURCE = 'https://ai.google.dev/gemini-api/docs/pricing'


def payload_for(source: bytes) -> dict:
    return {'model': MODEL, 'store': False,
            'input': [{'type': 'text', 'text': ANCHORED_SQUARE_PROMPT},
                      {'type': 'image', 'mime_type': 'image/png',
                       'data': base64.b64encode(source).decode()}],
            'response_format': {'type': 'image', 'mime_type': 'image/jpeg',
                                'aspect_ratio': '9:16', 'image_size': '1K'}}


def output_images(response: dict) -> list[dict]:
    return [part for step in response.get('steps', [])
            if step.get('type') == 'model_output'
            for part in step.get('content', [])
            if part.get('type') == 'image' and part.get('data')]


def cost_estimate(usage: dict) -> dict:
    """Only estimate when modality-resolved output and input counts are present."""
    details = usage.get('output_tokens_by_modality')
    inputs = usage.get('total_input_tokens')
    estimate = None
    if isinstance(details, list) and isinstance(inputs, int):
        images = sum(d.get('tokens', 0) for d in details if d.get('modality') == 'image')
        total_output = usage.get('total_output_tokens')
        thoughts = usage.get('total_thought_tokens', 0)
        supported = all(d.get('modality') in {'image', 'text'} for d in details)
        if (images > 0 and isinstance(total_output, int) and total_output >= images
                and isinstance(thoughts, int) and thoughts >= 0 and inputs >= 0 and supported
                and not usage.get('total_cached_tokens', 0) and not usage.get('total_tool_use_tokens', 0)):
            # Image-only response can omit internal text from modality details.
            # Aggregate output includes that text but excludes thought tokens.
            text = total_output - images
            estimate = round((inputs * 1.5 + images * 30 + (text + thoughts) * 7.5) / 1e6, 8)
    return {'actual_cost_usd': None, 'estimated_cost_usd': estimate,
            'cost_source': 'official_standard_rates_estimate_not_invoice',
            'pricing_source': PRICE_SOURCE, 'pricing_checked_date': '2026-10-11',
            'published_1k_image_output_only_usd': 0.0336,
            'billing_note': 'Actual charged amount is not returned by this API; input/thinking can add cost.'}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--key-file', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--track', required=True)
    p.add_argument('--source-image', type=Path)
    args = p.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    receipt_path = out / 'receipt.json'
    if receipt_path.exists():
        raise SystemExit('Existing receipt: refusing to repeat a potentially charged request')
    key = args.key_file.read_text().strip()
    if not key:
        raise SystemExit('Empty credential; no request sent')
    geometry = prepare(args.track, out, args.source_image)
    source = (out / 'source.png').read_bytes()
    payload = payload_for(source)
    receipt = {'model': MODEL, 'track': args.track, 'geometry': geometry,
               'input_mode': 'square-anchored', 'diagnostic_canvas_sent': False,
               'route': 'direct_google_gemini_api', 'endpoint': API,
               'request': {k: v for k, v in payload.items() if k != 'input'},
               'prompt': ANCHORED_SQUARE_PROMPT, 'status': 'prepared',
               'training_eligibility': 'not established; visual evaluation only'}
    receipt['request']['input_references'] = [{'local_file': 'source.png',
                                               'sha256': hashlib.sha256(source).hexdigest()}]
    save_json(receipt_path, receipt)
    started = time.monotonic()
    receipt['status'] = 'request_started'
    save_json(receipt_path, receipt)
    try:
        request = urllib.request.Request(API, data=json.dumps(payload).encode(),
                                         headers={'x-goog-api-key': key, 'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=240) as r:
            raw = r.read(40 * 1024 * 1024 + 1)
            if len(raw) > 40 * 1024 * 1024:
                raise ValueError('Response exceeds bounded 40 MiB reader')
            response = json.loads(raw)
        receipt.update(http_status=200, usage=response.get('usage', {}),
                       provider_status=response.get('status'), response_model=response.get('model'))
        receipt['response_metadata'] = {k: v for k, v in response.items() if k not in {'steps', 'output_image', 'usage'}}
        parts = output_images(response)
        for i, part in enumerate(parts):
            blob = base64.b64decode(part['data'], validate=True)
            with Image.open(io.BytesIO(blob)) as image:
                if image.width * image.height > 20_000_000:
                    raise ValueError('Unexpected oversized output')
                image.load()
                path = out / ('raw.png' if i == 0 else f'raw-{i}.png')
                image.save(path)
                receipt.setdefault('images', []).append({'file': path.name, 'dimensions': list(image.size),
                                                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        receipt['status'] = 'completed' if parts else 'no_image'
    except urllib.error.HTTPError as e:
        receipt.update(status='http_error', http_status=e.code,
                       error=e.read(16384).decode(errors='replace').replace(key, '[REDACTED]'))
    except Exception as e:
        receipt.update(status='uncertain_or_decode_error', error=(type(e).__name__ + ': ' + str(e)).replace(key, '[REDACTED]'))
    receipt['elapsed_seconds'] = round(time.monotonic() - started, 3)
    receipt.update(cost_estimate(receipt.get('usage', {})))
    save_json(receipt_path, receipt)
    print(json.dumps({k: receipt.get(k) for k in ('model', 'track', 'status', 'http_status', 'images',
                                                'usage', 'elapsed_seconds', 'estimated_cost_usd', 'error')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
