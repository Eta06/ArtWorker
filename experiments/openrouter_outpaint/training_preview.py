"""Read-only previews of the exact approved teacher-output input/target crops."""
from io import BytesIO
from PIL import Image


def preview(store, key, kind, fingerprint, revision):
    if kind not in ('input', 'target'):
        raise ValueError('Unknown preview kind')
    snapshot = store.snapshot()
    decision = snapshot['decisions'].get(key, {})
    if decision.get('raw_sha256') != fingerprint or str(decision.get('revision')) != revision:
        raise FileExistsError('Selection changed; refresh the page')
    candidate = next((r for r in snapshot['manifest']['candidates'] if r['id'] == key), None)
    if not candidate or decision.get('decision') != 'accepted':
        raise PermissionError('Only approved training candidates have previews')
    crop = candidate['conditioning']
    box = crop['box_px'] if kind == 'input' else crop['target_box_px']
    with Image.open(candidate['raw_file']) as raw:
        if list(raw.size) != crop['raw_dimensions']:
            raise ValueError('Raw image dimensions changed')
        # One raw image at a time, no persistent decoded-image cache or crop files.
        image = raw.crop(box)
        image.thumbnail((560, 780), Image.Resampling.LANCZOS)
        output = BytesIO()
        image.convert('RGB').save(output, format='JPEG', quality=88)
        return output.getvalue()
