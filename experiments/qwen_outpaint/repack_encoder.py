"""Repack native encoder by layer without changing any tensor bytes."""
import argparse
import hashlib
import json
import re
import struct
from pathlib import Path

from run_trial import MODEL, ROOT, header


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--component', choices=['text_encoder','transformer'], default='text_encoder')
    args = parser.parse_args()
    destination = MODEL / (args.component + '_layerwise')
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir()
    groups = {}
    source_count = 0
    metadata = None
    for path in sorted((MODEL / args.component).glob('*.safetensors')):
        data = header(path)
        shard_metadata = data.pop('__metadata__')
        if metadata is not None and metadata != shard_metadata:
            raise ValueError('Inconsistent shard metadata')
        metadata = shard_metadata
        with path.open('rb') as source:
            offset = 8 + struct.unpack('<Q', source.read(8))[0]
        for key, entry in data.items():
            match = re.match(r'(language_model\.layers\.\d+|visual\.blocks\.\d+|visual\.deepstack_merger_list\.\d+|transformer_blocks\.\d+)\.', key)
            group = match.group(1) if match else key.rsplit('.', 1)[0]
            groups.setdefault(group, []).append((key, entry, path, offset))
            source_count += 1
    records = []
    for index, (group, tensors) in enumerate(sorted(groups.items())):
        output = destination / f'{index:03d}.safetensors'
        new_header = {'__metadata__': metadata}
        offset = 0
        for key, entry, _, _ in tensors:
            size = entry['data_offsets'][1] - entry['data_offsets'][0]
            new_header[key] = dict(entry, data_offsets=[offset, offset + size])
            offset += size
        raw_header = json.dumps(new_header, separators=(',', ':')).encode()
        raw_header += b' ' * (-len(raw_header) % 8)
        digest = hashlib.sha256()
        def write(stream, data):
            stream.write(data)
            digest.update(data)
        with output.open('xb') as target:
            write(target, struct.pack('<Q', len(raw_header)))
            write(target, raw_header)
            for key, entry, path, start in tensors:
                remaining = entry['data_offsets'][1] - entry['data_offsets'][0]
                with path.open('rb') as source:
                    source.seek(start + entry['data_offsets'][0])
                    while remaining:
                        chunk = source.read(min(remaining, 8 * 1024**2))
                        if not chunk:
                            raise EOFError(path)
                        write(target, chunk)
                        remaining -= len(chunk)
        if {k: (v['dtype'], v['shape']) for k,v in header(output).items() if k != '__metadata__'} != {k: (v['dtype'], v['shape']) for k,v,_,_ in tensors}:
            raise ValueError('Repack header mismatch')
        records.append(dict(group=group, file=output.name, bytes=output.stat().st_size,
            tensor_count=len(tensors), sha256=digest.hexdigest()))
    result = dict(status='completed', method='bit-identical tensor payload copy; per-layer safetensors',
        source_manifest='experiments/qwen_outpaint/native_q4_manifest.json',
        source_repo='Qwen/Qwen-Image-2.1', source_revision='790c92633540aa0cb11d9abf19eb46d861714758',
        license='qwen-research', tensor_count=source_count, files=records,
        payload_bytes=sum(x['bytes'] for x in records), published_payloads=False)
    (destination / 'repack_manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    (ROOT / ('experiments/qwen_outpaint/layerwise_' + ('encoder' if args.component == 'text_encoder' else 'transformer') + '_manifest.json')).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(status='completed', tensor_count=source_count, files=len(records), bytes=result['payload_bytes'])))


if __name__ == '__main__':
    main()
