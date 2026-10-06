"""Small cross-version Python/Swift packing and quantized-matmul compatibility check."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import mlx.core as mx
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, default=Path('.build/koza-quant-probes'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    mx.set_cache_limit(0)
    generator = np.random.default_rng(17)
    weight = mx.array(generator.normal(size=(32, 128)).astype(np.float16))
    inputs = mx.array(generator.normal(size=(8, 128)).astype(np.float16))
    records = []
    for bits in (2, 3, 4):
        fixture = args.fixtures / f'int{bits}.safetensors'
        fixture.parent.mkdir(parents=True, exist_ok=True)
        if fixture.exists():
            raise FileExistsError(fixture)
        packed, scales, biases = mx.quantize(weight, group_size=64, bits=bits)
        decoded = mx.dequantize(packed, scales, biases, group_size=64, bits=bits)
        product = mx.quantized_matmul(inputs, packed, scales, biases, transpose=True, group_size=64, bits=bits)
        mx.save_safetensors(str(fixture), {'weight': packed, 'scales': scales, 'biases': biases,
                                            'input': inputs, 'dequantized': decoded, 'product': product})
        process = subprocess.run(['.build/flux2-native/Build/Products/Release/Flux2CLI',
                                 'artworker-quant-probe', '--path', str(fixture), '--bits', str(bits)], text=True, capture_output=True)
        print(process.stdout, end='', flush=True)
        print(process.stderr, end='', flush=True)
        record = json.loads(process.stdout.strip().splitlines()[-1])
        record['native_exit_code'] = process.returncode
        record['fixture_sha256'] = hashlib.sha256(fixture.read_bytes()).hexdigest()
        records.append(record)
        mx.clear_cache()
    (args.output / 'comparison.json').write_text(json.dumps({'seed': 17, 'weight_shape': [32, 128],
        'input_shape': [8, 128], 'group_size': 64, 'python_mlx': '0.32.3', 'swift_mlx': '0.31.6',
        'records': records, 'limitation': 'Small fixture compatibility, not full model semantic acceptance.'}, indent=2) + '\n')
    print(json.dumps(records))
    if any(record['native_exit_code'] != 0 for record in records):
        raise RuntimeError('At least one cross-version compatibility check failed; all results retained')


if __name__ == '__main__':
    main()
