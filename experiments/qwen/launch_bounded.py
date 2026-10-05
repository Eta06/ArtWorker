"""Run one recoverable local GPU experiment with a process-group timeout."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeout', type=float, default=900)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command[:1] == ['--']:
        command = command[1:]
    if not command:
        parser.error('Missing experiment command')
    root = Path(__file__).resolve().parents[2]
    lock_path = root / '.build/outpaint-gpu.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    if args.manifest.exists():
        raise FileExistsError(args.manifest)
    hashes = {}
    for item in command:
        path = Path(item)
        if path.suffix == '.py' and path.is_file():
            hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    with lock_path.open('a') as lock, args.log.open('x') as output:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        begun = time.monotonic()
        process = subprocess.Popen(command, cwd=root, stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        record = dict(command=command, cwd=str(root), code_sha256=hashes,
                      timeout_seconds=args.timeout, pid=process.pid, status='running')
        args.manifest.write_text(json.dumps(record, indent=2))
        print(json.dumps({'status': 'running', 'pid': process.pid, 'log': str(args.log)}), flush=True)
        timed_out = False
        try:
            code = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait()
        record.update(status='timeout' if timed_out else ('success' if code == 0 else 'failed'),
                      exit_code=code, elapsed_seconds=time.monotonic()-begun)
        args.manifest.write_text(json.dumps(record, indent=2))
        print(json.dumps(record), flush=True)
        raise SystemExit(124 if timed_out else code)


if __name__ == '__main__':
    main()
