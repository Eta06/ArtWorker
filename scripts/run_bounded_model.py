#!/usr/bin/env python3
"""Run one serial model job with macOS footprint, pressure and swap guards."""
from __future__ import annotations

import argparse
import ctypes
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import struct
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
LIBPROC = ctypes.CDLL('/usr/lib/libproc.dylib')
LIBPROC.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
LIBPROC.proc_pid_rusage.restype = ctypes.c_int


def footprint(pid: int) -> int:
    buffer = ctypes.create_string_buffer(1024)
    if LIBPROC.proc_pid_rusage(pid, 4, buffer) != 0:
        return 0  # Process can exit between snapshot and query.
    return struct.unpack_from('Q', buffer.raw, 16 + 7 * 8)[0]


def pressure_and_swap() -> tuple[int, int]:
    values = subprocess.check_output(['sysctl', '-n', 'kern.memorystatus_vm_pressure_level', 'vm.swapusage'], text=True).splitlines()
    match = re.search(r'used = ([\d.]+)([MG])', values[1])
    if match is None:
        raise RuntimeError('Cannot read swap usage; refusing unmonitored execution.')
    swap = int(float(match.group(1)) * (1024 ** (2 if match.group(2) == 'M' else 3)))
    return int(values[0]), swap


def group_memory(pid: int) -> tuple[int, int]:
    snapshot = subprocess.check_output(['ps', '-axo', 'pid,ppid,rss'], text=True)
    rows = [tuple(map(int, row.split())) for row in snapshot.splitlines()[1:] if row.strip()]
    ids = {pid}
    while True:
        before = len(ids)
        ids.update(child for child, parent, rss in rows if parent in ids)
        if before == len(ids):
            break
    return sum(rss * 1024 for child, parent, rss in rows if child in ids), sum(footprint(child) for child in ids)


def run(command: list[str], output: Path, max_gib: float, timeout: float, max_swap_mib: float = 256) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    telemetry_path = output / 'memory.jsonl'
    receipt_path = output / 'launch.json'
    if receipt_path.exists():
        raise RuntimeError('Existing run receipt; choose a fresh output directory.')
    lock_path = ROOT / '.build/outpaint-gpu.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        initial_pressure, initial_swap = pressure_and_swap()
        if initial_pressure != 1:
            raise RuntimeError('Memory pressure is already elevated; no model launched.')
        receipt = {'status':'running', 'command':command, 'footprint_limit_bytes':int(max_gib * 1024**3), 'swap_growth_limit_bytes':int(max_swap_mib * 1024**2), 'timeout_seconds':timeout, 'sample_interval_seconds':0.1, 'initial_swap_bytes':initial_swap}
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
        start = time.monotonic()
        peak_rss = peak_footprint = 0
        reason = 'complete'
        with (output / 'execution.log').open('w') as log, telemetry_path.open('w') as telemetry:
            child = subprocess.Popen(['nice', '-n', '15', *command], stdout=log, stderr=subprocess.STDOUT, start_new_session=True, cwd=ROOT)
            receipt['pid'] = child.pid
            receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
            try:
                while child.poll() is None:
                    rss, physical = group_memory(child.pid)
                    level, swap = pressure_and_swap()
                    elapsed = time.monotonic() - start
                    peak_rss = max(peak_rss, rss)
                    peak_footprint = max(peak_footprint, physical)
                    telemetry.write(json.dumps({'seconds':round(elapsed,3),'rss_bytes':rss,'phys_footprint_bytes':physical,'pressure_level':level,'swap_bytes':swap}) + '\n')
                    telemetry.flush()
                    if max(rss, physical) > receipt['footprint_limit_bytes']:
                        reason = 'memory_limit'
                    elif level != 1:
                        reason = 'system_memory_pressure'
                    elif swap - initial_swap > receipt['swap_growth_limit_bytes']:
                        reason = 'swap_growth_limit'
                    elif elapsed > timeout:
                        reason = 'wall_time_limit'
                    if reason != 'complete':
                        break
                    time.sleep(0.1)
            except BaseException:
                reason = 'monitor_error_or_interrupt'
                raise
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                if reason == 'complete' and child.returncode != 0:
                    reason = 'subprocess_exit'
                receipt.update({'status':'completed' if child.returncode == 0 and reason == 'complete' else 'stopped_or_failed', 'stop_reason':reason, 'exit_code':child.returncode, 'elapsed_seconds':time.monotonic()-start, 'peak_sampled_rss_bytes':peak_rss, 'peak_sampled_phys_footprint_bytes':peak_footprint, 'measurement_note':'RSS and phys_footprint overlap; do not add them. Process-group sums can double-count shared memory. Sampling is not a kernel-enforced limit.'})
                receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
        return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--max-gib', type=float, default=12)
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or args.max_gib <= 0 or args.timeout <= 0:
        parser.error('Command and positive limits required.')
    result = run(command, args.output, args.max_gib, args.timeout)
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result['status'] == 'completed' else 1)


if __name__ == '__main__':
    main()
