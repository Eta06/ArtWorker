"""Bounded two-request variant batch. Existing receipts are never resubmitted."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
from pathlib import Path
import subprocess
import sys

from benchmark import ROOT, key_usage, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--key-file', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    jobs, cells = [], []
    # Breadth first: cover every model before moving to its additional tiers.
    for index in range(max(len(m['profiles']) for m in plan['models'])):
        for model in plan['models']:
            if index >= len(model['profiles']):
                continue
            profile = model['profiles'][index]
            for track in profile['tracks']:
                cell = {'model': model['id'], 'profile': profile['tag'], 'track': track['track'],
                        'resolution': profile['resolution'], 'quality': profile['quality']}
                if track['blocked_by_prior_filter']:
                    cell['status'] = 'blocked_from_prior_filter'
                elif track['reuse_directory']:
                    cell.update(status='reused', directory=track['reuse_directory'])
                else:
                    directory = args.results / model['id'].replace('/', '--') / profile['tag'] / track['track']
                    cell['directory'] = str(directory)
                    if (directory / 'receipt.json').exists():
                        receipt = json.loads((directory / 'receipt.json').read_text())
                        cell['status'] = receipt['status']
                        if cell['status'] in ('prepared', 'request_started', 'uncertain_or_decode_error', 'no_image'):
                            raise SystemExit('Unresolved existing receipt; reconcile before resuming: ' + str(directory))
                    else:
                        cell['status'] = 'pending'
                        jobs.append(cell)
                cells.append(cell)
    print(json.dumps({'models': len(plan['models']), 'cells': len(cells), 'new_requests': len(jobs),
                      'reused': sum(c['status'] == 'reused' for c in cells),
                      'prior_blocked': sum(c['status'] == 'blocked_from_prior_filter' for c in cells)}), flush=True)
    if args.dry_run:
        return
    args.results.mkdir(parents=True, exist_ok=True)
    key = args.key_file.read_text().strip()
    prior_batch = args.results / 'batch.json'
    initial = (json.loads(prior_batch.read_text())['initial_key_usage']
               if prior_batch.exists() else key_usage(key))
    state = {'initial_key_usage': initial, 'cells': cells, 'status': 'running',
             'max_concurrent_requests': 2, 'stop_reason': None}
    save_json(args.results / 'batch.json', state)

    def run(cell):
        command = [sys.executable, str(Path(__file__).with_name('benchmark.py')),
                   '--key-file', str(args.key_file), '--model', cell['model'], '--track', cell['track'],
                   '--output', cell['directory'], '--request-timeout', '480']
        for control in ('resolution', 'quality'):
            if cell[control]:
                command += ['--' + control, cell[control]]
        process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        path = Path(cell['directory']) / 'receipt.json'
        receipt = json.loads(path.read_text()) if path.exists() else {}
        cell.update(status=receipt.get('status', 'preflight_failed'),
                    cost=receipt.get('reported_cost_usd'), seconds=receipt.get('elapsed_seconds'))
        if process.returncode:
            cell['preflight_error'] = (process.stderr[-1200:] or process.stdout[-1200:]).replace(key, '[REDACTED]')
        print(json.dumps(cell, ensure_ascii=False), flush=True)
        return cell

    stopped = False
    with ThreadPoolExecutor(max_workers=2) as executor:
        active = {}
        while jobs or active:
            while jobs and len(active) < 2 and not stopped:
                account = key_usage(key)
                spent = account['usage'] - initial['usage']
                # Reserve $3 per in-flight request, including the one about to launch.
                if spent + 3 * (len(active) + 1) > 45 or account['limit_remaining'] < 3 * (len(active) + 1):
                    stopped = True
                    state['stop_reason'] = 'budget_reservation'
                    break
                cell = jobs.pop(0)
                cell['status'] = 'dispatched'
                active[executor.submit(run, cell)] = cell
                save_json(args.results / 'batch.json', state)
            if not active:
                break
            completed, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in completed:
                active.pop(future)
                cell = future.result()
                if cell['status'] not in ('completed', 'http_error'):
                    stopped = True
                    state['stop_reason'] = 'unresolved_request'
                save_json(args.results / 'batch.json', state)
    state.update(status='paused' if stopped else 'completed', final_key_usage=key_usage(key))
    save_json(args.results / 'batch.json', state)
    print(json.dumps({'batch_status': state['status'], 'stop_reason': state['stop_reason'],
                      'remaining': len(jobs), 'account': state['final_key_usage']}), flush=True)


if __name__ == '__main__':
    main()
