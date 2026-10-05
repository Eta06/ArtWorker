"""Wall-time bound for one explicitly authorized local serial MaskFlow run."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-seconds", type=float, default=3600)
    parser.add_argument("trial_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    trial_args = args.trial_args[1:] if args.trial_args[:1] == ["--"] else args.trial_args
    if args.timeout_seconds <= 0 or "--execute-gpu" not in trial_args:
        parser.error("Positive bound and explicit --execute-gpu trial argument are required")
    command = [sys.executable, str(Path(__file__).with_name("run_maskflow_trial.py")), *trial_args]
    start = time.monotonic()
    child = subprocess.Popen(command, start_new_session=True)
    reason = "complete"
    try:
        code = child.wait(timeout=args.timeout_seconds)
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
        reason = "wall_time_limit" if isinstance(exc, subprocess.TimeoutExpired) else "interrupted"
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        code = 124 if reason == "wall_time_limit" else 130
    print(json.dumps({"bounded_run": reason, "exit_code": code,
                      "elapsed_seconds": time.monotonic()-start,
                      "checkpoint_policy": "Retained checkpoint.json and state files support --resume-from"}), flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
