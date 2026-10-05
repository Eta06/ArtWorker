#!/usr/bin/env python3
"""Commit a reviewed JSON file list atomically per file; never push."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import os
import subprocess
import time


def memory_gate() -> None:
    """Stop before a write when macOS reports elevated memory pressure."""
    if os.uname().sysname != "Darwin":
        return
    level = subprocess.check_output(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True).strip()
    if level != "1":
        raise RuntimeError(f"Stopped: macOS memory pressure level is {level}.")


def guarded_write(args: list[str], max_mib: int, telemetry: Path) -> None:
    memory_gate()
    cap = max_mib * 1024 * 1024
    start = time.monotonic()
    peak_kib = 0
    process = subprocess.Popen(["nice", "-n", "15", *args], start_new_session=True)
    try:
        while process.poll() is None:
            snapshot = subprocess.check_output(["ps", "-axo", "pid,ppid,rss,comm"], text=True)
            rows = []
            all_git_kib = 0
            for row in snapshot.splitlines()[1:]:
                columns = row.split(None, 3)
                if len(columns) != 4:
                    continue
                pid, parent, rss = map(int, columns[:3])
                rows.append((pid, parent, rss))
                if columns[3].endswith("/git") or columns[3] == "git":
                    all_git_kib += rss
            descendants = {process.pid}
            changed = True
            while changed:
                before = len(descendants)
                descendants.update(pid for pid, parent, rss in rows if parent in descendants)
                changed = len(descendants) != before
            rss_kib = sum(rss for pid, parent, rss in rows if pid in descendants)
            peak_kib = max(peak_kib, rss_kib)
            if rss_kib * 1024 > cap:
                raise RuntimeError(f"Stopped: Git process group RSS exceeds {max_mib} MiB.")
            if all_git_kib > 1024 * 1024:
                raise RuntimeError("Stopped: combined RSS of all Git processes exceeds 1 GiB.")
            if time.monotonic() - start > 60:
                raise RuntimeError("Stopped: a single Git write exceeded 60 seconds.")
            memory_gate()
            time.sleep(0.1)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, args)
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, 15)
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, 9)
                process.wait()
        raise
    finally:
        with telemetry.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"operation":args[1:],"peak_sampled_rss_kib":peak_kib,"seconds":round(time.monotonic()-start,3),"exit_code":process.returncode}) + "\n")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--files-file", required=True, type=Path)
    parser.add_argument("--resume", action="store_true", help="Skip clean files already committed; rebuild the receipt from history.")
    parser.add_argument("--delay-seconds", type=float, default=0.0, help="Optional pause between commits; continuous memory checks remain active.")
    parser.add_argument("--max-git-mib", type=int, default=256, help="Stop above this sampled Git process-group RSS (default: 256 MiB).")
    parser.add_argument("--max-files", type=int, help="Bound a pilot or publication batch.")
    args = parser.parse_args()
    if args.delay_seconds < 0:
        parser.error("Delay cannot be negative.")
    if args.max_git_mib < 64 or (args.max_files is not None and args.max_files < 1):
        parser.error("Memory limit must be at least 64 MiB and max-files must be positive.")
    if not re.fullmatch(r"[a-z][a-z0-9-]*", args.phase):
        parser.error("Phase must be a lowercase single word or hyphenated identifier.")
    root = Path(git("rev-parse", "--show-toplevel")).resolve()
    if Path.cwd().resolve() != root:
        parser.error("Run from the repository root.")
    if git("diff", "--cached", "--name-only"):
        parser.error("The index must be empty; preserve unrelated staged changes.")
    files = json.loads(args.files_file.read_text(encoding="utf-8"))
    if not isinstance(files, list) or not files or any(not isinstance(p, str) for p in files):
        parser.error("File list must be a nonempty JSON array of relative paths.")
    if len(files) != len(set(files)):
        parser.error("Duplicate paths in the file list.")
    for name in files:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or ".git" in path.parts:
            parser.error(f"Unsafe relative path: {name}")
        if not (root / path).is_file() or not (root / path).resolve().is_relative_to(root):
            parser.error(f"Missing file or external symlink: {name}")
        ignored = subprocess.run(["git", "check-ignore", "-q", "--", name])
        if ignored.returncode == 0:
            parser.error(f"Ignored file: {name}")
        if ignored.returncode != 1:
            parser.error(f"Could not check ignore rules: {name}")
    receipt = root / ".build" / "publication" / f"{args.phase}-commits.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    telemetry = receipt.with_name(f"{args.phase}-memory.jsonl")
    records = []
    remaining = []
    for name in files:
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", "--", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        if tracked and args.resume:
            if git("status", "--porcelain", "--", name):
                parser.error(f"Previously committed file has new changes: {name}")
            records.append({"file": name, "commit": git("log", "-1", "--format=%H", "--", name)})
        else:
            remaining.append(name)
    print(f"Preserved {len(records)} prior commits; {len(remaining)} files remain.", flush=True)
    if args.max_files is not None:
        remaining = remaining[:args.max_files]
    for index, name in enumerate(remaining, 1):
        guarded_write(["git", "-c", "gc.auto=0", "add", "--", name], args.max_git_mib, telemetry)
        staged = [p for p in git("diff", "--cached", "--name-only", "--no-renames", "-z").split("\0") if p]
        if staged != [name]:
            raise RuntimeError(f"Expected exactly one changed file: {name}")
        guarded_write(["git", "-c", "gc.auto=0", "commit", "-q", "-m", f"{args.phase}: {name}"], args.max_git_mib, telemetry)
        records.append({"file": name, "commit": git("rev-parse", "HEAD")})
        temporary = receipt.with_suffix(".tmp")
        temporary.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        temporary.replace(receipt)
        if index % 25 == 0 or index == len(remaining):
            print(f"Committed {len(records)}/{len(files)} files", flush=True)
        time.sleep(args.delay_seconds)
    print(f"Local commits complete; no push performed. Receipt: {receipt}", flush=True)


if __name__ == "__main__":
    main()
