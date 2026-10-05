"""Verify a pinned public Fill Q4 local download without loading tensor weights."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPO = "mflux-community/flux-1-dev-fill-mflux-q4"
REVISION = "eebfbaa12c95107169452c7d22622e04771192a3"


def verify(manifest_path):
    started = time.perf_counter()
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text())
    if (manifest["repo"] != REPO or manifest["revision"] != REVISION
            or manifest["resolved_revision"] != REVISION):
        raise ValueError("Expected the authorized exact Fill Q4 repository revision")
    local = Path(manifest["local_dir"]).resolve()
    if local != (ROOT / ".build/models/flux1-fill-q4").resolve():
        raise ValueError("Unexpected isolated model directory")
    issues = []
    verified = []
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Repository path escapes the local directory")
        target = local / relative
        if not target.is_file() or target.stat().st_size != entry["expected_bytes"]:
            issues.append({"path": entry["path"], "issue": "missing_or_size_mismatch"})
    # Avoid hashing partially downloaded large files; preserve all cache/partials.
    if issues:
        return {"status": "incomplete", "issues": issues, "manifest": str(path),
                "model_weights_loaded": False, "files_modified": False}
    for entry in manifest["files"]:
        target = local / entry["path"]
        size = target.stat().st_size
        digest = hashlib.sha256()
        blob = hashlib.sha1(b"blob " + str(size).encode() + b"\0")
        with target.open("rb") as stream:
            for data in iter(lambda: stream.read(16 * 1024 * 1024), b""):
                digest.update(data)
                if not entry["expected_lfs_sha256"]:
                    blob.update(data)
        sha256 = digest.hexdigest()
        check = {"path": entry["path"], "actual_bytes": size, "actual_sha256": sha256,
                 "size_matches": size == entry["expected_bytes"]}
        if entry["expected_lfs_sha256"]:
            check["lfs_sha256_matches"] = sha256 == entry["expected_lfs_sha256"]
            check["verified"] = check["size_matches"] and check["lfs_sha256_matches"]
        else:
            check["actual_git_blob_sha1"] = blob.hexdigest()
            check["git_blob_matches"] = blob.hexdigest() == entry["git_blob_id"]
            check["verified"] = check["size_matches"] and check["git_blob_matches"]
        verified.append(check)
        if not check["verified"]:
            issues.append({"path": entry["path"], "issue": "digest_mismatch"})
    result = {"status": "verified" if not issues else "verification_failed",
              "verification_completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "verification_seconds": time.perf_counter() - started,
              "model_weights_loaded": False, "device": "CPU file hashing only",
              "actual_total_bytes": sum(entry["actual_bytes"] for entry in verified),
              "verified_files": verified, "verification_issues": issues}
    manifest.update(result)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(path)
    # Keep each per-file expected LFS checksum/Git object alongside actual values.
    return {**result, "manifest": str(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / ".build/models/flux1-fill-q4.manifest.json")
    args = parser.parse_args()
    result = verify(args.manifest)
    print(json.dumps(result, indent=2), flush=True)
    raise SystemExit(0 if result["status"] == "verified" else 2)


if __name__ == "__main__":
    main()
