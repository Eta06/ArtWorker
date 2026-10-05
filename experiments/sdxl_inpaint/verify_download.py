"""Strict CPU file integrity verification; no tensors, model, network or GPU."""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
REPO = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
REVISION = "115134f363124c53c7d878647567d04daf26e41e"

def main():
    started = time.perf_counter()
    path = ROOT / ".build/models/sdxl-inpaint-fp16.manifest.json"
    manifest = json.loads(path.read_text())
    assert manifest["repo_id"] == REPO and manifest["revision"] == REVISION
    model = Path(manifest["local_dir"]).resolve()
    assert model == (ROOT / ".build/models/sdxl-inpaint-fp16").resolve()
    issues, checked = [], []
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        assert not relative.is_absolute() and ".." not in relative.parts
        target = model / relative
        if not target.is_file() or target.stat().st_size != entry["bytes"]:
            issues.append({"path": entry["path"], "issue": "missing_or_size_mismatch"})
    if issues:
        print(json.dumps({"status": "incomplete", "issues": issues}), flush=True)
        raise SystemExit(2)
    for entry in manifest["files"]:
        target = model / entry["path"]
        sha = hashlib.sha256()
        blob = hashlib.sha1(b"blob " + str(target.stat().st_size).encode() + b"\0")
        with target.open("rb") as stream:
            for data in iter(lambda: stream.read(16 * 1024 * 1024), b""):
                sha.update(data); blob.update(data)
        actual_sha, actual_blob = sha.hexdigest(), blob.hexdigest()
        matched = actual_sha == entry["sha256"] if entry["sha256"] else actual_blob == entry["git_blob_sha1"]
        checked.append({"path": entry["path"], "actual_bytes": target.stat().st_size,
                        "actual_sha256": actual_sha, "actual_git_blob_sha1": actual_blob,
                        "verified": matched})
        if not matched:
            issues.append({"path": entry["path"], "issue": "digest_mismatch"})
    manifest.update(status="verified" if not issues else "verification_failed",
                    verified=not issues, actual_total_bytes=sum(e["actual_bytes"] for e in checked),
                    verified_files=checked, verification_issues=issues,
                    verification_seconds=time.perf_counter() - started,
                    verification_completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    model_weights_loaded=False, device="CPU file hashing only")
    manifest["access"]["model_payload_downloaded"] = True
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)
    (ROOT / ".build/models/sdxl-inpaint-fp16.verification.json").write_text(json.dumps({
        "status":manifest["status"],"verification_seconds":manifest["verification_seconds"],
        "actual_total_bytes":manifest["actual_total_bytes"],"verified_files":checked,
        "verification_issues":issues,"model_weights_loaded":False}, indent=2) + "\n")
    print(json.dumps({"status":manifest["status"],"actual_total_bytes":manifest["actual_total_bytes"],
                      "file_count":len(checked),"issues":issues,
                      "verification_seconds":manifest["verification_seconds"]}), flush=True)
    raise SystemExit(0 if not issues else 2)

if __name__ == "__main__":
    main()
