"""Download only the pinned public FP16 files authorized by the parent task."""
from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
REPO = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
REVISION = "115134f363124c53c7d878647567d04daf26e41e"

def main():
    source = ROOT / "experiments/sdxl_inpaint/download_manifest.json"
    manifest = json.loads(source.read_text())
    assert manifest["repo_id"] == REPO and manifest["revision"] == REVISION
    assert not manifest["access"]["gated"]
    assert manifest["file_count"] == 18 and manifest["total_bytes"] == 6941218469
    model = ROOT / ".build/models/sdxl-inpaint-fp16"
    output = ROOT / ".build/models/sdxl-inpaint-fp16.manifest.json"
    manifest.update(status="downloading", local_dir=str(model),
                    download_started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    parent_authorized_public_download=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    command = [str(ROOT / ".build/dreamlite-venv/bin/hf"), "download", REPO,
               *manifest["allow_patterns"], "--revision", REVISION,
               "--local-dir", str(model), "--max-workers", "2"]
    environment = dict(os.environ)
    environment["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    log = ROOT / ".build/models/sdxl-inpaint-fp16.download.log"
    with log.open("a") as stream:
        completed = subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    manifest["download_exit_code"] = completed.returncode
    manifest["status"] = "download_complete_unverified" if completed.returncode == 0 else "download_failed"
    manifest["download_finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": manifest["status"], "exit_code": completed.returncode,
                      "log_path": str(log), "manifest_path": str(output)}), flush=True)
    raise SystemExit(completed.returncode)

if __name__ == "__main__":
    main()
