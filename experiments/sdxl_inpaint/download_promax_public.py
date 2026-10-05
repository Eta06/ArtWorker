"""Assemble the pinned official ProMax recipe via public HF CLI downloads.

Downloads are network/CPU only. Original model files and caches are preserved.
Tokenizer reuse is independently byte-verified before a new local hard link.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = {
    "xinsir/controlnet-union-sdxl-1.0": "801a4a3fa3d4c936f4feea95b98607bc6726f80c",
    "stabilityai/stable-diffusion-xl-base-1.0": "462165984030d82259a11f4367a4eed129e94a7b",
    "madebyollin/sdxl-vae-fp16-fix": "207b116dae70ace3637169f1ddd2434b91b3a8cd",
}

def digest_ok(path, entry):
    if not path.is_file() or path.stat().st_size != entry["bytes"]:
        return False
    sha = hashlib.sha256()
    blob = hashlib.sha1(b"blob " + str(path.stat().st_size).encode() + b"\0")
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            sha.update(data); blob.update(data)
    return sha.hexdigest() == entry["sha256"] if entry.get("sha256") else blob.hexdigest() == entry["git_blob_sha1"]

def safe_relative(value):
    path = Path(value)
    assert not path.is_absolute() and ".." not in path.parts
    return path

def main():
    manifest = json.loads((ROOT / "experiments/sdxl_inpaint/promax_download_manifest.json").read_text())
    assert manifest["file_count"] == 20 and manifest["download_bytes"] == 9618667533
    local = Path(manifest["local_dir"]).resolve()
    assert local == (ROOT / ".build/models/sdxl-promax-official").resolve()
    path = ROOT / ".build/models/sdxl-promax-official.manifest.json"
    manifest.update(status="downloading", parent_authorized_public_download=True,
                    download_started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    groups = {}
    for entry in manifest["files"]:
        assert EXPECTED[entry["repo_id"]] == entry["revision"]
        safe_relative(entry["path"]); relative = safe_relative(entry["destination_path"])
        if entry["action"] == "download":
            groups.setdefault((entry["repo_id"], entry["revision"]), []).append(entry)
        else:
            assert entry["action"] == "copy_verified_identical"
            source = Path(entry["source_local_path"]).resolve()
            assert source.is_relative_to((ROOT / ".build/models/sdxl-inpaint-fp16").resolve())
            if not digest_ok(source, entry):
                raise ValueError(f"Tokenizer reuse source is not byte-identical: {relative}")
            destination = local / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if not digest_ok(destination, entry):
                    raise FileExistsError(f"Preserving unexpected destination: {destination}")
            else:
                os.link(source, destination)
            entry["source_bytes_verified_before_link"] = True
    environment = dict(os.environ)
    environment["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    def write():
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(path)
    write()
    manifest["download_groups"] = []
    for (repo, revision), entries in groups.items():
        staging = local / ".downloads" / repo.replace("/", "--")
        log = ROOT / ".build/models" / f"sdxl-promax-{repo.split('/')[0]}.download.log"
        info = {"repo_id":repo,"revision":revision,"status":"downloading","log_path":str(log)}
        manifest["download_groups"].append(info); write()
        print(json.dumps({"phase":"download_group","repo_id":repo,"files":len(entries)}), flush=True)
        command = [str(ROOT / ".build/dreamlite-venv/bin/hf"),"download",repo,
                   *(e["path"] for e in entries),"--revision",revision,"--local-dir",str(staging),"--max-workers","2"]
        with log.open("a") as stream:
            result = subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                                    timeout=3600)
        info.update(status="complete" if result.returncode == 0 else "failed", exit_code=result.returncode)
        write()
        if result.returncode:
            manifest["status"] = "download_failed"; write()
            raise SystemExit(result.returncode)
        for entry in entries:
            source = staging / entry["path"]
            if not digest_ok(source, entry):
                manifest["status"] = "verification_failed"; write()
                raise ValueError(f"Pinned payload digest mismatch: {entry['path']}")
            destination = local / entry["destination_path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if not digest_ok(destination, entry):
                    raise FileExistsError(f"Preserving unexpected destination: {destination}")
            else:
                os.link(source, destination)
            entry["verified_before_link"] = True
    verified = []
    for entry in manifest["files"]:
        target = local / entry["destination_path"]
        matched = digest_ok(target, entry)
        verified.append({"path":entry["destination_path"],"actual_bytes":target.stat().st_size,
                         "verified":matched})
    manifest.update(status="verified" if all(e["verified"] for e in verified) else "verification_failed",
                    verified_files=verified, verified=all(e["verified"] for e in verified),
                    actual_total_bytes=sum(e["actual_bytes"] for e in verified),
                    download_finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    model_weights_loaded=False, gpu_used=False)
    write()
    print(json.dumps({"status":manifest["status"],"actual_total_bytes":manifest["actual_total_bytes"],
                      "manifest_path":str(path)}), flush=True)
    raise SystemExit(0 if manifest["verified"] else 2)

if __name__ == "__main__":
    main()
