#!/usr/bin/env python3
"""Download a bounded local cover set from previously observed YouTube Music DOM URLs.

Browser discovery is separate: no account cookies or private API calls are used.
The manifest, artwork, title/artist metadata and review page must remain ignored.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from PIL import Image, ImageStat

ROOT = Path(__file__).resolve().parents[2]
ALLOWED_HOSTS = {"yt3.googleusercontent.com", "lh3.googleusercontent.com"}


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def image_request_url(observed: str, size: int) -> str:
    parsed = urlparse(observed)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
        raise ValueError("Only observed Google album-image CDN URLs are allowed")
    # Preserve the observed asset identity; request its CDN size variant.
    if not re.search(r"=w\d+-h\d+", observed):
        raise ValueError("Observed image has no recognized width/height suffix")
    return re.sub(r"=w\d+-h\d+", f"=w{size}-h{size}", observed, count=1)


def fingerprint(image: Image.Image) -> tuple[str, str, float]:
    normalized = image.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
    digest = hashlib.sha256(normalized.tobytes()).hexdigest()
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    values = list(gray.get_flattened_data())
    bits = 0
    for y in range(8):
        for x in range(8):
            bits = (bits << 1) | int(values[y * 9 + x] > values[y * 9 + x + 1])
    stat = ImageStat.Stat(normalized)
    variation = sum(stat.stddev) / 3
    return digest, f"{bits:016x}", variation


def safe_candidate(candidate: dict) -> str:
    parsed = urlparse(candidate["album_url"])
    if parsed.scheme != "https" or parsed.hostname != "music.youtube.com":
        raise ValueError("Album URL must be from YouTube Music")
    if not re.fullmatch(r"/browse/MPRE[a-zA-Z0-9_-]+", parsed.path):
        raise ValueError("Source must have an observed album-release link")
    return parsed.path.rsplit("/", 1)[1]


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "experiments/cover_collection/results"):
        raise ValueError("Keep collection artifacts in the ignored results directory")
    output.mkdir(parents=True, exist_ok=True)
    (output / "covers").mkdir(exist_ok=True)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "schema_version": 1, "stage": args.stage, "target": args.target,
        "requested_cdn_size": args.size, "started_at": datetime.now(timezone.utc).isoformat(),
        "source": "authenticated YouTube Music rendered DOM",
        "rights_status": "not established; local evaluation candidates, not a released training dataset",
        "covers": [], "attempts": [],
    }
    # Resume only after validating the actual persisted files.
    for record in manifest["covers"]:
        local = output / record["file"]
        if hashlib.sha256(local.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Resume integrity failure: {record['file']}")
    manifest["target"] = args.target
    manifest["status"] = "running"
    manifest.pop("finished_at", None)
    atomic_json(manifest_path, manifest)
    seen_albums = {r["album_id"] for r in manifest["covers"]}
    seen_attempts = {(r["album_id"], r.get("asset_identity", "")) for r in manifest["attempts"]}
    seen_assets = {r["asset_identity"] for r in manifest["attempts"] if r.get("asset_identity")}
    seen_pixels = {r["normalized_rgb_sha256"]: r["id"] for r in manifest["covers"]}
    seen_dhash = [(int(r["dhash"], 16), r) for r in manifest["covers"]]
    start = time.monotonic()
    while len(manifest["covers"]) < args.target:
        candidates = json.loads(args.candidates.read_text())
        pending = []
        for candidate in candidates:
            try:
                album_id = safe_candidate(candidate)
            except (KeyError, ValueError):
                continue
            identity = (candidate.get("observed_image_url") or "").split("=", 1)[0]
            if album_id not in seen_albums and (album_id, identity) not in seen_attempts:
                pending.append((album_id, candidate))
        if not pending:
            if args.follow_seconds and time.monotonic() - start < args.follow_seconds:
                time.sleep(1)
                continue
            break
        for album_id, candidate in pending:
            if len(manifest["covers"]) >= args.target:
                break
            record = {"album_id": album_id, "time": datetime.now(timezone.utc).isoformat()}
            observed = candidate.get("observed_image_url") or ""
            asset_identity = observed.split("=", 1)[0]
            record["asset_identity"] = asset_identity
            seen_attempts.add((album_id, asset_identity))
            if not observed.startswith("https://"):
                record["status"] = "placeholder_not_loaded"
            elif asset_identity in seen_assets:
                record["status"] = "duplicate_asset"
            else:
                try:
                    request_url = image_request_url(observed, args.size)
                    with urlopen(Request(request_url, headers={"User-Agent": "ArtWorker-cover-collector/1.0"}), timeout=20) as response:
                        content = response.read(8 * 1024 * 1024 + 1)
                        record["http_status"] = response.status
                    if len(content) > 8 * 1024 * 1024:
                        raise ValueError("Response exceeded 8 MiB")
                    with Image.open(io.BytesIO(content)) as image:
                        image.load()
                        width, height = image.size
                        if width != height or not 512 <= width <= 4096:
                            raise ValueError(f"Expected square cover >=512 pixels; got {width}x{height}")
                        pixel_hash, dhash, variation = fingerprint(image)
                        if variation < 2:
                            raise ValueError("Near-solid image requires separate review")
                        duplicate = seen_pixels.get(pixel_hash)
                        if not duplicate:
                            from PIL import ImageChops
                            current = image.convert("RGB").resize((64, 64), Image.Resampling.LANCZOS)
                            for bits, previous in seen_dhash:
                                if (bits ^ int(dhash, 16)).bit_count() > 2:
                                    continue
                                with Image.open(output / previous["file"]) as prior:
                                    small = prior.convert("RGB").resize((64, 64), Image.Resampling.LANCZOS)
                                    mae = sum(ImageStat.Stat(ImageChops.difference(current, small)).mean) / 3
                                if mae <= 2:
                                    duplicate = previous["id"]
                                    break
                        if duplicate:
                            record.update(status="duplicate_pixels", duplicate_of=duplicate)
                        else:
                            extension = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(image.format)
                            if not extension:
                                raise ValueError("Unsupported image format")
                            cover_id = f"cover{len(manifest['covers']) + 1:04d}"
                            relative_file = f"covers/{cover_id}{extension}"
                            (output / relative_file).write_bytes(content)
                            cover = {**candidate, "id": cover_id, "album_id": album_id,
                                "file": relative_file, "asset_identity": asset_identity,
                                "request_image_url": request_url, "dimensions": [width, height],
                                "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
                                "normalized_rgb_sha256": pixel_hash, "dhash": dhash,
                                "downloaded_at": record["time"],
                                "native_source_resolution": "unknown; dimensions describe CDN response"}
                            manifest["covers"].append(cover)
                            seen_albums.add(album_id)
                            seen_pixels[pixel_hash] = cover_id
                            seen_dhash.append((int(dhash, 16), cover))
                            record.update(status="downloaded", cover_id=cover_id, bytes=len(content))
                    seen_assets.add(asset_identity)
                except Exception as error:
                    record.update(status="error", error=f"{type(error).__name__}: {error}"[:500])
            manifest["attempts"].append(record)
            manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
            manifest["downloaded_count"] = len(manifest["covers"])
            atomic_json(manifest_path, manifest)
            if len(manifest["attempts"]) % 25 == 0:
                print(json.dumps({"downloaded": len(manifest["covers"]), "attempts": len(manifest["attempts"]), "candidates": len(candidates)}), flush=True)
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["status"] = "target_reached" if len(manifest["covers"]) >= args.target else "candidates_exhausted"
    atomic_json(manifest_path, manifest)
    print(json.dumps({"status": manifest["status"], "downloaded": len(manifest["covers"]), "attempts": len(manifest["attempts"])}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--stage", default="defne")
    parser.add_argument("--follow-seconds", type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.target <= 10000 or args.size not in (512, 1024):
        parser.error("Bound target to 1..10000 and CDN size to 512 or 1024")
    run(args)
