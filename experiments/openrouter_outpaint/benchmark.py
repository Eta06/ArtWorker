"""Reference-canvas outpainting call using OpenRouter's Image API.

Credentials are read from a private file, never saved in receipts. No automatic
POST retries: a lost response can still represent an upstream generation.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
API = "https://openrouter.ai/api/v1/"
MODELS = [
    "openai/gpt-image-2",
    "google/gemini-3.1-flash-lite-image",
    "krea/krea-2-medium-turbo",
    "krea/krea-2-medium",
    "krea/krea-2-large",
    "microsoft/mai-image-2.5-pro",
    "qwen/qwen-image-3",
    "qwen/qwen-image-3-pro",
    "x-ai/grok-imagine-image-2.0",
    "bytedance-seed/seedream-5-0-pro",
    "bytedance-seed/seedream-5-0-lite",
    "microsoft/mai-image-2.6",
    "microsoft/mai-image-2.6-flash",
    "google/gemini-nano-banana-2.1",
    "tencent/hy-image-v3.5-preview",
    "bytedance-seed/seedream-5-0-flash",
    "black-forest-labs/flux-3-image",
    "openai/gpt-image-2.5-sunburst",
    "openai/gpt-image-2.5-flare",
]
PROMPT = (
    "Outpaint the album artwork into one seamless portrait image, aspect ratio 9:16. "
    "Reference 1 is the target layout: a 720x1280 canvas containing the original "
    "720x720 square at x=0, y=280, with solid green placeholder areas above and below. "
    "Reference 2 is the original square artwork. Keep that square centered, full "
    "width, at the same scale and position. Replace only the green placeholders "
    "with natural continuations of the existing scene. Preserve the original "
    "square's composition, people, faces, text, logos, color, grain, lighting and "
    "perspective. Continue lines, surfaces, shadows and clothing across both "
    "boundaries without visible seams or sudden color changes. Printed graphics "
    "on clothing must stay printed graphics, not become real objects. Do not add "
    "new people, new lettering, borders, gradients, blurred copies, mirrored "
    "copies or green tint. Return the complete expanded image, not a new cover."
)


def save_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def request_json(path: str, key: str, payload: dict | None = None, timeout: float = 240) -> tuple[dict, dict]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(API + path, data=data, headers={
        "Authorization": "Bearer " + key,
        "Content-Type": "application/json",
    })
    with urllib.request.urlopen(req, timeout=timeout if payload else 30) as response:
        raw = response.read(40 * 1024 * 1024 + 1)
        if len(raw) > 40 * 1024 * 1024:
            raise ValueError("Response exceeds bounded 40 MiB reader")
        return json.loads(raw), {k.lower(): v for k, v in response.headers.items()
                                 if k.lower() in {"x-request-id", "x-generation-id"}}


def key_usage(key: str) -> dict:
    d, _ = request_json("key", key)
    return {k: d["data"].get(k) for k in ("usage", "limit", "limit_remaining")}


def prepare(track: str, output: Path) -> dict:
    source_path = ROOT / "Player/Resources" / (track + ".jpg")
    original = Image.open(source_path).convert("RGB")
    width, height = original.size
    side = min(width, height)
    left, top = (width - side) // 2, (height - side) // 2
    source = original.crop((left, top, left + side, top + side))
    source = source.resize((720, 720), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (720, 1280), (0, 255, 0))
    canvas.paste(source, (0, 280))
    source.save(output / "source.png")
    canvas.save(output / "input.png")
    return {"original_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "original_dimensions": [width, height],
            "crop_xyxy": [left, top, left + side, top + side],
            "canvas_dimensions": [720, 1280], "source_rect_xyxy": [0, 280, 720, 1000]}


def data_url(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--track", choices=["track2", "track3"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", help="A supported native resolution tier")
    parser.add_argument("--quality", help="A supported native quality tier, including auto")
    parser.add_argument("--request-timeout", type=float, default=240)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / "receipt.json").exists():
        raise SystemExit("Existing receipt: refusing to repeat a paid request")
    key = args.key_file.read_text().strip()
    geometry = prepare(args.track, out)
    catalog, _ = request_json("images/models/" + args.model + "/endpoints", key)
    endpoint = catalog["endpoints"][0]
    supported = endpoint["supported_parameters"]
    max_references = supported.get("input_references", {}).get("max", 0)
    if max_references < 1:
        raise SystemExit("This endpoint does not accept source images; no paid outpaint request sent")
    names = ("input.png", "source.png") if max_references >= 2 else ("input.png",)
    prompt = PROMPT if len(names) == 2 else PROMPT.replace(
        "Reference 2 is the original square artwork. ", "")
    payload = {"model": args.model, "prompt": prompt,
               "aspect_ratio": "9:16", "input_references": [
                   {"type": "image_url", "image_url": {"url": data_url(out / name)}}
                   for name in names],
               "provider": {"only": [endpoint["provider_tag"]], "allow_fallbacks": False}}
    if "n" in supported:
        payload["n"] = 1
    if "resolution" in supported:
        values = supported["resolution"]["values"]
        resolution = args.resolution or ("1K" if "1K" in values else values[0])
        if resolution not in values:
            raise SystemExit("Unsupported resolution tier; no paid request sent")
        payload["resolution"] = resolution
    elif args.resolution:
        raise SystemExit("Endpoint has no resolution control; no paid request sent")
    if "quality" in supported:
        values = supported["quality"]["values"]
        quality = args.quality or ("medium" if "medium" in values else values[0])
        if quality not in values:
            raise SystemExit("Unsupported quality tier; no paid request sent")
        payload["quality"] = quality
    elif args.quality:
        raise SystemExit("Endpoint has no quality control; no paid request sent")
    # Receipt has hashes and paths instead of source payloads or authentication.
    public_payload = {**payload, "input_references": [
        {"local_file": name, "sha256": hashlib.sha256((out / name).read_bytes()).hexdigest()}
        for name in names]}
    receipt = {"model": args.model, "track": args.track, "geometry": geometry,
               "request": public_payload, "endpoint_snapshot": catalog,
               "status": "prepared", "before": key_usage(key),
               "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "training_eligibility": "not established; visual evaluation only"}
    save_json(out / "receipt.json", receipt)
    started = time.monotonic()
    receipt["status"] = "request_started"
    save_json(out / "receipt.json", receipt)
    print(json.dumps({"event": "request_started", "model": args.model, "track": args.track}), flush=True)
    try:
        response, headers = request_json("images", key, payload, timeout=args.request_timeout)
        receipt["response_headers"] = headers
        receipt["usage"] = response.get("usage", {})
        receipt["response_metadata"] = {k: v for k, v in response.items()
                                        if k not in {"data", "usage"}}
        receipt["response_data_metadata"] = []
        images = response.get("data", [])
        if not images:
            receipt["status"] = "no_image"
        for index, item in enumerate(images):
            receipt["response_data_metadata"].append({k: v for k, v in item.items()
                                                       if k not in {"b64_json", "url"}})
            if not item.get("b64_json"):
                raise ValueError("Image API did not return documented b64_json output")
            blob = base64.b64decode(item["b64_json"], validate=True)
            image = Image.open(io.BytesIO(blob))
            image.load()
            raw_path = out / ("raw.png" if index == 0 else f"raw-{index}.png")
            image.save(raw_path)
            receipt.setdefault("images", []).append({"file": raw_path.name,
                "dimensions": list(image.size), "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest()})
        if receipt.get("images"):
            receipt["status"] = "completed"
    except urllib.error.HTTPError as exc:
        receipt["status"] = "http_error"
        receipt["http_status"] = exc.code
        receipt["response_headers"] = {k.lower(): v for k, v in exc.headers.items()
            if k.lower() in {"x-request-id", "x-generation-id"}}
        receipt["error"] = exc.read(8192).decode(errors="replace").replace(key, "[REDACTED]")
    except Exception as exc:
        receipt["status"] = "uncertain_or_decode_error"
        receipt["error"] = (type(exc).__name__ + ": " + str(exc)).replace(key, "[REDACTED]")
    finally:
        receipt["elapsed_seconds"] = round(time.monotonic() - started, 3)
        try:
            receipt["after"] = key_usage(key)
        except Exception as exc:
            receipt["billing_lookup_error"] = type(exc).__name__
        cost = receipt.get("usage", {}).get("cost")
        receipt["reported_cost_usd"] = cost
        receipt["cost_source"] = "image_api_usage.cost" if cost is not None else "not_reported"
        if "after" in receipt:
            receipt["key_usage_delta_usd"] = receipt["after"]["usage"] - receipt["before"]["usage"]
        save_json(out / "receipt.json", receipt)
        print(json.dumps({k: receipt.get(k) for k in ["model", "track", "status", "elapsed_seconds", "reported_cost_usd", "key_usage_delta_usd", "http_status", "error"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
