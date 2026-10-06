"""Reconcile paid request receipts with generation billing; render raw comparisons."""
from __future__ import annotations

import argparse
from collections import defaultdict
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import urllib.parse

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmark import key_usage, request_json, save_json

ORDER = [("seedream", "Seedream 5.0 Flash"), ("nano", "Nano Banana 2.1"),
         ("hy", "Hy Image 3.5"), ("flux", "FLUX 3 Image"),
         ("sunburst", "Sunburst"), ("flare", "Flare")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    args = parser.parse_args()
    key = args.key_file.read_text().strip()
    rows = []
    for receipt_path in sorted(args.results.glob("*/*/receipt.json")):
        receipt = json.loads(receipt_path.read_text())
        row = {k: receipt.get(k) for k in ["model", "track", "status", "created_utc",
            "elapsed_seconds", "reported_cost_usd", "cost_source", "http_status", "error"]}
        row["run_directory"] = str(receipt_path.parent)
        row["source_sha256"] = receipt["geometry"]["original_sha256"]
        row["request_geometry"] = receipt["geometry"]
        headers = {k.lower(): v for k, v in receipt.get("response_headers", {}).items()}
        row["generation_id"] = headers.get("x-generation-id")
        row["provider"] = receipt["endpoint_snapshot"]["endpoints"][0]["provider_name"]
        if row["generation_id"]:
            try:
                billing, _ = request_json("generation?id=" + urllib.parse.quote(row["generation_id"]), key)
                data = billing["data"]
                row["generation_billing"] = {k: data.get(k) for k in ["id", "model", "provider_name",
                    "total_cost", "native_tokens_completion_images", "created_at", "generation_time"]}
                row["actual_cost_usd"] = data.get("total_cost")
                row["billing_matches_response"] = (Decimal(str(data["total_cost"])) ==
                    Decimal(str(receipt["reported_cost_usd"]))) if receipt["reported_cost_usd"] is not None else None
            except Exception as exc:
                row["billing_lookup_error"] = type(exc).__name__
        if row.get("actual_cost_usd") is None:
            row["actual_cost_usd"] = receipt.get("reported_cost_usd")
        image_path = receipt_path.parent / "raw.png"
        if image_path.exists():
            raw = Image.open(image_path).convert("RGB")
            source = Image.open(receipt_path.parent / "source.png").convert("RGB")
            normalized = raw.resize((720, 1280), Image.Resampling.LANCZOS)
            center = normalized.crop((0, 280, 720, 1000))
            row["raw_dimensions"] = list(raw.size)
            row["raw_sha256"] = hashlib.sha256(image_path.read_bytes()).hexdigest()
            row["expected_center_mae_255"] = round(float(np.abs(np.asarray(center, dtype=float) -
                                                    np.asarray(source, dtype=float)).mean()), 4)
            row["raw_center_pixel_exact_after_normalization"] = bool(np.array_equal(np.asarray(center), np.asarray(source)))
        rows.append(row)
    account = key_usage(key)
    initial_usage = min(Decimal(str(json.loads(p.read_text())["before"]["usage"]))
                        for p in args.results.glob("*/*/receipt.json"))
    reported_total = sum((Decimal(str(r["actual_cost_usd"])) for r in rows
                          if r["actual_cost_usd"] is not None), Decimal(0))
    account_delta = Decimal(str(account["usage"])) - initial_usage
    # Definitive HTTP failures are not billed by the Image API. Attribute zero
    # only when the dedicated key's full usage also reconciles to known charges.
    if account_delta == reported_total:
        for row in rows:
            if row["actual_cost_usd"] is None and row["status"] == "http_error":
                row["actual_cost_usd"] = 0
                row["cost_source"] = "definitive_http_error_and_dedicated_key_reconciliation"
    totals = defaultdict(lambda: Decimal(0))
    unknown = []
    for row in rows:
        if row["actual_cost_usd"] is None:
            unknown.append([row["model"], row["track"]])
        else:
            totals[row["model"]] += Decimal(str(row["actual_cost_usd"]))
    ledger = {"stage": "kehribar", "date": "2026-10-06", "currency": "USD", "runs": rows,
              "total_reported_usd": float(sum(totals.values(), Decimal(0))),
              "per_model_usd": {m: float(t) for m, t in totals.items()},
              "unknown_cost_runs": unknown, "key_usage_snapshot": {"usage": account["usage"]},
              "initial_key_usage_usd": float(initial_usage),
              "account_usage_matches_known_charges": account_delta == reported_total,
              "cost_note": "Generation total_cost, reconciled to Image API usage.cost. Credit purchase fees/tax and training compute excluded. Key usage can lag; it is not used to attribute individual requests.",
              "quality_note": "Raw outputs only, no center paste or color finishing. Center MAE is a placement/pixel-change diagnostic, not a perceptual quality score. Single sample per cover/model is not a general benchmark.",
              "training_eligibility": "Not established; evaluation images are not an approved training dataset."}
    save_json(args.ledger, ledger)
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 17)
    small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
    for track in ["track3", "track2"]:
        sheet = Image.new("RGB", (6 * 252, 522), "#14171c")
        draw = ImageDraw.Draw(sheet)
        for index, (slug, label) in enumerate(ORDER):
            path = args.results / slug / track / "raw.png"
            row = next((r for r in rows if Path(r["run_directory"]).parts[-2:] == (slug, track)), None)
            x = index * 252 + 6
            draw.text((x, 8), label, fill="white", font=font)
            if row:
                cost = row["actual_cost_usd"]
                draw.text((x, 32), f"${cost:.6f} | {row['elapsed_seconds']:.1f}s" if cost is not None else "Cost unconfirmed", fill="#bac6d6", font=small)
            if path.exists():
                im = Image.open(path).convert("RGB").resize((240, 427), Image.Resampling.LANCZOS)
                sheet.paste(im, (x, 65))
                draw.text((x, 498), f"RAW {row['raw_dimensions'][0]}x{row['raw_dimensions'][1]}", fill="#bac6d6", font=small)
            else:
                label = ("Blocked (HTTP 400)" if row and row.get("http_status") == 400
                         else row["status"] if row else "Not attempted")
                draw.text((x, 220), label, fill="white", font=small)
        sheet.save(args.results / (track + "-comparison.png"))
    print(json.dumps({"total_usd": ledger["total_reported_usd"], "per_model": ledger["per_model_usd"],
                      "unknown_cost_runs": unknown, "key_usage": ledger["key_usage_snapshot"]}, indent=2))


if __name__ == "__main__":
    main()
