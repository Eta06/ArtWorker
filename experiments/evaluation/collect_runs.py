#!/usr/bin/env python3
"""Normalize available trial artifacts without loading models or inventing missing runs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re

from PIL import Image, ImageDraw

from evaluate_outpainting import compare, font, write_json


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
INPUTS = HERE / "inputs/inputs.json"
MAC_NOTE = "All recorded inference here ran on an Apple Silicon Mac; none is an iPhone measurement."
MEMORY_NOTE = "Process RSS, sampled MPS allocations, MLX allocations and driver allocations measure different things. They may overlap; do not sum them or call them interchangeable device peak RAM."
SOURCE_NOTE = "Source equality compares the shared 512×512 Lanczos-resized reference, not the original 720×720 decoded cover pixel map."


def common_metadata(data: dict, omitted: tuple[str, ...]) -> dict:
    return {key: value for key, value in data.items() if key not in omitted}


def measurement(record: dict, key: str, kind: str, scope: str, unit: str = "bytes") -> dict | None:
    value = record.get(key)
    return {"field": key, "value": value, "unit": unit, "kind": kind, "scope": scope} if value is not None else None


def measurements(record: dict, descriptions: list[tuple]) -> list[dict]:
    return [result for args in descriptions if (result := measurement(record, *args)) is not None]


def paths(run: dict, manifest: Path) -> dict:
    result = dict(run)
    result["source_manifest"] = str(manifest.resolve())
    result["benchmark_platform"] = "macOS Apple Silicon"
    result["platform_caveat"] = MAC_NOTE
    if result.get("image"):
        path = Path(result["image"])
        result["image"] = str((manifest.parent / path).resolve() if not path.is_absolute() else path)
        result["image_exists_at_collection"] = Path(result["image"]).is_file()
    else:
        result["image_exists_at_collection"] = False
    return result


def mobile_o(data: dict, manifest: Path) -> list[dict]:
    shared = common_metadata(data, ("results",))
    result = []
    locked = bool(data.get("latent_lock"))
    lock_label = "latent-lock-" if locked else ""
    for order, record in enumerate(data.get("results", []), 1):
        common = {
            **shared, **record,
            "model": data["model_id"],
            "track_id": record["track"],
            "trial_id": f"mobile-o-{data.get('steps', '?')}step-{lock_label}seed{data.get('seed', '?')}",
            "column_label": f"Mobile-O\n{data.get('steps', '?')} step / seed {data.get('seed', '?')}" + ("\nlatent lock" if locked else "\ninstruction edit"),
            "status": record.get("status", "completed"),
            "elapsed_seconds": record.get("seconds"),
            "elapsed_seconds_scope": "generation as reported by Python runner; model loading excluded",
            "display_timing_label": "Generation (load excl.)",
            "generation_order_within_process": order,
            "memory_measurements": measurements(record, [
                ("mps_current_allocated_bytes", "MPS current allocation snapshot; not a measured peak", "after this generation"),
                ("mps_driver_allocated_bytes", "MPS driver allocation snapshot; not a measured peak", "after this generation"),
                ("sampled_peak_mps_driver_bytes", "sampled MPS driver allocation maximum", "this generation; sampled between operations/steps"),
                ("sampled_peak_mps_current_bytes", "sampled MPS current allocation maximum", "this generation; sampled between operations/steps"),
                ("process_peak_rss_bytes", "process maximum RSS reported by runner", "whole process up to this record"),
            ]),
            "memory_caveat": MEMORY_NOTE,
        }
        for variant, field in (("raw", "raw"), ("exact-source-composite", "composited")):
            if record.get(field):
                result.append(paths({**common, "variant": variant, "image": record[field]}, manifest))
        if not record.get("raw") and not record.get("composited"):
            result.append(paths({**common, "variant": "attempt"}, manifest))
    return result


def dreamlite(data: dict, manifest: Path) -> list[dict]:
    shared = common_metadata(data, ("runs",))
    trial = manifest.parent.name
    latent = data.get("experimental_latent_preservation", False)
    model_id = data.get("model", data.get("model_repo", "DreamLite"))
    model_variant = "Base" if "base" in str(model_id).lower() else "Mobile" if "mobile" in str(model_id).lower() else str(data.get("variant", "unknown variant"))
    method_label = "square context 1024² → crop" if "square-context" in trial or data.get("model_work_resolution") == [1024, 1024] else "latent blend" if latent else "coastal prompt" if "coastal-prompt" in trial else "diagnostic controls" if data.get("diagnostic_control") else "instruction edit"
    label = f"DreamLite {model_variant}\n{data.get('steps', '?')} step / seed {data.get('seed', '?')}\n{method_label}"
    result = []
    for order, record in enumerate(data.get("runs", []), 1):
        common = {
            **shared, **record,
            "model": model_id,
            "model_variant": model_variant,
            "track_id": record.get("track_id", data.get("track_id", "diagnostic:" + record.get("test", "unknown"))),
            "trial_id": "dreamlite-" + trial,
            "column_label": label,
            "elapsed_seconds": record.get("generation_seconds"),
            "elapsed_seconds_scope": "generation as reported by Python runner; model loading excluded",
            "display_timing_label": "Generation (load excl.)",
            "generation_order_within_process": order,
            "status": record.get("status", "completed" if record.get("raw_path") else data.get("status", "unknown")),
            "memory_measurements": measurements(record, [
                ("peak_process_rss_bytes", "sampled process RSS maximum", "whole process up to this record"),
                ("peak_mps_allocated_bytes", "sampled MPS current allocation maximum", "whole process up to this record"),
                ("peak_mps_driver_bytes", "sampled MPS driver allocation maximum", "whole process up to this record"),
            ]),
            "memory_caveat": MEMORY_NOTE + " Sampling may miss transient peaks.",
        }
        for variant, field in (("raw", "raw_path"), ("exact-source-composite", "composite_path")):
            if record.get(field):
                result.append(paths({**common, "variant": variant, "image": record[field]}, manifest))
        if not record.get("raw_path") and not record.get("composite_path"):
            result.append(paths({**common, "variant": "attempt"}, manifest))
    if not data.get("runs"):
        result.append(paths({**shared, "trial_id": "dreamlite-" + trial, "column_label": label, "variant": "attempt", "track_id": data.get("track_id", "unknown")}, manifest))
    return result


def flux(data: dict, manifest: Path) -> list[dict]:
    result = []
    shared = common_metadata(data, ("runs",))
    for record in data.get("runs", []):
        trial = f"flux2-{record.get('model', '?')}-{record.get('steps', '?')}step-seed{record.get('seed', '?')}"
        label = f"{record.get('model', 'FLUX.2')}\n{record.get('steps', '?')} step / seed {record.get('seed', '?')}"
        mlx_unit = "MiB" if str(record.get("mlx_profile_unit", "")).startswith("MiB") else "reported MB; byte divisor unconfirmed"
        memory = measurements(record, [
            ("mlx_peak_allocated_mb", "MLX cumulative peak allocation reported by runner, rounded to profile units", "whole command, including model loading", mlx_unit),
            ("process_maximum_rss_bytes", "process maximum resident set size from /usr/bin/time -l", "whole command including loading"),
            ("process_peak_memory_footprint_bytes", "macOS process peak memory footprint from /usr/bin/time -l", "whole command including loading"),
        ])
        result.append(paths({**shared, **record, "trial_id": trial, "column_label": label,
                             "elapsed_seconds_scope": record.get("elapsed_seconds_scope", "Scope unconfirmed in source runner; inspect process wall time and phase profile before comparing with load-excluded generation timings."),
                             "display_timing_label": "Chain incl. lazy load" if "lazy" in record.get("elapsed_seconds_scope", "") else "Profiled chain",
                             "memory_measurements": memory,
                             "source_memory_caveat": record.get("memory_caveat"),
                             "memory_caveat": MEMORY_NOTE}, manifest))
    return result


def qwen(data: dict, manifest: Path) -> list[dict]:
    trial = manifest.parent.parent.name
    turbo = bool(data.get("adapter_repo"))
    adapter_rank = data.get("adapter_rank")
    if turbo and adapter_rank is None:
        match = re.search(r"(?:^|[-_])r(\d+)(?:[.\-_]|$)", str(data.get("adapter_file", "")))
        adapter_rank = int(match.group(1)) if match else None
    adapter_label = f"Viggle r{adapter_rank}" if adapter_rank is not None else "Viggle unknown rank"
    label = f"Qwen 2.1 {adapter_label if turbo else 'Base'}\n{data.get('steps', '?')} step / Q{data.get('quantization', '?')} / seed {data.get('seed', '?')}"
    known_padding = data.get("known_region_encoding_padding")
    if known_padding == "edge" or "knownedge" in trial:
        label += "\nknown-latent edge padding"
    status = "completed" if data.get("status") == "success" else data.get("status", "unknown")
    phases = data.get("phases", {})
    generation = [phases.get(key) for key in ("conditioning_seconds", "denoising_seconds", "decode_seconds")]
    elapsed = sum(generation) if status == "completed" and all(value is not None for value in generation) else None
    common = {
        **data,
        "model": data.get("model_repo", "Qwen/Qwen-Image-2.1"),
        "trial_id": trial,
        "adapter_rank": adapter_rank,
        "column_label": label,
        "track_id": data.get("track", manifest.parent.name),
        "status": status,
        "attempt_elapsed_seconds_including_setup_and_load": data.get("elapsed_seconds"),
        "elapsed_seconds": elapsed,
        "elapsed_seconds_scope": "sum of conditioning, denoising and decoding phases; loading excluded; null until successful phase timings exist",
        "display_timing_label": "Cond+generate+decode",
        "model_load_seconds": phases.get("load_seconds"),
        "memory_measurements": measurements(data, [
            ("load_mlx_peak_bytes", "MLX peak allocation", "model loading phase"),
            ("conditioning_mlx_peak_bytes", "MLX peak allocation", "conditioning phase"),
            ("denoising_mlx_peak_bytes", "MLX peak allocation", "denoising phase"),
            ("decoding_mlx_peak_bytes", "MLX peak allocation", "decoding phase"),
            ("mlx_peak_bytes", "MLX peak allocation after reset following loading", "conditioning, denoising and decoding phase"),
            ("process_peak_rss_bytes", "process maximum RSS reported by resource.getrusage", "whole attempt including loading"),
        ]),
        "memory_caveat": MEMORY_NOTE,
    }
    result = []
    # A stale output from a prior attempt must not turn a failed/currently running record into success.
    if status == "completed":
        for variant, field in (("raw", "raw_path"), ("exact-source-composite", "composite_path")):
            if data.get(field):
                result.append(paths({**common, "variant": variant, "image": data[field]}, manifest))
    if not result:
        result.append(paths({**common, "variant": "attempt", "image": None}, manifest))
    return result


def collect() -> tuple[list[dict], list[dict], list[dict]]:
    sources = []
    mobile = ROOT / "experiments/mobile_o/results/trial.json"
    if mobile.exists():
        sources.append((mobile, mobile_o))
    locked = ROOT / "experiments/mobile_o/results_latent_lock/trial.json"
    if locked.exists():
        sources.append((locked, mobile_o))
    sources.extend((path, dreamlite) for path in sorted((ROOT / "experiments/dreamlite/results").glob("*/metrics.json"), key=lambda p: ("latent" in p.parent.name, p.parent.name)))
    sources.extend((path, flux) for path in sorted((ROOT / "experiments/flux2/results").glob("*.json")))
    sources.extend((path, qwen) for path in sorted((HERE / "results").glob("qwen*/track*/metrics.json")))
    sources.extend((path, qwen) for path in sorted((ROOT / "experiments/qwen/results").glob("*/track*/metrics.json")))
    runs, read_sources, errors = [], [], []
    seen = set()
    for path, reader in sources:
        if path.resolve() in seen:
            continue
        seen.add(path.resolve())
        try:
            data = json.loads(path.read_text())
            if reader is qwen and (data.get("shared_canvas_comparable") is False or "spatial" in path.parent.parent.name):
                read_sources.append({"path": str(path), "records": 0, "excluded": True,
                                     "reason": "Separate Qwen spatial FULL/GROW diagnostic; reported outside the standard model trial aggregate."})
                continue
            normalized = reader(data, path)
            runs.extend(normalized)
            read_sources.append({"path": str(path), "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(), "records": len(normalized)})
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            errors.append({"path": str(path), "error": str(error), "note": "Possibly an in-progress write; rerun collector later."})
    return runs, read_sources, errors


def column_keys(runs: list[dict]) -> list[tuple[str, str]]:
    return list(dict.fromkeys((run["trial_id"], str(run.get("seed", "?"))) for run in runs))


def find_run(runs: list[dict], key: tuple[str, str], track: str, variant: str) -> dict | None:
    candidates = [run for run in runs if (run["trial_id"], str(run.get("seed", "?"))) == key and run.get("track_id") == track]
    return next((run for run in candidates if run.get("variant") == variant), next((run for run in candidates if run.get("variant") == "attempt"), None))


def grid_sheet(runs: list[dict], tracks: list[dict], variant: str, destination: Path) -> Image.Image:
    columns = column_keys(runs)
    label_width, tile_width, tile_height, header = 96, 244, 624, 142
    image = Image.new("RGB", (label_width + len(columns) * tile_width, header + len(tracks) * tile_height), (20, 22, 26))
    draw = ImageDraw.Draw(image)
    title = "RAW MODEL OUTPUT" if variant == "raw" else "RESIZED 512 PX SOURCE COMPOSITED AFTER GENERATION"
    draw.text((12, 12), title, font=font(20), fill=(250, 250, 250))
    draw.text((12, 42), "Apple Silicon Mac. Center equality uses source_512.png, not original 720² pixels.", font=font(13), fill=(178, 183, 193))
    for index, key in enumerate(columns):
        run = next(run for run in runs if (run["trial_id"], str(run.get("seed", "?"))) == key)
        label = run.get("column_label", run.get("model", key[0]))
        draw.multiline_text((label_width + index * tile_width + 8, 70), label, font=font(13), fill="white", spacing=4)
    for row, track in enumerate(tracks):
        y = header + row * tile_height
        draw.multiline_text((8, y + 12), track["id"] + "\n" + track["title"][:12], font=font(12), fill=(200, 204, 211), spacing=6)
        for col, key in enumerate(columns):
            x = label_width + col * tile_width
            run = find_run(runs, key, track["id"], variant)
            draw.rectangle((x + 3, y + 3, x + tile_width - 3, y + tile_height - 3), fill=(31, 34, 40))
            usable = run and run.get("status") == "completed" and run.get("variant") == variant and run.get("image") and Path(run["image"]).is_file()
            if usable:
                try:
                    frame = Image.open(run["image"]).convert("RGB")
                    frame.thumbnail((228, 514), Image.Resampling.LANCZOS)
                    image.paste(frame, (x + (tile_width - frame.width) // 2, y + 10))
                except OSError:
                    usable = False
            if not usable:
                status = "no recorded run" if run is None else "last reported: " + run.get("status", "unknown")
                draw.multiline_text((x + 12, y + 235), status + "\nNo completed output", font=font(12), fill=(166, 170, 180), spacing=6)
            if run:
                seconds = run.get("elapsed_seconds")
                timing_label = run.get("display_timing_label", "Reported time")
                summary = f"{timing_label}: {seconds:.2f}s" if seconds is not None else "Generation: unavailable"
                if run.get("model_load_seconds") is not None:
                    summary += f"\nModel load: {run['model_load_seconds']:.2f}s"
                elif run.get("load_seconds") is not None:
                    summary += f"\nModel load: {run['load_seconds']:.2f}s"
                elif run.get("pipeline_initialization_seconds") is not None:
                    summary += f"\nPre-chain setup: {run['pipeline_initialization_seconds']:.2f}s"
                preservation = run.get("source_preservation", {})
                if preservation.get("mean_absolute_rgb_error_0_to_255") is not None:
                    summary += f"\nCenter RGB MAE: {preservation['mean_absolute_rgb_error_0_to_255']:.2f}"
                draw.multiline_text((x + 10, y + 535), summary, font=font(12), fill=(186, 191, 201), spacing=4)
    image.save(destination)
    return image


def html_grid(runs: list[dict], tracks: list[dict], destination: Path) -> None:
    columns = column_keys(runs)
    sections = []
    for variant, title in (("raw", "Raw model output"), ("exact-source-composite", "512×512 reference source composited after generation")):
        heads = ["<th>Cover</th>"]
        for key in columns:
            run = next(run for run in runs if (run["trial_id"], str(run.get("seed", "?"))) == key)
            heads.append("<th>" + html.escape(run.get("column_label", run.get("model", key[0]))).replace("\n", "<br>") + "</th>")
        rows = []
        for track in tracks:
            cells = [f"<th>{html.escape(track['id'])}<br>{html.escape(track['title'])}</th>"]
            for key in columns:
                run = find_run(runs, key, track["id"], variant)
                if run and run.get("status") == "completed" and run.get("variant") == variant and run.get("image") and Path(run["image"]).is_file():
                    uri = html.escape(Path(run["image"]).as_uri())
                    seconds = run.get("elapsed_seconds")
                    timing_label = html.escape(run.get("display_timing_label", "Reported time"))
                    time = f"{timing_label}: {seconds:.2f} s" if seconds is not None else "Generation timing unavailable"
                    metrics = html.escape(json.dumps(run, ensure_ascii=False, indent=2))
                    cells.append(f'<td><a href="{uri}"><img src="{uri}" loading="lazy"></a><p>{time}</p><details><summary>Exact metadata and measurements</summary><pre>{metrics}</pre></details></td>')
                else:
                    status = "No recorded run" if run is None else "Last reported: " + run.get("status", "unknown")
                    details = html.escape(json.dumps(run, ensure_ascii=False, indent=2)) if run else ""
                    cells.append(f"<td class=missing>{html.escape(status)}<br>No completed output<details><summary>Metadata</summary><pre>{details}</pre></details></td>")
            rows.append("<tr>" + "".join(cells) + "</tr>")
        sections.append(f"<h2>{title}</h2><div class=scroll><table><thead><tr>{''.join(heads)}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>")
    markup = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>ArtWorker actual outpainting trials</title>
<style>body{{background:#14161a;color:#eee;font:14px system-ui;margin:24px}}p{{max-width:1000px;line-height:1.5}}.scroll{{overflow:auto}}table{{border-collapse:separate;border-spacing:8px}}th{{font-weight:500;text-align:left;min-width:90px}}td{{background:#24272e;padding:12px;vertical-align:top;min-width:230px;width:230px;border-radius:8px}}img{{width:230px;max-height:520px;object-fit:contain}}.missing{{color:#a7aebc;padding-top:230px}}pre{{max-width:500px;white-space:pre-wrap;overflow-wrap:anywhere;font:11px monospace}}details{{margin-top:12px}}</style>
<h1>Actual cover outpainting trials</h1><nav><a href="track3-curated.png">Curated track3 image</a> · <a href="raw_contact_sheet.png">All raw outputs</a> · <a href="composite_contact_sheet.png">All source composites</a> · <a href="report.json">Measured report JSON</a> · <a href="runs.json">Source metadata JSON</a> · <a href="validation.json">Artifact validation</a></nav><p>{html.escape(MAC_NOTE)} {html.escape(MEMORY_NOTE)}</p><p>{html.escape(SOURCE_NOTE)} Raw and source-composited outputs are kept separate. Exact source compositing proves that the reference center is preserved, not that the generated areas are coherent. Prompts and masking methods differ by runtime; this is a task-oriented smoke test, not a controlled leaderboard. Square-context crop trials also use different model work resolution and source/context geometry. Boundary metrics are only crude seam flags.</p>{''.join(sections)}</html>"""
    destination.write_text(markup)


def curated_track3(runs: list[dict], destination: Path) -> list[dict]:
    selectors = [
        ("dreamlite-mobile-square-context-phone-seed42", "DreamLite Mobile", "4 steps / 1024² square context", "448×1008 crop → 512×1152", "Scene-specific prompt; no hard mask"),
        ("dreamlite-base-28step-portrait-coastal-prompt-seed42", "DreamLite Base", "28 steps / direct portrait", "Coastal prompt; no hard mask", "512×1152 model work resolution"),
        ("flux2-FLUX.2 klein-4b-base / int4-50step-seed42", "FLUX.2 Klein Base 4B", "50 steps / transformer int4", "Reference + soft-mask latent blend", "Small decoder; scene-specific prompt"),
        ("qwen21-viggle-v021-r256-6step-knownedge-q4", "Qwen 2.1 Viggle r256", "6 steps / Q4 / direct portrait", "Known-latent edge padding", "Masked Euler; scene-specific prompt"),
    ]
    selected = []
    for trial, title, setting, method, note in selectors:
        run = next((run for run in runs if run.get("trial_id") == trial and run.get("track_id") == "track3" and run.get("variant") == "exact-source-composite" and run.get("status") == "completed"), None)
        if run is None:
            raise ValueError(f"Curated track3 candidate is not completed: {trial}")
        selected.append({"run": run, "title": title, "setting": setting, "method": method, "note": note})
    tile_width, header, image_top, footer_top, height = 320, 118, 128, 822, 1038
    sheet = Image.new("RGB", (4 * tile_width, height), (19, 21, 25))
    draw = ImageDraw.Draw(sheet)
    draw.text((14, 10), "TRACK3 — ACTUAL ALBUM-COVER EXPANSION TRIALS", font=font(19), fill="white")
    draw.text((14, 38), "Mac measurements; methods differ. The resized source_512.png center is pasted back exactly.", font=font(13), fill=(183, 190, 201))
    for col, item in enumerate(selected):
        x = col * tile_width
        run = item["run"]
        draw.text((x + 10, 72), item["title"], font=font(16), fill="white")
        draw.text((x + 10, 97), item["setting"], font=font(12), fill=(185, 191, 201))
        frame = Image.open(run["image"]).convert("RGB")
        if frame.size != (512, 1152):
            raise ValueError(f"Curated candidate output size mismatch: {run['image']}")
        frame = frame.resize((304, 684), Image.Resampling.LANCZOS)
        sheet.paste(frame, (x + 8, image_top))
        lines = [item["method"], item["note"]]
        seconds = run.get("elapsed_seconds")
        if seconds is not None:
            lines.append(f"{run.get('display_timing_label', 'Reported time')}: {seconds:.2f}s")
        if run.get("model_load_seconds") is not None:
            lines.append(f"Separate model load: {run['model_load_seconds']:.2f}s")
        elif run.get("load_seconds") is not None:
            lines.append(f"Separate model load: {run['load_seconds']:.2f}s")
        if run.get("attempt_elapsed_seconds_including_setup_and_load") is not None:
            lines.append(f"Full attempt: {run['attempt_elapsed_seconds_including_setup_and_load']:.2f}s")
        draw.multiline_text((x + 10, footer_top), "\n".join(lines), font=font(12), fill=(198, 204, 214), spacing=7)
    draw.text((14, 1008), "Center pixel equality does not prove seamless continuation. These are task trials, not a controlled model ranking.", font=font(12), fill=(166, 175, 187))
    sheet.save(destination)
    return [{"title": item["title"], "trial_id": item["run"]["trial_id"], "image": item["run"]["image"]} for item in selected]


def validate(runs: list[dict], shared: list[dict]) -> dict:
    completed = [run for run in runs if run.get("status") == "completed"]
    missing = [run for run in completed if not run.get("image") or not Path(run["image"]).is_file()]
    composites = [run for run in shared if run.get("status") == "completed" and run.get("variant") == "exact-source-composite"]
    errors = [run for run in composites if run.get("source_preservation", {}).get("identical_pixels_fraction") != 1.0]
    geometry = [run for run in shared if run.get("status") == "completed" and run.get("output_size") != [512, 1152]]
    trials = {(run["trial_id"], run.get("track_id"), run.get("seed")) for run in shared if run.get("status") == "completed"}
    return {
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "source_reference": "source_512.png; Lanczos-resized 512×512 reference, not original 720×720 pixel map",
        "shared_canvas_completed_trials": len(trials),
        "shared_canvas_completed_image_records": sum(run.get("status") == "completed" for run in shared),
        "verified_exact_source_composites": len(composites) - len(errors),
        "all_completed_image_records": len(completed),
        "distinct_completed_image_paths": len({run["image"] for run in completed if run.get("image")}),
        "diagnostic_image_records_excluded_from_grid": sum(bool(run.get("diagnostic_control") or run.get("shared_canvas_comparable") is False) for run in runs),
        "failed_or_incomplete_records": sum(run.get("status") != "completed" for run in runs),
        "missing_completed_image_paths": [run.get("image") for run in missing],
        "source_equality_failures": [{"trial_id": run["trial_id"], "track_id": run.get("track_id"), "image": run.get("image")} for run in errors],
        "shared_geometry_failures": [{"trial_id": run["trial_id"], "track_id": run.get("track_id"), "output_size": run.get("output_size")} for run in geometry],
        "quality_caveat": "No automated perceptual quality score or winner is inferred from these validation checks.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=HERE / "aggregate")
    args = parser.parse_args()
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    runs, sources, errors = collect()
    document = {"collected_utc": datetime.now(timezone.utc).isoformat(), "platform_caveat": MAC_NOTE,
                "memory_caveat": MEMORY_NOTE, "source_equality_caveat": SOURCE_NOTE, "source_manifests": sources, "read_errors": errors, "runs": runs}
    manifest = destination / "runs.json"
    write_json(manifest, document)
    compare(manifest, INPUTS, destination, max(1, len(column_keys(runs))))
    report = json.loads((destination / "report.json").read_text())
    measured = report["runs"]
    grid_runs = [run for run in measured if not run.get("diagnostic_control") and run.get("shared_canvas_comparable") is not False]
    inputs = json.loads(INPUTS.read_text())
    if grid_runs:
        raw = grid_sheet(grid_runs, inputs["tracks"], "raw", destination / "raw_contact_sheet.png")
        composite = grid_sheet(grid_runs, inputs["tracks"], "exact-source-composite", destination / "composite_contact_sheet.png")
        combined = Image.new("RGB", (max(raw.width, composite.width), raw.height + composite.height), (20, 22, 26))
        combined.paste(raw, (0, 0))
        combined.paste(composite, (0, raw.height))
        combined.save(destination / "contact_sheet.png")
    html_grid(grid_runs, inputs["tracks"], destination / "comparison.html")
    validation = validate(measured, grid_runs)
    write_json(destination / "validation.json", validation)
    validation["curated_selection"] = curated_track3(grid_runs, destination / "track3-curated.png")
    write_json(destination / "validation.json", validation)
    print(json.dumps({"manifest": str(manifest), "comparison": str(destination / "comparison.html"), "records": len(runs),
                      "completed_image_records": sum(run.get("status") == "completed" and run.get("image_exists_at_collection", False) for run in runs),
                      "distinct_completed_image_paths": len({run['image'] for run in runs if run.get("status") == "completed" and run.get("image_exists_at_collection", False)}),
                      "diagnostic_records_excluded_from_grid": sum(bool(run.get("diagnostic_control")) for run in runs),
                      "noncompleted_records": sum(run.get("status") != "completed" for run in runs), "read_errors": errors}, indent=2))


if __name__ == "__main__":
    main()
