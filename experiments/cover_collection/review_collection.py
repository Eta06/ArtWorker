#!/usr/bin/env python3
"""Audit a private cover collection and build a paginated local review page."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from collect_covers import ROOT, atomic_json, fingerprint


def build(root: Path) -> dict:
    root = root.resolve()
    if not root.is_relative_to(ROOT / "experiments/cover_collection/results"):
        raise ValueError("Use the ignored local collection directory")
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["status"] != "target_reached" or len(manifest["covers"]) != manifest["target"]:
        raise ValueError("Do not mark a partial or running collection complete")
    visits = json.loads((root / "visits.json").read_text())
    by_page = {visit["url"]: visit["artist"] for visit in visits if visit.get("artist")}
    candidates = {c["album_url"]: c for c in json.loads((root / "candidates.json").read_text())}
    thumbs = root / "thumbs"
    thumbs.mkdir(exist_ok=True)
    byte_hashes, pixel_hashes, source_ids = set(), set(), set()
    dimensions = Counter()
    groups = defaultdict(deque)
    covers = manifest["covers"]
    for record in covers:
        local = (root / record["file"]).resolve()
        if not local.is_relative_to(root / "covers"):
            raise ValueError("Unsafe local image path")
        digest = hashlib.sha256(local.read_bytes()).hexdigest()
        if digest != record["sha256"] or digest in byte_hashes:
            raise ValueError(f"Corrupt or duplicate bytes: {record['id']}")
        byte_hashes.add(digest)
        if record["album_id"] in source_ids:
            raise ValueError("Repeated album release")
        source_ids.add(record["album_id"])
        with Image.open(local) as image:
            image.load()
            if list(image.size) != record["dimensions"] or image.width != image.height or image.width < 512:
                raise ValueError(f"Unexpected image dimensions: {record['id']}")
            pixel_hash, _, _ = fingerprint(image)
            if pixel_hash != record["normalized_rgb_sha256"] or pixel_hash in pixel_hashes:
                raise ValueError(f"Duplicate or changed decoded pixels: {record['id']}")
            pixel_hashes.add(pixel_hash)
            dimensions[f"{image.width}x{image.height}"] += 1
            ImageOps.exif_transpose(image).convert("RGB").resize((192, 192), Image.Resampling.LANCZOS).save(thumbs / f"{record['id']}.jpg", quality=85)
        if not record.get("album"):
            record["album"] = candidates.get(record["album_url"], {}).get("album", "")
        record["catalog_artist"] = by_page.get(record["discovery_page"], "")
        names = [a["name"] for a in record.get("artists", []) if a.get("name")]
        record["display_artist"] = " · ".join(names) or record["catalog_artist"] or "Sanatçı bilgisi yok"
        record["thumbnail"] = f"thumbs/{record['id']}.jpg"
        # Round-robin across observed artist labels; this is diversity sampling,
        # not a visual-quality or redistribution-eligibility judgement.
        groups[names[0] if names else record["catalog_artist"] or "unknown"].append(record)
    pilot = []
    while len(pilot) < min(100, len(covers)):
        for queue in groups.values():
            if queue and len(pilot) < min(100, len(covers)):
                pilot.append(queue.popleft())
    pilot_ids = {record["id"] for record in pilot}
    for record in covers:
        record["pilot_candidate"] = record["id"] in pilot_ids
    summary = {
        "stage": manifest["stage"], "verified_at": datetime.now(timezone.utc).isoformat(),
        "target": manifest["target"], "downloaded": len(covers), "all_files_verified": True,
        "unique_album_release_ids": len(source_ids), "unique_file_sha256": len(byte_hashes),
        "unique_normalized_rgb_sha256": len(pixel_hashes), "dimensions": dict(dimensions),
        "total_bytes": sum(record["bytes"] for record in covers),
        "candidate_album_links": len(candidates),
        "downloadable_candidate_links": sum(c.get("observed_image_url", "").startswith("https://") for c in candidates.values()),
        "visited_pages": len(visits), "distinct_catalog_artists_visited": len({v['artist'] for v in visits if v.get('artist')}),
        "artist_groups": len(groups), "pilot_count": len(pilot),
        "pilot_artist_groups": len({r['display_artist'] for r in pilot}),
        "raw_observation_status_counts": dict(Counter(r["status"] for r in manifest["attempts"])),
        "placeholder_observations": sum(not r.get("asset_identity", "").startswith("https://") for r in manifest["attempts"]),
        "real_duplicate_asset_observations": sum(r["status"] == "duplicate_asset" and r.get("asset_identity", "").startswith("https://") for r in manifest["attempts"]),
        "pixel_duplicate_observations": sum(r["status"] == "duplicate_pixels" for r in manifest["attempts"]),
        "download_errors": [r for r in manifest["attempts"] if r["status"] == "error" and r.get("asset_identity", "").startswith("https://")],
        "api_charge_usd": 0, "music_downloaded": 0,
        "native_detail": "CDN dimensions verified; original uploaded detail is not known",
        "training_status": "unexpanded, unreviewed source candidates; no teacher calls or training",
        "rights_status": manifest["rights_status"],
    }
    atomic_json(root / "audit.json", summary)
    atomic_json(root / "manifest.json", manifest)
    atomic_json(root / "pilot-100.json", {"stage": manifest["stage"], "selection": "round-robin observed artist groups; candidates only", "covers": pilot})
    data = [{k: r[k] for k in ("id", "album", "display_artist", "file", "thumbnail", "album_url", "dimensions", "pilot_candidate")} for r in covers]
    atomic_json(root / "gallery-data.json", data)
    dimension_text = " · ".join(f"{count} kapak {size.replace('x', '×')}" for size, count in dimensions.items())
    page = PAGE.replace("__COUNT__", str(len(covers))).replace("__SIZE__", f"{summary['total_bytes'] / 1024**2:.1f}").replace("__DIMENSIONS__", dimension_text)
    (root / "index.html").write_text(page)
    sheet = Image.new("RGB", (8 * 180, 6 * 210), "#101217")
    draw = ImageDraw.Draw(sheet)
    for index, record in enumerate(pilot[:48]):
        x, y = (index % 8) * 180, (index // 8) * 210
        with Image.open(root / record["thumbnail"]) as image:
            sheet.paste(image.resize((168, 168), Image.Resampling.LANCZOS), (x + 6, y + 6))
        draw.text((x + 6, y + 178), record["id"], fill="#eeeeee")
    sheet.save(root / "overview.jpg", quality=92)
    return summary


PAGE = r'''<!doctype html><html lang="tr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ArtWorker · Defne kapak koleksiyonu</title><style>
*{box-sizing:border-box}body{margin:0;background:#111318;color:#f3f4f7;font:15px system-ui}main{max-width:1440px;margin:auto;padding:36px 28px}header{display:flex;align-items:end;justify-content:space-between;gap:24px;flex-wrap:wrap}h1{font-size:32px;margin:6px 0 12px;letter-spacing:-1px}p{color:#aab1bd;line-height:1.6}.label{color:#e5a9a0;font-size:12px;letter-spacing:2px}.tools{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:26px 0 18px}input,select,button,.link{color:inherit;background:#22262e;border:1px solid #39404c;border-radius:10px;padding:12px 16px;font:inherit}input{flex:1;min-width:210px}button{cursor:pointer}button:disabled{opacity:.4;cursor:default}a{color:#f0beb6}#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:22px 16px}.card{margin:0;min-width:0}.card .art{display:block;padding:0;border:0;background:transparent;width:100%;text-align:left}.card img{width:100%;aspect-ratio:1;object-fit:contain;border-radius:9px;background:#22262e}.card h2{font-size:14px;line-height:1.4;margin:10px 0 5px}.card p{font-size:12px;margin:0;overflow-wrap:anywhere}.tag{color:#e5a9a0;font-size:11px}.paging{display:flex;justify-content:center;gap:20px;align-items:center;margin-top:32px}dialog{color:inherit;background:#181b22;border:1px solid #525966;border-radius:16px;max-width:min(900px,95vw);max-height:95vh;padding:20px}dialog::backdrop{background:#000b}dialog img{display:block;max-width:100%;max-height:68vh;margin:16px auto}#details{line-height:1.6}.notice{font-size:12px;color:#87909f}#count{color:#bdc3ce} @media(max-width:600px){main{padding:24px 16px}h1{font-size:26px}#grid{grid-template-columns:repeat(2,1fr);gap:18px 12px}}
</style><main><header><div><div class="label">ARTWORKER / DEFNE</div><h1>1.000 kapak. Yeni başlangıç.</h1><p>__COUNT__ doğrulanmış, farklı kapak · __SIZE__ MiB · Üretim harcaması $0</p></div><a class="link" href="pilot-100.json" download>İlk 100 adayın listesi ↓</a></header><p class="notice">Orijinal kapaklar henüz genişletilmedi. İlk 100 listesi sanatçı çeşitliliği için hazırlanmış aday seçimidir; kalite onayı değildir.</p><div class="tools"><input id="search" aria-label="Kapak ara" placeholder="Albüm, sanatçı veya kapak numarası ara"><select id="scope" aria-label="Koleksiyon filtresi"><option value="all">Tüm kapaklar</option><option value="pilot">İlk 100 aday</option></select><span id="count"></span></div><section id="grid" aria-label="Albüm kapakları"></section><div class="paging"><button id="prev">Önceki</button><span id="page"></span><button id="next">Sonraki</button></div><p class="notice">Dosyalar yerel koleksiyonda tutulur. __DIMENSIONS__. Yüklenen orijinalin ayrıntı çözünürlüğü bilinmiyor.</p></main><dialog id="viewer"><button id="close">Kapat</button><img id="full" alt=""><div id="details"></div><a id="source" target="_blank" rel="noopener">YouTube Music albüm sayfası ↗</a></dialog><script>
let records=[],page=0;const perPage=48;const $=s=>document.getElementById(s);const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));function filtered(){const q=$('search').value.toLocaleLowerCase('tr');return records.filter(r=>($('scope').value!=='pilot'||r.pilot_candidate)&&`${r.album} ${r.display_artist} ${r.id}`.toLocaleLowerCase('tr').includes(q))}function render(){const all=filtered(),pages=Math.max(1,Math.ceil(all.length/perPage));page=Math.min(page,pages-1);$('count').textContent=`${all.length} kapak`;$('page').textContent=`${page+1} / ${pages}`;$('prev').disabled=page===0;$('next').disabled=page===pages-1;$('grid').innerHTML=all.slice(page*perPage,(page+1)*perPage).map(r=>`<article class="card"><button class="art" data-id="${esc(r.id)}" aria-label="${esc(r.album||r.id)} kapağını aç"><img loading="lazy" src="${esc(r.thumbnail)}" alt="${esc(r.album||r.id)}"></button><h2>${esc(r.album||r.id)}</h2><p>${esc(r.display_artist)}</p><span class="tag">${esc(r.id)}${r.pilot_candidate?' · İlk 100':''}</span></article>`).join('')}$('grid').addEventListener('click',e=>{const b=e.target.closest('[data-id]');if(!b)return;const r=records.find(r=>r.id===b.dataset.id);$('full').src=r.file;$('full').alt=r.album;$('details').textContent=`${r.album||r.id} · ${r.display_artist} · ${r.dimensions.join('×')}`;$('source').href=r.album_url;$('viewer').showModal()});$('close').onclick=()=>$('viewer').close();$('prev').onclick=()=>{page--;render()};$('next').onclick=()=>{page++;render()};$('search').oninput=$('scope').onchange=()=>{page=0;render()};fetch('gallery-data.json').then(r=>{if(!r.ok)throw Error('Koleksiyon yüklenemedi');return r.json()}).then(r=>{records=r;render()}).catch(e=>{$('count').textContent=e.message});
</script></html>'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.collection), ensure_ascii=False, indent=2))
