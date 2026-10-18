"""Resume public YouTube Music release discovery; never download songs or infer art.

Requires ytmusicapi==1.12.3, installed in an ignored dependency directory.
A bounded catalog worker returns album-release cards and their observed CDN
URLs. No account, cookies, liked-list edits, or generation-provider keys are used.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import argparse
import gc
import json
from pathlib import Path
import re
import time
import unicodedata

import requests
from ytmusicapi import YTMusic
from collect_covers import ROOT, safe_candidate, image_request_url


def atomic_json(path,value):
    # Stream JSON chunks instead of allocating whole multilingual snapshots.
    temp=path.with_suffix('.tmp')
    with temp.open('w') as stream:json.dump(value,stream,ensure_ascii=False,separators=(',',':'))
    temp.replace(path)


def normalized(value):
    return re.sub(r'[^a-z0-9]', '', unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode().lower())


def discover(name):
    session=requests.Session()
    session.request=partial(session.request, timeout=25)
    api=YTMusic(requests_session=session, language='en', location='US')
    rows=[];errors=[]
    try:
        matches=api.search(name, filter='artists', limit=3)
        chosen=next((r for r in matches if normalized(r.get('artist',''))==normalized(name)),None)
        if not chosen:return {'artist':name,'status':'artist_not_matched','releases':[]}
        channel=chosen['browseId'];page=api.get_artist(channel)
        for kind in ('albums','singles'):
            section=page.get(kind,{}) or {}
            releases=list(section.get('results',[]))
            if section.get('browseId') and section.get('params'):
                try:
                    # This is the section MPAD browseId, not the video channelId.
                    releases+=api.get_artist_albums(section['browseId'],section['params'],limit=120)
                except Exception as error:errors.append({'section':kind,'error':type(error).__name__})
            for release in releases:
                browse=release.get('browseId','')
                thumbs=release.get('thumbnails',[])
                if not browse.startswith('MPRE') or not thumbs:continue
                image=max(thumbs,key=lambda t:t.get('width',0))['url']
                row=dict(album=release.get('title',''),album_url='https://music.youtube.com/browse/'+browse,
                    artists=[{'name':name,'id':channel}],catalog_artist=name,kind=kind,
                    observed_image_url=image,discovery_page='https://music.youtube.com/channel/'+channel,
                    origin='public_artist_catalog',context='vivid-diverse-artwork-selection',year=release.get('year'))
                try:safe_candidate(row);image_request_url(image,1024)
                except ValueError:continue
                rows.append(row)
        return {'artist':name,'status':'observed','channel_id':channel,'releases':rows,'errors':errors}
    except Exception as error:
        return {'artist':name,'status':'error','error':type(error).__name__,'releases':rows}
    finally:session.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artists',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-artists',type=int,default=40,help='Bound one process; rerun to resume next group')
    args=parser.parse_args();out=args.output.resolve()
    if not out.is_relative_to(ROOT/'experiments/cover_collection/results'):parser.error('Use ignored cover results')
    out.mkdir(parents=True,exist_ok=True)
    checkpoint=out/'catalog-discovery.json'
    observed=json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
    artists=json.loads(args.artists.read_text())
    def save():
        atomic_json(checkpoint,observed)
        # Round-robin artists and album/single ranks, not a single-artist block.
        pools=[];seen=set()
        for artist in artists:
            entries=[]
            for row in observed.get(artist,{}).get('releases',[]):
                identity=row['album_url']
                if identity in seen:continue
                seen.add(identity);entries.append(row)
            pools.append(entries)
        candidates=[pool[i] for i in range(max(map(len,pools),default=0)) for pool in pools if i<len(pool)]
        atomic_json(out/'candidates.json',candidates)
        atomic_json(out/'visits.json',[{'url':'https://music.youtube.com/channel/'+r['channel_id'],'artist':name}
            for name,r in observed.items() if r.get('channel_id')])
        print(json.dumps({'artists_completed':len(observed),'artists_target':len(artists),'release_candidates':len(candidates)}),flush=True)
    if not 1<=args.max_artists<=50:parser.error('Use at most 50 artists per process')
    remaining=[a for a in artists if a not in observed][:args.max_artists]
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures={executor.submit(discover,a):a for a in remaining}
        for future in as_completed(futures):
            result=future.result();observed[futures[future]]=result;save()
            del futures[future]
            gc.collect()
    save()


if __name__=='__main__':main()
