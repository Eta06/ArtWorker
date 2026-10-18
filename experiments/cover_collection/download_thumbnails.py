"""Bounded parallel yt-dlp thumbnail-only I/O in one Python process.

Use yt-dlp==2026.08.19, matching the previously validated local executable.
No browser cookie access, media downloads, or arbitrary extractor traversal.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from pathlib import Path
import re

from yt_dlp import YoutubeDL
from collect_covers import image_request_url


def download(entry,output):
    key=entry.get('id','')
    if not re.fullmatch(r'MPRE[a-zA-Z0-9_-]+',key):raise ValueError('Invalid observed release ID')
    thumbnails=entry.get('thumbnails',[])
    if len(thumbnails)!=1:raise ValueError('Expected exactly one observed cover')
    if entry.get('url')!=thumbnails[0]['url']:raise ValueError('Image and thumbnail URL must agree')
    image_request_url(thumbnails[0]['url'],1024)
    options={'skip_download':True,'writethumbnail':True,'quiet':True,'no_warnings':True,
             'outtmpl':str(output/'%(id)s.%(ext)s'),'socket_timeout':20,'retries':1,
             'overwrites':False,'cachedir':False,'noprogress':True}
    try:
        # These are complete image info records; do not allocate every media extractor.
        with YoutubeDL(options,auto_init=False) as ydl:ydl.process_ie_result(entry,download=True)
        return (output/(key+'.jpg')).is_file()
    except Exception as error:
        print(json.dumps({'id':key,'error':type(error).__name__}),flush=True)
        return False


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--info',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if not 1<=args.workers<=4:parser.error('At most four thumbnail workers')
    entries=json.loads(args.info.read_text())
    if len(entries)>50:parser.error('At most 50 entries per bounded batch')
    args.output.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results=list(executor.map(lambda entry:download(entry,args.output),entries))
    print(json.dumps({'thumbnail_downloads':sum(results),'entries':len(entries)}),flush=True)


if __name__=='__main__':main()
