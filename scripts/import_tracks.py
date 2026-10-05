"""Import local MP3s and embedded covers without changing the originals."""
import json
import pathlib
import shutil
import subprocess
import sys

source = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else '~/test-player').expanduser()
out = pathlib.Path(__file__).resolve().parents[1] / 'Player' / 'Resources'
out.mkdir(parents=True, exist_ok=True)
tracks = []
paths = sorted(source.glob('*.mp3'))
if len(paths) != 3:
    raise SystemExit('This prototype project expects exactly three MP3 files.')
for i, path in enumerate(paths, 1):
    info = json.loads(subprocess.check_output(['ffprobe', '-v', 'quiet', '-show_format', '-of', 'json', str(path)]))
    tags = {key.lower(): value for key, value in info['format'].get('tags', {}).items()}
    name = f'track{i}'
    shutil.copy2(path, out / f'{name}.mp3')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(path), '-an', '-frames:v', '1', str(out / f'{name}.jpg')], check=True)
    parts = path.stem.split(' - ', 1)
    artist, title = parts if len(parts) == 2 else ('Unknown artist', parts[0])
    tracks.append(dict(id=name, title=tags.get('title', title), artist=tags.get('artist', artist), album=tags.get('album', 'Local collection'), duration=float(info['format']['duration'])))
(out / 'tracks.json').write_text(json.dumps(tracks, ensure_ascii=False, indent=2))
print(f'Imported {len(tracks)} tracks into {out}')
