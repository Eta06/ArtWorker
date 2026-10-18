"""Do not hand unobserved assets or unsafe release names to yt-dlp."""
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch
import download_thumbnails as downloader


class ThumbnailInputTests(TestCase):
    def test_invalid_release_or_non_cover_host_never_reaches_downloader(self):
        url='https://yt3.googleusercontent.com/observed=w1024-h1024-l90-rj'
        entries=[{'id':'../escape','url':url,'thumbnails':[{'url':url}]},
                 {'id':'MPREvalid','url':'https://example.com/unknown','thumbnails':[{'url':'https://example.com/unknown'}]},
                 {'id':'MPREvalid','url':'https://example.com/unknown','thumbnails':[{'url':url}]}]
        with patch.object(downloader,'YoutubeDL') as mock:
            for entry in entries:
                with self.assertRaises(ValueError):downloader.download(entry,Path('/unused'))
            mock.assert_not_called()


if __name__=='__main__':main()
