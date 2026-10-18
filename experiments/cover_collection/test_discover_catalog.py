"""Public-catalog provenance controls; no network or account access."""
from unittest import TestCase, main
from unittest.mock import patch
import discover_catalog as catalog


class CatalogTests(TestCase):
    def test_expanded_discography_uses_observed_section_not_video_channel(self):
        class Fake:
            def __init__(self,**kwargs):pass
            def search(self,*args,**kwargs):return [{'artist':'Dua Lipa','browseId':'UCmusic'}]
            def get_artist(self,*args):return {'channelId':'UCvideo','singles':{'browseId':'MPADmusic','params':'observed-params','results':[]}}
            def get_artist_albums(self,browse,params,**kwargs):
                assert browse=='MPADmusic' and params=='observed-params'
                return [{'browseId':'MPRErelease','title':'Colorful single','thumbnails':[{'url':'https://yt3.googleusercontent.com/observed=w544-h544-l90-rj','width':544}]}]
        with patch.object(catalog,'YTMusic',Fake):result=catalog.discover('Dua Lipa')
        self.assertEqual(result['status'],'observed')
        self.assertEqual(result['releases'][0]['album_url'],'https://music.youtube.com/browse/MPRErelease')
        self.assertEqual(result['releases'][0]['artists'][0]['name'],'Dua Lipa')
        self.assertEqual(result['releases'][0]['origin'],'public_artist_catalog')

    def test_ambiguous_other_artist_is_not_silently_accepted(self):
        class Fake:
            def __init__(self,**kwargs):pass
            def search(self,*args,**kwargs):return [{'artist':'A tribute to Dua Lipa','browseId':'UCwrong'}]
        with patch.object(catalog,'YTMusic',Fake):result=catalog.discover('Dua Lipa')
        self.assertEqual(result['status'],'artist_not_matched');self.assertEqual(result['releases'],[])


if __name__=='__main__':main()
