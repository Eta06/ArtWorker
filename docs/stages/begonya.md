# Begonya — 1.000 yeni, renkli ve çeşitli kaynak kapak

18 Ekim 2026. Kullanıcı Dua Lipa ve Ariana Grande gibi farklı sanatçılardan
daha canlı ve çekici 1.000 yeni kapak istedi. Bu aşama yalnız kaynak keşfi ve
kapak indirmesidir; teacher üretimi ve öğrenci eğitimi başlatılmadı.

## Sonuç

- 177 sanatçı kataloğundan 1.000 yeni kaynak kapak indirildi.
- Dua Lipa ve Ariana Grande'den dörder yeni kapak var. Pop, K-pop, alternatif
  pop, elektronik, disko ve Latin katalogları seçkiye yayıldı.
- Sanatçı başına üst sınır 12 olarak ayarlandı; gerçekleşen en yüksek sayı 9.
- 998 dosya 1024×1024, biri 1000×1000 ve biri 900×900. Bunlar CDN yanıtının
  boyutlarıdır; orijinal yüklemenin doğal ayrıntı çözünürlüğü bilinmiyor.
- Toplam 220.919.626 bayt, yaklaşık 210,69 MiB.

İki önceki 1.000'er kapaklık Defne ve Erguvan arşivi ile mevcut kare kaynaklar
birlikte indekslendi. Toplam 3.443 dosya, tekrar eden kaynak kopyalarını da
içerir; 3.443 farklı albüm anlamına gelmez. Yeni kabul edilen dosyalarda eski
arşivle veya kendi aralarında tanımlı yöntemle tekrar tespit edilmedi.

## Keşif ve seçim

233 sanatçılık özel bir başlangıç listesi hazırlandı. 180 katalog sorgusu
yeterli havuzu sağladı: 179 sanatçı eşleştirildi, bir ad tam eşleşmediği için
otomatik kabul edilmedi. Kalan 53 başlangıç adı bu turda sorgulanmadı.
Sanatçı, albüm/single kartı, MPRE sürüm kimliği ve gözlenen Google kapak CDN
URL'si kamuya açık YouTube Music katalog yanıtlarından kaydedildi. Hesap
çerezleri veya beğeni listesi değiştirilmedi. Gözlenen havuzda 10.142 farklı
albüm/single bağlantısı var; bu sayı indirilen kapak sayısı değildir.

Keşif ytmusicapi 1.12.3 ile yapıldı. Sanatçı adının normalize tam eşleşmesi
aranır; tribute veya farklı sanatçı sonucu sessizce kabul edilmez. Genişletilmiş
diskografi için gözlenen bölümün MPAD browseId/params çifti kullanılır.
İlk küçük denemede video channelId kullanılması ayrıştırma hatası üretti;
bölüm kimliğine geçilerek düzeltildi. Kaynaklar:
[ytmusicapi katalog belgeleri](https://ytmusicapi.readthedocs.io/en/stable/reference/browsing.html),
[yt-dlp projesi](https://github.com/yt-dlp/yt-dlp).

Sanatçı havuzları sırayla iç içe geçirilir. Piksel düzeyindeki renk ön filtresi
colorfulness ≥25 ve yeterince aydınlık/doygun piksel oranı ≥%8 arar. Bu bir
güzellik modeli veya insan kalite onayı değildir. Fotoğraf, illüstrasyon, neon,
grafik tasarım ve farklı paletlerin birlikte kalması amaçlandı.

Sürüm kimliği, CDN varlık kimliği, normalize RGB SHA-256, pHash/dHash ve RGB
MAE/p95 ile tekrar kontrolü uygulanır. Bu tutucu bir benzerlik sezgisidir;
anlamsal olarak aynı tasarımın her kırpımını kesin yakalama iddiası yoktur.
Yalnız yeni indirilmiş, dışlanan kopyalar staging alanından kaldırıldı;
önceki arşivlerin görüntüleri değiştirilmedi.

Son gözlem günlüğü 1.000 kabul edilmiş indirme, 183 eski sürüm, 47 eski varlık,
432 bekleyen varlık takma adı, 57 piksel/algısal tekrar, 303 renk filtresi
dışlaması ve bir kare/boyut/ayrıntı denetimi hatası içerir. Bu satırların
hepsi HTTP isteği değildir ve ücretli üretim sayısı olarak yorumlanmamalıdır.

## İndirme ve bellek düzeltmeleri

yt-dlp 2026.08.19 thumbnail-only modu kullanıldı; ses/video indirilmedi.
İlk seri CLI işi 232 kapakta kalıcı kayıt alındıktan sonra daha hızlı işçi
kipine geçmek için kontrollü durduruldu. Dört sınırlı I/O işçisi tek Python
indirme sürecinde çalışır; görsel kod çözme ve tekrar karşılaştırması ana
süreçte sırayla yapılır. Tamamlanmış görüntü bilgi kayıtları için gereksiz
medya extractor başlatılması kapatıldı. Her istek gözlenen Google kapak URL'si
ve güvenli MPRE dosya kimliğiyle sınırlandırılır; genel medya indirme açılmaz.

İlk üç işçili katalog keşfi 129 kayıt civarında 512 MiB örnekleme sınırında
durduruldu. 40 sanatçılık ilk devam da 140 kayda ulaştığında bellek sınırında
durdu. Sınır yükseltilmedi. JSON kayıtları tüm metni bellekte birleştirmek
yerine parça parça yazılacak şekilde değiştirildi; keşif tek işçiye indirildi.
Son 40 sanatçılık devam 79,56 saniye, 116,14 MiB RSS / 93,41 MiB fiziksel
footprint ile bitti. İki önceki koruma duruşu ve kayıtları korunur.

İlk seri indirme işi 237,66 saniye, 312,00 MiB RSS / 259,13 MiB footprint;
dört işçili tamamlayan devam 308,33 saniye, 394,42 MiB RSS / 380,88 MiB
footprint ölçtü. Bu süreler başlangıç indeksleme ve tekrar kontrollerini de
içerir; eşit koşullu indirme hızı benchmark'ı değildir. RSS ve footprint
örtüşür, toplanmaz. Koruma 100 ms örnekleme ile süreç grubunu, sistem basıncını
ve swap artışını izledi. Örnekleme ani tepeyi önleyen çekirdek sınırı değildir;
ilk iki işin son örneği durdurma eşiğini hafif aştı. Sistem basıncı normal
kaldı; başlangıçta var olan swap sıfır değildi, işlerde artış görülmedi.

## Doğrulama ve yerel galeri

Son bağımsız tekrar denetimi bütün 1.000 dosyayı yeniden açtı; bayt SHA-256,
normalize piksel özeti, boyutlar ve arşivler arası karşılaştırma geçti.
62,90 saniye, 181,58 MiB RSS / 143,80 MiB footprint ölçüldü. Galeri derleme
12,07 saniye, 117,23 MiB RSS / 104,69 MiB footprint ile tamamlandı.

/begonya/ altında arama, büyütme, kaynak YouTube Music albüm bağlantısı ve
sayfalama sunulur. Sanatçılar arasında dolaşan ilk 100 listesi estetik onay
değildir. İlk 36, son 48'lik pilot ve 24 eşit aralıklı örnek yerel paftalarda
görsel incelendi; örnekler örtüşebilir, 1.000 kapağın tamamına insan estetik
onayı verilmedi. uniqueness-audit.json ve uniqueness-overview.jpg korunarak
galeri özeti ve pilot paftası ayrı hazırlandı.

Altı çevrimdışı test, Python/JavaScript sözdizimi ve git diff --check geçti.
Galeri, JSON, önizleme, ilk/son kaynak dosyası ve tekrar denetimi HTTP
200/no-store döndürdü. Önceki Browser Use URL politika engeli nedeniyle canlı
tarayıcı tıklama/yerleşim doğrulaması yapılmadı. Kaynak görsel incelemesi
tarayıcı testi değildir. Eski 390 eğitim adayının kararlarına yazılmadı.

Yeni teacher isteği: 0. Yeni üretim maliyeti: $0. Ses/video indirmesi: 0.
Yeni alt ajan: 0. Öğrenci eğitimi veya checkpoint: 0.

## Yayın ve sonraki adım

Kod, testler ve bu kayıt dosya başına bir commit, tek push ve RAM/basınç
denetimiyle yayımlanır. Kapaklar, sanatçı başlangıç listesi, gözlemler ve
manifestler özel results/.build dizinlerinde kalır. ArtWorker Space kaydı
güncellenir. Gelecek üretimde yeni kaynaklar aşama adıyla adlandırılmalı;
eski cover kimlikleriyle çakıştırılmamalı. Bu toplama aşaması üretim kuyruğunu
genişletmez veya ücretli model çağrısı başlatmaz.
