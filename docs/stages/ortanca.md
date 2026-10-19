# Ortanca — kaynak şeklinin değişmesi için ayrı denetim

19 Ekim 2026. Kullanıcı Nilüfer'in ilk Dua Lipa çıktısının yamulup
yamulmadığını ve eski onaylı / yeni sette benzer sorunlar olup olmadığını
sordu. Yeni görsel üretilmedi, provider çağrısı yapılmadı; insan kararları,
crop seçimleri, kaynaklar ve önceki fidelity dosyaları değiştirilmedi.

## Görsel bulgu ve önceki testin sınırı

nilufer-cover0001-sunburst, Radical Optimism (Extended Versions): kaynak,
aligned ve gerçek raw dosyaları açıldı. Kol/gövde geometrisindeki fark raw
dosyasında da görülüyor; yalnız önizleme boyutlandırması değildir. Önceki
test 181 inlier, %16,0 kaynak feature hull ve hizalı SSIM 0,479 vermişti.
Bu değerler şekil bozulmasının yüzdesi değildir.

Eski kaynak bulucu global similarity transform kullanıyor: kaydırma,
döndürme ve tek ölçek. Ayrı yatay/dikey ölçek veya shear ölçmüyor. Kaynak
bulundu etiketi şeklin korunduğu anlamına gelmez. Kaymanın hata sayılmaması
kuralı korunmalı. Düşük SSIM de tek başına yamulma kanıtı değil: eski
Platinum Hits örneğinde renk/grain değişimi de skoru düşürüyor.

## Kapsam ve salt okunur ön kontrol

Güncel Sümbül karar dosyasında hash ile eşleşen 390 kabul, 59 ret, 3 kararsız
var. Onaylı 390 çıktı ve yeni galerideki 638 çıktı, toplam 1.028 çift ayrı
manifestte donduruldu. PNG başlıkları ve boyutları eksiksiz; bütün girdi
source.png dosyaları 720x720. Bu dosya/boyut kontrolü şekil denetimi değildir.

Begonya'daki 1.000 indirilen kaynağın tamamı manifest SHA-256 değerleriyle
eşleşti. İndirmeden sonra yerel değişiklik saptanmadı; checksum upstream
görselin estetik veya geometrik doğruluğunu kanıtlamaz. Üretim indirme kodu
ham çıktıyı resize etmeden kaydediyor; giriş hazırlığı eşit kenarlı crop'u
720x720'ye ölçekliyor. Yeni analiz bu dosyalara yazmıyor.

## Yeni şekil taraması

deformation_audit.py karşılıklı SIFT eşleşmeleri ve RANSAC full-affine
modeli kullanır. Aynı affine inlier noktalarına en iyi similarity fit de
hesaplanır. Singular-value oranı, eksen ölçek oranı ve eksenler arası açı
ayrıca raporlanır. Konum, dönme ve eşit ölçek hata değildir.

En az 20 inlier, %35 inlier oranı, %20 hull, üç quadrant ve en çok iki
analiz pikseli median residual gereklidir. Desteksiz örnek belirsiz kalır.
En az %5 anisotropy veya 3 derece shear, ayrıca fit iyileşmesi gerektiren
bir şüphe eşiğidir; estetik karar veya otomatik eğitim reddi değildir.
%2,5-5 arası küçük değişimler sınırda raporlanır. Lokal yüz/anatomi/yazı
değişimleri bu global testle kesin olarak dışlanamaz; görsel kontrol gerekir.

Dört saf matematik testi geçti: konum/dönme/eşit ölçeği izinli bırakma,
unequal scale/shear yakalama, zayıf eşleşmeyi belirsiz tutma ve küçük sayısal
gürültüyü bozulma saymama. Gerçek feature zinciri için üç sentetik görsel
testi hazır; RAM normale dönmeden çalıştırıldığı iddia edilmez.

## Çalıştırma durumu ve koruma

Sistem bellek basıncı sürekli 2. Ağır OpenCV analizi henüz başlamadı; tüm
1.028 görüntü temiz veya bozuk ilan edilmedi. Özel bekleyici üç normal
basınç örneğinden sonra global ağır iş kilidi altında, 512 MiB RSS/footprint,
swap ve basınç korumalarıyla önce sentetik feature testlerini, sonra Dua
Lipa pilotunu ve tüm taramayı çalıştıracak. Basınçta durursa tamamlanmış
analiz dosyaları atlanarak devam eder; başka hata sessizce tekrar denenmez.
En fazla altı saat bekler. Üretimle global kilit paylaşılır, alt ajan yok.

Özel raporlar experiments/fal_outpaint/results/2026-10-19/ortanca/ altında:
manifest, envanter, indirme bütünlüğü, run-status; ilerledikçe measurements
ve summary. Ham dosyalar onarılmaz/warp edilmez. Mevcut insan kabulü ayrı
kalır; detector sonucu bir insan kararı gibi kaydedilmez.

Kod, iki test dosyası ve bu kayıt dört ayrı dosyadır. Git yayını yüksek
bellek basıncı nedeniyle bekliyor; kodun yayımlandığı veya taramanın
tamamlandığı iddia edilmez. Önceki Yasemin değişiklikleri de korunuyor.

## Kullanıcı düzeltmesi: yalnız yerel Mac

Mac basıncı normale dönmeyince açık izin almadan RyzenServer'a SSH ile
bakıldı ve izole geçici Python ortamı hazırlandı. Kullanıcı bunu açıkça
reddetti; bu seçim hataydı. Görsel aktarımı veya uzakta görsel analizi
başlamadı. Oluşturulan geçici dizin/ortam kaldırıldı; kontrolde sıfır görsel
dosyası ve o dizine bağlı sıfır çalışan süreç vardı. Bundan sonraki bu
tarama yalnız kullanıcının Mac'inde yapılır; başka sunucu kullanılmaz.
Eski yerel bekleyici yeniden açıldı. Bu sırada Mac basıncı 1'e döndü.
Global ağır iş kilidi Nilüfer'in 650 hedefindeki son üretimce tutulduğu
için ilk yerel analiz başlangıcı kilitte reddedildi; paralel ağır iş
açılmadı. Yerel bekleyici üretim kilidinin serbest kalmasını bekliyor.

## Tamamlanan yerel tarama

Üretim 650 başarılı çıktıda durdu. Son 12 dosya da manifeste eklendi;
390 eski onay + 650 yeni çıktı, toplam 1.040 çift tamamen Mac üzerinde
tarandı. İlk salt global test 512 MiB grup sınırında durdu. Genel testin
yanına, affine hizalamadan sonra çift yönlü tutarlılığı ölçülen DIS optical
flow ile yerel yapı değişimi kontrolü eklendi. Sadece küçük bir bölgenin
büküldüğü sentetik kontrol ilk median eşiğini geçemedi; alanın yarısının
değişmesini gerektiren median şartı kaldırıldı. Renk/ton değişimi kontrolü
bükülme sayılmıyor. Dört matematik + beş gerçek feature/flow testi geçti.

Analiz uzun kenarı 960 ile sınırlandı. 25'lik alt süreçler de bazı farklı
oranlarda bellek sınırına ulaştı ve koruma durdurdu. Sınır yükseltilmeden
beşlik alt süreçlere geçildi; ölçülmüş dosyalar tekrar hesaplanmadı.
Son devam parçası 24,51 saniye, örneklenen tepe RSS 459,72 MiB ile
tamamlandı. Bu değer bütün taramanın süresi veya başarısız girişimlerin
tepe değeri değildir. Rapor üretimi 2,78 saniye / 130,05 MiB RSS ölçtü.

| Sonuç | Eski onaylı | Yeni Nilüfer |
|---|---:|---:|
| Genel eşitsiz ölçek/shear adayı | 1 | 5 |
| Yerel yapı değişimi adayı | 9 | 58 |
| Sınırda | 2 | 8 |
| Güvenilir konum/şekil ölçülemeyen | 35 | 59 |
| Global test var, yerel test belirsiz | 5 | 7 |
| Belirgin geometrik fark saptanmayan | 338 | 513 |

Toplam 73 aday, 116 belirsiz/sınırda, 851 belirgin fark saptanmayan.
851 sayı anatomik/yazısal kusursuzluk garantisi değildir. Adaylar da
otomatik ret veya kesin bükülme listesi değildir; yerel yeniden çizim,
grafik ekleme ve doku farkları ayrıca görsel değerlendirme gerektirir.
Eski 390 insan kararının hash'i korundu. Hiçbir seçim değiştirilmedi.

Özel shape-v2 dizininde all-results.csv 1.040, suspects.csv 73 ve
uncertain.csv 116 benzersiz kimlik içeriyor. 19 karşılaştırma sayfası üretildi;
ilk dört sayfa (16 aday) görsel olarak incelendi. Dua Lipa'da kol/gövde,
My Love (2024)'te basılı daire/çarpı oranı, Girl 2.0'da iç içe çerçeve
çizgileri/yazı oranı değişimi görüldü. Change Street ve NAKAMA örneklerinde
arka plan yeniden çizimi de katkı yapıyor; bunlar bedensel bükülme olarak
kesinleştirilmedi. Bu görsel spot kontrol tüm adayların insan onayı değildir.

Karşılaştırmalar yalnız konum/dönme/eşit ölçek ile hizalanır; eşitsiz
ölçek veya yerel bükülme düzeltilerek saklanmaz. Öğretmen çıktısı,
kaynak ve eğitim verisi dosyaları değiştirilmez. Rapor ve özel görseller
normal Git'e girmez; yalnız kod, test ve bu aşama kaydı yayımlanır.
