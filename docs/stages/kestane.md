# Kestane — eksik kaynak alanları ve retleri yeniden inceleme

18 Ekim 2026. Kullanıcı, güzel bulduğu ancak kaynak bölgesi bulunamayan
çıktılardaki notlarının incelenmesini, önce turuncu alanların bulunmasını ve
ardından eski retlerini yeni üst/alt kontrolleriyle yeniden değerlendirmeyi istedi.

## Bulgular ve yöntem

Mevcut 452 çıktıdaki 19 başarısız hizalamadan 17'sinde kullanıcı notu vardı:
16 kabul, bir ret. ORB/SIFT birçok kapakta doğruya yakın bir global dönüşüm
bulmuş, ancak küçük merkez logosu veya düz arka plan nedeniyle özelliklerin
mekânsal kapsama eşiğinde elenmişti. Temel source_fidelity kayıt eşikleri
değiştirilmedi. Diğer iki, notsuz başarısız kayda bu geçişte müdahale edilmedi.

Yeni çevrimdışı recover_registration aracı reddedilmiş global dönüşüm
hipotezlerini, özellik noktalarından ayrı yoğun piksel kanıtıyla kontrol eder.
Bilgi içeren kenar alanlarında SSIM ve parlaklık korelasyonu, görünür kaynak
oranı, eşleşme artığı ve ORB/SIFT köşe uzlaşması kullanılır. Düz görüntü,
ilgisiz içerik, bozuk dönüşüm ve aşırı kırpılma negatif kontrollerdir. Bu yöntem
genel bir semantik kaynak bulma modeli değildir; eski özellik hipotezlerine
bağımlıdır ve yeni kapaklarda evrensel başarı iddiası taşımaz.

Gerçek 17 çıktıda 13 yoğun piksel kanıtıyla desteklenen sınır adayı, iki kısmen
kırpılmış referans, iki daha fazla yeniden çizilmiş referans konumu bulundu.
Kısmi referansların yaklaşık %82,4 ve %82,7'si görünürdür. Bu görünürlük
oranları korunmuş piksel yüzdesi değildir. Yeniden çizilmiş içeriklerde sınır
yalnızca temkinli bir konum önerisidir. SSIM ve RGB toleransı insan kalite
kararının yerini almaz; güçlü konum kanıtı renklerin aynen kaldığını göstermez.

Ek ECC affine inceltmesi iki sorunlu içerikte yaklaşık %5 eksen farklılığına
yöneldi; uniform dönüşüme geri yansıtma SSIM'i kötüleştirdi. Bu değişiklik
benimsenmedi. Kaynağı sonuca uydurmak için yerel bükme veya renk düzeltmesi
yapılmadı. Orijinal audit, ham çıktı ve kaynak dosyaları korundu.

## İnceleme davranışı

Sürüm kimliği, ham çıktı/kaynak SHA-256 ve eski audit SHA-256 ile bağlı özel
öneri manifesti oluşturulur. Galeri bunu ayrı registration_recovery kanıtı
olarak gösterir; eski audit_status ve sayısal sonuçları güvenilir diye yeniden
yazmaz. Üç karşılaştırma paftasında 17 orijinal/işaretli çıktı yerel olarak
görsel incelendi. Kısmi ve yeniden çizilmiş adaylar ayrıca uyarılır.

Yeni kaynak alanlarını kontrol et düğmesi 17 öneriyi açar. Space turuncu sınırı
gösterir. Bu kaynak alanını kullan düğmesi yalnızca sınır onayını kaydeder;
iyi/kötü kararı, not ve sıra korunur. Onay öncesinde otomatik eğitim kabulü
yoktur; üst/alt eğitim kontrolleri kapalıdır. Onay belirli öneriye bağlıdır;
öneri veya çıktı değişirse önceki onay geçmez. Onaylanmış ret yine eğitim
dışındadır. Şüpheli örneğin sınırını onaylamak içerik korunması onayı değildir.

Reddettiklerimi yeniden incele düğmesi tüm 83 reddi yeni ve sabit bir sırada
açar; önceki arama/model filtresini temizler. Üst/alt dahil et ve çıkarma,
hover önizlemesi, notlar ve otomatik kayıt kullanılabilir. 366 kabul, 83 ret,
üç kararsız kararın dosya içeriği ve mevcut crop/trim tercihleri değiştirilmedi.

## Doğrulama ve kaynak kullanımı

22 JavaScript akış, 13 Python kayıt/HTTP, dört galeri, beş kaynak kurtarma ve
15 temel kaynak eşleşmesi testi geçti: toplam 59. Test kararları yalnızca
geçici fixture'lara yazıldı. Gerçek karar dosyasının SHA-256 değeri yayın
öncesi/sonrası aynı doğrulandı. Yeni sınırı onaylamadan adaylığa geçmeme,
öneri değişince onayı geçersiz sayma, ret kararını koruma, salt okunur Space
önizlemesi, retleri yeniden açma ve eski audit'in değişmemesi denetlendi.

17 görselin analizi 3,31 saniyede tamamlandı; örneklenen süreç grubu tepe RSS
153,94 MiB ve fiziksel footprint 121,44 MiB. Bu iki ölçü örtüşür, toplanmaz.
512 MiB/basınç/swap denetimi açık kaldı. Galeri güncellemesi 32,86 MiB
örneklenmiş RSS ile tamamlandı. Yalnızca bir sonraki görsel çifti önden yüklenir.

Site, üç script, karar uçları ve yeni işaretli kaynak varlığı HTTP 200/no-store
ile doğrulandı; kayıt sunucusu kontrollü yeniden başlatıldı. Önceki Browser
Use URL politika engeli nedeniyle canlı tarayıcı tıklama/görsel doğrulaması
yapılmadı. Yerel çıktı paftalarının incelemesi tarayıcı doğrulaması değildir.

## Yayın ve sonraki adım

Kod ve bu aşama kaydı dosya başına bir commit, tek push ve RAM denetimiyle
yayımlanır; özel notlar, kararlar, öneri manifesti ve görüntüler results/.build
altında kalır. Yeni provider üretimi, ücretli istek, model yükleme veya öğrenci
eğitimi yapılmadı. Kullanıcı kaynak adaylarını ve eski retlerini inceleyebilir;
veri hazırlama aşaması kaydedilmiş sınır/crop/target_trim tercihlerini izlemelidir.
