# Kilim — onaylı referanslar ve müdahalesiz kullanım karşılaştırması

18 Ekim 2026. Kullanıcı kaynak alanı önerilerini onayladıktan sonra tüm kabul
edilmiş örneklerin gerçek eğitim referanslarını görmek ve hangi teacher'ın
daha az müdahale gerektirdiğini öğrenmek istedi. Yeni üretim yapılmadı.

## Mevcut insan kararları

452 mevcut çıktıda 367 kabul, 82 ret, üç kararsız kayıt var. 17 kurtarılmış
kaynak önerisinin tamamı kullanıcı tarafından onaylanmış. Kabul edilmiş 367
örnek de geçerli kırpım ve kalan üretim alanıyla aday manifestinde bulunuyor;
bekletilen kabul yok. Bu kayıt eğitim başladığı anlamına gelmez.

| Model | Mevcut çıktı | Kabul | Müdahalesiz kabul | İlk otomatik kaynak eşleşmesi |
| --- | --- | --- | --- | --- |
| Sunburst 2.5 | 313 | 293 (%93,6) | 271 (%86,6) | 297 (%94,9) |
| Nano Banana 2.1 | 139 | 74 (%53,2) | 69 (%49,6) | 136 (%97,8) |

Müdahalesiz kabul: insan kabulü, ilk güvenilir kaynak eşleşmesi, kaynak
kurtarması gerekmemesi ve üst/alt girdi dahil etme veya hedef çıkarma tercihi
olmaması birlikte aranır. Payda modelin tüm mevcut çıktılarıdır. Bu ölçü
kusursuz orijinal piksel korunması, hız ya da genel model puanı değildir.
Özellikle üst/alt tercihi her zaman model hatası için verilmiş olmayabilir.

Genel insan kabulü ve müdahalesiz akışta Sunburst önde; yalnız otomatik
kaynak eşleşmesinde Nano biraz önde. Nano çoğunlukla Sunburst'ın üretmediği
zor kapakları işlediği için bunlar eşit kapaklarla kontrollü A/B sonuçları
değildir. Reddedilen/üretilemeyen API istekleri mevcut çıktı paydasına dahil
edilmedi; yeni bir ücret veya hız karşılaştırması yapılmadı.

## Yeni görünüm

/sumbul/approved/ ve ana arşivde Onayladığım referanslar ve eğitim çiftleri
bağlantısı eklendi. Tüm kabul edilenler listelenir; eksik kabul varsa sessizce
elenmek yerine bekleme gerekçesi gösterilir. Sayfa başına dokuz çift,
arama/model filtresi, karar notları ve güncel model sayıları vardır.

Solda teacher çıktısından alınan gerçek girdi kırpımı, sağda aynı çıktının
hedef kırpımı gösterilir. Kaydedilmiş crop_extensions ve target_trim
uygulanır. Mavi sınır girdinin hedefteki yerini; isteğe bağlı turuncu poligon
orijinal kaynak eşleşmesini gösterir. Hedef kırpıldıysa koordinatlar hedef
başlangıcına göre ötelenir. Kaynak görüntü kareye esnetilmez.

Küçültülmüş JPEG önizlemeleri istek anında oluşturulur; eğitim görüntüsü
olarak kaydedilmez ve ham kaynak dosyaları değiştirilmez. En fazla
560×780 önizleme, sırayla tek ham görüntü ve lazy loading kullanılır.
Kalıcı çözümlenmiş görüntü önbelleği veya toplu kırpım arşivi oluşturulmaz.

Önizleme uçları yalnız güncel kabul edilmiş, kayıt/maske denetimlerinden
geçmiş adaylar için açılır. Çıktı SHA-256 ve karar revision değerleri isteğe
bağlanır. Eski seçim 409, ret veya bekletilen aday 403 döndürür. Sayfa hiçbir
kararı değiştirmez; yeni kararlardan sonra Seçimleri yenile kullanılır.

## Doğrulama

25 JavaScript, dört önizleme, 13 kayıt/HTTP ve dört galeri testi geçti:
toplam 46. Yeni kontroller kabul/ret/bekleme, eski fingerprint, kırpılmış
hedef koordinatları, teacher girdi pikselleri, hedef çıkarma, güncel revision,
HTTP içerik türü ve müdahalesiz paydanın hesaplanmasını kapsar.

367 adayın dosyası ve girdi/hedef koordinatlarının iç içe geçerliliği
kontrol edildi. Kısmi referans, hedef tarafı çıkarılmış örnek ve Nano
örneği olmak üzere üç gerçek çiftin önizlemeleri HTTP'den alınıp yerel
paftada görsel incelendi. Sunucunun bu isteklerde örneklenen tepe RSS'i
58,44 MiB; galeri yayımlayıcısınınki 32,66 MiB. Seçim dosyasının baytları
aynı kaldı. HTTP 200/no-store ve JavaScript sözdizimi doğrulandı.

Önceki Browser Use URL politika engeli nedeniyle canlı tarayıcı tıklama
ve yerleşim kontrolü yapılamadı. Yerel kırpım paftası tarayıcı görüntüsü
değildir. Arşiv veya eğitim kalitesine yeni otomatik kabul iddiası eklenmedi.

## Yayın

Dokuz dosya, dosya başına bir commit ve tek push/RAM denetimiyle yayımlanır.
Görseller, notlar, kararlar ve yerel görsel doğrulama paftası Git'e girmez.
ArtWorker Space aşama kaydı güncellenir. Model yüklenmedi, provider çağrısı
ve öğrenci eğitimi yapılmadı.
