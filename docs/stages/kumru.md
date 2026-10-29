# Kumru Atlanan dört kapak için alan önerileri

29 Ekim 2026. Kullanıcı not yazmadığı dahil dört atlanan kapağın alanını
bulmayı istedi; eşleşme güvenilir olmazsa merkezden tam kare istedi ve
son kontrolleri açıkça kendisine bıraktı. İnsan kararı otomatik değiştirilmedi.

İlk mevcut ORB/SIFT ve yoğun piksel doğrulaması dört örneği de unresolved
bıraktı. Mevcut full-affine kayıtları ve özellik başlangıçlarından ECC
iyileştirmesi ayrıca denendi. Bu yeni opt-in yöntem geometri önerir;
değişmemiş içerik yüzdesi veya estetik kabul değildir. Sonuçlar:

- Once Twice Melody: merkez yazı üzerinden aday; dış çerçeve yeniden
  çizilmiş. Merkez korelasyonu yaklaşık 0,917. edited_reference olarak
  gösterilir, kusursuz eşleşme sayılmaz.
- Wild: bilgi taşıyan piksellerde korelasyon yaklaşık 0,994 ve SSIM 0,974;
  yine yalnız kullanıcı onayı bekleyen sınır önerisidir.
- Clarity ve My Love: ayrıntılı denemede değişmiş yapı adayları elde edildi;
  kullanıcının son isteğiyle bu zayıf adaylar yerine merkezden tam kare
  sunuldu. 864x1152 çıktıda [0,144,864,1008]. Bunlar otomatik eşleşme
  değildir; metrics null, is_source_match false olarak işaretlendi.

Not yazılmayan kapakların unutulmaması için CLI'ye skipped kapsamı eklendi; bu kapsam boş notlu pending kayıtları
da alır. Eski noted varsayılanı korundu. Merkez kare fallback opt-in ve
açık etiketlidir; sahte aligned veya fark görüntüsü/SSIM üretilmez.

Weather kaynak ve çıktı görselleri açıldı. Kaynakta başı kadraj dışında
kalan kişi üst alanda tamamlanmamış; başsız gövde görünümü var. Kullanıcının
ret kararı korundu. Bu görsel yeniden üretilmedi.

9 recovery, 4 galeri Python testi ve 28 controller JavaScript testi geçti.
Yeni kapsam boş notlu atlamaları, piksel iyileştirmesini, ilgisiz/özelliksiz
görselleri reddetmeyi, tam merkez kareyi ve eşleşme metriği uydurmamayı,
manuel alan onayının pending kalite kararını değiştirmemesini doğrular.
Tüm test yazmaları geçici fixture dosyalarında.

Nilüfer'e dört review_pending öneri yayımlandı. Son HTTP doğrulamasında
51 kabul, 1 ret, 4 pending, dört boş registration_confirmation vardı;
51 eğitim adayı ve sıfır blocked. İnsan karar dosyası hash'i ve dört eski
audit hash'i değişmedi. Dört placement ve sayfa/API HTTP 200 verdi.
3:4 grubundaki Yeni kaynak alanlarını kontrol et (4) düğmesinden erişilir.
Kullanıcı alanı ayrıca onaylar ve iyi/kötü kararını kendisi verir.

Çalışma yalnız Mac üzerinde ve 512 MiB guard altında yapıldı. Son kurtarma
2,29 saniye / 139,17 MiB tepe örneklenen RSS; yayın 1,04 saniye / 35,84 MiB.
Özel denemeler ve yedekler ignored results/.build içinde korundu. Yeni
ücretli üretim, eğitim, alt ajan veya uzak sunucu kullanımı yok. Tarayıcı
yerleşim testi yapılmadı; dört adayın placement dosyaları yerelde açıldı.
