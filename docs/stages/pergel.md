# Pergel — manuel inceleme sunucusunun bağlantıda takılması

29 Ekim 2026. Nilüfer sayfası ilk açılış kontrolünde HTTP 200 vermesine
rağmen daha sonra kullanıcıda açılmadı. Canlı kontrolde süreç çalışıyor,
tek kabul edilmiş TCP bağlantısı açık duruyor ve yeni sayfa/API istekleri
zaman aşımına uğruyordu. Sunucu standart tek iş parçacıklı HTTPServer idi;
boş tarayıcı bağlantısı diğer bütün istekleri bekletebiliyordu.

ReviewServer artık ThreadingHTTPServer kullanır. Boş bağlantıya 10 saniye
zaman aşımı uygulanır. Statik görseller ve sayfa istekleri birbirini
bekletmez. ReviewStore karar okuma, revizyon kontrolü, fsync/atomik kayıt ve
türetilmiş manifest oluşturmayı RLock altında seri tutar. Aynı revizyona
gelen iki farklı kararın biri conflict olur; ayrı görsellerin kararları
birbirini silmez. Idempotent request_id ve çıktı hash kuralları korunur.

18 sunucu/karar testi geçti. Yeni regresyonlar açık ve boş bir TCP
bağlantısı varken sayfa/API erişimini, eşzamanlı iki farklı kararın
kalıcılığını ve aynı revizyonda tek başarılı kayıt olmasını doğrular.
Yazma testleri sadece geçici fixture dosyalarındadır; canlı sete test oyu
gönderilmedi.

Sunucu aynı port ve galeride yeniden başlatıldı. Gerçek 54147 portunda
boş bağlantı açıkken sayfa, data.json, review-api ve JavaScript 200 verdi;
sayfa yaklaşık 1 ms, kayıt API'si 44 ms yanıtladı. Gerçek raw görseli de
200 verdi. Yeni galeride 650 çıktı ve sıfır kayıtlı karar vardı. Bu HTTP
kontrolü tarayıcının görsel yerleşimini doğrulamaz.

Kısayollar koddan kontrol edildi: D/sağ ok iyi, A/sol ok kötü, S atla,
Space kaynak alanını aç/kapat. Not kutusundan Enter ile kısayollara dönülür.
Tarayıcı yenilemesi için Command-R yeterlidir. Kaynak görseller, seçimler,
çıktılar ve mevcut deneyler değiştirilmedi; üretim veya eğitim başlatılmadı.
Alt ajan veya uzak sunucu kullanılmadı.
