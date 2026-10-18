# Sedir — seçim sayacı ve tek yönlü eğitim hedefi

18 Ekim 2026. Kullanıcı kaç çıktıyı seçtiğini görmek ve teacher çıktısının yalnızca üstü veya altı iyi olduğunda kötü genişlemeyi eğitim verisinden çıkarmak istedi.

## Davranış

Galeri ve hızlı inceleme başlığında dosyaya kaydedilmiş kabul, ret, kararsız ve incelenmemiş sayıları ayrı gösterilir. Eski parmak iziyle kayıtlar sayılmaz. Varsayılan pending ile hiç kaydı olmayan çıktı ayrılır; seçilen sayısı açık kabul kararlarının sayısıdır, otomatik olarak eğitim adaylığı onayı anlamına gelmez.

Üstü sil / sadece alta genişlet ve Altı sil / sadece üste genişlet düğmeleri iki inceleme görünümüne eklendi. Üstü sil hedefi kaynak bölgesinin üstünden çıktı altına, Altı sil çıktı üstünden kaynak bölgesinin altına kırpar. Tek yönlü hedefin yatay sınırları kaynak bölgesinin sınırlayıcı dikdörtgenidir; böylece yanlara ait hatalı genişlemeler de tek yönlü hedefe taşınmaz. Kaynak pikselleri bozulmaz, ham görüntü dosyası silinmez veya yeniden yazılmaz.

Bu işlem eğitim girdisine alan dahil etmekten ayrıdır. Silinen taraf aynı anda girdiye dahil edilemez; sil düğmesi o taraftaki dahil et seçimini kapatır. Diğer tarafın daha önceki seçimi korunur. İki tarafın çıkarılması veya tutulan hedefin tamamının girdiye dahil olması üretilecek bölge bırakmazsa kabul edilmiş örnek blocked listesinde kalır.

Sil düğmelerinde de hover/focus önizlemesi var. Kırmızı alan eğitimden çıkarılan kısım, yeşil sınır kalan hedef, mavi kesikli sınır girdidir. Turuncu tespit poligonu değişmez. Önizleme kayıt göndermez. Tıklama karar/not/sırayı değiştirmeden mevcut kalıcı işlem kuyruğuyla otomatik kaydeder; çıktı SHA-256, revision ve request_id denetimleri korunur.

Yeni target_trim tercihleri karar dosyası, seçim yedeği ve aday manifestinde taşınır. conditioning kaydında ham görüntüde target_box_px ve box_px; kırpılmış hedefin target_dimensions bilgisi; maskeyi doğru konuma koymak için input_box_in_target_px bulunur. Kırpılmış hedefe geçerken maske koordinatları yeni başlangıca göre ötelenir. Eski istemcilerin alanı göndermemesi mevcut hedef tercihini sıfırlamaz. Yeni veri hazırlama tüketicisi target_box_px bilgisini kullanmalıdır. Toplu görüntü/mask dosyaları üretilmedi, eğitim başlatılmadı.

## Doğrulama

Yirmi çevrimdışı JavaScript akış testi ve on bir Python kayıt/HTTP testi geçti. Yeni testler sayımda fingerprint ve pending/unseen ayrımını; üstü çıkarma ve çakışan dahil et seçimini kapatma; karar/sıranın korunması; kayıt sonrası geri yükleme; hover ile hedef kırpma önizlemesini; kırpılmış hedefin maske koordinatlarını; eski istemciyi; iki tarafı çıkarma ve geri almayı; idempotent isteğin farklı hedefle tekrarını reddetmeyi ve asimetrik kaynakta yalnızca dikey hedefi kapsar.

Site, üç script ve karar uçları HTTP 200/no-store ile doğrulandı. Kayıt sunucusu güncel şemayla kontrollü yeniden başlatıldı. Mevcut karar dosyası sıfırlanmadı, kullanıcı seçimleri değiştirilmedi ve gerçek arşive test kararı yazılmadı. JavaScript sözdizimi ve git diff --check geçti. Önceki Browser Use URL politikası engeli nedeniyle farklı bir tarayıcı yolu denenmedi; canlı görsel ve tıklama doğrulaması yapılamadı.

## Yayın

Sekiz dosya dosya başına bir commit, tek push ve RAM/basınç denetimiyle yayımlanır. Ham görseller, kullanıcı notları, kararlar ve süreç kayıtları özel results/.build dizinlerinde kalır. Yeni teacher üretimi, model yükleme veya öğrenci eğitimi başlatılmadı.
