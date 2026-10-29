# Nar Kaynak bulunamasa da merkez kare seçimi

29 Ekim 2026. Kullanıcı güvenilir kaynak bölgesi bulunamayan görsellerde
Kareyi kenarlara oturt düğmesinin çalışmasını istedi.

Düğme artık kayıtlı eşleşme olmadan da etkin. Kullanıcı basınca kısa kenarı
dolduran merkez kare seçilir; tespit edilmiş kaynak veya otomatik eşleşme
olarak gösterilmez. center_square tercihi hash/revizyon, yerel yedek,
outbox, not düzenleme ve sunucu kaydında korunur. Sonradan kaynak testi
değişse bile seçilen merkez kare kaymaz. Mod kapatıldığında kullanılmayan
uzantılar temizlenir ve önceki eşleşme durumuna dönülür.

Önizleme için boş/gizli SVG mevcut; eşleşme olmayan yerde uydurulmuş
poligon gösterilmez. Hover kaydetmez; tıklama manuel referansı kaydeder.
Uzantı/kırpma ve Space seçilmiş kareyle çalışır. Eşleşme puanı gösterilmez.
Kalite kararı değişmez; eğitim adayı için ayrıca insan kabulü gerekir.
Manifest manuel referansı is_source_match false olarak ayırır.

30 controller JavaScript ve 23 karar/sunucu Python testi geçti. Yeni
regresyonlar bilinmeyen kaynakta etkin düğme ve boş overlay, merkez kare,
yatay/dikey uzantı, hover, yenileme, kapatma ve sonradan değişen kaynak
testine rağmen kalıcılığı doğrular.

Nilüfer dosyaları güncellendi, yerel sunucu yeniden başlatıldı. Üç JS
dosyası ve API 200 verdi; 101 insan kararı vardı ve kayıt dosyası hash'i
yeniden başlatmada değişmedi. Canlı sete test oyu gönderilmedi. Tarayıcı
yerleşimi görsel olarak doğrulanmadı. Üretim, eğitim, alt ajan veya uzak
sunucu çalıştırılmadı. Özel karar yedeği .build içinde kaldı.
