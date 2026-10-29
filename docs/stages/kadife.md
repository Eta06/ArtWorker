# Kadife Kare referansı kısa kenara oturtma

29 Ekim 2026. Kullanıcı Be The One çıktısında alt alanı girdiye dahil
ederken sol/sağda dar hedef şeritleri kaldığını gösterdi. Önceki seçim
yalnız tespit edilmiş kaynak kutusunun genişliği boyunca aşağı uzuyordu.

İncelemeye isteğe bağlı Kareyi kenarlara oturt düğmesi eklendi. Önce
kaynağın görünen tamamını kapsayan, çıktı kısa kenarını dolduran bir kare
hesaplanır; ardından üst/alt veya sol/sağ dahil etme ve hedef kırpma
uygulanır. Görsel pikselleri ölçeklenmez veya warp edilmez. Kare tuval
içinde kaydırılarak kaynağı kapsar; tamamen sığmıyorsa seçenek kapanır,
kaynak sessizce kırpılmaz. Eski kayıtlar varsayılan kapalı davranışı korur.

Kare modu açıkken turuncu çizgi seçilmiş kare referansı, mavi çizgi
uzantıları da içeren eğitim girdisini gösterir. Özgün tespit poligonu
data.json içinde değişmeden kalır. Hover/klavye odağı kaydetmeden önizler;
tıklama square_fit değerini anında ve görüntü hashine bağlı kaydeder.
Not düzenleme, yenileme, sonraki iyi/kötü kararı, outbox ve idempotent
request_id bu tercihi korur. Tam sunucu onayı olmadan bekleyen kayıt silinmez.

Sunucu, JavaScript ve dışa aktarılan manifest aynı reference_box_px ve
box_px koordinatlarını kullanır. Onaylılar ekranı kareyi hedefe göre çizer;
önizleme URL'si doğru galeri route'unu izler. Kareye elle oturtulmuş örnek
modelin müdahalesiz kabul istatistiğine eklenmez.

Be The One için gerçek koordinatlarla salt okunur prova:
864x1152 hedefte referans [0,146,864,1010], 864x864 kare.
Altı dahil edilince girdi [0,146,864,1152]; yanlarda hedef şeridi kalmaz,
yalnız üstte 864x146 = 126.144 piksel üretim alanı kalır. Bu prova canlı
görselin kararını veya crop tercihini değiştirmedi.

27 hızlı inceleme/geometri ve 4 onaylı ekran JavaScript testi; 21
karar/sunucu ve 4 eğitim önizleme Python testi geçti, toplam 56.
Yeni kapsam kareyi kısa kenara oturtmayı, yatay/dikey uzantıları, tuval
kenarında kapsama, sığmayan kutuyu reddetme, salt okunur hover,
yenileme kalıcılığı, eski not istemcisinin tercihi koruması ve onaylı
önizlemenin doğru route/koordinatları kullanmasını içerir.

Nilüfer statik dosyaları atomik yenilendi ve yerel sunucu yeniden
başlatıldı. Karar dosyası özel yedeklendi; yeniden başlatma öncesi/sonrası
hash aynı, dokuz insan kaydı korundu. Sayfa, güncel dört JS dosyası ve
review-api HTTP 200 verdi. Canlı sete test oyu gönderilmedi. Tarayıcıda
görsel yerleşim testi yapılmadı; controller/HTTP kanıtı bu iddia değildir.
Kaynaklar, üretilmiş görseller ve eski set değiştirilmedi; model, eğitim,
alt ajan veya uzak sunucu çalıştırılmadı.
