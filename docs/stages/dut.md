# Dut — eğitim girdisinin üst ve alt alanını seçme

18 Ekim 2026. Kullanıcı, teacher'ın kaynak alanı dışında yeniden yerleştirdiği logo gibi öğeleri öğrenci girdisine dahil ederek, sadece kalan dış bölgeyi üretme hedefi oluşturmak istedi. Bu, teacher'ın orijinali koruduğuna dair bir iddia değildir; öğrenci girdisinin teacher çıktısından nereden kesileceğine ilişkin ayrı seçimdir.

## Davranış

Hızlı ve ayrıntılı inceleme pencerelerinde Üstü de dahil et ve Altı da dahil et düğmeleri eklendi. Üst seçimi kesiti hedef görselin üst kenarına, alt seçimi alt kenarına kadar uzatır. Yatay sınırlar tespit edilen alanın sınırlayıcı dikdörtgeninden gelir ve çıktı içinde tutulur. Dönüşlü veya hafif kesilmiş eşleşmede sınırlar aşağı/yukarı yuvarlanıp çıktı boyutlarına kırpılır. Teacher'ın ham görüntüsünden aynı pikseller kesileceği için bu işlem yeni renk düzeltmesi veya kaynak warp uygulamaz.

Turuncu poligon değişmez tespit kaydıdır. Mavi kesikli dikdörtgen eğitim girdisi için seçilen kesittir. Üst/alt düğmesine basmak ilgili sınırı görünür yapar; Space iki sınırı birlikte açıp kapatır. Yeni görselde sınırlar yine kapalı başlar. Karşılaştırma sırasında renkleri etkilememesi için poligon dolgusu kaldırıldı. Güvenilir kaynak bulunmadığında üst/alt seçimi kapalıdır.

Alan seçimi karar vermeden ve sırayı ilerletmeden otomatik kaydedilir. Not ve mevcut kabul/ret/kararsız kararı korunur. Seçim id ve raw_sha256 ile bağlıdır; sayfa yenilendiğinde geri yüklenir ve sonraki karar/not isteklerinde taşınır. Eski bir sekmenin crop_extensions içermeyen isteği mevcut alan seçimini sıfırlamaz. Yeni istemci, kayıt onayında üst/alt tercihlerinin de eşleşmesini ister. Eski bekleyen işlem biçimi desteklenir.

İndirilen seçim yedeği ve eğitim adayı manifesti bu tercihi ve conditioning kaydını taşır: teacher_output_crop, hedef çözünürlüğü, [sol, üst, sağ, alt] piksel kesiti ve doldurulacak dış piksel sayısı. Dikdörtgen Pillow crop gibi sağ/alt hariç koordinat sistemindedir. Tüm görüntü girdi olarak seçilmişse üretilecek dış bölge kalmaz; kabul kararı korunur fakat örnek no_region_left_to_outpaint nedeniyle blocked listesine alınır. Varsayılan kesit geçersizse aday oluşturulmaz.

Bu aşama seçim arayüzünü ve veri hazırlama tarifini uygular. Toplu crop/mask görüntü dosyaları çıkarılmadı, eğitim veya yeni teacher üretimi başlatılmadı. Öğrencinin gerçek eğitim hazırlığı bu metadata kaydını kullanmalıdır.

## Doğrulama

On dört çevrimdışı JavaScript akış testi ve sekiz Python kayıt/HTTP testi geçti. Yeni testler alta dahil etmenin kararı veya sırayı değiştirmeden kaydını, çıktı parmak iziyle bağlı geri yüklemeyi, kayıt sonrası kararlara tercihin taşınmasını, kaynak poligonunun değişmemesini, eski istemcinin seçimi sıfırlamamasını, üst/alt/tam-görsel engelini, geri almayı, geçersiz/güvenilmez seçimin reddini ve dönüşlü sınırların kırpılmasını kapsar.

Yerel site, yeni script ve karar uçları HTTP 200/no-store ile doğrulandı. Sunucu güncel kayıt şemasıyla kontrollü yeniden başlatıldı. Mevcut karar dosyası silinmedi veya sıfırlanmadı; gerçek arşive sahte karar yazılmadı. JavaScript sözdizimi ve git diff --check geçti. Önceki Browser Use URL politikası engeli nedeniyle başka bir erişim yolu denenmedi; canlı görsel/tıklama doğrulaması yapılmadı.

## Yayın

Dokuz dosya, dosya başına bir commit ve tek push ile yayımlanır. Git işlemleri boyunca RAM/basınç denetimi açıktır. Özel kapaklar, ham çıktılar, kullanıcı kararları ve notları Git dışında tutulur. Mevcut S/A/D/ok ve Space kısayolları korunur.
