# İpek — eğitim kesiti için hover önizlemesi

18 Ekim 2026. Kullanıcı üstü/altı dahil et düğmesine basmadan, seçimin dışa aktarılacak eğitim kesitini nasıl değiştireceğini görmek istedi.

Düğme üzerine fare gelince veya klavye odağı gelince, bir sonraki tıklamanın oluşturacağı kesit mor kesikli sınırla gösterilir. Zaten seçili bir düğmenin önizlemesi o tarafın çıkarılmasını gösterir. Kesitin boyutu ve kaydedilmediği yazıyla belirtilir. Bulunan kaynak poligonu turuncu olarak korunur. Önizleme hiçbir karar, not, alan tercihi, sıra veya dışa aktarma dosyasını değiştirmez; sunucuya istek göndermez.

Fare çıkınca veya odak ayrılınca kayıtlı mavi kesit ve kullanıcının güncel sınır görünürlüğü geri gelir. Önizleme sırasında Space ya da ayrıntı görünümünün görünürlük kutusu değiştirilirse bu tercih geri dönüşte korunur. Düğmeye tıklamak önizlemeyi kaldırıp mevcut otomatik kayıt davranışını uygular. Güvenilir kaynak bulunmayan veya kayıt sırasında devre dışı olan düğmeler önizleme yapmaz.

On yedi çevrimdışı JavaScript denetleyici testi geçti. Yeni testler kaydetmeden hover, fare çıkınca eski kesit/gizlilik durumuna dönüş, klavye odağıyla seçimi kaldırma önizlemesi ve önizleme sırasında değişen Space/checkbox görünürlüğünü kapsar. Sözdizimi ve git diff --check geçti. Yerel index ve training-crop.js HTTP 200/no-store ile doğrulandı. Mevcut kararlar okunup incelenmedi, sıfırlanmadı ve arşive sahte karar yazılmadı. Önceki Browser Use URL politikası engeli nedeniyle canlı tarayıcı hover/görsel doğrulaması yapılmadı.

Dört dosya dosya başına bir commit, tek push ve RAM/basınç izlemesiyle yayımlanır. Özel veri ve kullanıcı kararları Git dışında kalır. Eğitim ve yeni görsel üretimi başlatılmadı.
