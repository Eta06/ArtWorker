# Sarmaşık — kaynak kapaklarda aynı görselin varyantları

18 Ekim 2026. Kullanıcı Begonya koleksiyonunda aynı kapak olup olmadığını
yeniden kontrol etmemizi istedi. Bu aşama çevrimdışı denetimdir; kapaklar,
kaynak manifestler ve insan eğitim kararları değiştirilmez. Yeni indirme,
teacher çağrısı, eğitim, alt ajan veya üretim harcaması yoktur.

## Önceki denetimden farkı

Begonya'nın önceki sıkı RGB/piksel denetimi kabul edilmiş 1.000 kapakta tekrar
tespit etmemişti. Bu sonuç aynı fotoğrafın belirgin şekilde yeniden
renklendirilmiş veya yazısı değiştirilmiş sürümlerinin yokluğu anlamına
gelmiyordu. Yeni denetim özellikle bu görsel ailelerini araştırır.

İki eski 1.000'er kapaklık Defne ve Erguvan arşivi ile mevcut kare kaynaklar
3.443 dosya içerir. Normalize RGB özetiyle yinelenen referans kopyaları
birleştirildiğinde 2.931 referans görüntü indekslenir. Yeni 1.000 kapağın
tamamı eski referanslara ve kendinden önceki yeni kapaklara karşı aranır.

## Yöntem ve sınırlar

Tam görüntü ve %94/%85/%75 merkez kırpımlarının pHash özetleri komşu aramada
kullanılır. Her kaynak için en yakın 12 adaydan Hamming uzaklığı en çok 20
olanlar ayrıntılı karşılaştırılır. 512 pikselde lüminans SSIM/NCC ve SIFT
özellikleriyle tek bir ölçek/dönüş/öteleme dönüşümü denenir. Geometri,
özelliklerin görüntüye yayılımı ve geçerli örtüşme denetlenir. Renkli RGB
farkı kaydedilir; yeniden renklendirilmiş görseli sırf bu fark büyük diye
dışlamayız. İşaretleme bir inceleme önerisidir, otomatik silme kararı değildir.

Bu sınırlı komşu araması bütün olası dönüşümler için eksiksizlik kanıtı
değildir. Büyük kompozisyon değişimleri, merkez dışı kırpımlar, yerel
dönüşümler veya hash aramasına girmeyen çiftler kaçabilir. Tespit edilmeyen
tekrar kalmadığı iddia edilmez. Görsel aile kararı eğitim kalitesi onayı
veya kullanıcının mevcut kabul/ret kararı değildir.

Sentetik kontroller yeniden kodlama/boyutlandırma, merkez kırpma, büyük renk
kayması ve benzer paletli farklı içerik durumlarını sınar. Dört test geçti.
Tespit edilen aday çiftler ayrıca yerel paftalarda görsel incelendi.

## Sonuç ve kayıt

1.000 kaynak için 12.000 ayrıntılı çift karşılaştırması tamamlandı. İşaretlenen
20 çiftin tamamı görsel incelendi. Aynı fotoğraf/illüstrasyonun renk, yazı veya
kadraj varyantı olan 9 ailede 18 yeni kapak bulundu: Little Mix — Between Us,
Peggy Gou — I Hear You, Wicked: For Good, CHVRCHES — Screen Violence,
Cannons — Shadows, Wicked: The Soundtrack, Kylie Minogue — DISCO,
ILLENIUM — Fallen Embers ve Rina Sawayama — Hold The Girl. Her aileden birini
tutmak 9 fazlalığı azaltır, koleksiyon 991 olur; bu işlem uygulanmadı.

YOASOBI E-SIDE ve THE BOOK serilerinde dörder kapak aynı tasarım şablonunu
kullanır. Bu 8 kapak ayrı iki şablon ailesi olarak işaretlendi; farklı
albümler oldukları için doğrudan kopya sayısına katılmadı. Toplam 11 aday
çift bu iki aile içindedir; çift sayısı kapak veya aile sayısı değildir.
Eski arşive karşı bu yöntemle eşleşme önerisi bulunmadı.

Tam tarama 567,20 saniye, örneklenen tepe RSS 296,73 MiB ve fiziksel footprint
276,91 MiB ile tamamlandı. Bunlar örtüşen ölçümlerdir, toplanmaz. 512 MiB
koruma sınırı aşılmadı; sistem basıncı normal, örneklerde swap sabit kaldı.
Kaynak kodu sözdizimi, dört test ve git diff denetimi geçti.

Son sayımlar, çiftler, görsel değerlendirme JSON'u ve karşılaştırma paftası
özel results/2026-10-18/sarmasik altında saklanır. Başlangıç dosyalarının
hashleri denetim sırasında doğrulandı.
Hiçbir kapak otomatik elenmedi; temizleme veya yerine yeni kaynak toplama
bu denetimin parçası değildir. Görseller ve özel manifestler Git'e eklenmez.

Kod, test ve bu kayıt dosya başına bir commit ve tek push ile yayımlanır;
Space'e gerçek sonuçlar eklenir. Git yazıları mevcut RAM/basınç korumasıyla
çalıştırılır. Sonraki adım, aynı fotoğraf varyantlarından eğitimde kaçının
tutulacağına karar vermektir; benzer şablonlu farklı albümler ayrı ele alınır.
