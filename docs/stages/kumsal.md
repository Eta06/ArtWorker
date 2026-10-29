# Kumsal STAYC merkez kare ve birleşik veri kontrolü

29 Ekim 2026. Kullanıcı STAYC I WANT IT kaydına merkez kare seçilmesini
istedi. Mevcut accepted kararı, notu ve uzantı/kırpmaları koruyan yeni
request_id ile yerel review-api kullanıldı. square_fit ve center_square
true oldu; revizyon 1'den 2'ye çıktı. Başka hiçbir karar değişmedi.
1152x864 çıktıda referans [144,0,1008,864], 864x864 manuel kare.

Her iki set güncel ReviewStore mantığıyla salt okunur denetlendi:

| Set | Çıktı | Kabul ve aday | Ret | Kararsız | Geometri bloğu |
|---|---:|---:|---:|---:|---:|
| Eski Sümbül | 452 | 390 | 59 | 3 | 0 |
| Yeni Nilüfer | 650 | 643 | 7 | 0 | 0 |
| Toplam | 1102 | 1033 | 66 | 3 | 0 |

1.033 adayın gerçek PNG boyutları ve SHA-256 değerleri, girdi/hedef
kutuları, hedef içindeki göreli koordinatlar ve kalan dış alan hesabı
kontrol edildi. Eksik dosya, hash/boyut uyuşmazlığı, boş kırpma, hedef dışı
girdi veya görüntü dışı kutu saptanmadı. Aynı öğretmen çıktısından kırpılan
girdi/hedef prensibi korundu; fiziksel girdi/hedef paketlemesi veya eğitim
başlatılmadı. Tam aynı öğretmen çıktısı hash'inde tekrar yok; bu farklı
albüm sayısı veya train/validation ayrımı iddiası değildir.

Tespit poligonu 254 eski, 201 yeni adayda tuval sınırını aşıyor. Bu sayılar
özgün tespit poligonunu sayar; seçilmiş manuel referansın taştığı anlamına
gelmez. reference_box görünür görüntü sınırına kesilir; training_crop
girdiyi tutulmuş hedefe de sınırlar. Öğretmen çıktısındaki mevcut pikseller
kullanılır, tuval dışına crop edip siyah padding eklenmez. Taşma tek
başına bu çiftin piksel tutarlılığını bozmaz; tespit kaybolan orijinal
içeriği geri kazandırdığı iddiası da taşımaz.

Ignored results içinde birleşik aday manifesti ve denetim raporu
oluşturuldu. Her iki karar dosyası hash'i denetim boyunca sabit kaldı.
Denetim 1,93 saniye / 46,44 MiB örneklenen tepe RSS ile 512 MiB guard
altında tamamlandı. STAYC'nin önceki kararı özel .build yedeğinde saklandı.
Yeni görsel üretimi, eğitim, alt ajan veya uzak sunucu çalışması yok.
