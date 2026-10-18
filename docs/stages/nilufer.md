# Nilüfer — fal Sunburst ile farklı oranlarda üretim

18 Ekim 2026. Kullanıcı yeni Begonya koleksiyonundaki 1.000 kaynağı fal
üzerinden Sunburst ile işleme izni verdi. Ret alan kapak atlanır; Gemini veya
başka bir modele aktarılmaz. Eski üretim klasörleri ve insan kararları korunur.

## Model ve oranlar

Canlı fal kataloğunda `openai/gpt-image-2.5/sunburst/edit` doğrulandı. Sunburst
`high`, tek çıktı, PNG, opaque arka plan ve varsayılan açık içerik kontrolü
kullanılır. Eski başarılı iki referanslı old-canvas promptu temel alındı.
Yalnız hedef oran/boyut, kaynak konumu ve sınır yönleri uyarlanır. İki referans
sırası yeşil yer tutuculu hedef kanvas, ardından 720×720 kaynak kapaktır.
Orijinalin korunması promptta istenir; korunma puanı eğitim için otomatik
kabul/ret eşiği değildir.

| Oran | İstenen çıktı |
|---|---|
| 3:4 | 864×1152 |
| 4:3 | 1152×864 |
| 9:16 | 720×1280 |
| 16:9 | 1280×720 |
| 2:3 | 768×1152 |
| 3:2 | 1152×768 |
| 4:5 | 896×1120 |
| 5:4 | 1120×896 |
| 1:2 | 720×1440 |
| 2:1 | 1440×720 |
| 1:3 | 720×2160 |
| 3:1 | 2160×720 |

Hazır presetler kare seçenekler dışında 3:4, 4:3, 9:16 ve 16:9 içerir. Diğer
oranlar özel genişlik/yükseklik alanıyla istenir. 1:1 ve auto kullanılmaz.
Canlı belgelerde boyutlar 16'nın katı, en uzun kenar en çok 3840 piksel,
uzun/kısa kenar en çok 3:1 ve toplam 655.360–8.294.400 piksel olmalıdır.
Bu tur yaklaşık 0,88–1,56 megapiksel aralığında kalır; eşit piksel veya eşit
fiyat iddiası yoktur. İlk dört istek 3:4, 4:3, 9:16 ve 16:9 oranlarındadır.

1.000 kaynak 12 orana sırayla dağıtılır: ilk dört oranda 84'er, kalan sekiz
oranda 83'er istek planlanır. Her kaynak bir oran alır; her kapak 12 kez
üretilmez. 720×720 kaynak kanvasa ortalanır. Bazı boyutlarda dört kenarda,
diğerlerinde yalnız yatay veya dikey kenarlarda boşluk bulunur. Gerçek üretim
yerleşimi daha sonra serbest konumlu kaynak bölgesi tespitiyle ölçülmelidir.
Bu oran dağılımı tek başına her yönde kalite veya öğrenci kararlılığı kanıtı
değildir.

## Ayrı ve kalıcı kuyruk

Özel `experiments/fal_outpaint/results/2026-10-18/nilufer/` altında state,
kaynak kimlikleri, çıktı klasörleri ve tekil receipt'ler tutulur. Her işin
kimliği `nilufer-coverNNNN` olur; kaynak kimliği ayrıca `begonya:coverNNNN`
olarak saklanır. Orijinal manifest ve kaynak hashleri doğrulanır. Sarmaşık'ta
bulunan varyantlar kaynak koleksiyondan silinmedi; bu başlangıç planı bütün
1.000 kayıt içindir, 1.000 başarılı çıktı garantisi değildir.

Tek hafif Python süreci en çok dört bilinen fal isteğini açık tutar; yerel
model veya GPU çalıştırmaz. Ücretli POST öncesi `submitting` durumu yazılır.
Yanıt belirsizse yeni harcama durur; körlemesine tekrar yapılmaz. Bilinen
istek kimliğiyle devam yalnız GET durum/sonuç okumaları yapar. Fal'ın sunucu
otomatik retry özelliği `X-Fal-No-Retry: 1` ile kapatılır. Retlerde başka
kapağa geçilir; hesap/ücret/yetki hatası, beklenmeyen çıktı boyutu ve diğer
beklenmeyen hatalar yeni gönderimi durdurup bilinen işleri toplar. STOP dosyası
yeni gönderimi keser; bitmekte olan isteklerin çıktıları yine kaydedilir.

Kaynaklar, kanvaslar, prompt hashleri, çıktı hashleri, gerçek boyutlar ve
istek kimlikleri kalıcıdır. HTTP anahtarı yalnız özel ignored dosyadan okunur;
receipt, log ve Git'e yazılmaz. CDN çıktıları geldikçe yerel diske indirilir.
512 MiB süreç grubu RAM sınırı, basınç ve swap koruması kullanılır.

## İlk doğrulama ve ücret

İlk dört isteğin üçü tamamlandı; biri içerik kontrolü nedeniyle reddedildi ve
atlanmış olarak kaydedildi. Tamamlananların üçünde de gerçek boyut istenen
boyutla aynıydı. Dua Lipa'nın dikey ve Kesha'nın yatay çıktıları yerel dosya
olarak açılıp incelendi; bu iki örnekte yeşil yer tutucu kalmamıştı. Bu inceleme
bütün veri setinin estetik onayı veya kaynak alanı tespiti ölçümü değildir.
Özel oranlar için sıradaki sekiz kaynak küçük devam turunda gönderildi.
İlk 12 kaynak toplam 7 çıktı ve 5 içerik reddi verdi; yedi çıktının boyutu da
istekle aynıydı. Üretilen oranlar 3:4, 4:3, 16:9, 2:3, 5:4, 2:1 ve 3:1.
Diğer beş orandaki ilk kaynak içerik reddi aldığı için o oranlar henüz gerçek
bir çıktıyla doğrulanmış değildir. 3:1 ve 2:3 özel ölçülü iki çıktı da ayrıca
yerel dosya olarak açılıp incelendi. Diğer çıktıların tamamına estetik onay
verilmedi; kaynak bölgesi ölçümleri henüz yapılmadı.

İlk dört işin koordinatörü 61,49 saniye; örneklenen tepe RSS 67,83 MiB ve
fiziksel footprint 52,30 MiB ile bitti. Bu ölçümler örtüşür; toplanmaz.
Koruma örneklemelidir, çekirdek düzeyinde ani RAM sınırı değildir.
Sekiz yeni kaynaklık devam 119,66 saniye, 66,59 MiB RSS / 49,25 MiB footprint
ile tamamlandı; basınç normal ve swap sabitti. Ardından kalan kuyruğu en çok
dört açık istekle işleyen, RAM korumalı ayrı bir süreç başlatıldı. Sürecin
üst süresi altı saattir; hesap/teknik duruşta tamamlanmamış kayıtlar korunur.

Canlı pricing API `USD 1 / units` döndürür; bu görüntü başına 1 dolar demek
değildir. Tamamlanan cevapların `x-fal-billable-units` başlığı, bu oranla
çarpılarak istek bazında kaydedilir. İlk iki gözlenen örnek $0,0559 ve
$0,0451 bildirdi. Bunlar sağlayıcı başlığındaki tutarlardır; uzlaştırılmış
fatura olarak sunulmaz. Başlık yoksa ücret null/bilinmiyor kalır, sıfır
varsayılmaz. Verilen anahtar ile bakiye okuma HTTP 403 döndürdü; hesap
bakiyesi bu aşamada doğrulanamadı. Hesaba kredi ekleme veya ödeme yapılmadı.
İlk yedi başarılı çıktının bildirilen tutarları toplam $0,3497. Beş reddin
ücret başlığı yoktur; bunların ücreti sıfır varsayılmadı, bilinmiyor kaldı.

Kaynak bölgesi tespiti ve insan incelemesi sonraki aşamadır. Eğitim girdisi
teacher çıktısının tespit edilen bölgesinden kırpılacak, hedef aynı teacher
çıktısı olacak; eski orijinal kapak farklı piksellerle hedefe eşleştirilmez.
Şu aşamada hiçbir çıktı otomatik eğitim onayı almaz, eğitim başlatılmaz.

## Doğrulama ve kayıtlar

Yedi çevrimdışı test; oran kısıtları, yönlere uygun prompt, URL/yetki sınırı,
ret ve hesap hatası ayrımı, bilinmeyen ücret, bilinen isteği POST etmeden
geri okuma ve kayıp gönderim yanıtında kalıcı duruş durumlarını doğrular.
İlk test denemesinde çoklu context manager yazım hatası düzeltildi; sonra
yedi test geçti. Python sözdizimi ve diff denetimi de geçti.

Kod, test ve bu kayıt dosya başına bir commit ve tek push ile yayımlanır.
Space'te başlangıç ve ölçümler kaydedilir; kaynaklar, anahtar, kanvaslar ve
üretilen görseller özel ignored sonuçlarda kalır. Yeni alt ajan açılmadı.

Kaynaklar: [Sunburst edit API](https://fal.ai/models/openai/gpt-image-2.5/sunburst/edit/api),
[boyut ve kalite kuralları](https://fal.ai/gpt-image-2.5),
[kuyruk API'si](https://fal.ai/docs/documentation/model-apis/inference/queue),
[retry başlığı](https://fal.ai/docs/documentation/model-apis/common-parameters),
[bakiye yetkileri](https://fal.ai/docs/platform-apis/v1/account/billing).

## Bellek basıncı nedeniyle duruş ve sonuç kurtarma

18 Ekim'deki sonraki canlı kontrolde ana süreç durmuş bulundu. Koruma kaydı
585,47. saniyede sistem bellek basıncının 2'ye yükseldiğini gösterdi; işin
o andaki RSS'i yaklaşık 58,55 MiB, footprint'i 56,80 MiB idi. Süreç kendi
512 MiB sınırını aşmadı. Sistem basıncını hangi uygulamanın artırdığı bu
denetimde belirlenmedi; başka süreçler kapatılmadı veya koruma gevşetilmedi.

Durma anındaki 35 çıktı ve 24 ret korunmuştu. Basınç tekrar 1 olduğunda
bilinen üç fal istek kimliğinin sonucu yalnız GET ile geri alındı; yeni
üretim POST'u yapılmadı. Böylece yerel çıktı sayısı 38 oldu. Kurtarma işi
7,91 saniye, 46,23 MiB tepe RSS / 31,20 MiB footprint ile tamamlandı.

63. kaynağın kesinti anında `submitting` kaydı vardı, kalıcı request_id
yoktu. Kaynak zamanından filtrelenmiş fal request geçmişi yalnız bilinen
üç kimliği döndürdü. Geçmişte görülmemesi isteğin kesin kabul edilmediğini
kanıtlamadığı için 63. kaynak `uncertain` olarak korundu; yeniden gönderilmedi.
Kuyruk `paused_uncertain_submission`: 38 tamamlanmış, 24 ret, 1 belirsiz,
937 bekleyen. Yeni üretim yeniden başlatılmadı. Özel pressure-recovery.json
ve önceki state kopyası saklandı; tamamlanmış çıktı veya insan kararı silinmedi.
