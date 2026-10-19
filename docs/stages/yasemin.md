# Yasemin — 650 çıktı sınırı ve oranlara göre eğitim incelemesi

19 Ekim 2026. Kullanıcı üretimin 650 başarılı görselde durmasını, yeni
çıktılar için mevcut insan inceleme akışının hazırlanmasını ve farklı
oranların araya karışmadan sırayla incelenmesini istedi.

## Başarılı çıktı sınırı

Önce geçici STOP dosyasıyla açık işler toplandı; yeni gönderimler kesildi.
604 başarılı kayıt sonrasında kuyruk `completed_target=650` ile devam etti.
Sınır state içinde kalıcıdır ve yeniden başlatmada da uygulanır. Her açık
isteğin başarılı olabileceği varsayılarak hedefte bir yer ayrılır; tamamlanmış
sayı ile açık işlerin toplamı hedefe ulaştığında yeni POST gönderilmez.
Retler başarılı sayıyı artırmaz. Hedefe ulaşıldığında kalan kaynaklar pending
kalır; durum completed_target_reached olur. İlk belirsiz kayıt ayrı tutulur.

Test, hedefe ayrılmış tek açık iş IN_PROGRESS iken başka kapağın
gönderilmediğini ve başarıyla bitince pending kaynağa dokunulmadan durduğunu
doğrular. Prompt, high kalite ve 12 oran değiştirilmedi. Son ara kontrolde
638 başarılı çıktı vardı; bu kayıt tüm turun bittiği iddiası değildir.

## Ayrı canlı inceleme

Yeni yerel sayfa `http://127.0.0.1:54147/nilufer/` altında çalışır. Eski
Sümbül'ün sunucusu, medya arşivi ve insan karar dosyaları bu sayfaya taşınmaz
ve sıfırlanmaz. Yeni galerinin kararları kendi training-reviews.json ve
türetilmiş training-admission.json dosyalarına kaydedilir. Tüm kabul/ret ve
notlar çıktı hashine bağlanır; revizyon, aynı origin, outbox ve disk kaydı
kuralları korunur. Canlı siteye test oyu yazılmadı; yazma testleri geçici
fixture dosyalarında çalışır.

Orijinal ve genişletilmiş görsel aynı anda gösterilir. A/sol ok kötü,
D/sağ ok iyi, S kararsız/atla, Space kaynak sınırını açıp kapatır. Kaynak
alanı hesabı sürerken görsel incelenebilir; güvenilir alan bulunmadan eğitim
geometrisi veya otomatik eğitim kabulü uydurulmaz.

Varsayılan grup 3:4. Her grup kendi hızlı inceleme kuyruğuna sahiptir;
farklı oran yeni görsel geldi diye araya girmez. Seçili oran yenilemede
hatırlanır. Tur bitince aynı orana yeni gelenler veya sonraki oran kullanıcı
düğmesiyle seçilir. Sıra: 3:4, 4:3, 9:16, 16:9, 2:3, 3:2, 4:5, 5:4,
1:2, 2:1, 1:3, 3:1. Eski galerilerde oran gruplaması zorla açılmaz.

## Dört düğme ve dışa aktarılan geometri

Dikey çıktıda üstü/altı girdiye dahil et ve üstü/altı hedeften çıkar
düğmeleri korunur. Yatay çıktıda bunlar solu/sağı dahil et ve solu/sağı
çıkar olur. Hover veya klavye odağı kaydetmeden önizler; tıklama geometriyi
anında kaydeder. Çıkarılan kenarın girdiye dahil edilmesi çakışması temizlenir.
Kaynak poligonu kanıtı değiştirilmez; eğitim girdisi teacher çıktısından
alınan ayrı bir dikdörtgendir.

Sunucu eski top/bottom seçimlerini okumaya devam eder; yeni yatay left/right
alanları da katı bool doğrulamasıyla kabul edilir. Girdi, tutulmuş hedef,
hedef içindeki göreli girdi koordinatları ve kalan üretim alanı birlikte
dışa aktarılır. Dış alan kalmayan seçim eğitim adayına eklenmez. Onaylı
girdi/hedef önizlemesi aynı teacher dosyasını kullanır; orijinal kapak farklı
piksellerle hedefe yapıştırılmaz.

## Kaynak alanı testi ve yeni dosyalar

Fal completed kayıtları mevcut galeriye uyarlanmıştır. Ayarlar, oran,
provider, ücret bilgisi ve gerçek kaynak/çıktı referansları receipt'ten
okunur. Ücret bilinmiyorsa bilinmiyor kalır. Offline pipeline mevcut
source_fidelity serbest konumlu ORB/SIFT analizini, oran sırasıyla ve tek
çocuk süreçte çalıştırır; her beş testte galeri atomik yenilenir. Yeni
çıktılar üretildikçe aynı state'ten eklenir. API çağrısı, anahtar okuma,
yeni üretim, eğitim veya alt ajan bu analizde yoktur.

İlk pipeline denemesi ve üretim koordinatörü sistem bellek basıncı 2
olduğunda durdu. Basınç yeniden normal olduktan sonra mevcut kayıtlardan
devam edildi; korumalar gevşetilmedi. Analiz pipeline'ı artık yüksek
basınçta bekler; aynı offline analizi normal basınçta sürdürebilir. Çocuk
analizde 384 MiB süreç RSS sınırı ve basınç kontrolü, üretimde 512 MiB
koruma korunur. Pipeline kilidi ikinci örneği engeller. Örneklenen analiz
çocuğu yaklaşık 152 MiB RSS ölçtü; bütün koşunun tepe değeri değildir.

Ara yerel kontrolde 638 görsel yayınlanmıştı; 268 kayıt güvenilir kaynak
alanı bulmuş, 356 kayıt analiz bekliyordu. Bunlar o anın sayılarıdır.
3:4 ve 4:3 örneklerinin gerçek placement dosyaları açılarak incelendi.
Bir hizalama ölçümünün başarılı olması estetik kalite veya insan kabulü
anlamına gelmez. Kalan testler tamamlanınca da insan kararları belirleyicidir.

## Doğrulama ve yayın

11 kuyruk, 15 karar/sunucu, 4 galeri ve 4 training preview Python testi;
25 hızlı inceleme/geometri/gruplama ve 3 onaylı çift JavaScript testi geçti.
Yeni kapsam yatay kırpmanın kalıcı ve doğru koordinatlarla kaydedilmesini,
hover'ın salt önizleme olmasını, ayrı route kaydını, fal metadata'sını,
oran değişiminin açık seçim gerektirmesini ve yenilemede oran korunmasını
doğrular. Sayfa içi JavaScript ve Python sözdizimi kontrol edildi.

Yeni sayfa, data, karar API'si, manifest, scriptler ve onaylılar yolu HTTP
200/no-store döndürdü. Önceki tarayıcı URL politika sınırı nedeniyle canlı
tarayıcı tıklama/yerleşim testi yapılmadı; controller testleri ve yerel
placement görselleri tarayıcı ekran görüntüsü olarak sunulmaz.

On dakikalık otomasyonun mevcut PAUSED durumu korundu; yalnız tanımı yeni
650 hedefi ve offline galeri kontrolüne uyarlandı. Duraklatılmış görevin
çalıştığı iddia edilmez. Üretim ve yeni galerinin offline pipeline'ı ayrı
çalışan süreçlerdir. Kaynaklar, öğretmen çıktıları, insan seçimleri, anahtar,
özel state ve otomasyon yapılandırması Git'e girmez. Değişen kod/test/kayıt
dosyaları dosya başına bir commit, tek push ve RAM kontrolüyle yayımlanır;
Space'e gerçek sonuçlar eklenir.

Son kontrolde sistem basıncı 2 olduğundan Git koruması ilk dosya
eklenmeden yayını durdurdu; bu aşamanın commit/push işlemi henüz yapılmadı.
638 başarılı çıktı ve dört bilinen açık istek bulunuyordu. Özel `.build`
başlatıcısı basınç normal olana kadar en fazla altı saat bekler; üç normal
örnekten sonra boş kilitleri doğrulayarak benzersiz kayıtla yalnız bir
korumalı çalıştırma başlatır. STOP veya değişen hedefte çıkış yapar.
512 MiB, swap ve basınç korumaları ile 650 sınırı aynen korunur. Bu bekleyici
duraklatılmış heartbeat'i açmaz; üretimin çalıştığı iddiası değildir.

Sonraki canlı kontrolde basınç normale döndü; bekleyici aynı kuyruğu
korumayla sürdürdü ve üretim tam 650 başarılı çıktıda durdu. Son state:
650 completed, 186 refused, 1 held_uncertain, 163 pending; açık iş kalmadı.
Yeni POST gönderimi durdu. Şekil denetiminin son 12 dosyayı da kapsayan
1.040 çiftlik yerel taraması Ortanca kaydında ayrı tutulur.
