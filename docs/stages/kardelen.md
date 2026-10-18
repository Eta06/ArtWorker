# Kardelen — indirme kesintisinden sonra aynı sonuçla devam

19 Ekim 2026. Kullanıcının canlı durum sorusu üzerine Nilüfer'in dosyaları,
süreçleri, guard kayıtları ve sistem bellek basıncı yeniden denetlendi.
Mevcut Sunburst high promptu, 12 oran, kaynak koleksiyon ve insan seçimleri
değiştirilmedi; yeni alt ajan açılmadı.

## Bulunan duruş

595 görüntü yerel diskteydi, 175 kapak reddedilmişti. Dört bilinen fal
isteği ve 225 bekleyen kayıt vardı; 63. kapak ayrı tutulan belirsiz olarak
korunuyordu. State eski running bilgisini taşımasına rağmen süreç yoktu.
Guard, subprocess_exit ile bitmişti. Traceback CDN indirmesindeki
ConnectionResetError hatasını gösterdi. Önceki RAM basıncı duruşundan farklı
olarak bu hata sırasında sistem basıncı normaldi ve bellek sınırı aşılmadı.

## Düzeltme

Durum, sonuç ve çıktı indirme GET işlemlerinde ConnectionError ailesi ve
http.client.IncompleteRead artık yakalanır. Kalıcı istek kimliği korunur;
aynı GET okuması sonraki kontrolde yinelenir. Yeni bir görüntü üretim POST'u
yapılmaz. Son bilinen ücret başlığı ve okuma hatasının türü/zamanı receipt'e
yazılır. Ücretli POST hatasının belirsizliği, hesap hataları, bozuk çıktı
boyutları veya disk hataları bu değişiklikle sessizce yutulmaz.

Dokuz eski test ve bir yeni test, toplam on test geçti. Yeni test iki
ayrı hata durumunu (connection reset ve eksik HTTP gövdesi) canlandırır:
aynı generation kimliğiyle dört GET çağrısı, iki indirme denemesi ve sıfır
üretim POST'u; doğru nihai durum ve tek ücret kaydı doğrulanır.

## Kurtarma ve devam

Yeniden başlatmadan önce eski çalışan süreç bulunmadığı, global ağır iş
kilidi ve kuyruk kilidinin serbest olduğu doğrulandı. İlk 775 kaydın yalnız
bilinen sonuçları okundu: dört çıktı da geri alındı, toplam 599 oldu.
Kurtarma 9,40 saniye, örneklenen tepe RSS 51,67 MiB / fiziksel footprint
36,73 MiB ile bitti. Bu iki bellek ölçümü örtüşür; toplanmaz.

Kurtarılan dört yanıtın ücret başlığı yoktu. Ücretleri bilinmiyor olarak
korundu; ücretsiz veya ikinci kez üretilmiş olarak raporlanmadı. O andaki
bilinen sağlayıcı tutarları toplam $29,0013; retler, belirsiz iş ve bu dört
sonucun bilinmeyen tutarları buna dahil değildir.

Ardından benzersiz resume2 guard/başlatma kaydıyla kalan 225 kapak için
aynı kuyruk yeniden başlatıldı. En çok dört açık istek, 512 MiB süreç grubu
sınırı, sistem basıncı ve swap koruması korunur. Daha önceki üretimler tekrar
gönderilmez; ayrı tutulan belirsiz kayıt atlanır. Bu kayıt bütün yeni
çıktıların tamamlandığı veya eğitim için onaylandığı anlamına gelmez.

On dakikalık heartbeat'in yerel yapılandırması PAUSED bulundu; uygulamadaki
kayıt da açıldı. Neden duraklatıldığı belirlenmedi; mevcut duraklatma
kendiliğinden kaldırılmadı. Yeniden çalışan üretim süreci bu görevden
bağımsızdır. Kontrolün otomatik devam ettiği iddia edilmez.

Kod, test ve bu kayıt dosya başına bir commit, tek push ve RAM korumasıyla
yayımlanır. Space'e gerçek durum eklenir. Anahtar, görseller, özel state,
receipt'ler ve otomasyon yapılandırması Git'e eklenmez.
