# Mimoza — kalıcı otomatik kayıt ve kaynak sınırı kısayolu

18 Ekim 2026. Kullanıcı kararların anında kaydedilmesini, Nano Banana çıktılarının büyük bir etiketle belirtilmesini, güvenilir kaynak bölgesinin isteğe bağlı gösterilmesini ve Space yerine S ile atlanmasını istedi. Kullanıcının açık isteği doğrultusunda geçmiş kaybolan kararlar araştırılmadı, kurtarma taraması yapılmadı.

## Otomatik kayıt

Yeni her karar, ağ isteğinden önce tarayıcıdaki kalıcı bekleyen işlem kuyruğuna yazılır. Ardından mevcut SHA-256/revision denetimiyle yerel dosyaya kaydedilir; tam işlem kimliği, karar, not ve görsel parmak izi onayı gelmeden hızlı inceleme ilerlemez. Bekleyen işlemler aynı origin üzerinde sayfa yenilenince, bağlantı geri gelince ve dört saniyelik yeniden denemelerde sırayla gönderilir. Sunucudaki request_id denetimi yanıtı kaybolan bir isteğin tekrarının kararı iki kez değiştirmesini engeller. Sonraki not kaydı bir kabul kararını pending durumuna çeviremez. Başka sekmeyle revision çakışmasında dosyanın üzerine otomatik yazılmaz; kullanıcının yeniden karar vermesi gerekir.

Notlar yazarken cihazda taslak olarak tutulur ve 700 ms duraklamadan sonra ayrıca dosyaya otomatik kaydedilir. Karar düğmesine basmak not beklemesini iptal eder ve notu kararla birlikte gönderir. Hızlı incelemenin sırası, konumu ve taslakları sayfa yenilemesinden sonra devam etmek üzere tarayıcıda tutulur. Dosya onayı ve bekleyen kayıt mesajı hem galeride hem hızlı inceleme penceresinde görünür. Tarayıcı depolaması kullanıcı tarafından temizlenirse onaylanmamış istemci taslakları korunamaz; onaylanmış kararların asıl kaydı özel results dizinindeki dosyadır.

İlk kontrolde yerel sunucu dinlemiyordu. Kayıt sunucusu terminal isteğinden bağımsız, ayrı oturumda ve log dosyasına yazarak yeniden başlatıldı; üst süreci PID 1 olarak doğrulandı. İlk HTTP kontrolü sunucu hazır olmadan geldiği için bağlantı reddedildi; hazır olduktan sonraki kontroller başarılıydı. Giriş sayfası ve script yanıtları eski kodun yeniden kullanılmasını önlemek için no-store başlığı döndürüyor. Makine yeniden başlatıldığında sunucunun yeniden çalıştırılması gerekir; yeni login servisi kurulmadı.

## Görsel ve kısayollar

Nano Banana çıktılarının hızlı ve ayrıntılı görünümünde büyük sarı NANO BANANA etiketi var. Kartlarda da model adı belirginleştirildi. Orijinal ve genişletilmiş görüntü yan yana kalır.

Space veya kaynak bölgesi düğmesi yalnızca güvenilir eşleşme varsa turuncu çokgeni açar/kapatır. S veya emin değilim düğmesi atlar. A/sol ok kötü, D/sağ ok iyi kararıdır. Kaynak sınırı her yeni görselde kapalı başlar; object-fit:contain görünümünde SVG xMidYMid meet ile aynı en-boy oranını korur. Güvenilir eşleşme yoksa tahmini sınır çizilmez. Metin yazarken kısayollar devre dışı kalır.

## Doğrulama ve sınırlar

On iki çevrimdışı JavaScript akış testi ve beş Python kayıt/HTTP testi geçti. Testler kaynak sınırının karar vermeden açılıp kapanmasını, güvenilmez sınırın gösterilmemesini, Nano etiketini, ağdan önce kalıcı niyeti, yenileme sonrası yeniden göndermeyi, peş peşe bekleyen işlemlerin doğru revision ile gitmesini, otomatik not kaydını, eksik onayda ilerlememeyi ve sunucu yeniden açıldıktan sonra aynı request_id ile güvenli tekrarı kapsıyor. Testler yalnızca geçici fixtures kullanır; gerçek arşive yapay inceleme kararı yazılmadı. Yerel sayfa, iki script ve karar API HTTP 200/no-store ile doğrulandı. Script sözdizimi ve git diff --check geçti.

Önceki Browser Use URL politikası engeli nedeniyle farklı bir tarayıcı erişim yolu denenmedi. Canlı klavye ve görsel yerleşim doğrulaması yapılmadı; SVG hizası ve etiket okunabilirliği canlı gözle henüz doğrulanmış değil. Yeni üretim veya öğrenci eğitimi başlatılmadı. Eğitim adaylığı ve güvenilir kaynak bölgesi şartları değişmedi.

## Yayın

Yedi değişen dosya, dosya başına bir commit ve tek push ile kaydedilir. RAM/basınç izleme açıktır. Yerel fotoğraflar, karar dosyaları, kuyruklar, notlar ve süreç kayıtları Git dışında kalır.
