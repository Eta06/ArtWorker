# Akasya — klavyeyle hızlı eğitim verisi inceleme

18 Ekim 2026. Kullanıcı orijinal ve genişletilmiş kapakları aynı anda görerek iyi/kötü/emin değilim kararını kısayollarla vermek, her karara isteğe bağlı not eklemek ve son eğitim seçimlerini kaydetmek istedi.

## Uygulama

Sumbul sayfasına Hızlı incele düğmesi ve iki sütunlu inceleme penceresi eklendi. Mobil genişlikte de orijinal ve tam genişletilmiş görüntü yan yana kalır; görüntüler oranı korunarak sığdırılır. Görseller küçük önizleme yerine orijinal dosya ve raw.png üzerinden gösterilir. Yalnızca sıradaki bir görsel çifti önceden yüklenir.

Sol ok veya A kötü, sağ ok veya D iyi, Space emin değilim / atla kararıdır. Karar ve isteğe bağlı 500 karakterlik not, mevcut fingerprint ve revision kontrollü kayıt hizmetine tek istekte gönderilir. Kayıt başarılı olduktan sonra sıradaki görsele geçilir. Hata durumunda not ve geçerli görsel korunur. İstek sürerken ikinci karar, kapatma veya Escape ile yanlışlıkla ilerleme engellenir. İyi/kötü kararı iki görsel yüklenene kadar kapalıdır; yükleme hatasında atlama mümkündür.

Metin alanlarında yazarken, IME kullanımında, değiştirme tuşlarıyla veya basılı tutulan tuşun tekrarında karar kısayolları çalışmaz. Not alanında Enter kısayollara geri döndürür. Önceki düğmesi karar düzeltmeye izin verir. Normalde filtredeki incelenmemiş/kararsız görsellerden sabit sıra oluşturulur; açık kabul/ret filtresi varsa o kararlar yeniden incelenebilir. Yeni kararlar bu sırayı yeniden sıralamaz. Atlananlar aynı turda döngüye girmez; tur sonunda tekrar incelenebilir. Tur sonunda karar yedeği ve eğitim adayı manifesti indirilebilir.

Kayıt/admission politikası değişmedi: pending ve rejected örnekler eğitim dışında, accepted fakat güvenilir kayıt bölgesi bulunmayan örnekler blocked durumundadır. Model üretimi veya eğitim başlatılmadı. Önceden önerilen teacher görüntüsünden kaynak kesiti çıkarma veri hazırlama yöntemi bu arayüz aşamasında uygulanmadı.

## Doğrulama

Beş çevrimdışı JavaScript denetleyici testi geçti: notlu atlamanın kayıttan sonra ilerlemesi; hata ve hızlı çift basış; incelenmemiş sıra ile açık karar filtresi; düzenlenebilir alan/IME/modifier/tekrar korumaları; görseller yüklenmeden karar engeli, aynı görünümde önce/sonra dosyaları ve notun başka kapağa taşınmaması. Mevcut dört karar/kayıt Python testi de geçti. Bunlar tarayıcı görsel testleri değildir.

Yerel index, iki JavaScript dosyası ve karar API HTTP 200 döndürüyor; JavaScript sözdizimi ve diff boşluk kontrolleri geçti. Gerçek arşive test kararı eklenmedi. Önceki aşamada Browser Use URL politikası yerel sayfaya erişimi reddettiğinden tarayıcı erişimi başka yoldan denenmedi; canlı klavye ve görsel düzen doğrulaması henüz yapılmadı.

## Yayın

Altı dosya dosya başına bir commit, tek push ve sürekli RAM/basınç kontrolüyle yayımlanır. Özel kararlar, manifestler, görseller ve notlar Git'e alınmaz. Yerel inceleme hizmeti aynı 127.0.0.1:54146/sumbul/ adresinde kaldı. Sonraki iş kullanıcının hızlı incelemeyi tamamlaması ve kabul ettiği adaylardan ayrı veri hazırlama aşamasının kurulmasıdır.
