# Kayrak — ret listesini doğrudan açma

18 Ekim 2026. Kullanıcı eski retlerini üstü/altı dahil et ve üstü/altı sil
kontrolleriyle yeniden değerlendirmek istedi. /sumbul/?review=rejected
bağlantısı sayfa açılışında ret filtresini seçer. Mevcut Reddettiklerimi yeniden
incele düğmesi bütün retleri sabit bir hızlı inceleme sırasına alır.

Son dosyada 82 ret var. 80'inde kaynak eşleşmesi hazır olduğu için dört eğitim
alanı kontrolü kullanılabilir. İkisinde kaynak alanı henüz bulunamadığından
kontroller kapalı kalır; bu kayıtlar listeden gizlenmez. Alan tercihini değiştirmek
ret kararını otomatik kabule çevirmez. Kullanıcı kurtardığı örneği ayrıca iyi
olarak işaretlemelidir. Kararlar ve notlar mevcut otomatik kayıt akışını izler.

JavaScript sözdizimi, git diff --check ve doğrudan bağlantıda HTTP 200/no-store
doğrulandı. Gerçek karar dosyasına yazılmadı, sunucu yeniden başlatılmadı.
Önceki Browser Use URL politika engeli nedeniyle canlı tarayıcı yerleşim/tıklama
doğrulaması yapılmadı. Yeni model veya görüntü üretimi yok.

İki dosya dosya başına bir commit, tek push ve RAM denetimiyle yayımlanır;
ArtWorker Space aşama kaydı güncellenir. Kullanıcı verileri özel arşivde kalır.
