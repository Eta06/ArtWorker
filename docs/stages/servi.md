# Servi — belirsiz işi ayırarak üretime devam ve on dakikalık kontrol

18 Ekim 2026. Kullanıcı üretimin yeniden başlatılmasını ve her on dakikada
bir kontrol edilmesini açıkça istedi. Nilüfer'in mevcut 1.000 kaynaklık
kuyruğu, fal Sunburst high, prompt ve 12 kare olmayan oran korunarak devam eder.
Yeni model, kaynak toplama, Gemini fallback veya eğitim başlatılmaz.

## Kuyruk düzeltmesi

Tek bir belirsiz gönderim bütün diğer kapakları gereksiz bekletiyordu.
Yeni `--hold-uncertain` seçeneği, kalıcı fal request_id bulunmayan eski
`submitting`/`uncertain` kaydını `held_uncertain` olarak ayrı tutar. Kaynak ve
receipt silinmez; gerekçe ve zaman kaydedilir. Bu kayıt yeniden `pending`
yapılmaz ve hiçbir sonraki kapakla beraber tekrar POST edilmez.

Receipt içinde request_id ve geçerli fal durum/sonuç URL'leri zaten varsa,
kayıt ayrı tutulmak yerine `submitted` olarak kurtarılır ve yalnız GET ile
okunur. Seçenek verilmeden eski belirsizlik koruması çalışmaya devam eder.
Yeni bir POST sırasında yanıt kaybolursa o iş yine belirsiz duruş üretir;
güvenli sonraki devam bu kaydı ayrı tutabilir. Eksik kimlik için başarı,
başarısızlık veya sıfır ücret uydurulmaz.

63. kapak ayrı tutulduktan sonra 64. ve sonraki kapaklar fal'a gönderildi.
Başlangıç kontrolünde 38 çıktı, 25 ret, 1 ayrı tutulan belirsiz, 4 açık istek
ve 932 bekleyen vardı; çalışan süreç ve yeni fal istek kimlikleri doğrulandı.
Süreç RAM/basınç/swap koruması altında, en çok dört açık isteğe devam eder.
512 MiB sınırı veya sistem basıncı duruşu gevşetilmedi. Yaklaşık 66 MiB RSS
ve normal basınç gözlendi; bu bir yeni tam koşu tepe ölçümü değildir.

## On dakikalık kontrol

Bu sohbet üzerinde ArtWorker 10 dk üretim kontrolü adlı aktif heartbeat
oluşturuldu. Kaydedilen yapılandırmada on dakikalık aralık ve bu sohbet
hedefi doğrulandı. Kontrol; gerçek state, süreç, guard/log, çıktı dosyaları,
sayılar ve macOS RAM basıncını okur. Normal ilerlemede sessiz kalır; durma,
önemli hata, tamamlanma veya kullanıcı eylemi gerektiğinde haber verir.

Teknik duruşta sistem basıncı normalse ve ikinci süreç bulunmuyorsa aynı
kuyruk benzersiz guard/başlatma kaydıyla yeniden başlatılabilir. Bilinen
kimlikler tekrar üretilmez; belirsizler ayrı tutulur. Bakiye/yetki veya
beklenmeyen boyut hatasında ücretli tekrar yapılmaz. RAM basıncı yüksekken
başlatılmaz; başka uygulamalar kapatılmaz. Kuyruk bitince son rapor verilip
heartbeat durdurulur. Yerel dosyalara erişen kontrol için bilgisayar ve
uygulama açık kalmalıdır; bu çalışma şekli
[OpenAI Docs](https://learn.chatgpt.com/docs/automations?surface=app) ile de
kontrol edildi. Otomasyonun oluşturulması ilk zamanlanmış turun çalıştığı
anlamına gelmez; ilk gelecek tetikleme bu aşama içinde beklenmedi.

## Doğrulama ve yayın

Dokuz çevrimdışı test geçti. Yeni testler belirsiz kapağın tekrar
gönderilmemesini, sonraki kapağın ilerlemesini ve receipt'teki bilinen
kimliğin yalnız GET ile kurtarılmasını doğrular. Canlı yeniden başlatmada
yeni istek kimlikleri ve çalışan koruma süreci doğrulandı. Eski seçimler,
kaynak manifestler ve üretim dosyaları korunur; yeni alt ajan açılmadı.

Kod, test ve bu kayıt dosya başına bir commit, tek push ile yayımlanır;
Space'e sonuç eklenir. Anahtar, otomasyonun yerel yapılandırması, kaynak
görseller ve üretim receipt'leri özel ignored dizinlerde kalır.
