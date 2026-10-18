# Çiğdem — insan kararıyla eğitim verisini eleme

18 Ekim 2026. Kullanıcı kötü bulduğu çıktıları seçerek eğitim verisinin dışında tutmak istedi. Mevcut /sumbul/ arşivinin görünümü korundu; ücretli üretim başlatılmadı ve ham görseller değiştirilmedi. Arşivdeki son kayıt 452 çıktı ve teacher_pipeline_budget_stop durumunda.

## Davranış

Her çıktının detayında Eğitime al, Eğitimden çıkar ve Kararsız / incelenmedi kararları var. İsteğe bağlı 500 karakterlik not, sonraki incelenmemiş çıktıya geçme, karar filtresi, kart durum etiketleri ve karar sayıları eklendi. Kararlar anında yerel sunucuya gönderilir; kaydedilmeyen karar başarılı gösterilmez. JSON yedek ve eğitim adayı manifesti indirilebilir.

Kararlar görsel kimliği ve raw_sha256 ile bağlıdır. Aynı kimliğin görüntüsü değişirse önceki karar taşınmaz. Sekmeler arasında eski sürümle gelen kayıt 409 ile reddedilir. Eski beğeni kayıtları tarayıcıdan silinmez, eğitim kabulüne otomatik çevrilmez. İlk kurulumda hiçbir çıktı kabul edilmez.

Yerel HTTP sunucusu eski arşiv dizinini 127.0.0.1:54146 üzerinden sunar. training-reviews.json kararların asıl kaydı, training-admission.json türetilen aday listesidir; ikisi de Git dışında kalan özel results dizinindedir. Sadece açıkça kabul edilen ve güvenilir kaynak eşleşmesi bulunan çıktılar candidates listesine girer. Reddedilenler ve incelenmeyenler excluded listesinde kalır. Kabul edilip eşleşmesi güvenilmez bulunanlar blocked listesinde bekler. Manifest eğitim başlatmaz; henüz öğrenci eğitim hattı çalıştırılmadı.

## Doğrulama

Dört karar/kayıt testi geçti: açık kabul ve güvenilir eşleşme koşulu; yeniden açma, kararsıza alma ve değişen çıktıya karar taşımama; eşzamanlı karar çakışması ve geçersiz isteğin önceki kaydı bozmaması; gerçek HTTP üzerinden kayıt ve yabancı origin isteğinin reddi. Dört mevcut galeri testi de geçti. JavaScript sözdizimi ve git diff --check kontrolleri geçti.

Yerel index, JavaScript, karar okuma ve manifest uçları HTTP 200 döndürüyor. Başlangıçta 0 karar, 0 aday ve 452 inceleme bekleyen çıktı var; gerçek arşive sahte insan kararı eklenmedi. Browser Use aracı yerel sayfaya erişimi URL politikası nedeniyle reddetti. Alternatif tarayıcı veya dolaylı erişim denenmedi; canlı tıklama ve görsel düzen doğrulaması bu aşamada yapılamadı.

## Kullanım

Yerel inceleme sunucusunu depo kökünden başlat:

```sh
python3 experiments/openrouter_outpaint/review_server.py --directory experiments/openrouter_outpaint/results/2026-10-06 --gallery experiments/openrouter_outpaint/results/2026-10-11/sumbul --port 54146
```

Aynı portta eski bir statik sunucu çalışıyorsa önce o sunucunun kapatılması gerekir. Tarayıcıda mevcut Sumbul sayfasını yenile. Eğitimden çıkar kararı ham çıktıyı silmez; karar geri alınabilir. Kararların yerel dosyaları ve yedekleri üçüncü taraflara yayımlanmaz.

Yayın dosya başına bir commit, tek push ve sürekli RAM/basınç kontrolüyle yapılır. Sonraki adım kullanıcının görsel incelemesi ve kabul edilen adayların gerçek eğitim hazırlığında tekrar dosya/checksum doğrulamasından geçirilmesidir.
