# ArtWorker — outpainting model araştırması

İlk araştırma: 30 Eylül 2026; deney durumu güncellemesi: 5 Ekim 2026. Bu belge ilk aday seçiminin kaynak araştırmasını ve sonradan yapılan kontrolleri ayırır. Qwen 2.1, Klein, DreamLite, Mobile-O, Viggle v0.3, Qwen ControlNet ve FLUX.1 Fill ağırlıkları indirildi; gerçek yerel denemeler ayrı [deney raporunda](OUTPAINT_TRIAL_RESULTS.md) tutuluyor. Player'a henüz bir model bağlanmadı.

## İlk aday seçimi

İlk kalite karşılaştırmasına **Qwen-Image-2.1**, mobil ürün ve küçültme araştırmasına **FLUX.2 Klein 4B** ile **HiDream-O1-Image Full** alınmalı. Bu bir değerlendirme önerisidir; albüm kapağı outpainting testinin sonucu değildir.

30 Eylül'de incelenen Artificial Analysis genel görüntü düzenleme sıralamasında Qwen-Image-2.1 açık ağırlıklı modeller arasında öndeydi. Aynı sıralama bütün modellerde GPT Image 2.5 Sunburst (max)'ı önde gösteriyordu. Bu tarihli gözlem 5 Ekim'de yeni bir leaderboard doğrulaması değildir; ayrıca kare albüm kapağını telefon ekranına genişletirken orijinal pikselleri koruyan özel bir outpainting testi değil. [Sıralama ve yöntem](https://artificialanalysis.ai/image/leaderboard/editing)

## 5 Ekim: gerçek görev sonucu

**Henüz kabul edilmiş bir outpainting öğretmeni yok.** Daha yeni adapter veya görev için eğitilmiş model kullanmak bu kapakta tek başına yeterli olmadı. Viggle v0.3 altı adım ve resmî yedi Turbo + iki Base adımı çalıştırıldı; korkuluk birleşimi düzelmedi. Kaynak sınırı sampler, compositor, VAE projection ve LanPaint-style kontrollerinde bazı renk/doku ölçümleri iyileşti, geometri başarısı sağlanmadı. [Tamamlanan kontrol özeti](../experiments/qwen/BOUNDARY_FIX_RESULTS.md), [v0.3 sonuçları](../experiments/qwen/TURBO_TAIL_RESULTS.md).

İndirilen eğitimli Qwen ControlNet'in tam ve sparse koşulları ile gerçek kaynak-reference varyantları çalıştırıldı. Yerelleştirilmiş tamamlanmış hint'ler tek sahne kompozisyonunu iyileştirdi; korkuluğun dış kenarı kaynakta beklenen konumdan **−17,404 px yatay kaydı** ve fazladan metal bant kaldı. Küçük açı farkı doğru birleşim anlamına gelmedi. Native FLUX.1 Fill Q4, 50 adımda üstte yazı ve altta ikinci araba/kişi sahnesi üretti; bütün görüntü kabul edilmedi. Bunlar tek kapak/seed üzerinde yerel bulgular, genel model sıralaması değildir. [ControlNet](../experiments/qwen/CONTROLNET_RESULTS.md), [yerelleştirilmiş sonuç incelemesi](../experiments/qwen/boundary_eval/2026-10-05/controlnet-localized40-comparison/REVIEW.md), [Fill50](../experiments/flux1_fill/FILL_RESULTS.md).

**Sonraki saved-top32 kontrolü asıl korkuluk kusurunu belirgin düzeltti.** Ortak gerçek conditioning/noise/sigma tensorları aynen geri yüklenip yalnız kaynak üst 32 px şeridindeki 64 hint hücresine weight1 verildi. Aynı dış kenarın yatay endpoint hatası **−17,404 → −2,086 px**, yakın açı farkı **+0,341°** oldu. Son satırda zayıf edge/3,930 px jog ve su/asfalt dikişleri kaldığı için bütün görüntü hâlâ kabul edilmedi; bu gerçek yerel geometri ilerlemesidir. [Saved-top32 gerçek inceleme](../experiments/qwen/boundary_eval/2026-10-05/controlnet-saved-top32-comparison/REVIEW.md).

Fill'in iki bağımsız local-context50 çağrısı üst gölü daha makul devam ettirdi; alt iki soluk bant ve doku hatası kaldı. Native SDXL 9-channel inpaint20 de gerçek çalıştı, dış alan neredeyse siyaha çöktü; maske/noise kontrollerinde terslik bulunmadı. Bunlar başarılı streaming veya kabul edilmiş öğretmen değil. MaskFlow için **2511'e uygun izole MLX-Gen portu ve model indirme**, ProMax author outpainting tarifi için ağırlık indirme başladı; henüz GPU kalite sonuçları yok. Qwen 2.1 ile 2511'in packed genişliği eşit olsa da codec'leri uyumsuz, adapter doğrudan taşınamaz. [Local Fill kayıtları](../experiments/flux1_fill/runs/2026-10-05/track3-local-context50-seed42/metrics.json), [SDXL sonucu](../experiments/sdxl_inpaint/SDXL_RESULTS.md), [MaskFlow hazırlık audit'i](../experiments/maskflow/research/2026-10-05/FEASIBILITY.md), [ProMax tarifi](../experiments/sdxl_inpaint/PROMAX_FEASIBILITY.md).

Gerçek altı-adımlı growing localized-control deneyi de tamamlandı: 0 / 0,5 / 1 scale kollarında gelecek target tokenları hem base hem control girdisinden gerçekten çıkarıldı. Bütün sahne tutarlı kaldı; korkuluk profilindeki basamak/ek kenar hiçbir kolda çözülmedi. Bu, büyüyen compute kanıtı; kalite kabulü değil. Tam tuval ControlNet/Fill sonuçlarıyla ayrı tutulur. [Gerçek growing görüntü incelemesi](../experiments/qwen/boundary_eval/2026-10-05/controlnet-growing-localized6-comparison/REVIEW.md). Küçük öğrenci eğitimi ve fiziksel iPhone inference yapılmadı. Öğretmen seçimi önce bütün sahne, kaynak sınırı ve farklı kapak/seed kalite kabulünü geçmeli.

## Kısa liste

| Aday | Doğrulanan özellik | ArtWorker için rolü / sınırı |
| --- | --- | --- |
| Qwen-Image-2.1 | 20 Eylül 2026; birleşik üretim ve düzenleme. Yaklaşık 7B görüntü transformer'ı yanında Qwen3-VL-8B encoder ve VAE bulunuyor. | Güncel kalite referansı. “7B” toplam çalışma yükünü anlatmıyor. Research License nedeniyle ticari kullanım ayrıca lisans gerektiriyor. [Model](https://huggingface.co/Qwen/Qwen-Image-2.1), [mimari](https://github.com/QwenLM/Qwen-Image-2.1), [lisans](https://github.com/QwenLM/Qwen-Image-2.1/blob/main/LICENSE) |
| FLUX.2 Klein 4B Base / distilled | 15 Ocak 2026; 4B varyantları Apache 2.0. Base adaptasyon için, distilled sürüm az adımlı üretim için sunuluyor. Ayrı Qwen3-4B encoder ve VAE gerekiyor. | Yerel çalışma ve öğrenci model geliştirme için ilk pratik aday. Hazır iPhone performansı doğrulanmadı. 9B varyantının lisansı farklı. [Resmî repo](https://github.com/black-forest-labs/flux2), [distilled model](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B), [base model](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) |
| HiDream-O1-Image Full | 8 Mayıs 2026; 8B birleşik pixel-level mimari, ayrı zorunlu text encoder / VAE yok. Kod ve modeller MIT. Full sürümü edit için öneriliyor. | Mimari sadeliği nedeniyle mobil küçültme araştırmasına değer. Genel editing sıralaması Qwen'den düşük; özel outpainting ve iOS portu henüz doğrulanmadı. Opsiyonel prompt agent ek model yüküdür. [Resmî repo](https://github.com/HiDream-ai/HiDream-O1-Image), [model](https://huggingface.co/HiDream-ai/HiDream-O1-Image) |
| FLUX Outpainting | 14 Mayıs 2026; görsel, hedef canvas ölçüleri ve kaynak konumu alan özel genişletme servisi; 4MP'ye kadar. | Göreve doğrudan uygun kalite karşılaştırması. Sunulan erişim API; telefonun içinde çalışacak açık ağırlık paketi doğrulanmadı. [Resmî duyuru](https://bfl.ai/blog/outpainting-extend-any-image-in-any-direction), [dokümantasyon](https://docs.bfl.ai/flux_tools/flux_outpainting) |

## Diğer incelenen adaylar

- **HunyuanImage-3.0-Instruct:** genel editing sıralamasında güçlü; 80B toplam / 13B aktif MoE. Aktif parametre sayısı depolanması gereken bütün ağırlıkların boyutunu göstermez. Telefon için doğrudan başlangıç değil. [Model kartı](https://huggingface.co/tencent/HunyuanImage-3.0-Instruct)
- **FireRed-Image-Edit-1.1:** Apache 2.0 lisanslı edit modeli; yaklaşık 20B görüntü modeli yanında büyük bir encoder var. Lisans açısından öğretmen / kalite kontrol adayı; burada üstün outpainting sonucu ölçülmedi. [Repo](https://github.com/FireRedTeam/FireRed-Image-Edit), [model kartı](https://huggingface.co/FireRedTeam/FireRed-Image-Edit-1.1)
- **Krea 2 Raw / Turbo:** Haziran 2026'da yayımlanan estetik üretim adayları. Resmî kullanım ağırlıklı olarak text-to-image; outpainting için ek yöntem gerekiyor. Custom lisansın gelir ve türev adlandırma koşulları var. [Teknik rapor](https://www.krea.ai/blog/krea-2-technical-report), [lisans](https://www.krea.ai/krea-2-licensing)
- **Ideogram 4:** Haziran 2026; ücretsiz ağırlıkların lisansı noncommercial. Çıktılarla rekabetçi model geliştirme kısıtı nedeniyle varsayılan distillation öğretmeni seçilmemeli. [Repo](https://github.com/ideogram-oss/ideogram4), [model lisansı](https://github.com/ideogram-oss/ideogram4/blob/main/model_licenses/LICENSE-IDEOGRAM-4-NON-COMMERCIAL)
- **FLUX.1 Fill:** native maskeli üretim için yerel Q4/50-step, exterior-only prompt ve iki local-context çağrısı çalıştırıldı; bütün görüntü kalite kabulü sağlanmadı. FLUX.1 dev Non-Commercial koşulları geçerli. **PowerPaint / BrushNet** burada çalıştırılmadı. [Gerçek Fill deneyleri](../experiments/flux1_fill/FILL_RESULTS.md), [Fill](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev), [PowerPaint](https://github.com/open-mmlab/PowerPaint), [BrushNet](https://github.com/TencentARC/BrushNet)

## iPhone ve distillation

Bulunan FLUX.2 Swift MLX uygulaması outpainting zincirleri içeriyor, ancak mevcut proje macOS 15+ hedefli ve Apple Silicon Mac için belgelenmiş. MLX-Swift'in iOS desteği, bu Mac projesinin hazır iPhone portu olduğu anlamına gelmiyor. Telefon üzerinde süre, en yüksek bellek kullanımı ve ısınma ölçülmedi. [Package.swift](https://github.com/VincentGourbin/flux-2-swift-mlx/blob/main/Package.swift), [README](https://github.com/VincentGourbin/flux-2-swift-mlx), [outpainting rehberi](https://github.com/VincentGourbin/flux-2-swift-mlx/blob/main/docs/INPAINTING_GUIDE.md)

Resmî Diffusers checkpoint dosyaları birlikte sayıldığında Qwen-Image-2.1 için yaklaşık **33,1 GB**, Klein 4B için **16,0 GB** tutuyor. Qwen DiT ve encoder BF16; VAE checkpoint'i F32. Aynı transformer'ın farklı paket biçimindeki kopyaları iki kez sayılmadı. Bunlar dosya boyutlarıdır; runtime bellek ölçümü değildir. Aktivasyon, attention buffer, cache ve player belleği bunlara dahil değil. Quantization ve bileşenleri sırayla yükleme bu bütçeyi değiştirebilir. [Qwen dosyaları](https://huggingface.co/Qwen/Qwen-Image-2.1/tree/main), [Klein dosyaları](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/tree/main)

Klein'ın distilled 4B sürümü daha az sampling adımı kullanıyor fakat hâlâ 4B bir görüntü modeli. Adım azaltma ile parametre / bellek küçültme ayrı hedefler. Encoder, VAE, aktivasyonlar ve quantization overhead toplam bellek bütçesine dahil edilmeli. Telefon hedefi için daha küçük öğrenci mimarisi ve quantization birlikte değerlendirilmelidir; yalnız “distilled” adı yeterli değildir.

Qwen'ın lisansı çıktılardan eğitilen dağıtılabilir modeller için atıf koşulu içeriyor; ticari kullanım izinleri öğretmen seçiminden önce çözülmeli. Apache / MIT adayları bu nedenle ayrı tutuldu.

## Outpainting yöntemi ve seçim testi

LanPaint ayrı bir model değil, farklı modelleri maskeli üretime uyarlayan bir sampling yöntemi. Güncel kod Qwen-Image-2.1, Klein ve başka modelleri destekliyor. Makaledeki karşılaştırmalar güncel modellerin albüm kapağı genişletme yarışması değil; makale distilled modellerde kalite düşüşü de bildiriyor. Dört adımlı sürümü LanPaint'e bağlayınca aynı kaliteyi alacağımız varsayılmamalı. [Kod](https://github.com/scraed/LanPaint), [makale](https://arxiv.org/html/2502.03491v3)

5 Ekim'de Qwen Turbo üzerinde izole LanPaint-style düzeltme denendi: kontrol/tek/çift düzeltme **6/9/12 NFE** yaptı. Tek düzeltmede kompozisyon değişti, birleşim kaldı; çift düzeltmede korkuluk koptu. Bu ayarlar reddedildi; resmî workflow'un bütün ayarlarını test etmiş değiliz. [Deney ve sınırları](../experiments/qwen/LANPAINT_RESULTS.md).

Entegrasyondan önce önerilen test:

1. Projedeki üç kapakta aynı telefon oranı, aynı kaynak konumu ve aynı çözünürlük kullan.
2. Her aday için kapak başına üç çıktı üret. Tek seferde genişletme ile dışarı doğru şeritler halinde üretmeyi ayrı karşılaştır.
3. Orijinal kareyi sonuç üzerine aynen yerleştir ve bu bölgenin piksel eşitliğini kontrol et. Prompt veya latent maske tek başına garanti değil; VAE dönüşümü korunmuş pikselleri değiştirebilir.
4. Kenar birleşimleri, renk / doku devamlılığı, kompozisyon, yinelenen nesneler ve istenmeyen yazıları anonim karşılaştır. Süre ve en yüksek bellek kullanımını kaydet.
5. Üç kapak ilk eleme için yeterli; distillation kararı öncesi farklı stillerde daha geniş bir örnek kümesi kullan.

Dışarı doğru yavaşça açılma hissi, tamamlanmış geniş görsel üzerinde Swift animasyonuyla da verilebilir. Bunun tekrar tekrar model çalıştırmaktan daha iyi olacağı henüz ölçülmedi; şerit üretimiyle karşılaştırılacak bir uygulama seçeneğidir.
