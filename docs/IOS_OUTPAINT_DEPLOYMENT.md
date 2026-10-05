# ArtWorker — iOS'ta yerel outpainting çalıştırma

İlk araştırma ve kaynak kontrolü: **30 Eylül 2026**; deney durumu güncellemesi: **5 Ekim 2026**. Bu rapor yayımlanan ölçümleri, kaynak kodundan görülen davranışı ve ArtWorker için önerilen deneyi ayrı tutar. Fiziksel iPhone üzerinde inference henüz ölçülmedi. Simulator'da derleme veya Mac'te üretim, telefon performansının kanıtı değildir.

## Tamamlanan doğrulamalar ve sonraki iOS adımı

**Bu oturumda yalnız araştırma yapılmadı:** Mobile-O ve DreamLite ağırlıkları indirildi, Mac'te kapaklar üzerinde çalıştırıldı; yerel Swift MediaGenerationKit çağrısı iOS Simulator için iki mimaride derlendi. DreamLite Mobile'ın ilk üç kapak outpainting deneyi başarısız, Base'in tek kapaktaki kontrolü daha tutarlı çıktı; Mobile-O'nun ilk instruction-only genişletmeleri de kaliteli devamlılık sağlamadı. Bunları yayımlanan telefon hızlarından ayrı okumalıyız. [DreamLite deneyleri](/Users/emir/Documents/ChatGPT/ArtWorker/experiments/dreamlite/README.md), [Mobile-O deneyleri](/Users/emir/Documents/ChatGPT/ArtWorker/experiments/mobile_o/README.md), [başarılı iOS SDK derlemesi](/Users/emir/Documents/ChatGPT/ArtWorker/experiments/ios_sdk/README.md)

**5 Ekim Mac kontrolleri:** kaynak sampler, CPU compositor, VAE projection, Viggle v0.3/tail, LanPaint-style, eğitimli ControlNet tam/sparse/reference/localized ve native FLUX.1 Fill50 gerçek çıktılarla sınandı. Localized40 tek sahne üretse de dış korkuluk **−17,404 px** kaydı ve ekstra bant kaldı; Fill50 yazı ve ikinci araba/kişi sahnesi üretti. Hiçbiri bütün görüntü kabulünü geçmedi. Native Fill'in en yüksek kaydedilmiş faz MLX peak'i yaklaşık **10,33 GiB**, VAE consistency projection peak'i **40,60 GiB** idi; bunlar Mac ölçümleri, iOS bütçesi değil. Projection derivative kolunun final çıktısı yok. [Güncel deney özeti](OUTPAINT_TRIAL_RESULTS.md), [localized40](../experiments/qwen/boundary_eval/2026-10-05/controlnet-localized40-comparison/REVIEW.md), [Fill50](../experiments/flux1_fill/FILL_RESULTS.md), [projection](../experiments/qwen/PROJECTION_RESULTS.md).

Saved-top32, yalnız 64 source-collar hint hücresini değiştirerek asıl dış rail hatasını **−17,404 → −2,086 px** azalttı; kaynak dış feature'ı doğrulandı. Küçük son-satır jog ve su/asfalt dikişleri kaldı. Yerel kalite ilerlemesi var, ürün kabulü yok. Native Fill local-context50 **iki bağımsız çağrıda 320,980 s/100 NFE** yaptı; alt soluk bantlar/doku hatası kaldı, streaming değil. Native SDXL9-channel20/CFG8/strength0,99 **25,630 s wrapper/19 U-Net çağrısı/38 CFG branch sample** ile çalıştı, ağır siyah dış alan üretti; mask/noise audit'i input tersliği bulmadı. Bunların Mac süreleri telefon tahmini değildir. [Top32 gerçek inceleme](../experiments/qwen/boundary_eval/2026-10-05/controlnet-saved-top32-comparison/REVIEW.md), [local Fill kayıtları](../experiments/flux1_fill/runs/2026-10-05/track3-local-context50-seed42/metrics.json), [SDXL sonucu](../experiments/sdxl_inpaint/SDXL_RESULTS.md).

Yeni growing localized6 suite tamamlandı: 0 / 0,5 / 1 control scale, altı NFE ve **12.288 target-token-forward/kol**, toplam wrapper **211,185 s**. Gelecek tokenlar gerçekten yoktu; genel sahne tutarlı, korkuluk profil basamağı/ek kenar kaldı. Bu Mac compute kanıtı, kabul edilmiş kalite veya telefon performansı değil. Tam tuvalde sparse hint kullanmak büyüyen compute değil. Yeni growing-top32 ve portrait canvas-reference yalnız CPU hazır. MaskFlow2511 için doğru codec'e uygun izole MLX-Gen port/model indirmesi ve ProMax outpainting tarifi indirmesi başladı, GPU sonuçları yok. Player'a model bağlanmadı, öğrenci eğitilmedi, ücretli cloud işi ve fiziksel iPhone inference yapılmadı. Önce öğretmenin kalite kabulü, ardından küçük öğrencinin gerçek cihaz ölçümü gerekir; adapter ve export bulunması bütün iPhone'larda iyi performans anlamına gelmez. [Growing gerçek görüntü incelemesi](../experiments/qwen/boundary_eval/2026-10-05/controlnet-growing-localized6-comparison/REVIEW.md), [MaskFlow hazırlık audit'i](../experiments/maskflow/research/2026-10-05/FEASIBILITY.md), [ProMax tarifi](../experiments/sdxl_inpaint/PROMAX_FEASIBILITY.md), [distillation kalite koşulu](DISTILLATION_PLAN.md).

1. **Klein 4B distilled** için Mac kalite deneyi ile iOS entegrasyonunu ayrı değerlendir. MediaGenerationKit'in yerel Swift çağrısı artık derleme düzeyinde doğrulandı; sıradaki runtime doğrulaması fiziksel iPhone'da maskeli üretim ve bellek ölçümü. Bu, Klein'ın bütün iPhone'larda hızlı çalışacağı anlamına gelmiyor.
2. **DreamLite**'ın küçük denoiser ve dört adımlı sürümü dar görevli öğrenci için mimari referans. Gerçek Mac sonuçları, hazır Mobile sürümünün bu kapak görevine doğrudan uygun olmadığını gösterdi; Base kontrolü ve görev adaptasyonu karşılaştırılmalı. Kapak piksellerini kilitleyen hazır outpainting API'si bulunmadı.
3. **Mobile-O**'yu mobil sistem tasarımı karşılaştırması olarak kullan. Mac'te gerçekten çalıştırıldı; native appte editing akışı README ile uyuşmuyor ve mevcut genişletme deneyi kaliteyi karşılamadı.
4. Kendi ticari öğrenci modelimiz için **Apache/MIT öğretmen ve eğitim verisi izinlerini** esas al. DreamLite/Mobile-O'nun araştırma lisanslarını ticari öğrenciye kendiliğinden taşıma.

Geniş eski/yeni cihaz desteği için aşağıda küçük görev modeli ve üç çözünürlüklü Core ML planı var. Telefon performansı henüz ölçülmedi.

## Gerçek iOS runtime seçenekleri

| Yol | Doğrulanan durum | ArtWorker açısından karar |
| --- | --- | --- |
| **Core ML / Swift** | Apple'ın resmî diffusion Swift paketi iOS hedefli; model dönüştürme, sıkıştırma ve bileşenleri sırayla yükleme örnekleri var. | Küçük, sabit şekilli ve Core ML'e uygun öğrenci model için güçlü ürün yolu. Her modern DiT için otomatik ANE desteği yok. |
| **MLX Swift / Metal GPU** | Resmî MLX Swift örnekleri iOS ve macOS'ta Stable Diffusion/LLM/VLM çalıştırıyor. | Modern encoder ve henüz Core ML'e dönüştürülmemiş mimaride esnek. CPU/GPU yolunu ANE gibi sunmamalıyız. |
| **Draw Things MediaGenerationKit** | Public Swift package iOS 16+ hedefli, `.local` backend, `UIImage`/`CIImage`, image/mask, preview ve cancellation API'si var. | Klein için ilk entegrasyon deneyi. Tek başına derlenmesi model kalitesi veya telefon bellek yeterliliğini kanıtlamaz. |
| **flux-2-swift-mlx** | Klein ve masked inpainting/outpainting zincirleri içeriyor; mevcut package macOS 15+ hedefli. | Mac kalite deneyine uygun; hazır iPhone portu değil. |

Kaynaklar: [Apple diffusion paketi](https://github.com/apple-aiml-research/ml-stable-diffusion), [MLX Swift](https://github.com/ml-explore/mlx-swift), [MediaGenerationKit platformları](https://github.com/drawthingsai/media-generation-kit/blob/main/Package.swift), [Flux Swift platformları](https://github.com/VincentGourbin/flux-2-swift-mlx/blob/main/Package.swift).

### MediaGenerationKit: en yakın hazır Swift SDK yolu

SDK, Klein 4B için dört adımlı yerel örnek sunuyor; image ve mask input'larını aynı `generate` çağrısına alıyor. Configuration'da `preserveOriginalAfterInpaint`, `tiledDecoding`, `tiledDiffusion`, batch ve mask blur alanları var. Maske desteğinin public API'de bulunması, her checkpointte aynı iyi sonucu vereceğini garanti etmez. Dikdörtgen canvas, maske yönü ve kaynak piksel korunumu gerçek inference ile kontrol edilmeli. [SDK kılavuzu](https://github.com/drawthingsai/media-generation-kit), [pipeline kaynak kodu](https://github.com/drawthingsai/draw-things-community/blob/main/Libraries/MediaGenerationKit/Sources/MediaGenerationPipeline.swift)

**Maske sözleşmesi:** pinli dependency `d473a2f148b3e7dc9b90d0b7cfccc5cda999eb66` kodunda `.mask()` input'u yalnız grayscale UInt8'e çevrilip native generator'a gönderiliyor. Generator byte'ın alt üç bitini sınıf olarak okuyor: `3` koru/skip, `1` boş alanda üret, `2`/`4` inpaint. Standart Diffusers `0/255` maskesinin `255 & 7 == 7` olması nedeniyle burada farklı davranışı var. Native encoding veya alpha-derived otomatik mask kullanılmalı ve SDK dönüştürmesinden sonra byte değerleri doğrulanmalı. API'nin `.mask()` adından standart polarity çıkarılamaz. [Input decoder](https://github.com/drawthingsai/draw-things-community/blob/d473a2f148b3e7dc9b90d0b7cfccc5cda999eb66/Libraries/MediaGenerationKit/Sources/MediaGenerationExecutionUtilities.swift), [generator sınıfları](https://github.com/drawthingsai/draw-things-community/blob/d473a2f148b3e7dc9b90d0b7cfccc5cda999eb66/Libraries/LocalImageGenerator/Sources/LocalImageGenerator.swift)

Outpainting için doğrudan yol, boş dış alanı tamamen transparent ve orijinal kareyi opaque tutan RGBA canvas'ı mask vermeden göndermek. `ImageConverter.tensor` böylece source için `251` (`3 | 248`), boş alan için `1` native mask değerlerini kendisi çıkarıyor. Bunun Klein ile düzgün continuation ürettiği henüz bu SDK üzerinde ölçülmedi. [Alpha mask kaynak kodu](https://github.com/drawthingsai/draw-things-community/blob/d473a2f148b3e7dc9b90d0b7cfccc5cda999eb66/Libraries/LocalImageGenerator/Sources/ImageConverter.swift)

**Lisans farkı:** `draw-things-community` doğrudan GPLv3; `media-generation-kit` ise LGPLv3. SDK README'si, bu pakete geçen dependency kodunun LGPLv3 olarak yeniden lisanslandığını açıkça belirtiyor. Bu yolu kullanmak ile community kodunu doğrudan player'a kopyalamak aynı lisans durumu değil. Dağıtımda bildirim ve LGPL yükümlülükleri ayrıca uygulanmalı. [SDK lisansı](https://github.com/drawthingsai/media-generation-kit/blob/main/LICENSE), [community lisans açıklaması](https://github.com/drawthingsai/draw-things-community#license)

Draw Things resmî sürüm notları **22 Ocak 2026**'da Klein model/import desteğini doğruluyor. Buna rağmen bu araştırmada Klein 4B için yayımlanmış, cihaz/çözünürlük/adım/bellek koşulları tam bir iPhone benchmark'ı bulunmadı. [Sürüm notları](https://drawthings.ai/downloads/)

**Gerçek derleme sonucu:** `experiments/ios_sdk` altında SDK revision `8868a9685d9c299816f43ef53efd455ffca437f0` pinli minimal Swift package **Xcode 27 / iOS Simulator 27.0'da arm64 ve x86_64 için başarıyla derlendi**. Son RGBA automatic-mask/optional explicit-mask kodu değişikliği incremental build ile yeniden doğrulandı. `OutpaintCompileProbe.swift`, yerel backend, UIKit input, mask, configuration ve result API'sini derleme düzeyinde doğruluyor. Model indirme/inference, fiziksel cihaz deploy'u veya bellek/hız ölçümü yapmadı. [Deney ve yeniden çalıştırma komutu](/Users/emir/Documents/ChatGPT/ArtWorker/experiments/ios_sdk/README.md), [validation.json](/Users/emir/Documents/ChatGPT/ArtWorker/experiments/ios_sdk/validation.json)

### Core ML: “dönüştürdük” ile “ANE'de iyi çalışıyor” ayrı sonuçlar

`MLModelConfiguration.computeUnits = .all`, izin verilen compute cihazlarını açar; tüm işlemlerin ANE'ye yerleştiğini kanıtlamaz. Aynı modelin `.all`, `.cpuAndGPU` ve `.cpuAndNeuralEngine` seçeneklerini hedef cihazda karşılaştırmak gerekir. `MLComputePlan` cihaz yerleşimini ve operasyon maliyetini incelemek için resmî API sunar; Xcode performans raporu ve Instruments ile çalışırken doğrulanmalı. [Apple performans analizi](https://developer.apple.com/documentation/coreml/analyzing-a-core-ml-model-s-performance-in-xcode), [compute device usage](https://developer.apple.com/documentation/coreml/mlcomputeplandeviceusage)

Apple'ın resmî SD3 MMDiT dönüştürme yolunda hâlen FP32 ve CPU/GPU gereksinimi yazıyor. Küçük UNet için kullanılan split-einsum/ANE tarifini yeni Qwen veya Klein DiT'ye aynen uygulamak kanıtlanmış bir çözüm değil. [Apple SD3 bölümü](https://github.com/apple-aiml-research/ml-stable-diffusion#using-stable-diffusion-3)

Core ML optimizasyonları arasında iOS 18+ için 4-bit blockwise weight quantization, grouped-channel palettization ve INT8 LUT bulunuyor. W4 dosya boyutu, aktivasyonları da dört bit yapmaz. A17 Pro/M4 ve daha yeni destekli cihazlarda INT8 aktivasyon + INT8 LUT yolu avantaj sağlayabilir; calibration ve layer sensitivity testi gerekir. [Apple sıkıştırma özellikleri](https://apple.github.io/coremltools/docs-guides/source/opt-whats-new.html), [joint compression](https://apple.github.io/coremltools/docs-guides/source/opt-joint-compression.html)

Palettization her zaman hız kazandırmaz: grouped-channel LUT sayısı arttıkça runtime yavaşlayabilir. Post-training 4-bit, 6-bit ve bazı hassas katmanları FP16 bırakılan mixed-bit sürümleri kapak testinde ayrı değerlendirilmeli. Yalnız PSNR veya paket boyutuyla seçim yapmamalıyız. [Apple palettization performansı](https://apple.github.io/coremltools/docs-guides/source/opt-palettization-perf.html)

### MLX Swift: iOS mümkün, bulunan Flux portu Mac için

Resmî MLX Swift StableDiffusionExample, SDXL Turbo kullanıyor. Dokümanı, yaklaşık 4 GB kullanılabilir belleğin altında constrained modda bileşenleri sırayla yükleyip boşalttığını ve yalnız bir diffusion adımı yapabildiğini söylüyor. Bu örnek, çalışma şekli için referans; Klein 4B'nin aynı bütçeye sığdığının kanıtı değil. [Resmî örnek](https://github.com/ml-explore/mlx-swift-examples/blob/main/Applications/StableDiffusionExample/README.md)

Flux Swift MLX deposunun kendi Mac bellek önerisi Klein 4B için INT4/QINT8'de **16 GB**. Bu, 4-bit ağırlıkların teorik boyutundan büyük; mevcut loader, encoder, VAE ve çalışma buffer'ları birlikte değerlendirilmiş bir proje gereksinimi. Bu sınırı iOS'a taşıyamayız. [Projede bellek gereksinimleri](https://github.com/VincentGourbin/flux-2-swift-mlx#requirements)

## Hazır küçük ve distilled modeller

### DreamLite: doğrudan küçük image-edit adayı

Base ve Mobile aynı **389M denoiser** mimarisini kullanıyor; yanında **Qwen3-VL-2B** encoder ve **2,5M TinyVAE** var. Mobile dört adım ve CFG olmadan çalışıyor; Base 28 adım ve image/text guidance kullanıyor. Yazarlar 4-bit encoder + FP16 UNet/VAE ile **iPhone 17 Pro'da 1024×1024 generation/edit için yaklaşık 3 saniye** bildiriyor. Bu bizim ölçümümüz değil; model yükleme, cold start, sürekli kullanım ve player ile birlikte bellek ayrıca ölçülmeli. [Model ve yazar ölçümü](https://github.com/ByteVisionLab/DreamLite), [mimari/makale](https://arxiv.org/html/2603.28713v1)

**Canlı erişim düzeltmesi:** GitHub README'si hâlâ erişim formu/gated yazıyor; 30 Eylül 2026'da Hugging Face API kontrolünde `DreamLite-mobile` ve `DreamLite-base` **gated=false** döndü. Mobile main SHA `a48c656c291fe98f74a0405961c9e4e3eee5d6c8`, Base main SHA `a9a0f151ffd99d3c37f3fd0472f5e8f1b31215aa`. Ağırlık erişiminin açık olması, CC BY-NC 4.0 ve ağırlık koşullarını değiştirmiyor. [Mobile checkpoint](https://huggingface.co/carlofkl/DreamLite-mobile), [Base checkpoint](https://huggingface.co/carlofkl/DreamLite-base), [weights license](https://github.com/ByteVisionLab/DreamLite/blob/main/WEIGHTS_LICENSE)

DreamLite `deploy/` gerçek Swift inference ve Core ML export kodu içeriyor. Text/image hidden states için `mlx-swift-lm` kaynak değişiklikleri gerekiyor; sıradan `generate` metin çağrısı yeterli değil. VAE encoder ve image-conditioned generation akışı mevcut. Ancak verilen encoder export'u 1024 kare sabit şekilli; telefon oranına geçerken canvas/source boyutları ve model shape'leri birlikte ele alınmalı. Native API'de hard spatial mask / known-latent restore akışı bulunmadı. Dış alanları üretmek için görev adaptasyonu ve son kaynak compositing gerekir. [iOS deploy kılavuzu](https://github.com/ByteVisionLab/DreamLite/blob/main/deploy/README.md), [Swift pipeline](https://github.com/ByteVisionLab/DreamLite/blob/main/deploy/DreamLitePipeline.swift), [VAE encoder export](https://github.com/ByteVisionLab/DreamLite/blob/main/deploy/export_vae_encoder.py)

**Bu oturumdaki Mac deneyi:** dört adımlı Mobile, ortak 512×1152 canvas'ta üç kapakta yaklaşık 1,23–4,67 saniyede çıktı verdi; dış bölgeler tekrar eden nesne/harfler, collage ve siyah boşluklar içerdi. Source-latent restore kontrolü tek kapakta merkez değişimini azalttı, dış alanı düzeltmedi. Sonraki sahneye özel kontrollerde normal editing çalıştı; Base 28-step tek kapak portrait outpaint yaklaşık 32,36 saniyede tutarlı kıyı/yol devamlılığı üretti, renk ve birleşim izleri kaldı. Bu sonuç üç kapak kalite başarısı veya iPhone ölçümü değil. Unquantized BF16 encoder kullanılan Mac belleği, yayımlanan 4-bit iOS encoder ölçümüyle eş tutulmamalı. [İndirilen revision'lar, ölçümler ve görsel sonuçlar](/Users/emir/Documents/ChatGPT/ArtWorker/experiments/dreamlite/README.md)

### Mobile-O: iOS kaynak kodunu README'den ayrı kontrol ettik

Yazarlar iPhone 15/16/17 Pro için yaklaşık **3–4 saniye**, **2 GB'dan az bellek** bildiriyor. Bu yayımlanan genel image-generation iddiası; albüm kapağı outpainting ölçümü değil. On-device paket Qwen2-0.5B MLX 4-bit dil modeli yanında Core ML diffusion/connector/vision/VAE bileşenleri içeriyor. Model etiketi bütün sistemin 0.5B olduğu anlamına gelmiyor. Yaklaşık 3,6 GB indirme/5 GB disk gereksinimi belgelenmiş. [Mobile-O](https://github.com/Amshaker/Mobile-O), [iOS app kılavuzu](https://github.com/Amshaker/Mobile-O/blob/main/Mobile-O-App/README.md)

**Kaynak bulgusu, SHA `91c255080a0130846fe5b5cd54ff5af3c05ab2b9`:** Swift `ChatViewModel`, `generate` için text-to-image çağırıyor; attached image için understanding çağırıyor. README'deki `edit + image` dalı kodda yok. `MobileOGenerator.GenerationParameters` image veya spatial mask almıyor, yalnız metin hidden states ile generation yapıyor; latent ve decode 512 kare. Python editing script'inde `pixel_values` koşulu var, fakat bu iOS akışına bağlanmamış. Dolayısıyla hazır image-edit/outpaint app diye sunmak doğru değil. [ChatViewModel](https://github.com/Amshaker/Mobile-O/blob/91c255080a0130846fe5b5cd54ff5af3c05ab2b9/Mobile-O-App/app/MobileO/ViewModels/ChatViewModel.swift), [Swift generator](https://github.com/Amshaker/Mobile-O/blob/91c255080a0130846fe5b5cd54ff5af3c05ab2b9/Mobile-O-App/app/MobileO/Models/Generation/MobileOGenerator.swift), [Python edit](https://github.com/Amshaker/Mobile-O/blob/main/infer_image_editing.py)

**Lisans kaynakları uyuşmuyor:** repo/app kaynakları CC BY-NC-SA 4.0 yazarken indirilen Python ve iOS HF model kartları CC BY-NC 4.0 belirtiyor. Ağırlıklar için dosya/kart/revision koşulları ayrı kontrol edilmeli; repo ifadesini bütün ağırlıklara otomatik olarak atamamalıyız. Her iki kaynak da NC kısıtı taşıyor; ticari ürün/öğrenci için izin açıklığa kavuşturulmalı. [Repo lisansı](https://github.com/Amshaker/Mobile-O/blob/main/LICENSE.txt), [Python ağırlık kartı, pinli revision](https://huggingface.co/Amshaker/Mobile-O-0.5B/blob/92b5ef6a90a770c631153c3c32be0bbe049e2a8d/README.md), [iOS ağırlık kartı, pinli revision](https://huggingface.co/Amshaker/Mobile-O-0.5B-iOS/blob/56828ca7076d594dc38239627f0c4edb41a508b1/README.md)

**Bu oturumdaki Mac deneyi:** yüklenen tam Mobile-O modelinde **1.664.722.343 parametre** sayıldı. MPS'te 512×1152'ye uyarlanmış Python image-conditioned akışı üç kapakta 20 adım/seed 42 ile yaklaşık **4,74–4,99 saniye** çalıştı, fakat instruction-only outpainting yaklaşımı üçünde de tekrar eden nesneler/yazılar ve kötü birleşimler üretti. Bu sonuç, iPhone hızı veya bütün olası Mobile-O yöntemlerinin başarısızlığı iddiası değildir. Modelin küçük/çabuk olması, kaynak geometriyi genişletebildiğini kanıtlamadı. Son kaynak compositing, çalışma çözünürlüğündeki 512 kare `source_512` ile byte eşitliğini sağladı; orijinal 1280×720 asset'in bütün piksellerini koruma testi değildi. Modelin native spatial mask'i yoktu. [Deney kaydı](/Users/emir/Documents/ChatGPT/ArtWorker/experiments/mobile_o/README.md)

### Sadece text-to-image araştırmalarını outpainting diye değerlendirmemek

| Araştırma | Yayımlanan ölçüm | Bu işe sınırı |
| --- | --- | --- |
| SnapGen++ (2026) | iPhone 16 Pro Max, 1024 kare, 4 adım: 0.3B/1,2 s; 0.4B/1,8 s; 1.6B mixed-bit/6,7 s. | T2I; resmî checkpoint doğrulanmadı. Küçük öğrenci ve Core ML mimari örneği. |
| SnapGen | 379M; iPhone 16 Pro Max 1024 kare yaklaşık 1,4 s. | T2I; resmî ağırlık doğrulanmadı. |
| Google MobileDiffusion | 520M toplam; premium iOS/Android cihazda tek adım 512 kare yaklaşık 0,5 s. | T2I; resmî Google checkpoint doğrulanmadı. |
| EdgeFusion | Exynos 2400 için iki adım 512 kare, 738,5 ms. | Android NPU ölçümü; iPhone kanıtı değil. |

Kaynaklar: [SnapGen++](https://arxiv.org/html/2601.08303v1), [SnapGen](https://snap-research.github.io/snapgen/), [Google MobileDiffusion](https://research.google/blog/mobilediffusion-rapid-text-to-image-generation-on-device/), [EdgeFusion](https://arxiv.org/pdf/2404.11925).

## Belleği ve süreyi düşürmek için uygulama planı

Önerilen plan, henüz ArtWorker'da ölçülmüş optimizasyon sonucu değildir:

- İlk canvas'ı **512×1024** veya telefon oranına yakın, modelin downsampling hizasına uygun bir boyutta tut. Ekranın fiziksel piksel çözünürlüğünde diffusion başlatma. Kalite uygun bulunursa daha yüksek canvas ayrı test olsun.
- Tek output, tek batch, önce küçük prompt/token bütçesi. Distilled checkpoint için doğru scheduler/CFG kullan; Base tarifini dört adıma indirmek distillation değildir.
- Prompt embedding'i sabit görevde; image-conditioned embedding'i kapak bazında önbellekle. Metin/image encoder → VAE encode → denoiser → decode şeklinde bileşenleri sırayla yükle/boşaltmayı dene. Cache tutmanın peak RAM ve cold latency etkisini birlikte ölç.
- Önceden quantize edilmiş checkpoint yükle. Telefonda önce BF16 ağırlık yükleyip sonradan quantize etmek, açılış peak belleğini gereksiz büyütebilir.
- VAE decode sırasında tiled decoding dene. Tiled diffusion'ı ayrı değerlendir: küçük bölgeler global kompozisyonu bozabilir; VAE tiling ile aynı işlem değil.
- Kapak kimliği + hedef oran + model/config hash'ini sonuç cache anahtarı yap. Aynı şarkıda tekrar üretme. Şarkı değişince çalışmayı iptal et; ses oynatmayı inference kuyruğundan ayır.
- Üretilen canvas'ın üstüne çalışma çözünürlüğündeki kareyi aynen composite et; bu kareyle byte eşitliğini test et. Orijinal yüksek çözünürlüklü asset'i ayrıca Swift UI katmanında gösterme önerisi, çalışma canvas'ının byte testiyle ayrı tutulmalı. Latent mask, VAE veya `preserveOriginalAfterInpaint` ayarı tek başına piksel garantisi olarak kabul edilmesin.
- Yavaş dışarı açılma, tek canvas üzerinde reveal animasyonu ve gerçekten şerit üretimi olarak ayrı karşılaştırılsın. Çok sayıda model çağrısı hem ısınmayı hem hata birikimini artırabilir; sonuç testle seçilsin.

Telefonun RAM etiketi uygulamanın erişebileceği bütçeyi vermez. `os_proc_available_memory` ile canlı bütçe kontrol edilmeli. Increased Memory Limit entitlement yalnız destekli cihazlarda ek limit sağlar; ek RAM garanti değildir ve uygulama onsuz da düzgün davranmalıdır. [Apple memory entitlement](https://developer.apple.com/documentation/bundleresources/entitlements/com.apple.developer.kernel.increased-memory-limit)

## Eski ve yeni iPhone'larda iyi performans hedefi

**Ürün hedefi:** desteklediğimiz telefonlarda ses, dokunma ve ekran animasyonu akıcı kalırken kapağın dışını kaliteli biçimde tamamlamak. Aynı büyük modelin bütün telefonlarda aynı sürede çalışacağını varsaymak bu hedefi karşılamıyor. Geniş kapsama için görev odaklı küçük öğrenci model, cihazın ölçülmüş kapasitesine göre çözünürlük ve kapak başına tek üretim gerekir. Aşağıdaki profiller bir araştırma/uygulama önerisidir; henüz cihazda doğrulanmış destek veya hız garantisi değildir.

### Hangi cihazlara gerçekten ulaşabiliriz?

- **Mevcut ArtWorker minimumu iOS 17.** `ArtWorker.xcodeproj/project.pbxproj` hem Debug hem Release için `IPHONEOS_DEPLOYMENT_TARGET = 17.0` içeriyor. iPhone XR/XS/XS Max, iOS 17 destekli eski cihazlar arasında; iPhone 8/8 Plus/X için bu app hedefini düşürmek gerekir. Apple'ın iOS 17 cihaz tablosu ve iOS 16 listesi bu işletim sistemi sınırını doğruluyor. Bu, listedeki her telefonda seçilen AI modelinin kullanılabilir hızda çalışacağına dair kanıt değildir. [Apple iOS 17 cihaz tablosu, s. 6](https://www.apple.com/environment/pdf/Longevity_by_Design-June_2024-Apple.pdf), [iOS 16 cihaz listesi](https://support.apple.com/en-nz/103267)
- **SDK minimumu model minimumu değildir.** MediaGenerationKit manifesti iOS 16; 30 Eylül 2026'da incelenen güncel `mlx-swift` ana paket manifesti iOS 17. `mlx-swift-examples` manifestindeki iOS 16 değeri, bağlı MLX paketinin sınırını kaldırmaz. Tam dependency graph ve kullanılacak revision pinlenmeli. Eski cihaz kapsaması için ayrıca iOS 16 uyumlu, MLX gerektirmeyen Core ML export'u denenebilir; Klein veya DreamLite'ın mevcut export'unu doğrudan bu cihaza uygun ilan edemeyiz. [SDK manifesti](https://github.com/drawthingsai/media-generation-kit/blob/main/Package.swift), [MLX manifesti](https://github.com/ml-explore/mlx-swift/blob/main/Package.swift), [örnek manifesti](https://github.com/ml-explore/mlx-swift-examples/blob/main/Package.swift)
- **Sıkıştırma paketi OS'ye göre seçilmeli.** iOS 16 temel palettization/8-bit ağırlık sıkıştırmasını, iOS 17 daha yeni çalışma zamanı optimizasyonlarını, iOS 18 ise 4-bit blockwise ve grouped-channel gibi ek biçimleri destekliyor. Tek bir iOS 18 export'unu iOS 16'ya yüklemek plan değildir. Minimum deployment target, model operation set'i ve compression biçimi ayrı doğrulanmalı. [Apple'ın sürüm tablosu](https://apple.github.io/coremltools/docs-guides/source/opt-whats-new.html)

**Önerilen ilk doğrulama tabanı:** mevcut iOS 17 hedefini koruyarak bir XR/XS sınıfı eski cihazı, bir orta nesli ve yeni bir Pro cihazı aynı küçük öğrenci üzerinde test etmek. Eski cihazı en baştan ölçmek, yalnız son model Pro'da iyi sonuç alıp ürünün genelini bunun üzerine kurma riskini azaltır. Bu seçim telefonlara bugün destek verdiğimiz anlamına gelmez. iPhone 8/X kapsaması ayrıca istenirse app API uygunluğu ve ayrı iOS 16 export'u incelenmeli; fiziksel cihaz testi geçmeden minimumu düşürmemeliyiz.

### Büyük edit modelinden küçük outpaint öğrencisine

Öğretmen kalite üretimi için Klein/Qwen gibi güçlü edit modelleriyle karşılaştırılabilir; telefondaki ortak taban için **kabaca 100–400M denoiser aralığında mimari deneyi** ve küçük VAE daha anlamlı bir başlangıç hipotezi. Bu aralık hedef tasarım bütçesi; eğitilmiş veya kalitesi kanıtlanmış bir ArtWorker modeli değil. “0.5B” etiketiyle bütün pipeline boyutunu karıştırmamak gerekiyor: denoiser, VAE ve koşullama bileşenleri birlikte sayılmalı.

Görev yalnızca “bu kareyi koru, dışını devam ettir” ise çok milyarlı genel dil/VLM encoder'ını her üretimde çalıştırmak gerekmeyebilir. Sabit görev için text embedding'i önceden hesaplayabiliriz. Daha ileri aşamada öğrenciye **kaynak latent + yerleşim maskesi + hedef geometriyi** doğrudan koşul olarak vererek genel dil encoder'ını mimariden çıkarabiliriz. Bu, hazır modelden encoder'ı silmekle yapılmaz; öğrenci bu koşullarla eğitilmeli/distill edilmeli. Görsel koşul veya VAE encode hâlâ gerekiyorsa sonucu kapak bazında cache edilir; farklı kapakların image embedding'i aynı olamaz. Dış alanda diffusion loss, kenar devamlılığı ve kaynak latent restore, eğitim/değerlendirme görevine dahil edilir. Son ekranda orijinal kare ayrı yüksek çözünürlüklü katman olarak korunur.

### Üç sabit Core ML export adayı

Sabit tensor shape'leriyle aynı öğrenci ve aynı telefon oranı için şu başlangıç export'ları denenebilir. Şu anda yalnız aday boyutlar; kalite, ANE yerleşimi veya inference süreleri ölçülmedi.

| Profil | Canvas | Piksel sayısı | Amaç |
| --- | --- | ---: | --- |
| Küçük | 256×576 | 147.456 | Eski/dar bellekli cihaz için dış alan üretimini mümkün kılmayı denemek. |
| Dengeli | 384×864 | 331.776 | Orta cihazda kalite ve bellek dengesi. |
| Ayrıntılı | 512×1152 | 589.824 | Ölçümü geçen güçlü cihazda daha ayrıntılı dış alan. |

Üçü de 1:2,25 oranında; 8 veya 16 downsampling hizasına uygun. Büyük profil küçüğün dört katı piksele sahip, fakat bu **dört kat gecikme veya RAM** sonucunu vermez: attention, bileşen yükleme ve backend'e göre ölçek değişir. Hedef telefonun gerçek aspect ratio'su farklıysa final crop/padding uygulanır; örnek oran bütün ekranlar için bire bir iddia değildir.

**Runtime ayrımı:** bu tablo özel Core ML öğrenci export'u içindir. İncelediğimiz Draw Things `LocalImageGenerator` image-input yolu boyutlarda 64 hizası istiyor; 256×576 ve 512×1152 buna uyuyor, **384×864 uymuyor**. O SDK ile orta boyut deneyi 384×896 padded canvas + final crop şeklinde ele alınmalı. Modelin latent pack ve position-ID şartları da ayrıca kontrol edilmeli. [64 hizası kontrolünün kaynağı](https://github.com/drawthingsai/draw-things-community/blob/d473a2f148b3e7dc9b90d0b7cfccc5cda999eb66/Libraries/LocalImageGenerator/Sources/LocalImageGenerator.swift)

Telefon adıyla sabit profil atamak tek başına yeterli değil. Başlangıç profili cihaz/iOS için laboratuvarda ölçülen p95 süre ve peak bellek tablosundan seçilir; o oturumdaki kullanılabilir bellek, ısı ve Low Power Mode profili düşürebilir. `MLComputePlan` ile operator yerleşimi incelenir; Core ML `.all` kullanmak ANE hızını garanti etmez. Metal yolunda `recommendedMaxWorkingSetSize` GPU kaynakları için performans rehberidir, app'in jetsam limiti değildir; `maxBufferLength` ise tek buffer sınırıdır, toplam RAM bütçesi değildir. [GPU çalışma kümesi](https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize), [tek buffer sınırı](https://developer.apple.com/documentation/metal/mtldevice/maxbufferlength)

### Üretim sırasında player'ın davranışı

1. Kapak açılınca önce orijinal kare ve varsa tamamlanmış cache gösterilir. Aynı kapak/config için bir generation oturumu yürütülür. Kullanıcının istediği gerçek büyümede preview, aktif-token sampler durumundan gelir; tamamlanmış cache'i açma animasyonu ayrı davranıştır. İkisi için kalite ve decode/UI maliyeti ayrı ölçülmeli. Gerçek growing sampler henüz Swift player'a bağlanmadı.
2. Inference `MainActor` dışında, seri ve en fazla bir aktif iş olarak yürür. Şarkı değişince cancellation istenir; eski iş iptali geç bitse bile generation token/kapak kimliği kontrolü sonucunun yeni kapağa uygulanmasını engeller. Ses oynatma, seek ve volume inference kuyruğunu beklemez.
3. Model/VAE/encoder bileşenleri mümkünse sırayla yüklenir. Tek kapak için gereken koşullar tutulur; memory warning veya bütçe daralmasında tensor/model cache boşaltılır ve sonraki iş küçük profile iner. Bellek yetersizse iş başlatılmaz; player orijinal kapağı göstermeye devam eder. Bu durum AI outpaint tamamlandı diye sunulmaz.
4. `thermalStateDidChangeNotification` ve `NSProcessInfoPowerStateDidChange` izlenir. Önerilen politika: `.fair` durumunda gereksiz prefetch'i ertele; `.serious` durumda sonraki işi küçült/ertele, `.critical` durumda generation'ı durdur. Low Power Mode'da arka plan ön üretimini durdur. Eşikler ve yeniden yükseltme davranışı telefon testinde ayarlanır; sistemin kendi throttling'i olmadan sabit hız varsayılmaz. [Apple'ın güç/ısıya tepki kılavuzu](https://developer.apple.com/documentation/xcode/responding-to-power-notifications)
5. Küçük canvas'ın üretilmiş dış alanı ekrana büyütülür; orijinal kapak kaynak çözünürlüğünde kalır. Birleşim yumuşatma kaynak karenin içini değiştirmeden dış tarafta yapılır. Böylece düşük çözünürlük profili kapak yazısını ve merkezdeki özgün ayrıntıları gereksiz yere bulanıklaştırmaz.

Kullanılabilirlik iki farklı ölçüm gerektirir: cache hit/reveal'ın akıcılığı ve ilk kapağın gerçek üretim gecikmesi. Güzel animasyon yavaş üretimi ölçüm tablosundan saklamamalı. Eski/orta/yeni cihazda cold ve warm ölçümler, arka arkaya kapak üretimi, ses + kaydırma/seek, Low Power Mode ve ısınma senaryoları birlikte geçmeden “genel olarak iyi performans” sonucuna varmayız.

## Gerçek cihaz kabul ölçümü

Bir fiziksel iPhone bağlandığında her aday/ayar için şu kayıtlar tutulmalı:

1. Model dosyası, revision, quantization; iPhone modeli, iOS sürümü; canvas, steps, seed, CFG/scheduler.
2. Cold model load/compile, prompt encoder, image encode, denoising, VAE decode ve toplam süre ayrı.
3. `phys_footprint` / Instruments peak memory, kullanılabilir bellek; memory warning, jetsam/çökme, cancellation.
4. Bir kez üretim ve arka arkaya üretim; thermal state, Low Power Mode, player ses sürekliliği ve scroll/seek responsiveness.
5. Çalışma çözünürlüğündeki kaynak kareyle byte eşitliği ve orijinal yüksek çözünürlüklü asset'in UI katmanında korunumu ayrı; dikiş izi, renk/doku devamlılığı, yeni nesneler/yazılar; aynı üç kapakta aynı placement ve en az üç seed.
6. Küçük/dengeli/ayrıntılı profil için ayrı p50/p95 toplam süre ve peak bellek; daha düşük profile geçiş, model cache boşaltma ve eski işin sonucunu reddetme davranışı. En eski hedef cihaz test grubuna baştan dahil edilir.

Kabul eşiklerini ölçümden önce belirleyip tabloya yazmalıyız. Bu raporda fiziksel cihaz görmeden gecikme, peak RAM veya batarya tüketimi uydurulmadı.
