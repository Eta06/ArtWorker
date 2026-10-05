# ArtWorker — hazır distilled modeller ve kendi mobil outpainting öğrencimiz

İlk araştırma: **30 Eylül 2026**; deney durumu: **5 Ekim 2026**. Birincil model kartları, kaynak kodu, yayınlar ve canlı Hugging Face metadata kullanıldı. Bu belgede belirtilen yazar benchmark'ları ArtWorker ölçümü değildir. Önerilen model boyutu, veri adedi ve kalite eşikleri mühendislik hedefidir; eğitilmiş bir ArtWorker modeli veya ölçülmüş iPhone performansı olarak okunmamalıdır. Tamamlanan yerel indirme/inference deneyleri [deney raporunda](/Users/emir/Documents/ChatGPT/ArtWorker/docs/OUTPAINT_TRIAL_RESULTS.md); geniş eski/yeni iPhone kapsaması ve profil seçimi [iOS deployment planında](/Users/emir/Documents/ChatGPT/ArtWorker/docs/IOS_OUTPAINT_DEPLOYMENT.md) ele alınır.

## Karar

**Kalite öğretmeni ile telefonda çalışan öğrenci ayrı seçilmeli.** Qwen-Image-2.1'in hızlı LoRA'ları mevcut; bunlar yaklaşık 7B denoiser'ı ve büyük encoder'ı küçültmüyor. İlk telefon adayı yalnız Klein 4B olmamalı: yeni bulunan **DreamLite Mobile**, gerçekten küçük bir denoiser, hazır dört adımlı sürüm ve Swift/Core ML/MLX deployment referansı sunuyor. **Mobile-O** da küçük sistem tasarımı için değerli; ancak mevcut Swift uygulamasının image editing bağlantısı eksik.

**Önce kalite kabulü, sonra distillation.** İlk büyük/küçük model karşılaştırmaları ve 5 Ekim sınır kontrolleri çalıştırıldı; bütün sahne ve birleşim için kabul edilen öğretmen henüz yok. v0.3/tail, sampler, compositor, VAE projection ve LanPaint-style kolları geometriyi çözmedi. ControlNet localized40 kompozisyonu iyileştirdi ama dış korkuluk kenarı **−17,404 px** kaydı ve ek bant üretti; native Fill50 yazı/ikinci sahne tekrarladı. Bu sonuçları teacher target yapmak aynı kusurları öğrenciye aktarabilir. Farklı kapak/seed'lerde geometri, doku ve semantik kalite geçmeden teacher veri pilotu başlatılmamalı. **ArtWorker öğrenci eğitimi veya distillation yapılmadı.** [Tamamlanan deneyler](OUTPAINT_TRIAL_RESULTS.md), [localized40 incelemesi](../experiments/qwen/boundary_eval/2026-10-05/controlnet-localized40-comparison/REVIEW.md), [Fill50](../experiments/flux1_fill/FILL_RESULTS.md).

Sonraki **saved-top32** kontrolü ortak tensorları aynen kullanıp yalnız 64 source-collar hint hücresini değiştirdi; aynı dış metal kenarında hata **−17,404 → −2,086 px**, yakın açı **+0,341°**. Büyük bant hatası belirgin azaldı, son-satır jog ve su/asfalt dikişleri kaldı. Bu öğretmen davranışını doğru yönde geliştiren gerçek bulgu, teacher kabulü değil. Native Fill iki local-context50 çağrısı ve SDXL9-channel kontrolü de bütün kaliteyi geçmedi. [Gerçek top32 incelemesi](../experiments/qwen/boundary_eval/2026-10-05/controlnet-saved-top32-comparison/REVIEW.md), [SDXL sonucu](../experiments/sdxl_inpaint/SDXL_RESULTS.md).

İlk karşılaştırmalar sonrası önerilen sıra:

1. Tamamlanan büyük/hızlandırılmış/küçük aday denemelerinden sonra, önce kaynak sınırı ve bütün sahne kalite kabulünü geçen öğretmen/sampler elde et.
2. Seçimi aynı kapaklarda birden fazla seed ve daha geniş stil holdout'unda doğrula. Genel instruction editing başarısı hard-mask outpainting başarısı değildir.
3. Küçük modele maskeli outpainting öğret. Bu aşamada daha büyük öğretmenin seçilmiş çıktıları ve dışı bilinen gerçek geniş görüntüler kullanılır.
4. Küçük model yeterli kaliteye ulaşınca onun sampling adımlarını 8/4'e indir.
5. Quantization, Core ML/MLX export ve gerçek cihaz profiling yap.

Ticari ürün için başlangıç öğretmeni **Klein 4B Base (Apache 2.0)** ve izinleri açık küçük öğrenci daha az lisans belirsizliği taşır. DreamLite, Mobile-O ve Qwen 2.1 prototip araştırması için kullanılabilir; bunların mevcut ağırlıklarının lisansları ticari dağıtım için aynı serbestliği sağlamıyor. Bu tercih kalite sonucuyla birlikte yeniden değerlendirilmelidir.

## “Distilled” tam olarak neyi küçültüyor?

| İşlem | Kazanç | Kendiliğinden sağlamadığı şey |
| --- | --- | --- |
| Step distillation | Örneğin 40/50 denoiser geçişi yerine 4/6/8 geçiş | Parametre sayısı, encoder boyutu veya uygulama ağırlığının azalması |
| Guidance distillation | CFG için conditional/unconditional çift forward yerine tek geçiş | Modelin küçük mimariye dönüşmesi |
| Architecture / knowledge distillation | Daha az katman/kanal veya farklı, küçük öğrenci | Öğretmenin bütün kalitesinin kayıpsız korunması |
| Quantization / palettization | Ağırlık depolama ve uygun kernel ile bellek/bant genişliği tasarrufu | Sampling adımının azalması; her cihazda aynı hız veya tam integer compute |
| LoRA fine-tuning | Az sayıda trainable parametreyle göreve adaptasyon | Büyük base checkpoint'in inference sırasında ortadan kalkması |

Mobil hedef için **küçük mimari + az adım + sıkıştırma** birlikte gerekir. Bir 6-step LoRA indirmek, 7B model yerine 680 MB'lık bağımsız bir image model kullanmak anlamına gelmez.

## Hazır hızlı / distilled varyantlar

| Aile / varyant | Doğrulanan reçete | ArtWorker için anlamı |
| --- | --- | --- |
| Qwen-Image-2.1 Base | Yaklaşık 7.1B görüntü transformer'ı; ayrı büyük Qwen3-VL encoder ve VAE; resmî örnek 40 adım | Kalite referansı. Bütün sistem “7B” değil. [Resmî repo](https://github.com/QwenLM/Qwen-Image-2.1) |
| Alibaba PAI **Qwen-Image-2.1-Fun-Acc** | 24 Eylül; PDD ile **4 NFE**, rank-64 BF16 LoRA; T2I ve instruction edit | İlk resmi sağlayıcı hız adapter'ı. Base'in bütün ağırlıkları gerekir. Yazar edit çıktılarında daha koyu/yumuşak ayrıntı görülebildiğini söylüyor. [Model](https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Acc-LoRAs), [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun) |
| **Viggle Turbo v0.2.1** | 24 Eylül; DMD, **6 geçiş**, CFG yok; 1–3 referansla edit. Önerilen r256 adapter yaklaşık 1.3 GB, r128 yaklaşık 680 MB | Yazar yaklaşık 5× toplam hız bildiriyor; karmaşık edit/küçük yazı kaybı var. Aynı transformer+encoder+VAE kalır. Eski r64 4-step ile v0.2.1 karıştırılmamalı. [Model / reçete](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo) |
| **Viggle Turbo v0.3** | Altı geçiş veya resmî yedi Turbo + iki Base adımı; r256 adapter, CFG yok | 5 Ekim'de gerçek erken-büyüme kontrolü çalıştı; altı/dokuz adım kaynak korkuluk birleşimini çözmedi. Mimari küçülmedi. [Deney](../experiments/qwen/TURBO_TAIL_RESULTS.md) |
| **Turbo8 Qwen-Image-2.1** | 27 Eylül; 8 adım, CFG=1, rank-128 LoRA; trajectory initialization + DMD2 + regression anchor | Yazar 192 held-out örnekte ayrıntılı karşılaştırma yayımlamış; uzun metinde kayıp bildiriyor. Öğrenci mimari boyutu değişmiyor. [Model ve eğitim açıklaması](https://huggingface.co/chriswritescode/Turbo8-LoRA-Qwen-Image-2.1), [yazarın deney raporu](https://cstech.dev/articles/turbo8-distilling-qwen-image-2-1-to-8-steps) |
| **Klein 4B Base** | Step/guidance distill edilmemiş; örnek 50 adım, guidance=4; Apache 2.0 | Task fine-tune ve öğretmen olarak esnek. [Model kartı](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) |
| **Klein 4B distilled** | 4 adım, guidance=1; yine aynı yaklaşık 4B transformer; Apache 2.0 | Hazır hız karşılaştırması; büyük encoder ve VAE hâlâ bütçede. [Model kartı](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) |
| **HiDream-O1 Full / Dev** | Full 50; distilled Dev 28 adım; birleşik yaklaşık 8B; MIT | Dev küçük parametreli model değil. Yazar editing için **Full** öneriyor. Dev-2604 özellikle T2I odaklı; bu etiketten outpainting üstünlüğü çıkarılamaz. [Resmî repo](https://github.com/HiDream-ai/HiDream-O1-Image) |
| **DreamLite Base / Mobile** | Aynı **0.39B U-Net**, Base 28 adım, Mobile 4 adım/no CFG; ayrı Qwen-VL encoder + küçük codec | Gerçek küçük denoiser ve hazır iOS deployment referansı nedeniyle güçlü mobil araştırma adayı. Yazar iPhone 17 Pro'da 1024×1024 için yaklaşık 3 saniye bildiriyor; ArtWorker'da doğrulanmadı. [Repo](https://github.com/ByteVisionLab/DreamLite), [deployment](https://github.com/ByteVisionLab/DreamLite/blob/main/deploy/README.md) |

Bu hız adapter'ları için önerilen scheduler, sigma listesi, CFG ve base revision korunmalı. Varyantlar üst üste yüklenmemeli. Bir model ailesindeki eski Lightning LoRA yeni mimariye doğrudan uygulanamaz. **ModelTC Lightning** sayfasında Qwen-Image / Edit-2509 / Edit-2511 adapter'ları var; kontrol edilen listede Qwen 2.1 sürümü doğrulanmadı. [Primary release listesi](https://github.com/ModelTC/LightX2V-Qwen-Image-Lightning)

### Quantized paketler ayrı bir kategori

- **Unsloth Qwen-Image-2.1 GGUF:** Q2–Q8 seçenekleri var. GGUF dosyası yalnız denoiser; encoder ve VAE ayrıca gerekiyor. Dört bit encoder ile de bütün sistem tek küçük dosyaya dönüşmüyor. GGUF olması Core ML formatında veya hazır iOS pipeline olması demek değil. [Paket](https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF)
- **Comfy-Org Qwen 2.1:** BF16/INT8 transformer, BF16/INT8/W4A8 encoder paketleri bulunuyor. Bunlar ComfyUI runtime için; ANE'de aynı bit düzeni otomatik olarak kullanılamaz. [Paket](https://huggingface.co/Comfy-Org/Qwen-Image-2.1)
- **MLX Community Klein 4B 4-bit:** encoder, transformer ve VAE için MLX paketleri var. Bunların bulunması her iPhone'da yeterli bellek/hız kanıtı değildir. [Paket](https://huggingface.co/mlx-community/FLUX.2-Klein-4B-4bit)
- **Qwen PE-T2I Pocket 0.8B:** yalnız prompt rewriter, image generator değil. Büyük prompt-enhancement modelini azaltabilir; denoiser veya zorunlu visual encoder yerine geçmez. [Model kartı](https://huggingface.co/ML-Intern-lab/Qwen-Image-2.1-PE-T2I-Pocket-0.8B)

## Yeni küçük mobil modeller hakkında kritik düzeltmeler

### DreamLite: bu araştırmanın en önemli yeni adayı

0.39B etiketi **U-Net** içindir; bütün sistem değil. `diffusers` revision'ında 30 Eylül canlı HF tree kontrolünde:

| Bileşen | Dosya boyutu |
| --- | ---: |
| text_encoder/model.safetensors | 4,255,140,312 byte |
| unet/diffusion_pytorch_model.safetensors | 780,074,688 byte |
| vae/diffusion_pytorch_model.safetensors | 4,903,270 byte |

Base ve Mobile aynı bileşen dosya boyutlarına sahip. Bu toplam yaklaşık **5.04 GB indirilen FP16 ağırlıktır**, runtime RAM ölçümü değildir. iOS reçetesi encoder'ı MLX dört bit, U-Net/VAE'yi FP16 Core ML yapıyor. [Canlı dosyalar](https://huggingface.co/carlofkl/DreamLite-mobile/tree/diffusers)

Repo README'sinin “gated / form gerekli” bölümü canlı erişimle uyuşmuyor: 30 Eylül API kontrolünde Base/Mobile `gated=false, private=false`; yetkisiz weights HEAD kontrolünde 200 görülmüş. Dolayısıyla ağırlıkların erişimi ayrıca canlı kontrol edilmelidir, eski README nedeniyle deneme gereksiz yere durdurulmamalıdır.

Kod Apache 2.0; ağırlıklar CC BY-NC 4.0. LoRA eğitim kodu ve source/target edit dataset arayüzü var; kendi dataset sınıfı doldurulmalı. Native hard-mask outpainting checkpoint'i doğrulanmış değil. [LoRA guide](https://github.com/ByteVisionLab/DreamLite/blob/main/lora/README.md), [weights license](https://github.com/ByteVisionLab/DreamLite/blob/main/WEIGHTS_LICENSE)

### Mobile-O: önemli mimari, mevcut app'te edit yoluna dikkat

Mobile-O FastVLM/Qwen2-0.5B, SANA-600M decoder, VAE ve conditioning projector birleştiriyor. Kart toplam yaklaşık 1.6B parametre bildiriyor; **0.5B toplam sistem etiketi değil**. iOS checkpoint'i ve Mac/iPhone app kaynakları var. Yazar 512×512 için yaklaşık 3–4 saniye ve 2 GB altında memory bildiriyor; bu ArtWorker ölçümü değil. [Model kartı](https://huggingface.co/Amshaker/Mobile-O-0.5B), [iOS kartı](https://huggingface.co/Amshaker/Mobile-O-0.5B-iOS), [makale](https://arxiv.org/abs/2602.20161)

Kaynak incelemesinde Python instruction editing yoluna karşın mevcut Swift app'te attachment understanding'e, generate text-to-image'e bağlanıyor. Bu nedenle indirmek hazır outpainting düğmesi sağlamaz. Kod/app ve weights lisansları dosya bazında kontrol edilmeli: repo CC BY-NC-SA 4.0, HF model kartı CC BY-NC 4.0 gösteriyor. [Repo](https://github.com/Amshaker/Mobile-O), [app](https://github.com/Amshaker/Mobile-O/tree/main/Mobile-O-App)

## Maskeli görevi distillation sırasında korumak

Birçok hızlı image-edit modeli “arka planı değiştir” türü instruction eğitiminden geliyor. ArtWorker'ın sözleşmesi farklı:

```text
Girdi: orijinal kare + kaynak konumu + hedef aspect ratio + üretilecek alan maskesi
Çıktı: kapağın dışını tutarlı devam ettiren görüntü
Koruma: orijinal karede piksel değişimi = 0
```

Son piksel koruması **orijinali final çıktı üzerine aynen composite ederek** garantilenmeli. VAE encode/decode veya prompt sadakati tek başına aynı byte'ları sağlamaz. Model dış alanı ve sınır devamlılığını öğrenir; Swift renderer kaynak kareyi korur. Kaynak kenarının dışındaki dar bir bandın renk/doku geçişi ayrıca değerlendirilir.

Yeni merkezden dışarı **tek sampling akışı** hedefi için source/mask yanında spatial noise/time map koşulu da değerlendirilmelidir. Standart global-timestep modelden yalnız final pixel target aktarmak, erken tamamlanan yakın bölge + noisy uzak bölge davranışını kendiliğinden kazandırmaz. Öğrencinin task adaptasyonunda heterojen noise durumları görülmeli; step distillation bu spatial koşulları korumalı. Aktif-token/KV caching ise hız için ayrı tasarımdır. İlk Qwen aktif-canvas denemesinde geç açılan dış bantlar bozuldu; 5 Ekim'de aynı ağırlıklarla daha erken açılma ağır kusurları azalttı. Bu, önce inference-only sampler kontrollerinin tamamlanmasını destekler; eğitim her durumda zorunlu denemez. Kaliteli davranışı doğrulamadan kusurlu sampler'ı distill etmemeliyiz. [Mekanizma araştırması ve gerçek prototip](SPATIAL_STREAMING_RESEARCH.md)

**MaskFlow** (Ağustos 2026), Qwen-Image-Edit-2511 üzerinde mask-aware flow matching ve sınır iyileştirmesi için bir primary örnek. 50-step SFT yanında eşleşen 8/16-step DMD LoRA'ları ve MaskEdit-10k dataset'i var. Öğrencinin source/mask condition'ını distillation boyunca koruma reçetesi incelenmeli. Base modelin büyüklüğünü azaltmaz ve Qwen 2.1'e adapter'ı doğrudan taşınamaz. MIT adapter/kod lisansı base model koşullarını ortadan kaldırmaz. [Kod](https://github.com/ReyChiaro/MaskFlow), [makale](https://arxiv.org/abs/2608.06929)

5 Ekim'de exact recipe/2511 uyumluluk incelemesi tamamlandı ve **izole MLX-Gen port/model indirmesi başladı**. 2511 RGB16-channel/8× codec ile mevcut2.1 RGBA64-channel/16× codec farklı; packed64 etiketi latent uyumu sağlamıyor. Maske/CFG/Soft-Poisson reçetesini koruyan yerel port hazırlanıyor, henüz GPU sonucu veya yeni eğitim yok. ProMax author outpainting tarifi ağırlıkları da indiriliyor; teacher kalite üstünlüğü ölçülmedi. [MaskFlow feasibility](../experiments/maskflow/research/2026-10-05/FEASIBILITY.md), [ProMax hazırlığı](../experiments/sdxl_inpaint/PROMAX_FEASIBILITY.md).

**LanPaint** çeşitli base modelleri training-free maskeli üretime bağlayabilir; kendisi öğrenci checkpoint'i değil. Makale distilled modelde kalite gerilemesi bildiriyor. Base + mask sampler ile distilled + aynı sampler ayrı deney satırları olmalı; düşük adım sayısı otomatik olarak eşdeğer outpainting demek değil. [Kod](https://github.com/scraed/LanPaint), [makale](https://arxiv.org/html/2502.03491v3)

5 Ekim'deki izole Qwen Turbo LanPaint-style kontrolü 6/9/12 NFE ile çalıştı ve birleşim kalite kabulünü geçmedi. Bu inference-only deney, resmî workflow'un bütün ayarları veya öğrenci eğitimi değil. Frozen-VAE projection'ın latent optimizasyonu da model parametresi öğrenmedi. [LanPaint-style](../experiments/qwen/LANPAINT_RESULTS.md), [projection](../experiments/qwen/PROJECTION_RESULTS.md).

## Kendi öğrencimiz için iki uygulanabilir yol

### Yol A — en kısa araştırma prototipi

**DreamLite Mobile / Base'e maskeli albüm genişletme öğretmek.** Source-as-reference, hedef canvas ve maskeyi koşul yapan küçük bir adaptasyon; önce Base'in 28-step kalitesi, sonra Mobile 4-step sürümü karşılaştırılır. Hazır küçük backbone ve Apple deployment kodu, büyük DiT'yi baştan port etmekten daha az iş bırakabilir. Bu bir mühendislik çıkarımıdır; deney sonucu değil.

Mevcut instruction edit ve boş dış alan kontrolleri Mac'te çalıştırıldı: hazır Mobile kaliteyi karşılamadı, Base'in tek kapak devamı daha tutarlı olsa da sınır izi kaldı. Sonraki LoRA/conditioning adaptasyonu bir eğitim önerisi; burada yapılmadı. Hazır image-edit LoRA kodunda source/target tensor + source PIL + prompt arayüzü bulunduğu için kabul edilmiş teacher verisiyle adaptasyon somut bir başlangıç. Ağırlıklar noncommercial olduğundan bu yol önce araştırma prototipidir; ticari ürün için izin alınmadan dağıtım seçimi yapılamaz. [Gerçek kontroller](../experiments/dreamlite/README.md).

### Yol B — lisansı ve mimarisi bize uygun gerçek küçük öğrenci

**Klein 4B Base öğretmen + küçük maskeli öğrenci** öneriyorum. Kullanıcının geniş iPhone desteği hedefi için ortak denoiser bütçesi **100–400M mimari deneyi** olarak ele alınmalı. Bunun hazır, kaliteli bir outpainting checkpoint'i olduğu iddia edilmiyor; pretrained bir modeli structural pruning ile küçültüp recovery/task training yapmak veya küçük pretrained backbone'a mask/source koşulu öğretmek gerekir. Apache 2.0 SANA-600M, karşılaştırma/başlangıç backbone'u ve daha güçlü cihazlar için isteğe bağlı profil adayıdır; bütün eski cihazların ortak modeli değildir. SANA'nın mevcut T2I checkpoint'i tek başına masked outpaint yapmaz. [SANA repo](https://github.com/NVlabs/Sana), [600M checkpoint](https://huggingface.co/Efficient-Large-Model/Sana_600M_512px_diffusers)

ArtWorker tek işi yaptığı için devasa genel amaçlı visual-language encoder zorunlu ürün kararı olmamalı. Kaynak kareyi latent/patch koşulu olarak doğrudan denoiser'a vermek, maskeyi ayrıca vermek ve generic “continue the artwork” text embedding'ini önceden hesaplamak araştırılabilir. Bu yaklaşım 4B/8B caption encoder yükünü kaldırabilir; ancak anlam ve kompozisyon kalitesini veri/eğitimle geri kazanmak gerekir. İsteğe bağlı küçük vision encoder eklenirse onun parametre ve aktivasyonları bütçeye dahil edilir.

Önerilen aşamalar:

1. **Koşullu çok adımlı öğrenci:** mevcut küçük pretrained denoiser'ın source/mask/placement koşullarını eğit; önce 16–28 adımda kaliteyi sağla.
2. **Task knowledge transfer:** geniş gerçek görüntülerden center-crop görevleri + büyük öğretmenin seçilmiş wide-canvas sonuçlarıyla flow/diffusion training. Hedef yalnız dış alan olsa da global kompozisyon koşulu kaybolmamalı.
3. **Kendi öğrencisinden step distillation:** küçük öğrencinin kaliteli çok adımlı sürümünü öğretmen yapıp 8 → 4 adımlı sürüm eğit. Aynı latent/mimari üzerinde ODE/consistency/SenseFlow/DMD daha somut uygulanır.
4. **Sıkıştırma:** FP16 baseline → mixed 8/4 bit → gerekirse quantization-aware fine-tuning; her aşamada aynı holdout ve gerçek cihaz ölçümü.

**Latent uyumsuzluğu kritik:** Qwen/Klein ve SANA aynı latent codec/dimension kullanmıyor. Qwen hız LoRA'sını SANA'ya yükleyemeyiz; teacher velocity vektörü farklı latent uzayında doğrudan öğrenci velocity loss'u olamaz. İlk aktarım teacher görüntülerini **pixel space'e decode edip öğrencinin codec'iyle encode ederek** veri hedefi yapmalı. Doğrudan feature/velocity KD istenirse ayrıca öğrendiğimiz bridge veya ortak latent tasarımı gerekir. Bu, küçük öğrencinin kendi teacher'ından step distill edilmesini daha basit kılar.

## Hangi distillation yöntemi?

| Yöntem | Uygun kullanım | Buradaki sınır |
| --- | --- | --- |
| Offline teacher targets + conditional flow/diffusion fine-tuning | Büyük modelden farklı küçük mimariye task aktarımı; teacher ayrı çalışır | Teacher seçim hatalarını öğrenebilir; çeşitli sonuçları tek piksel-MSE hedefinde ortalamamak gerekir |
| ODE / trajectory regression | Aynı latent uzayında teacher noise levels / denoiser geçişlerini öğrenciye öğretme | Çok teacher forward ve trajectory depolama gerekebilir; tek başına aşırı az adımda kalite düşebilir |
| Consistency / sCM / MeanFlow | Az adımlı model için alternatif başlangıç | Kaynak yöntemlerin T2I başarıları maskeli editing başarısının kanıtı değil; condition'lar eğitimde korunmalı |
| DMD2 | Few-step örnek dağılımını öğretmenin dağılımına yaklaştırma | Generator + real teacher + fake-score/critic bellek maliyeti, stabilite ve GAN ayrıntıları var; LoRA model yükünü ortadan kaldırmaz |
| SenseFlow | Flow-based DiT'de dağılım eşleme ve intra-segment guidance | Önce task ve latent uyumu gerekir; resmi CUDA eğitim kodu iPhone veya Mac eğitim reçetesi değil |
| Progressive step distillation | Çok adımlı küçük modelden kademeli 8/4 adım | Daha az adım eşdeğer kalite garantisi vermez |
| Structural pruning + recovery KD | Katman/kanal azaltarak gerçek parametre küçültme | Rastgele katman silme kaliteyi bozabilir; export/kernel shape ve recovery training gerekir |

[DMD2 paper](https://arxiv.org/abs/2405.14867), [resmî SDXL eğitim rehberi](https://github.com/tianweiy/DMD2/blob/main/experiments/sdxl/README.md), [SenseFlow](https://github.com/XingtongGe/SenseFlow), [few-step practical guide](https://github.com/alibaba-damo-academy/T2I-Distill)

**SANA-Sprint** küçük flow backbone için continuous-time consistency + LADD training referansı sunuyor; resmî reçetede 1.6B modelin iki adımlı inference'i ve training script'i var, sonuç tablosunda 0.6B deney de bulunuyor. Bu T2I sonucu; maskeli ArtWorker öğrencisi ayrıca eğitilir. [Reçete](https://github.com/NVlabs/Sana/blob/main/asset/docs/sana_sprint.md)

Gerçek parametre küçültme örnekleri **BK-SDM** (U-Net residual/attention bloklarını kaldırıp feature/output KD) ve **TinyFusion** (sığ DiT ve recovery KD). İkisi de Qwen 2.1/Klein için hazır edit adapter'ı değil. TinyFusion'ın “masked KD” ifadesi image inpainting maskesi anlamına gelmiyor. Sadece ağırlıkları sıfırlayan unstructured pruning, dense mobile kernel boyutunu otomatik küçültmez; layer/channel kaldırılması ve export shape'i ölçülmelidir. [BK-SDM](https://github.com/Nota-NetsPresso/BK-SDM), [TinyFusion](https://github.com/VainF/TinyFusion)

Eğitim yönteminin kod lisansı ile model lisansı ayrı kontrol edilmelidir. Örneğin resmî DMD2 kodu CC BY-NC-SA 4.0; SenseFlow repo'su Apache 2.0. Makaledeki algoritmanın fikir olarak kullanılması, noncommercial repo kodunun ticari ürüne kopyalanmasıyla aynı lisans durumu değildir. [DMD2 license](https://github.com/tianweiy/DMD2/blob/main/LICENSE.md), [SenseFlow license](https://github.com/XingtongGe/SenseFlow/blob/main/LICENSE)

İlk araştırma için full DMD2'dan önce supervised task adaptasyonu ve trajectory/consistency denemesi daha ölçülebilir. Bunun gerekçesi model seçimi, condition tasarımı ve küçültme kaybını aynı anda üç büyük ağlı training stabilitesiyle karıştırmamaktır. DMD2 sonraki kalite iyileştirme seçeneği olmalı.

## Eğitim verisini nasıl üretiriz?

Üç mevcut albüm kapağı **ürün demosu ve ilk eleme** için yeterli; genel öğrenciyi eğitmek veya iyi genelleme iddiası için yeterli değil. Veri tasarımı önerisi:

1. Kullanım hakkı belli geniş görüntüleri topla: kendi işlerimiz, izinli artwork, uygun lisanslı fotoğraf/illüstrasyon. Tipografi, portre, minimal düz renk, doku, kolaj ve geometrik tasarım ayrı etiketlensin.
2. Geniş görüntünün kare bölümünü source olarak kes; dışı bilinen tam görüntü ground truth olur. Center ve hafif kaydırılmış source, farklı telefon oranları, hem üst/alt hem yan extension üret.
3. Albüm görünümüne benzeyen kare örnekler için öğretmen 2–3 farklı genişletme üretsin; seam/nesne/yazı hatası olanları filtrele. Sadece güzel seçilmiş sonuçlardan oluşan kapalı test setine karşı kendi eğitimimizi ölçmeyelim.
4. Teacher çıktılarını source üzerine hard composite yap; hedefte orijinal merkez değişmesin. Mask, bounding box, ratio, source hash, seed, checkpoint SHA, sampler/adımlar ve lisans provenance kaydet.
5. **Split kaynak görüntü/album bazında** olsun. Aynı kaynak için farklı crop/seed train ve test'e dağılmasın. Farklı sanatçı/stil/source grubu holdout ayrıca tutulsun.

İlk mühendislik pilotu için önerilen ölçek: **500–1,000 kaynak**, mask/ratio çeşitleriyle birkaç bin pair; 100+ kaynaklık ayrı holdout. Başarılı olursa **10–50 bin source/pair** düzeyinde daha geniş task adaptasyonu araştırılır. Bunlar kanıtlanmış minimum eğitim miktarı değil; veri ve compute yatırımı öncesi kalite trendini görmek için aşamalı hedeflerdir. Milyon ölçeği ancak pilotun learning curve'ü gerektirdiğinde düşünülmelidir.

Supervised flow/diffusion objective ana sinyal olabilir; mask dışındaki bilinen alan ve sınır için ek ağırlıklandırma denenir. Teacher target'larında bütün dış alan için yalnız deterministik MSE kullanmak farklı makul kompozisyonları ortalayarak bulanıklık oluşturabilir. Tek seed ezberinden kaçınmak için noise/seed çeşitliliği, distribution training ve insan değerlendirmesi korunmalı.

## Compute: bu Mac'te neler gerçekçi?

Mevcut makine 36 GiB unified memory. Büyük checkpoint'lerin quantized inference'ı ile büyük model **eğitimi** aynı bellek işi değil. Root deneyinin model başına gerçek ölçümü ayrıca kaydedilmeli; dosya boyutu RAM tahmini yerine kullanılmamalı.

Tipik mixed-precision Adam örneğinde parameter 2 byte + gradient 2 + FP32 master 4 + iki optimizer moment 8 = **yaklaşık 16 byte/trainable parameter**, activation ve geçici buffer hariç. Bu rejimde yalnız trainable state:

| Parametre | Yaklaşık training state |
| --- | ---: |
| 0.6B | 9.6 GB |
| 1B | 16 GB |
| 4B | 64 GB |
| 7B | 112 GB |

Bu sabit bir framework gereksinimi değil; FP32/FP16 master düzeni, sharding, 8-bit optimizer, LoRA ve checkpointing değiştirir. DMD ek teacher/critic ve activation gerektirir. Bu nedenle 36 GiB Mac'te Qwen'ın tam DMD eğitiminin rahat sığacağını söylemek yanlış olur.

Somut bir topluluk örneği olan Turbo8, 3,100 teacher trajectory, 3,000 initialization step ve 2,500 DMD2 step'i **bir RTX PRO 6000 Blackwell 96 GB** üzerinde eğittiğini; trajectory/eval için ayrı Ada kartı kullandığını bildiriyor. Bu minimum gereksinim değil, yayımlanan denemedir. DMD2'nin orijinal SDXL referans eğitimi 8 node × 8 GPU; bu da minimum değil. [Turbo8 eğitim](https://huggingface.co/chriswritescode/Turbo8-LoRA-Qwen-Image-2.1), [DMD2 guide](https://github.com/tianweiy/DMD2/blob/main/experiments/sdxl/README.md)

Pratik kaynak planı:

- Mac: veri hazırlama, teacher inference'in sığıp sığmadığı, küçük öğrenci inference, exporter ve iOS profiling. Küçük train/LoRA pilotu MPS/MLX operator desteği ve peak memory ölçülerek denenebilir; CUDA training script'inin doğrudan çalıştığı varsayılmaz.
- Büyük teacher: önce offline target üret; training sırasında Qwen/Klein'ı öğrenciyle aynı anda tutmak zorunlu olmasın.
- Küçük öğrenci CUDA pilotu: 24/48 GB kart, küçük batch + checkpointing ile profil çıkarmak için aday donanımdır; yeterli olacağı önceden garanti edilmez. Çok büyük DMD teacher için 80/96 GB veya sharding seçenekleri ayrıca ölçülür.
- Ücretli GPU işine veya çok günlü training'e başlamadan önce 100–300 update pilotuyla loss, örnek kalite, update süresi ve peak memory ölç. Kaynak maliyetinin sayılarını bundan çıkar.

Süre tahmini formülü: `teacher_targets = kaynak_adedi × seed_adedi × ölçülen_saniye/örnek`. Örneğin 10,000 kaynak ve 2 seed için **20,000 inference** gerekir; bunun kaç saat olduğu bizim donanımda ölçülen inference süresine bağlıdır. İlk birkaç seed'in prefill/warmup süresi dahil ve hariç ayrı kaydedilmelidir. Training için `update_adedi × ölçülen_saniye/update` aynı şekilde kullanılır. Bu ölçümler olmadan “bir gecede distill olur” veya sabit GPU-hour iddiası yapılmamalıdır.

## iOS hedefleri: kanıt ile hedefi ayır

**Ortak öğrenci hedefi:** yaklaşık **100–400M denoiser**, büyük VLM'siz veya küçük condition encoder'lı ve küçük codec'li görev mimarisi; 4–8 adım karşılaştırması. Bu ölçülmemiş bir tasarım aralığıdır. Bu büyüklükte kaliteli albüm outpainting'in kendiliğinden hazır olduğu varsayılmaz; structural pruning + recovery/task training veya küçük backbone'a koşul öğretme araştırılır. İlk kalite tutmazsa adım/çözünürlük/mimari ödünleşimi açıkça ölçülür.

Mevcut app iOS 17 hedefli; eski XR/XS sınıfı **A12** cihazlar kapsama araştırmasının ilk test grubuna alınmalı. OS uyumluluğu, küçük öğrenci bile olsa yeterli inference hızı veya bellek kanıtı değildir. Eski, orta ve yeni cihazlarda aynı üretim süresi vaat edilmez. Telefon adı, toplam RAM etiketi veya Simulator sonucu yeterli profil seçme verisi değildir; gerçek p95 süre, peak footprint, ısı ve player akıcılığı belirleyicidir. [Cihaz/OS sınırı ve deployment planı](/Users/emir/Documents/ChatGPT/ArtWorker/docs/IOS_OUTPAINT_DEPLOYMENT.md)

Aynı özel Core ML öğrenci için üç **sabit export adayı**:

| Profil | Canvas | Kullanım hipotezi |
| --- | --- | --- |
| Küçük | **256×576** | Eski/dar bellekli cihazda ilk üretim; kaynak kare UI'da özgün çözünürlüğünde korunur |
| Dengeli | **384×864** | Orta cihazda dış alan ayrıntısı ve bellek dengesi |
| Ayrıntılı | **512×1152** | Gerçek ölçümü geçen güçlü cihaz |

Bunlar aynı 1:2.25 oranının örnekleri; bütün iPhone ekranlarının bire bir oranı değildir. Codec, latent packing ve position-ID gereksinimleri doğrulanır; final crop/padding gerçek ekran oranına uyar. **Bu boyutlar özel Core ML öğrenci içindir.** İncelenen Draw Things SDK image-input yolu 64 hizası istiyor: 384×864 bu şartı karşılamıyor; bu SDK için **384×896 padded canvas + final crop** denenmeli. 512 kare benchmark süresi 512×1152 süresi olarak sunulmamalıdır. [64 hizası ve profil ayrımı](/Users/emir/Documents/ChatGPT/ArtWorker/docs/IOS_OUTPAINT_DEPLOYMENT.md)

**İsteğe bağlı güçlü-cihaz profili:** denoiser **0.4–1.0B** ve yaklaşık **2 GB peak app memory** araştırma hedefi yalnız ölçümü geçen yüksek kapasiteli cihazlar için düşünülebilir. Bu eski/orta cihazların ortak tabanı veya her cihaz için iOS memory limiti değildir. Ek mimari büyüklüğü, ayrıntılı çözünürlükten bağımsız deney değişkenidir; aynı küçük öğrenci yüksek çözünürlükte de denenebilir.

Yalnız 4-bit denoiser'ın teorik ham ağırlığı 100M için 50 MB, 400M için 200 MB; isteğe bağlı 600M için 300 MB ve 1B için 500 MB'dır. **VAE/encoder, scales/metadata, activation/cache, Core ML compilation, player ve image buffer dahil değil.** Ortak modelin gerçek peak bütçesi en eski hedef cihaz üzerinde ölçülerek belirlenir; bu ham boyutlardan güvenli app RAM limiti çıkarılmaz.

Apple'ın weight-only 4-bit quantization API'si ağırlıkları sıkıştırır; bütün arithmetic integer'a dönüşmez. Palettization veya W8A8 gibi seçeneklerin kernel davranışı ve latency'si cihaz/compute unit'e göre değişir. ANE, GPU ve hybrid graph aynı model için ölçülmelidir. [Apple compression API](https://apple.github.io/coremltools/docs/source/coremltools.optimize.coreml.post_training_quantization.html), [optimization overview](https://apple.github.io/coremltools/docs-guides/source/opt-overview.html), [workflow](https://apple.github.io/coremltools/docs-guides/source/opt-workflow.html)

Device kabul ölçümü:

- Hedeflediğimiz **gerçek en düşük cihaz** ve en güçlü cihazda cold/warm latency; Simulator süreleri cihaz benchmark'ı değil.
- Peak resident/physical footprint, model load/unload, 10 ardışık track change sırasında memory baskısı.
- Audio çalarken dropped frame, kesinti ve UI responsiveness.
- İlk üretim ve tekrar cached artwork açılışı; iptal edilen generation ve hızlı track değiştirme.
- 5–10 tekrar sonra thermal state/enerji ve latency değişimi.
- FP16 → mixed 8/4-bit sonrası seam, texture ve kompozisyon kaybı.

Albüm sonucu once-per-cover üretip cache'lemek, her playback frame'inde inference yapmaktan farklı compute bütçesi sağlar. “Yavaşça dışarı büyüme” animasyonu bir üretilmiş canvas'ın reveal'i veya gerçek strip-by-strip generation olabilir; ikisinin sınır/kompozisyon kalitesi ayrıca karşılaştırılır. Model boyutuna karar verirken sürekli video üretimini bu bütçeyle karıştırmayalım.

## Kalite kapıları ve deney kayıtları

İlk turda model/adım/quantization başına üç kapak × üç seed. En iyi görünen tek sonucu seçerek kıyas yapılmamalı. İkinci turda en az 30–100 stil açısından farklı, hiç eğitime girmemiş kaynakla değerlendirme öneriyorum.

| Kontrol | Ölçüm / kabul fikri |
| --- | --- |
| Orijinalin korunması | Kaynak rectangle byte eşitliği; final composite sonrası %100 |
| Seam | Sınırın iki tarafında lokal gradient/renk sürekliliği + insan değerlendirmesi |
| Genel kompozisyon | Boş alan, yinelenen kişi/nesne, anlamsız yazı, konu uyumu; kör tercih |
| Çeşitlilik | Aynı source için farklı seed; öğrenci aynı düzeni her seferinde kopyalamıyor mu? |
| İlerleme davranışı | Whole-canvas vs kademeli şerit; önceki bölgelerde tutarlılık |
| Sistem | Çözünürlük, süre, peak memory, compute unit, cihaz, thermal state |

Her run manifest'inde model repo+SHA, dosya hash'i, dtype/quantization, pipeline commit, prompt, source placement, mask, seed ve output hash tutulmalı. LoRA'yı merge etmek/sıkıştırmak çıktı farkı yaratabildiği için merged/unmerged bir deney değişkenidir. Teacher şablonu aynı kalırken insan değerlendirme isimleri rastgeleleştirilmeli.

## Tamamlanan karşılaştırmadan sonraki çalışma

1. **Tamamlanan baseline'lar:** Qwen/Klein Base, Viggle/Klein-distilled ve DreamLite/Mobile-O gerçek Mac çıktıları mevcut; yeni ControlNet/Fill kontrolleri ayrı kayıtlı. Fun-Acc burada çalıştırılmış sayılmaz.
2. **Sıradaki kalite hedefi:** source-exact bütün sahne ve sınır geometrisini farklı kapak/seed'lerde doğrula. Growing localized6 tamamlandı ve birleşim kaliteyi geçmedi. Tam-tuval saved-top32 yerel dış kenarı belirgin iyileştirdi; growing-top32 ve canvas-reference yalnız CPU hazır. MaskFlow/ProMax hazırlıkları başladı, inference sonucu yok. Tek iyi kompozisyon veya küçük MAE teacher kabulü değil. [Gerçek growing incelemesi](../experiments/qwen/boundary_eval/2026-10-05/controlnet-growing-localized6-comparison/REVIEW.md).
3. **Kabul sonrası veri pilotu:** izinleri uygun öğretmen, kaynak-grupları ayrılmış source/canvas/mask veri ve dışı bilinen gerçek geniş görüntüler.
4. **Küçük task öğrencisi:** önce supervised adaptasyon; kalite tutarsa kendi öğrencisinden 8/4-step distill.
5. **Apple export ve fiziksel ölçüm:** mimari, quantization, büyüme/caching ve player birlikte doğrulanır. Mevcut adapter'lar veya 100–400M hedefi mobil performans kanıtı değildir.

Bu sıra, “büyük modelin turbo LoRA'sı var mı?” sorusuna hazır adaylarla; “telefonda gerçekten küçük ve kaliteli model nasıl olur?” sorusuna ayrı mimari ve task eğitim planıyla yanıt veriyor. Albüm outpainting kalitesi ve gerçek iPhone ölçümü model seçimindeki son kanıt olmalı.
