# ArtWorker — indirilen modellerin gerçek outpainting denemeleri

İlk karşılaştırma: 30 Eylül 2026; diagnostic güncelleme: 5 Ekim 2026. Apple M4 Max / 36 GiB unified memory üzerinde yerel inference. Fiziksel iPhone ölçümü yok. Model ağırlıkları indirildi ve gerçek görüntüler üretildi; player'a model henüz bağlanmadı. **28 ortak-canvas denemesi** eski aggregate'de tutuluyor; sonraki sınır, ControlNet ve native Fill deneyleri ayrı kaydedildi.

## İlk sonuç

Hazır mobil modeller bu üç kapakta güvenilir outpainting sağlamadı. **Qwen 2.1 + Viggle 6-step**, araba/deniz kapağında dış alanı anlamlı devam ettirdi; aynı yöntemin Base 40-step çıktısında da sınır izi kaldı. **Klein 4-step** daha kısa sürdü ve sahneyi devam ettirdi, fakat yeni korkuluk ve grafik çerçeve hataları üretti. **DreamLite Base 28-step**, küçük mimariyle anlamlı yol/deniz üretilebildiğini gösterdi; Mobile dört adımlı sürümde bütünlük ve birleşim zayıfladı.

Bu ilk eleme, üç kapak ve tek seed için gözlemdir. Genel model sıralaması, istatistiksel kalite ölçümü veya tüm iPhone'larda çalışma garantisi değildir. Maske, prompt, codec ve çalışma şekilleri farklı olduğundan sonuçlar bütün model ailesini başarısız ilan etmek için kullanılamaz.

## 5 Ekim: tamamlanan sınır ve model kontrolleri

**Bütün görüntü için kalite kabulü henüz geçilmedi.** Aşağıdaki bulgular gerçek kaydedilmiş raw/composite görüntülere dayanıyor. Kaynak eşitliği veya düşük reconstruction MAE, dış alanın kaliteli olduğunu kanıtlamıyor.

- **Sampler:** soft source clamp ve dinamik VAE context re-encoding çalıştırıldı. Dinamik bağlam kaynak-sınır reconstruction ve ton geçişini iyileştirdi; korkuluk yönü/dirseği kaldı. [Rapor](../experiments/qwen/BOUNDARY_SAMPLER_RESULTS.md).
- **CPU compositor:** üç kapakta harmonic renk düzeltme, DIS/Farneback flow ve cubic remapping denendi. Renk düzeltme bazı ton birleşimlerini azaltıyor; geometriyi veya track2'nin eksik gövde devamını düzeltmiyor. [Rapor](../experiments/qwen/COMPOSITOR_RESULTS.md).
- **VAE projection:** sekiz pixel-consistency update tamamlandı; kaynak-sınır MAE düştü, korkuluk düzelmedi. Peak **40,60 GiB** idi. Derivative kolu timeout ile altı/sekiz update'de kesildi, final görüntüsü yok; checkpointed-decoder sürümünün yalnız CPU kontrolleri mevcut. Bu model eğitimi değil, çıktı latent optimizasyonu. [Rapor](../experiments/qwen/PROJECTION_RESULTS.md).
- **Viggle v0.3:** altı adım ile resmî yedi Turbo + iki Base adımı çalıştırıldı. Yeni adapter/tail sınır geometri hatasını çözmedi. **LanPaint-style** kontrolü 6/9/12 gerçek NFE kullandı; tek düzeltmede dirsek kaldı, çift düzeltmede korkuluk koptu. [Turbo/tail](../experiments/qwen/TURBO_TAIL_RESULTS.md), [LanPaint-style](../experiments/qwen/LANPAINT_RESULTS.md).
- **Eğitimli ControlNet:** native 20/40-step ve known-bridge40, korkuluk yakın birleşimini iyileştirirken beyaz dış bantlar ve uyumsuz sahne üretti. Sparse koşullama beyazlığı kaldırdı, yol/deniz/korkuluk panellerini tekrarladı. Gerçek kare reference + güçlü outpaint prompt kombinasyonu da bu tekrarları kaldırmadı. [Tam/sparse deneyler](../experiments/qwen/CONTROLNET_RESULTS.md), [reference40 incelemesi](../experiments/qwen/boundary_eval/2026-10-05/controlnet-reference40-comparison/REVIEW.md).
- **Yerelleştirilmiş hint40:** üstte su/bitki ve altta asfaltla tek sahne elde edildi. Üretilen korkuluk dış kenarı **−17,404 px yatay kaydı**, fazladan metal bant ve doku birleşimleri kaldı. Küçük **+0,300°** yakın açı farkı kalite başarısı değil. Hint'lerin yalnız 97 target konumunda pozitif olması compute sparsity değil: her adım bütün 2304 target + 1024 reference image tokenı işlendi. [Bağımsız gerçek görüntü/array incelemesi](../experiments/qwen/boundary_eval/2026-10-05/controlnet-localized40-comparison/REVIEW.md).
- **Saved-top32:** localized40'ın gerçek ortak tensorları aynen geri yüklenip yalnız 64 known-source üst hint hücresi değiştirildi. Asıl dış metal kenarı hatası **−17,404 → −2,086 px**, yakın açı farkı **+0,341°**; geniş ekstra bant belirgin azaldı. Yine de zayıf son-satır edge'i/3,930 px jog ve su/asfalt dikişleri kaldı. Bu en belirgin yerel mekanik ilerleme, bütün görüntü kabulü değil. Tam tuval40, 92.160 target-token-forward. [Gerçek dış-edge incelemesi](../experiments/qwen/boundary_eval/2026-10-05/controlnet-saved-top32-comparison/REVIEW.md).
- **Native FLUX.1 Fill Q4/50:** ilk tam tuval 347,407 s/50 NFE'de yazı ve ikinci sahne üretti. Matched exterior-only prompt tekrarları kaldırdı, üst korkuluğu köprü gibi eğdi ve alt asfaltı değiştirdi. İki bağımsız **local-context50** çağrısı **320,980 s wrapper / 100 NFE / 89.600 target-token-forward** yaptı; üst göl makul, altta iki soluk bant/doku kusuru kaldı. Hiçbiri kalite kabulü değil; iki local çağrı gerçek tek-akış büyüme sayılmaz. [Fill raporu](../experiments/flux1_fill/FILL_RESULTS.md), [local-context gerçek kayıtları](../experiments/flux1_fill/runs/2026-10-05/track3-local-context50-seed42/metrics.json).
- **Native SDXL inpaint:** 9-channel20/CFG8/strength0,99, **25,630 s wrapper**; scheduler 19 gerçek U-Net çağrısı/38 CFG branch sample kullandı. Dış alan ciddi siyah çöküş gösterdi. Kaydedilmiş maske, ilk input, gerçek noise ve raw postprocess audit'i bunu ters maske veya compositor siyahı olarak desteklemiyor. [Gerçek sonuç ve initialization audit'i](../experiments/sdxl_inpaint/SDXL_RESULTS.md).
- **Gerçek growing localized6:** 0 / 0,5 / 1 control scale kolları tamamlandı; üç-kol wrapper toplamı **211,185 s**. Her kol altı NFE, 12.288 target + 6.144 reference image token forward ve her base/control chain için 19.470 joint query yaptı. Gelecek target tokenları gerçekten çıkarıldı. Bütün sahne tutarlı, korkuluk profilinde basamak ve ekstra kenar kaldı; üç finalde sabit evaluator `invalid-feature` verdi, bu sıfır hata değil. Scale0 eski cached akışa yakın, bit-identical değil; finalized aynı-stage clean-latent karşılaştırması max0,0625/mean0,002002 fark verdi, yalnız caching'e bağlanamaz. [Gerçek ara/final görüntü incelemesi](../experiments/qwen/boundary_eval/2026-10-05/controlnet-growing-localized6-comparison/REVIEW.md).

İlk known-collar kontrolü artık saved-top32 ile gerçek inference sonucu verdi; **growing-top32 ve portrait canvas-reference** için yalnız CPU hazırlığı var, gerçek GPU kalite sonucu yok. MaskFlow'un 2511'e uygun izole port/model indirmesi ve ProMax outpainting tarifi ağırlık indirmesi başladı; bunlar henüz üretilmiş sonuç değil. Öğrenci eğitimi, ücretli cloud işi, fiziksel iPhone inference ve Player model entegrasyonu yapılmadı. [MaskFlow uyumluluk araştırması](../experiments/maskflow/research/2026-10-05/FEASIBILITY.md), [ProMax hazırlığı](../experiments/sdxl_inpaint/PROMAX_FEASIBILITY.md), [mekanizma ve compute ayrımı](SPATIAL_STREAMING_RESEARCH.md).

## Deneyin ortak geometrisi

- Projedeki üç JPEG'in ortasındaki 720×720 kapak kesildi; **512×512 `source_512.png`** ortak çalışma referansı oldu. JPEG dosyaları değiştirilmedi.
- Ortak sonuç **512×1152**, kaynak rectangle **[0, 320, 512, 832]**, seed **42**. Üstte ve altta 320 px yeni alan.
- Ham model sonucu ve final source composite ayrı saklandı. Final sonuçların kaynak eşitliği, yeniden boyutlandırılmış 512 kareye karşı ölçülür; orijinal 720 kareyle aynı pixel array olduğu iddia edilmez.
- Piksel koruma final composite ile sağlanır. Bu, dış alanın kaliteli veya birleşimin görünmez olduğu anlamına gelmez.
- Model checkpoint/revision, pipeline, prompt, mask, quantization, süre ve bellek kayıtları deney manifest'lerinde. Başarısız runtime/indirme denemeleri model kalite sonucu diye sayılmadı.

| Kapak | Test ettiği sorun |
| --- | --- |
| Un Día / Future Nostalgia | Siyah arka plan, tek kişi/ay, tipografinin tekrar etmemesi |
| karambol | Cyan duvar, iki kişi, büyük beyaz yazı; aşağıda kıyafet devamı |
| Aklın Hep Bende | Yol/deniz/korkuluk perspektifi, gece tonları, tek araba/kişi |

## Tamamlanan süre ölçümleri

Süre scope'u satıra göre açıkça yazılıdır. Qwen'de generation fazları ve process toplamı ayrı; Klein'da lazy ağırlık yükleme inpainting chain'in içinde; DreamLite/Mobile-O modelleri aynı process'te yüklenip peş peşe üretilmiştir. Bunlar eşit cold-start koşulları olan bir hız yarışması değildir.

| Aday / yöntem | Un Día | karambol | Aklın Hep Bende | Süre scope'u |
| --- | ---: | ---: | ---: | --- |
| Mobile-O / 20-step instruction edit | 4,91 s | 4,74 s | 4,99 s | Generation; 2,86 s model load hariç |
| Mobile-O / 20-step known-latent lock | 5,80 s | 8,10 s | 8,57 s | Generation; 3,20 s load hariç |
| DreamLite Mobile / 4-step ilk portrait | 4,66 s | 1,31 s | 1,23 s | Generation; 4,20 s load hariç; warmup farkı var |
| DreamLite Mobile / 4-step square context → crop | 5,13 s | 3,72 s | 4,20 s | Generation; 1024 kare çalışma, sonra crop/resize |
| DreamLite Mobile / 4-step ayrıntılı coastal prompt | — | — | 7,78 s | Generation; 4,24 s load hariç |
| DreamLite Base / 28-step aynı coastal prompt | — | — | 32,36 s | Generation; 2,48 s load hariç |
| Klein 4B distilled / INT4 / 4-step | 26,0 s | 23,5 s | 26,2 s | Chain; lazy encoder/transformer load dahil |
| Klein 4B Base / INT4 / 50-step / CFG4 | — | — | 550,6 s | Chain; lazy load dahil; guidance iki forward kullanıyor |
| Qwen 2.1 Base / Q4 / 40-step | 405,64 s | 192,29 s | 334,92 s | Tam attempt, model load dahil |
| Qwen 2.1 Viggle v0.2.1 r128 / Q4 / 6-step | 69,27 s | 54,66 s | 52,73 s | Tam attempt, model load dahil |
| Qwen 2.1 Viggle v0.2.1 r256 / Q4 / 6-step | — | 45,17 s | 48,23 s | Tam attempt, model load dahil; önerilen tam-rank adapter |

Qwen r128 track3 generation fazları toplamı **41,53 s**, full attempt **52,73 s**. Qwen Base track1 generation fazları **386,42 s**, full attempt **405,64 s**. Checkpoint'in önerilen **r256** adapter'ı ayrıca indirildi ve çalıştırıldı; bu adapter aynı büyük sistemi kullanıyor, yalnız adım sayısını azaltıyor. Tek ölçümlerde cache, warmup ve ilk Qwen denemelerine eşlik eden CPU derleme yükü farklı: tek satırdan kesin hız oranı çıkarılmaz.

## Görüntülerde görülen farklar

**Mobile-O:** ilk portrait çıktılarda kişi/ay/araba/yazı tekrarları ve kötü birleşimler. Known-latent restore, ham kaynak bozulmasını azalttı; dış alandaki semantik sorunu çözmedi. 0.5B etiketi bütün sistem değil; yüklenen modelde **1.664.722.343 parametre** sayıldı. Native iOS app'te mevcut generate yolu T2I, attachment yolu understanding; Python edit akışı Swift'e henüz bağlanmamış. [Deney](../experiments/mobile_o/README.md)

**DreamLite:** ilk Mobile portrait denemeleri kolaj gibi oldu. Kontrol için resmi 1024 kare boyutta sıradan araba rengi değiştirme çalıştı; modelin bütün image editing kabiliyetini reddetmedik. Base 28-step portrait, yol ve denizi anlamlı üretti; gündüz/gece renk uyuşmazlığı ve sınır izi vardı. Aynı ayrıntılı prompt Mobile'a verilince ilk sonuca göre iyileşti, fakat farklı kanal/bitki ve doku sorunları kaldı. Bu gözlemde prompt ve distillation birlikte kaliteyi etkiliyor. [Deney ve kontroller](../experiments/dreamlite/README.md)

DreamLite kare-context yolunda source 448 kare olarak 1024 canvas'a yerleştirildi; 448×1008 alan crop edilip 512×1152'ye büyütüldü ve kaynak tekrar aynen kondu. Bu, aynı model çalışma çözünürlüğü değildir: **1,049 MP** üretimden crop, doğrudan portrait'te **0,590 MP**. Cyan arka plan iyileşti, fakat siyah çerçeve çizgileri/boşlukları ve yanlış aşağı devamı sürdü. Verilen iOS export'unun kare sabit şekli için pratik bir araştırma yolu; çözülmüş outpainting motoru değil.

**Klein distilled:** siyah arka planın uzantısı temiz; cyan duvar ve gövde devamı anlamlı. karambol'un aşağısında yatay siyah bant kaldı. Yol/deniz kapağında doğru malzemeler var, fakat aşağıda ikinci korkuluk/geometri eklendi. Kaynak-referans + RePaint tarzı latent blending, 32 px kaynak içi soft mask rampası, nötr noise canvas kullanıldı; bu LanPaint'in tam sampler'ı değil. [Runtime, ağırlık ve deneme](../experiments/flux2/README.md)

**Klein Base:** aynı INT4 ve prompt ile 50-step/CFG4 track3 denendi. Distilled çıktıda altta eklenen ikinci korkuluk kayboldu, deniz/asfalt devamı daha sade oldu. Üstte su tonunda ve altta asfalt texture'ında kaynak birleşimi hâlâ görünür. Süre yaklaşık 9 dakika; bu modelin yalnız adımlarını dörde indirmek, eğitilmiş dört-adımlı model yapmak değildir.

**Qwen:** siyah kapakta Base/Turbo temiz ve ek kişi/ay yok. Yol/denizde 6-step r128/r256 ve Base 40-step doğru su/asfalt devamı üretti; merkez üst kenarında düz yatay birleşim izi kaldı. Base'de de aynı iz görüldüğü için yalnız turbo adapter'a bağlanamaz. karambol r128 Turbo aşağıda siyah bant, önerilen r256 ise düz cyan bant üretti; **40-step Base'de bu bant kayboldu ve kıyafet/gövde devamı tutarlı oldu**. Sorun yalnız adapter rank kırpılmasına indirgenemez. [Deney](../experiments/qwen/README.md)

**Qwen sınır kontrolü:** r256 track3 için yalnız known-source VAE context'i gray padding yerine source kenarlarının uzatılmasıyla encode edildi. Vision ve reference prefix aynı tutuldu. Top/bottom 16 px ham kaynak MAE sırasıyla **18,02→8,24** ve **12,25→4,87**; iç kaynak MAE **3,25→3,21**. Görselde sınır halkası azaldı, su texture birleşimi tamamen kaybolmadı. Süre **46,14 s** full attempt. Bu, sampler/codec hazırlığının kaliteye etkisini gösteriyor; final merkez eşitliği zaten composite ile garanti ediliyor.

## Bellek: dosya boyutu, RSS ve GPU allocation aynı şey değil

| Aday | Ölçüm | Scope / sınır |
| --- | --- | --- |
| Mobile-O ilk tur | MPS driver son snapshot yaklaşık 6,64 GB | Transient peak ölçümü değil |
| Mobile-O latent-lock | Sampled MPS driver yaklaşık 6,65 GB | Sampler'ın gördüğü maximum; daha kısa peak kaçabilir |
| DreamLite Mobile portrait | Sampled MPS driver yaklaşık 6,37 GB | FP16/BF16 encoder tam sistem; Mac |
| DreamLite Mobile square context | Sampled MPS driver yaklaşık 7,44 GB | Daha büyük kare çalışma |
| DreamLite Base portrait | Sampled MPS driver yaklaşık 9,46 GB | 28-step aynı küçük denoiser mimarisi |
| Klein distilled | MLX cumulative allocation peak **7.513 MiB**, yaklaşık **7,88 GB** | Lazy load dahil; macOS footprint ayrıca yaklaşık 12,4 GB |
| Qwen Q4 | MLX load peak **26,31 GB**; inference phase peak yaklaşık **11,63–11,69 GB** | On-load quantization açılış peak'ini büyütüyor |

Bu farklı ölçümler tek bir “gerçek RAM” sütununda karşılaştırılmaz ve toplanmaz. RSS, macOS footprint, MPS/MLX aktif allocation, cache ve driver allocation ayrı kayıtlardır. iPhone için `phys_footprint`, Instruments, available memory, jetsam ve ses sürekliliği ayrıca ölçülmeli.

Qwen Base Q4 aktif ağırlıkları DiT **4,148 GB**, encoder **4,778 GB**, VAE **1,351 GB**: toplam **10,277 GB**. VAE resmi checkpoint'te F32. Viggle r128 **0,680 GB**, r256 **1,359 GB** ekstra adapter; 680 MB dosya bütün bağımsız model değildir. DreamLite 389M etiketi yalnız U-Net; ayrı 2B VLM bütçede. [Hazır distillation / full-stack açıklaması](DISTILLATION_PLAN.md)

## iPhone ve kendi öğrencimiz için karar

Geniş cihaz desteği için büyük modele yalnız turbo adapter takmak yeterli görünmüyor. Önce farklı kapak/seed'lerde kabul edilen bir öğretmen/sampler gerekir; mevcut başarısız sonuçları distill etmek geometrik ve semantik hataları aktarabilir. Sonra **kapağın dışını devam ettirmeye özel, küçük source/mask koşullu öğrenci** araştırılmalı. Büyük VLM'yi kaldırmak sabit text embedding cache'inden ibaret değil: görüntü conditioning'ini koruyan küçük encoder veya yeni koşul katmanı eğitilmeli.

1. Kaliteli, izinleri uygun geniş görüntülerden merkez kare → tam görüntü pair'leri; büyük öğretmenden seçilmiş ek hedefler. Kaynak grubu bazında train/test ayrımı; üç demo kapağı eğitim veri seti sayılmaz.
2. Öğretmen ve öğrenci farklı codec kullanıyorsa önce **pixel target → öğrenci codec → task fine-tune**. Qwen/Klein LoRA veya velocity hedefini başka latent uzayına doğrudan taşımayız.
3. Önce çok adımlı küçük task modelinin kalitesi; sonra kendi latent uzayında 8/4-step consistency/trajectory veya dağılım distillation.
4. Ortak **100–400M denoiser** mimari/pruning hedefi; ölçülmüş hazır model değil. 256×576, 384×864, 512×1152 sabit Core ML export adayları; cihaz bütçesine göre seçilir.
5. Prequantized loading, mixed-bit katmanlar, encoder offload/cache, cancellation, kapak cache'i, thermal/bellek durumuna göre profil düşürme. Kaynak kapak UI'da orijinal yüksek çözünürlüğünde ayrı layer kalır.

Klein 4B/Base Apache 2.0 olduğu için ticari öğretmen başlangıcında uygun aday. DreamLite/Mobile-O araştırma ağırlıkları noncommercial; Qwen 2.1 Research License. Lisanslar ve kendi küçük backbone seçimi [distillation planında](DISTILLATION_PLAN.md) kaynaklarıyla ayrıldı. HiDream-O1 Full/Dev araştırıldı, bu oturumda indirilip inference yapılmadı; CUDA odaklı runtime nedeniyle Mac karşılaştırmasına hazır aday diye sunulmadı.

MediaGenerationKit minimal fixture **iOS Simulator arm64+x86_64 için derlendi**. Model inference veya telefon hızı ölçmedi. SDK'da alpha-derived mask ve native sınıf kodları incelendi; standart 0/255 grayscale maske varsayımı yanlış olabilir. [iOS planı, SDK doğrulaması ve cihaz kabul matrisi](IOS_OUTPAINT_DEPLOYMENT.md)

## Tek üretimin içinde merkezden dışarı büyüme

Kullanıcının isteği, yeni boyut için tekrar tekrar ayrı üretim çağırmak değil; **aynı sampling akışında hesaplanan/generatif alanın kapağın çevresinden dışarı büyümesi**. Başlangıçtaki ayrı tur açıklaması bu isteği karşılamıyordu.

Spatial time/noise conditioning ile yakın bölgeler önce, uzak bölgeler sonra netleştirilebilir. AsyncPatch ve Patch Forcing bunu araştırıyor. Ancak bölgesel zamanlama dense modelin bütün tuvali hesaplamasını kendiliğinden engellemez. Gerçek compute tasarrufu için aktif target tokenları, attention/cache ve sampler birlikte değişmeli. [Birincil kaynaklar, mevcut modellerin sınırları ve deney tasarımı](SPATIAL_STREAMING_RESEARCH.md)

Qwen'in temiz reference-prefix cache'i ve dinamik target uzunluğu, tek model yüklemesi/döngü içinde aktif token alanını büyüten bir prototip için incelendi. Yeni tokenların geç diffusion adımında nasıl başlatılacağı ve noise conditioning eğitim dağılımı kalite riski. Mevcut temel karşılaştırma tek tam canvas üretimidir; aktif-canvas denemesi ayrı tutulur. Modelin kendi başına hazır merkezden dışarı stream özelliği olduğu iddia edilmez.

**Gerçek prototip sonucu:** aynı 512 kare reference-prefix, prompt, seed/noise ve sigma schedule ile iki **6-NFE** döngüsü karşılaştırıldı. FULL her adım 2304 target token; GROW 2 adım 1280, 2 adım 1792, 2 adım 2304 token hesapladı. Aktif pencere **512×640 → 512×896 → 512×1152**. Gelecek tokenlar yalnız maskelenmedi, transformer girdisinden gerçekten çıkarıldı. Sabit source-prefix cache ve tam hedefe göre mutlak RoPE konumları korundu.

Target token-forward toplamı **13.824 → 10.752** (%22,2 az); denoising **29,38 → 24,33 s**. Tek eşleştirilmiş ölçüm; sıra, warmup ve thermal etkisi olabilir. Load/conditioning pair için bir kez yapıldı, preview capture/decode core süre dışında. **GROW final kalitesi başarısız:** son bantlarda ızgara izleri, ek korkuluk ve küçük hayalî işaretler. Geç eklenen tokenlar source latent kenarından tahmin + mevcut sigma'da noise ile başlatıldı; bu eğitilmemiş bir heuristic. Daha az hesap gerçek, kalite korunumu henüz çözülmüş değil.

İç alanlar aynı global sigma'da son adıma kadar netleşmeye devam etti; erken preview tamamlanmış/dondurulmuş halka demek değil. Sampling state'inden alınan gerçek predicted-clean latentler 2/4/6. adımda kaydedildi; adil core süre karşılaştırması için VAE decode iki döngüden sonra yapıldı. **Canlı UI streaming ölçülmedi.** [Prototip ve tam kayıt](../experiments/qwen/SPATIAL_RESULTS.md), [ara aşamalar](../experiments/qwen/spatial_runs/q4/track3/growth-stages.png), [eşleştirilmiş son görüntüler](../experiments/qwen/spatial_runs/q4/track3/full-vs-grow.png)

**5 Ekim güncellemesi:** aynı ağırlıklarla FULL/OLD/EARLY-SOURCE/EARLY-FRONTIER dört kollu kontrol çalıştı. Dış alanları ilk üç adımda açmak, track3 finalindeki ağır ızgara ve sahte alt işaretleri kaldırdı; yeni ufuk/kıyı ve sınır doku değişimleri kaldı. Her arm 6 NFE, erken plan 12.288 target-token-forward (%11,1 FULL'den az). Sıfırdan eğitim zorunlu olduğu sonucu desteklenmiyor; genel kapak kalitesi/telefon hızı henüz doğrulanmış değil. Yeni denemeler eski 28-trial aggregate'e karıştırılmadı. [Yeni kontrol raporu](../experiments/qwen/QUALITY_ABLATION_RESULTS.md), [gerçek 6. adım karşılaştırması](../experiments/qwen/quality_runs/2026-10-05/track3/final-comparison.png).

Ek track2 FULL/EARLY-FRONTIER pair'de üstteki kopya başlık azaldı; iki finalde de alt kıyafet/gövde devamı kesilip cyan panel kaldı. Toplam altı yeni inference arm / 36 NFE runtime kontrollerini geçti, fakat robust bütün-kapak kalitesi çözülmedi. Modelin edit/task conditioning ve few-step kapasitesi, geç aktivasyondan ayrı araştırma konusu. [İkinci kapak](../experiments/qwen/quality_runs/2026-10-05/track2/final-comparison.png).

Canvas maliyeti için ek bir kısa Klein kontrolü yapıldı: aynı 512 source, prompt, seed, INT4 ve **4 NFE**, hedef **512×640**. Whole chain **13,74 s**, denoising **9,93 s**; önceki 512×1152 track3 whole chain **26,26 s**. Bu iki tekil ölçüm aynı cold-start koşulunda benchmark değildir; daha küçük hedefin aynı NFE'de daha az iş yapabildiğini gösterir. Merkezden dışarı sampling kanıtı olarak sayılmaz. [Kontrol kaydı](../experiments/flux2/compute_area_control.json)

**Kaynak sınırı geometri kontrolü:** EARLY-FRONTIER ile iki yeni matched arm çalıştı; yalnız sabitlenen target kaynak latentlerinin VAE encode bağlamı değişti. Orijinal kare reference prefix/noise/prompt/sigmas aynı kaldı, baseline önceki finali birebir yeniden üretti. Kareyi edge-padded geniş tuvalde encode etmek üst/alt 16 px raw kaynak MAE'yi **14,33→8,72 / 10,05→4,98** azalttı ve korkulukta büyük zikzakı hafifletti. Küçük konum/eğim farkı, su/bitki sınır izi kaldı; alt asfaltta koyu yatay geçiş belirginleşti. Bu, kısmi bir sınır iyileşmesi; tüm görüntü için başarı değil. Her arm 6 NFE / 12.288 target token, finite ve exact-source. Bu iki diagnostic arm eski aggregate ve altı-arm kalite süitinden ayrı. [Rapor ve gerçek closeup](../experiments/qwen/GEOMETRY_RESULTS.md).

## Yeniden çalıştırma ve kanıt

```sh
.build/dreamlite-venv/bin/python experiments/evaluation/collect_runs.py
```

Normalizer mevcut manifest'leri okur; eksik trial üretmez. [Karşılaştırma HTML'i](../experiments/evaluation/aggregate/comparison.html), [normalize run kayıtları](../experiments/evaluation/aggregate/runs.json), [piksel/kenar raporu](../experiments/evaluation/aggregate/report.json). Sınır row-difference metriği yalnız kaba tanıdır; görsel kalite skoru değildir. Diagnostic kare kontrolleri ayrı tutulur. Ham/composite PNG'ler yan yana incelenebilir.

Final standard aggregate: **28 trial / 56 görüntü / 28 exact-source composite**, ayrıca 8 DreamLite diagnostic görüntü kaydı. 64 image record / 62 ayrı dosya; iki Base portrait kaydı aynı kontrol dosyalarını referanslıyor. Eksik dosya, source eşitliği veya geometri hatası yok; kaynak üç JPEG hash'i değişmedi. Spatial FULL/GROW tamamen ayrı tutuldu. [Doğrulama](../experiments/evaluation/aggregate/validation.json), [zor kapakta dört aday](../experiments/evaluation/aggregate/track3-curated.png)
