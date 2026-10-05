# Tek üretim akışında merkezden dışarı outpainting

İlk araştırma: 30 Eylül 2026; deney güncellemesi: 5 Ekim 2026. Bu belge, kullanıcının düzelttiği soruyu ele alır: modelin **üretim işleminin kendisi**, tek bir inference oturumu/sampling akışı içinde kapağın kenarından dışarı doğru ilerleyebilir mi? İstenen, 1000×1000 → 1100×1100 → … şeklinde ayrı ayrı bitmiş görseller üretmek veya bitmiş büyük görseli animasyonla açmak değildir.

**Evet, bunu mümkün kılan model ve sampling tasarımları var.** Ancak şu anda denediğimiz Qwen 2.1, Klein ve DreamLite checkpoint'lerinde güvenilir bir “merkezden dışarı üret” ayarı doğrulanmadı. Üretim sırasını değiştirmek ile hesaplamayı azaltmak iki ayrı teknik gereksinimdir. Mac'te canvas-maliyeti kontrolü ve **tek akışta gerçekten büyüyen aktif-token prototipi çalıştırıldı**; hesap tasarrufu görüldü fakat final kalite bozuldu. Fiziksel telefon benchmark'ı yapılmadı.

## Kullanıcının örneğinin gerçek karşılığı

Hedef 1400×2400 ise, kapağı bu koordinat sisteminde başlangıçta bilinen bir bölge olarak tutabiliriz. Kapaktan uzaklığa göre çevreyi bantlara ayırırız. Bilinen merkez temiz kalır; ilk bant önce tamamlanır, sonraki bant daha sonra tamamlanır. Bitmiş bantlar sonraki üretime bağlam verir. Hedefin koordinatları en baştan belli olabilir; gerçek tensorun her karede yeniden boyutlandırılması bu davranış için zorunlu değildir.

Bir diffusion modelinde bunun doğal ifadesi, tek global noise level yerine bölgesel noise level'dır: `σ(x, y, s)`. Burada `s` tek inference akışının ilerlemesi, `(x,y)` görsel konumudur. Aynı adımda kaynak kapağın `σ=0`, yakın bandın düşük, uzak bandın yüksek noise level'da bulunması mümkündür. Halkaları geometrik daire yapmak gerekmez: telefon oranına uygun üst/alt şeritler ve köşeler de sıralanabilir. Bu bölüm, aşağıdaki mekanizmalardan proje için yapılan bir tasarım çıkarımıdır.

Aynı adım sayısı, aynı işlem maliyeti değildir: 1280×1280 canvas 1.638.400 piksel, 1400×2400 canvas 3.360.000 piksel içerir. Sabit VAE/patch düzeninde daha büyük canvas daha çok latent token üretir; dense DiT attention ve MLP maliyeti büyür. Bu aritmetikten tam süre oranı çıkarılamaz. Kaynak kapağın merkezde sabit olması da mevcut dense runtime'ın merkez tokenlarını hiç hesaplamadığı anlamına gelmez. Dolayısıyla “iki boyuta çıkarmak aynı adım sayısında” olabilir, “aynı sürede/aynı hesapla” sonucu ayrıca ölçülmelidir.

Yerel küçük alan kontrolü: Klein INT4/4 NFE, aynı 512 source/prompt/seed ile 512×640 hedef whole-chain **13,74 s**, denoising **9,93 s**; 512×1152 track3 whole-chain **26,26 s**. Tekil ölçümlerde cache/CPU yükü aynı değil; kesin hız oranı değil, boyutun işlem maliyetini etkileyebildiğine dair kanıt. Bu kontrol merkezden dışarı üretim testi değildir. [Mac ölçümü](../experiments/flux2/compute_area_control.json)

## Doğrudan ilgili güncel araştırmalar

### AsyncPatch Diffusion — Haziran 2026

[Makale](https://arxiv.org/html/2606.07079v1), özellikle §3.1–3.2 ve §5.5.

AsyncPatch, piksel veya latent token için ayrı timestep tanımlar. UNet'te timestep'in 2D harita olarak FiLM'e verilmesi ve iç katmanlarda küçültülmesi kullanılır. Tek sampler içinde temiz bilinen bölge, denoise edilen aktif grup ve tamamen gürültülü gelecek gruplar birlikte bulunabilir. Sıralı grup schedule'ı açıkça tanımlanmıştır; merkezden dışarı grup sırası bizim uyarlamamız olur. Model bu heterojen noise durumlarıyla **eğitilmiştir**; metindeki “pretrained model” sıradan herhangi bir mevcut checkpoint'e bir eklenti anlamına gelmez.

Önemli hesaplama ayrımı: joint score bütün tokenlardan tahmin edilir; yalnız aktif bölgeye reverse update uygulamak UNet forward maliyetini ortadan kaldırmaz. Yazarların 8×8 patch raster deneyi 16 adım/patch, toplam 1024 sampling adımı kullanır. Dolayısıyla yöntem, kendiliğinden düşük gecikmeli telefon çözümü değildir. Yayımlanan deneyler ImageNet/LSUN üzerindedir; yüksek kaliteli albüm düzenleme kanıtı değildir.

### Patch Forcing — CVPR 2026

[Makale](https://arxiv.org/html/2604.19141v1), [resmî kod](https://github.com/CompVis/patch-forcing).

Patch Forcing, DiT AdaLN conditioning'i token başına timestep'e genişletir. Heterojen timesteps ile eğitimde oluşan dağılım uyuşmazlığını özel timestep sampler ile düzeltir; ek bir patch difficulty head kolay bölgelerin daha önce ilerlemesini sağlar. Bu yöntem doğal olarak gerçek bölgesel üretim sırasını destekler. Geometrik merkezden dışarı sıra ise ayrıca seçilecek bir schedule'dır.

Makalenin Appendix A.1 kısmı bu sampler'ların esas amacını kalite artışı olarak ayırır; hızlandırma için RAS prediction/KV caching ile birleştirme deneyini sunar. Yalnız spatial update mask'ini değiştirmek hız kanıtı değildir. Resmî kodun `integrators.py` dual-loop akışı da tüm `xt` üzerinden `sample_fn` çağırdıktan sonra maskeli Euler güncellemeleri yapar. İncelenen kod revision'ı: `82554948508424db2667b07e0ffb6828cec7c129`.

5 Ekim kaynak kontrolü: [Appendix A.2](https://arxiv.org/html/2604.19141v1#A1.SS2.SSS0.Px2), pretrained PixArt-α üzerinde tokenwise timestep broadcast ve mevcut variance head ile **eğitimsiz, keşif amaçlı** uyarlama gösteriyor. Deneyler CFG olmadan; özel spatial eğitim daha sağlam sonuçlar sağlayabiliyor. Bu yüzden spatial timestep için eğitim her durumda zorunlu denemez. Qwen'in global-schedule six-step adapter'ında veya fiziksel olarak sonradan eklenen dış tokenlarda bu sonuç doğrulanmış değil.

### Just-in-Time — geç aktive edilen tokenları hizalama

[Makale](https://arxiv.org/abs/2603.10744), [resmî kod, 818ed8d](https://github.com/Wenhao-Sun77/Just-in-Time/tree/818ed8de7333f0c83e3f88a006576036c0f931e8).

Bu training-free yöntem sparse anchor tokenlardan velocity tahmini alıp diğer konumlara taşıyor; yeni aktif tokenları mevcut noise seviyesine hizalamak için micro-flow kullanıyor. Yalnız orijinal kapağın kenarını tekrarlayan initializer'a göre doğrudan ilgili bir araştırma yolu. Kamu kodu FLUX.1-dev ve Klein Base 9B içeriyor; Qwen desteği planned. Qwen'de burada çalıştırılmış bir port yok.

**Kapsam farkı:** JiT anchorları baştan tüm uzamsal alanı, sınırlar dahil, örnekliyor. Sadece merkezde aktif token bulunan uyarlama uzak sahne için aynı desteği alamaz. Merkez yoğun + dışarıda seyrek anchor ile sonradan yoğunlaşan alan ayrı bir seçenek; dış bölgenin bir kısmını ilk adımdan hesapladığı açıkça belirtilmeli. Full latent state'i ucuz tensor işlemleriyle sürdürmek, tüm tokenları her adımda DiT'ye vermekle aynı maliyet değil.

Resmî `pipeline_flux_JiT.py` bridge'i `1/relax_steps` oranında tek blend yapıyor; makalenin algoritması hedefe doğrudan atıyor. Bir port bu seçimi açıkça kaydetmeli. Aynı fikirleri Qwen'e taşımak conditional flow'un doğru örneklendiğini veya kaliteli outpaint elde edildiğini kendiliğinden kanıtlamaz. [Bridge kodu](https://github.com/Wenhao-Sun77/Just-in-Time/blob/818ed8de7333f0c83e3f88a006576036c0f931e8/flux/pipeline_flux_JiT.py#L431-L457)

### RAS — gerçek bölgesel hesaplama azaltma

[Resmî kod](https://github.com/microsoft/RAS), [makale](https://arxiv.org/abs/2502.10389).

Region-Adaptive Sampling, bazı DiT tokenlarını aktif hesaplar, diğer bölgelerde önceki prediction/KV durumunu yeniden kullanır. Bu, yalnız sonucu maskelemekten farklı olarak modelin yaptığı işi azaltır. Düzenli dense refresh, biriken hatayı düzeltmek için kullanılır. Makalenin SD3 ve Lumina ölçümleri Qwen 2.1 veya Klein için doğrulanmış hız değildir.

RAS tek başına “önce iç halka tamamlanır, sonra dış halka” semantiği sağlamaz: bölgeler aynı global diffusion zamanında kalabilir. Proje açısından spatial noise schedule ile active-token caching'in birbirini tamamlayan iki ayrı bileşen olduğu sonucu çıkar. Yeni bir Qwen/Klein portunda RoPE konumlarının, context K/V'nin ve source conditioning'in doğru kalması; kalite ve gerçek GPU süresinin ölçülmesi gerekir.

### RandAR — farklı bir mimari yolu

[Makale](https://arxiv.org/abs/2412.01827), [resmî uygulama](https://github.com/ziqipang/RandAR).

RandAR, üretilecek tokenın konumunu position instruction token ile bildirerek decoder-only autoregressive modelde keyfî görsel üretim sırasını destekler. KV cache ile önceki tokenlar tekrar üretilmeden yeni tokenlar eklenebilir; makale zero-shot inpainting/outpainting gösterir. Kapak tokenlarını önce yerleştirip yeni tokenları yakın banttan uzak banda sıralamak, tek üretim oturumunda gerçek büyüme için uygun bir mimari çıkarımdır.

Kamuya açık 0.3B/0.7B class-conditional ImageNet checkpoint'leri güçlü bir mekanizma referansıdır. Bunlar Qwen düzeyinde albüm editörü veya telefonda ölçülmüş bir ürün olarak sunulamaz. Ayrıca autoregressive block/token üretimi birden fazla iç model çağrısı içerebilir; fark, her çağrının daha büyük bir görseli baştan yeniden outpaint etmemesidir.

## Birbirine karıştırılmaması gereken yöntemler

| Yöntem | Gerçek üretim sırası değişir mi? | Otomatik hesaplama tasarrufu var mı? |
|---|---|---|
| Bitmiş görseli halka animasyonuyla açmak | Hayır | Hayır |
| Ayrı ayrı büyüyen outpainting çağrıları | Çağrılar arasında | Hayır; toplam iş artabilir |
| Full canvas'ta spatial noise schedule | Evet | Hayır; forward hâlâ dense olabilir |
| Aktif tokenlar + cached context | Kendi başına şart değil | Evet; doğru uygulanırsa daha az hesap yapılır |
| Token/block autoregressive üretim | Evet; yeni tokenlar eklenir | Önceki bağlam KV cache ile yeniden kullanılabilir |
| Bütün sahneyi düşük çözünürlükten yüksek çözünürlüğe götürmek | Detay üretim sırası değişir | Token sayısı erken aşamada azalabilir; bu, mekânsal kapsam büyümesi değildir |

[RALU](https://github.com/ignoww/RALU) örneği, düşük çözünürlük → seçili bölgelerde yüksek çözünürlük → tam yüksek çözünürlük akışında gerçek token azaltır. Bu, bütün sahnenin önce kaba taslağını üretir; kapağın sınırından yeni alan yaratma sorusunun doğrudan cevabı değildir. [Progressive Artwork Outpainting](https://eadcat.github.io/ProOutWeb/) ise ardışık local window üretimleri kullanır; kullanıcının ayrı üretim çağrısı istemediği koşulu karşılamaz.

## İndirdiğimiz mevcut modellerde ne değiştirilebilir?

Bu bölüm, yerel kod incelemesi ve yukarıdaki çalışmalardan teknik çıkarımdır. Bir kalite veya performans iddiası değildir.

- **Qwen 2.1:** İncelediğimiz MFLUX uygulamasında temiz text/reference prefix için `t=0`, bütün target image tokenları için ortak sampled timestep kullanılır. `_select_modulation_rows` image tokenlarına aynı modulation satırını broadcast eder. Bunu ring başına timestep embedding/modulation ve ring başına Euler `Δσ` ile değiştirmek teknik olarak mümkündür. Ağırlıklar yüklenebilir, fakat target image'ın birçok farklı noise level'da bulunduğu yeni durumun doğru tahmin edildiği kanıtlanmış olmaz. Robust sonuç için spatial schedule eğitimi/fine-tuning değerlendirilmelidir. Yerel kanıt: `.build/model-research/mflux/src/mflux/models/qwen21/model/qwen21_transformer/qwen21_transformer.py`, `_step`, `_select_modulation_rows`.
- **Klein:** İncelenen Swift transformer API'si timestep `[B]` alır ve ortak timestep/guidance modulation üretir. Per-image-token modulation, output normalization ve sampler adımları birlikte uyarlanmalıdır. Temiz reference görüntü koşullandırması olması, target halkaların ayrı zamanlarda üretilebildiği anlamına gelmez. Yerel kanıt: `.build/model-research/flux-2-swift-mlx/Sources/Flux2Core/Transformer/Flux2Transformer.swift`.
- **DreamLite:** Mevcut Mobile pipeline UNet'e batch başına global scalar timestep verir. Spatial time FiLM'in katmanlara taşınması, heterojen noise durumlarıyla kalite kontrolü ve gerekirse eğitim değerlendirilmelidir. Küçük öğrenci mimarisini bu amaçla baştan tasarlamak, bütün büyük Qwen runtime'ını telefona taşımaktan ayrı bir seçenektir. Yerel kanıt: `.build/model-research/DreamLite/dreamlite/pipelines/dreamlite/pipeline_dreamlite_mobile.py`.

Sadece dış halkaya sonradan pure noise ekleyip global sampler'ı düşük noise level'dan devam ettirmek güvenilir bir kısayol değildir: yeni alan yüksek noise level'da, modelin timestep conditioning'i düşük seviyede olur. Tersine, hedef canvas'ı baştan ayırıp bütün alanı her adımda hesaplayarak çıktıyı ring mask ile güncellemek gerçek piksel ilerlemesi yaratabilir, fakat dense hesap maliyeti devam eder ve aynı conditioning uyuşmazlığı ortaya çıkabilir.

[Differential Diffusion](https://differential-diffusion.github.io/), inference sırasında bölgeye göre edit strength ve iç içe maskeler kullanma örneğidir. Training gerektirmeyen bir sampler deneyine ilham verir; spatial-time-trained model ve aktif token hesabının yerini otomatik olarak tutmaz. Özellikle yeni içeriği yoktan üretmek ile mevcut görselin farklı bölgelerini farklı güçte değiştirmek aynı başlangıç koşulu değildir.

## İlk deney tasarımı ve sonraki araştırma

Bu ilk tasarımın full/active-canvas ve erken aktivasyon kontrolleri artık aşağıdaki gerçek deneylerde kayıtlı. Per-token spatial timestep ve dense-refresh/KV tasarımı ise burada uygulanmış sayılmaz.

İlk deney mevcut checkpoint üzerinde **tek inference oturumu ve sabit toplam model evaluation sayısı** ile yapılmalı. Önce standart full-canvas baseline kaydedilir. Sonra aynı kaynak, prompt, seed, hedef boyut, NFE ile bir merkezden dışarı spatial-update ablation çalıştırılır. Bu kontrolde ara latentler, ring noise/time map'leri, gerçek model call sayısı, token sayısı, peak bellek, wall time ve final kalite kaydedilir. Eski UI reveal animasyonuyla karıştırılmaması için ara görüntüler doğrudan sampling state'inden decode edilir.

Naif maskeli kontrolün amacı, “maskeyi ilerletmek yetiyor mu?” sorusunu ölçmektir. Şimdiden hız kazancı veya doğru conditional distribution iddiası yapılmamalı. Qwen üzerinde per-token conditioning deneyi ayrı bir varyant olmalıdır; bu da eğitim olmadan kalite garantisi taşımaz. 4-step distilled checkpoint'te bantların yeterli denoise zamanı bulamaması ayrıca sınanmalı; daha fazla ara frame üretmek modelin gerçek üretim adımlarını artırmadan daha fazla hesap bilgisi yaratmaz.

Gerçek maliyet hedefi için ikinci prototip, tamamlanmış bölgeleri bağlamda tutarken **yalnız aktif bant tokenlarının attention query/MLP hesabını** çalıştırır; context K/V ve prediction cache'i tutar, gerektiğinde dense refresh yapar. Böylece profil gerçekten azalan token hesaplamasını gösterirse hız iddiası kurulabilir. RAS'tan alınan fikrin Qwen/Klein'e uygulanması yeni bir porttur; kalite doğrulaması olmadan resmî destek gibi sunulmamalı.

Uzun vadeli küçük öğrenci modelin eğitiminde source mask + ring/time map conditioning'i baştan olmalı. Eğitim örnekleri temiz merkez, bitmiş yakın bölge ve henüz noisy uzak bölge içermeli; farklı aspect ratio ve band sıraları görülmeli. Öğrenciye önce yalnız final teacher piksellerini öğretmek, kendi başına bu sampling kabiliyetini kazandırmaz. Spatial schedule hedefi distillation planının bir parçası olmalıdır.

## Gerçek tek-döngü aktif-canvas denemesi

[Standalone script](../experiments/qwen/run_spatial_trial.py), [tam deney](../experiments/qwen/SPATIAL_RESULTS.md), [pair metriği](../experiments/qwen/spatial_runs/q4/track3/paired_metrics.json).

Qwen 2.1 Q4 + Viggle r256 ile tek model yüklemesi/conditioning ardından iki eşleştirilmiş 6-NFE döngüsü çalıştı. FULL tam alanı, GROW kaynak çevresindeki aktif pencereyi **512×640 → 512×896 → 512×1152** olarak işledi. Sırasıyla 1280→1792→2304 target token forward'a girdi; henüz açılmayan dış tokenlar gerçekten yoktu. Her loop kendi sabit-prefix cache'ini bir kez oluşturdu; mutlak hedef koordinatları ve önceki latent durumları korundu. Installed runtime veya ağırlıklar değiştirilmedi.

FULL **13.824**, GROW **10.752** target-token-forward; %22,2 daha az. Core denoising **29,38 s → 24,33 s**; tek pair'de %17,2 daha kısa. Bu genel model/telefon benchmark'ı değil; order/warmup etkisi olabilir. Kaynak kare iki sonuçta da aynen korundu, floating çıktılar finite. Denoise MLX peak FULL **9,00 GiB**, GROW **8,46 GiB**; ortak conditioning **10,84 GiB**, ortak on-load quantization peak **24,50 GiB**. Byte/GiB ile standard rapordaki decimal GB birbirine karıştırılmamalı.

**Kalite kapısı geçilmedi.** Sonradan açılan dış bantlarda örgü/ızgara izleri, ek üst korkuluk ve küçük hayalî semboller çıktı. Geç aktivasyon source latent edge extrapolation + mevcut sigma'da ortak noise ile başlatıldı; eğitilmemiş initializer. Bu durum, gerçek compute büyümesinin teknik olarak çalışması ile kaliteli bölgesel generation'ın ayrı sorunlar olduğunu gösteriyor.

İç alanlar tek global sigma ile son adıma kadar refine edildi; bu deney yakın halkaları erken bitirip kalıcı dondurmadı. 2/4/6. adımda gerçek predicted-clean latent kaydı var; adil süre kıyaslaması için preview'lar döngüler bitince decode edildi. Dolayısıyla resimler sampling state kanıtı, **canlı kullanıcı arayüzü streaming testi değil**. Daha hızlı core süre, preview decode/streaming overhead dahil toplam ürün gecikmesi gibi sunulmamalı.

[Gerçek ara aşamalar](../experiments/qwen/spatial_runs/q4/track3/growth-stages.png), [FULL/GROW karşılaştırması](../experiments/qwen/spatial_runs/q4/track3/full-vs-grow.png).

## 5 Ekim incelemesi: ikinci adımdaki bozulma neyi gösteriyor?

Bu bölüm kayıtlı görüntüler, metrikler ve sampler kodunun yeniden incelemesidir; yeni inference veya yeni hız ölçümü değildir.

**Erken tahmin ile final kaliteyi ayırmak gerekiyor.** FULL ve GROW'un ikinci adım PNG'lerinde su/asfalt yumuşak, düzenli doku ve kaynak sınırında pütürler var. FULL finalinde bunlar büyük ölçüde toparlanıyor. GROW'un ilk açılan 64 piksellik üst/alt bantları da sonradan iyileşiyor. Son GROW'daki ağır ızgara ve hayalî işaretler daha dışarıdaki, özellikle en son eklenen bantlarda yoğunlaşıyor. Aynı checkpoint/Q4/adapter/VAE'nin FULL finalini düzgün üretebilmesi, sorunu yalnız modelin genel kalitesine bağlamayı desteklemiyor.

İkinci adım görüntüsü bitmiş bir 512×640 outpaint değil. Script satır 324'te, ikinci forward'ın **giriş** durumundan `z_sigma - sigma*v` temiz sonuç tahmini alınıyor; sigma yaklaşık **0,9648**. İkinci Euler güncellemesi sonrası sigma hâlâ **0,9275**. Sigma bir noise-mix katsayısıdır; yüzde tamamlanma ölçüsü değildir. İkinci adımda yeni token eklenmiyor; ilk başlangıçta sigma=1 olduğu için edge guess katsayısı sıfır. Dolayısıyla ikinci adım kusurlarını geç token initializer'ına yüklemek yanlış olur.

**Kalıcı dış-bant kusuru için güçlü hipotezler:**

- Son bant sadece beşinci ve altıncı forward'a katılıyor. Yeni alan, yaklaşık sigma=0,6464'te modele girerken yakın alan birkaç üretim adımı geçirmiş oluyor. Checkpoint tüm aktif hedef için tek sigma kullanıyor.
- Yeni bölgenin temiz latent bileşeni bilinmiyor; kod satır 207–209'da kapağın ilk/son latent satırını dışarı doğru tekrarlıyor. Bu, sahnenin gerçek devamı değil. Satır 273–274'te bu tahmin + sigma ile ölçekli ortak noise karıştırılıyor; doğru conditional flow marginal olduğu gösterilmiş değil.
- Yeni tokenlar mevcut tokenların attention bağlamını da değiştiriyor. İlk sahne kararları gelecekteki bağlam olmadan veriliyor. Aynı konum/noise korunması bu dağılım farkını tek başına çözmüyor.

Görsel inceleme bu üç etkinin payını ayıramaz. Izgara görünmesi tek başına VAE hatası veya quantization hatası kanıtı değildir; FULL erken tahmininde de görülen kusurların finalde kaybolması özellikle önemlidir.

**En ucuz anlamlı sonraki kontroller:** aynı kaynak/prompt/seed/adım sayısını koruyarak bütün dış alanları yüksek-sigma bölümünde daha erken aktive etmek; ardından daha iyi, koşullu latent başlangıcını tek başına değiştirmek; son olarak uygun sampler ile ek düzeltme adımı kullanmak. Sonuçlar final dış-bant kalitesi ve toplam maliyetle karşılaştırılmalı. Erken aktivasyon hesap tasarrufunu azaltabilir. Bunların hiçbiri ayrı ayrı bitmiş outpaint çağrılarını zincirlemeyi gerektirmez.

İlk kontrolün yükseklik planı `[640,896,1152,1152,1152,1152]`. Böylece en dış bant sigma≈0,9275'te girer ve iki yerine dört forward görür. Hedef token toplamı 12.288; FULL'e göre token farkı %11,1. Bu plan aşağıdaki 5 Ekim denemesinde çalıştırıldı. Aynı global schedule'ı korumak tek değişken olarak aktivasyon zamanını sınadı.

Bu inceleme sırasında önerilen Viggle v0.3 kontrolü daha sonra **5 Ekim'de çalıştırıldı**: aynı erken büyüme planında v0.2.1 altı, v0.3 altı ve resmî yedi Turbo + iki Base adımlı dokuz-step mod karşılaştırıldı. Gerçek adapter scale değişimi ve Base'e geçişte prefix-cache yenilemesi doğrulandı. NFE **6/6/9**, target-token-forward **12.288/12.288/19.200**; korkuluk dirseği, üst su bandı ve alt asfalt geçişi kaldı. Yeni adapter/tail çözüm olmadı. Kartın düşük-noise düğümlerini koruma kuralı, keyfî 12 uniform adımı hâlâ uygun bir kontrol yapmıyor. [Gerçek sonuçlar](../experiments/qwen/TURBO_TAIL_RESULTS.md), [yayıncının kullanım kuralları](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo#rules-that-matter).

Bu kontroller yetmezse spatial timestep conditioning ile eğitim/fine-tuning değerlendirilir. Şimdiki kanıt, bütün modeli değiştirmek veya büyük distillation çalışmasını hemen başlatmak için yeterli değildir. Öğrenci önce kaliteli sampler davranışını öğrenmeli; kusurlu büyüme akışını distill etmek aynı hataları aktarabilir.

## 5 Ekim: final kalite için dört kontrollü deneme

[Runner](../experiments/qwen/run_spatial_ablation.py), [sonuç raporu](../experiments/qwen/QUALITY_ABLATION_RESULTS.md), [gerçek final görüntüler](../experiments/qwen/quality_runs/2026-10-05/track3/final-comparison.png).

Tek model yüklemesi; aynı track3/seed42/source-prefix/noise/prompt/6-sigma schedule. FULL ve OLD sonuçları 30 Eylül çıktılarıyla piksel/SHA256 eşleşti. Her arm altı transformer çağrısı yaptı; her kaynak composite birebir ve floating sonuçlar finite.

| Arm | Aktif yükseklik | Hedef token toplamı | Gerçek 6. adım gözlemi |
| --- | --- | ---: | --- |
| FULL | 1152×6 | 13.824 | Tutarlı su, korkuluk ve asfalt; kaynak sınırında doku geçişi var |
| OLD | 640×2, 896×2, 1152×2 | 10.752 | Ağır üst/alt ızgara, tekrarlanan korkuluk ve sahte alt işaretler |
| EARLY-SOURCE | 640, 896, 1152×4 | 12.288 | Ağır ızgara/işaretler kayboldu; üstte yeni kıyı/bitki/korkuluk kompozisyonu |
| EARLY-FRONTIER | Aynı erken plan | 12.288 | Ağır ızgara/işaretler kayboldu; daha temiz doku, üstte yeni ufuk/gökyüzü |

EARLY-FRONTIER yeni tokenları önceki forward'ın `z_sigma - sigma*v` tahmininin aktif üst/alt sınır satırlarını tekrarlayarak başlatır. Learned continuation modeli değil; source-edge gibi bir heuristic. Hiçbir ek denoiser/VAE çağrısı yapılmadı. Henüz açılmamış target tokenlar gerçekten forward dışında kaldı; eski durum ve mutlak konumlar korundu.

**Desteklenen sonuç:** erken aktivasyon bu kapakta final dış-bant kusurunu belirgin azalttı. Bu, mevcut ağırlıklarla sampler değişikliğinin fayda sağlayabildiğini gösterir. Aktif sınırın genel olarak daha iyi başlatıcı olduğu veya bütün kapakların çözüldüğü sonucu tek örnekten çıkarılamaz. Yeni ufuk/kıyı gibi kompozisyon değişimleri ve kaynak doku geçişleri hâlâ değerlendirilmelidir. ArtWorker player'a model bağlanmadı; fiziksel iPhone/UI streaming testi yok.

Bugünkü core süreleri FULL/OLD/EARLY-SOURCE/FRONTIER sırasıyla **55,44 / 54,83 / 55,05 / 46,22 s**. Yüksek sistem bellek baskısı ve sıra/warmup etkisi nedeniyle hız sıralaması kurulmaz. Yeni runner her adım temiz tahmin hesaplıyor; eski runner ile mutlak süre kıyaslaması yapılmaz. Kaynak kapağın aynen kalması yalnız koruma kontrolüdür, dış alanın kalite skoru değildir. [Gerçek sampling ara durumları](../experiments/qwen/quality_runs/2026-10-05/track3/growth-stages-early-frontier.png)

**İkinci kapak kontrolü:** track2'de aynı matched FULL/EARLY-FRONTIER altı-adımlı pair çalıştırıldı. Erken/frontier yöntemi bu seed'de üstteki kopya KARAMBOL başlığını kaldırdı. Ancak **iki yöntemde de alt gövde/kıyafet devamı kısa bir uzatmadan sonra kesilip cyan panel oluyor**. Bu sorun tam alan baseline'ında da bulunduğu için yalnız geç token aktivasyonuyla açıklanamaz. Kaynak konumu/edit conditioning, task/prompt sadakati ve six-step model kapasitesi ayrıca ayrıştırılmalı. Bu kapakta robust outpaint kalite kapısı geçilmedi. [İkinci kapağın gerçek finalleri](../experiments/qwen/quality_runs/2026-10-05/track2/final-comparison.png)

Toplam altı inference arm / 36 model çağrısı tamamlandı; bütün çıktılar finite ve ortak 512 kaynak pikselleri birebir. Orijinal JPEG hash'leri korundu. Bütün GPU süreçleri çıktı; model/player entegrasyonu veya eğitim yapılmadı. [Doğrulama](../experiments/qwen/quality_runs/2026-10-05/validation.json)

## 5 Ekim: kaynak sınırındaki korkuluk kayması

[İki kollu geometri kontrolü](../experiments/qwen/GEOMETRY_RESULTS.md), [gerçek 6× birleşim](../experiments/qwen/geometry_runs/2026-10-05/track3/rail-join-6x.png).

Korkuluk kırılması ham VAE decode'da da var; FULL'de de görüldü. Kaynak yerleşimi (`y=320:832`), mutlak token konumları ve RoPE incelemesinde kayma bulunmadı. Decoder tile birleşimleri bu sınıra denk gelmiyor. Kaynağı tek başına kare olarak encode etmek, latent kenarlarının geniş tuvaldeki komşulukla uyuşmamasına katkı yapabilir. Bu hipotez aynı altı-adımlı EARLY-FRONTIER akışında sınandı.

Kontrol önceki finali piksel/SHA256 birebir yeniden üretti. İkinci kolda yalnız target'ta sabitlenen kaynak latentleri, 512×1152 edge-padded tuvalin merkezinden encode edildi; text/image reference prefix aynı orijinal kare olarak kaldı. Prompt, seed/noise, büyüme planı ve toplam 6 NFE / 12.288 hedef-token hesabı aynı. İki finalde kaynak 512 kare pikselleri aynen korundu; warp/feather/çizgi boyama yapılmadı.

**Kısmi sonuç:** ham üst 16 px kaynak MAE **14,33→8,72**, alt 16 px **10,05→4,98**. Gerçek raw/composite closeup'ta büyük zikzak azaldı; üretilen korkuluk ile kaynak arasında küçük açı/konum farkı ve su/bitki birleşimi kaldı. Ayrıca alt kaynak sınırında koyu yatay asfalt geçişi belirginleşti. MAE bir geometri skoru değildir; bu kol tüm görüntü için kalite kapısını geçmedi. Kalan konum/eğim sürekliliği ve alt doku geçişi birlikte değerlendirilmeden varsayılan yöntem yapılmamalı. [Doğrulama](../experiments/qwen/geometry_runs/2026-10-05/track3/validation.json)

## 5 Ekim: büyüme ile kalite düzeltmesinin maliyetini ayırmak

Tamamlanan soft/dinamik kaynak sampler kontrolleri gerçek erken aktif-token büyümesini korudu: altı NFE, **12.288 target-token-forward**. Dinamik bağlam iki ek VAE decode/encode kullandı; daha düşük sınır reconstruction hatası korkuluk geometri kabulü sağlamadı. LanPaint-style büyüyen kontrolün ilave düzeltmeleri gerçek NFE'yi **6 → 9 → 12**, hedef-token işini **12.288 → 19.200 → 26.112** artırdı; daha fazla hesap da birleşimi çözmedi. [Sampler raporu](../experiments/qwen/BOUNDARY_SAMPLER_RESULTS.md), [LanPaint-style raporu](../experiments/qwen/LANPAINT_RESULTS.md).

Compositor ve frozen-VAE projection **üretim sonrası** onarım deneyleri; yeni dış bant üretmezler. Renk/MAE iyileşmeleri geometry başarısı değil. VAE consistency tamamlandı, derivative kolu final çıktı olmadan timeout oldu; bu incomplete kol kazanım olarak sayılmaz. [Compositor](../experiments/qwen/COMPOSITOR_RESULTS.md), [projection](../experiments/qwen/PROJECTION_RESULTS.md).

Eğitimli Qwen ControlNet'in tam, sparse, reference ve localized40 kolları ile native FLUX.1 Fill50 **bütün target alanını her adımda hesapladı**. Localized40 tek sahneye geçti ancak dış korkuluk kenarı kaynak tanjantına göre **−17,404 px yatay kaydı** ve ekstra bant kaldı. Her 40-step reference/localized kolunda 92.160 target + 40.960 reference image token forward var; yalnız hint alanını küçültmek hesap tasarrufu değil. Fill50'de 115.200 target-token-forward var, yeni yazı ve ikinci sahne kaliteyi bozdu. [ControlNet](../experiments/qwen/CONTROLNET_RESULTS.md), [localized40 gerçek inceleme](../experiments/qwen/boundary_eval/2026-10-05/controlnet-localized40-comparison/REVIEW.md), [Fill50](../experiments/flux1_fill/FILL_RESULTS.md).

**Saved-top32 kontrolünde asıl metal dış kenarının birleşimi belirgin iyileşti:** ortak gerçek tensorlar aynen geri yüklendi, yalnız kaynak üst 32 px şeridindeki 64 hint hücresine weight1 verildi. Dış feature hatası **−17,404 → −2,086 px**, yakın açı farkı **+0,341°** oldu. Son satırda 3,930 px zayıf-edge jog, su/asfalt dikişleri kaldı; bütün kalite kabulü yok. Bu hâlâ tam tuval40/92.160 target-token-forward sonucudur. Aynı collar'ın gerçek growing-top32 uyarlaması yalnız CPU hazır; henüz GPU sonucu yok. [Gerçek paired dış-edge incelemesi](../experiments/qwen/boundary_eval/2026-10-05/controlnet-saved-top32-comparison/REVIEW.md).

Native Fill local-context yöntemi **iki bağımsız 512×448 çağrı** yaptı: toplam 100 NFE, 89.600 target-token-forward ve 320,980 s wrapper. Üst göl daha makul olsa da alt soluk bantlar/doku kusuru kaldı. Bu daha küçük pencere üretimi, kullanıcının istediği tek sampling akışındaki büyüme değil. Native SDXL inpaint20/CFG8/strength0,99 da 19 U-Net çağrısı/38 CFG branch sample ile tamamlandı, dış alan siyaha çöktü; maske/noise audit'i bu hatayı açıklayan bir input tersliği bulmadı. [Local Fill kayıtları](../experiments/flux1_fill/runs/2026-10-05/track3-local-context50-seed42/metrics.json), [SDXL deneyi](../experiments/sdxl_inpaint/SDXL_RESULTS.md).

Gerçek **altı-adımlı growing localized-control** suite tamamlandı: control scale **0 / 0,5 / 1**, aktif target sayıları **1280 → 1792 → 2304×4**. Gelecek target tokenları hem base hem control girdisinde yoktu; mutlak konumlar korundu, 2/4/6 preview'ları gerçek sampler state'inden alındı. Her kol **12.288 target**, **6.144 reference image token forward** ve her chain için **19.470 joint query** yaptı. Tam reference prefix her çağrıda yeniden hesaplandı; scale0 bile control branch'i hesapladığından bare-base hız benchmark'ı değil. Üç-kol wrapper **211,185 s** sürdü. [Runner](../experiments/qwen/controlnet_port/run_growing_localized_control_trial.py), [gerçek görüntü ve compute incelemesi](../experiments/qwen/boundary_eval/2026-10-05/controlnet-growing-localized6-comparison/REVIEW.md).

Üç final de genel sahneyi tutarlı devam ettirdi; korkuluk profilindeki basamak ve ek metal kenar kaldı. Sabit source-tangent diagnostic üçünde de `invalid-feature` verdi; null ölçümler sıfır hata değil, görüntü incelemesi ayrıca başarısız birleşimi doğruladı. Eski cached kontrol ve uncached scale0 kaynak/noise/sigma/codec provenance'ını paylaşır ancak bit-identical değildir. Finalized incelemedeki aynı-stage clean-latent karşılaştırması max abs **0,0625**, mean abs **0,002002**; fark yalnız cache'e bağlanamaz. Gerçek compute büyümesi gösterildi, kaliteli streaming kabul edilmedi. Turbo + eğitimli ControlNet + büyüme birlikte eğitilmiş bir sistem değil. Portrait canvas-reference yalnız CPU hazırlık/doğrulama düzeyinde.

MaskFlow'un doğru **Qwen-Image-Edit-2511** codec/transformer'ına izole MLX-Gen portu ve ağırlık indirmesi başlatıldı; mevcut Qwen2.1 loop'una adapter takmak aynı yöntem olmaz. ProMax author outpainting tarifi için ağırlık indirmesi de başladı. İkisinin GPU sonucu yok, hazır spatial stream veya kabul edilmiş teacher değiller. [MaskFlow exact-recipe hazırlığı](../experiments/maskflow/research/2026-10-05/FEASIBILITY.md), [ProMax hazırlığı](../experiments/sdxl_inpaint/PROMAX_FEASIBILITY.md).

Bu deneylerde öğrenci eğitimi, fiziksel iPhone inference veya canlı SwiftUI generation streaming yapılmadı. Tek-sahne öğretmen kalitesi ile gerçek büyüyen hesap birlikte kabul edilmeden ürün yöntemi seçilmeyecek.

## iOS sonucu

Sabit canvas üzerindeki time map, sabit shape Core ML export'uyla kavramsal olarak uyumludur; ancak dense forward otomatik küçülmez. İncelediğimiz DreamLite export'ları mevcut sabit tensor şekillerine bağlıdır. Değişken active-token sayısı veya büyüyen tensor için yeniden export ve runtime uyarlaması gerekir. Core ML genel olarak flexible shape destekler; bunun her modelde aynı ANE yerleşimi/hız anlamına geldiği söylenemez. Önceden belirlenmiş `EnumeratedShapes`, sınırsız dinamik şekle göre cihaz optimizasyonu için daha elverişli bir seçenektir; dinamik reshape gibi desteklenmeyen katmanlara dönüşüm ANE kullanımını değiştirebilir. [Apple flexible input shapes](https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html), [Apple ANE ve flexible shape açıklaması](https://apple.github.io/coremltools/docs-guides/source/faqs.html)

MLX'in dinamik dizileri/Metal yolu aktif token prototipini daha esnek kılabilir; token gather/scatter, cache ve dikkat kernel maliyetleri ölçülmeden telefonda hızlı kabul edilmemeli. Üretim ilerlemesini göstermek için VAE'yi her adımda çalıştırmak da ek maliyet yaratır. Birkaç tamamlanma eşiğinde küçük preview decode etmek ayrı bir ürün tercihi olur. Burada hiçbir iPhone süresi veya bütün cihazlarda eşit performans sözü verilmemiştir.

Pratik yön: mevcut modellerde araştırma prototipi ile gerçek bölgesel üretim sırasını sınamak; kaliteyi koruyan küçük öğrenciye spatial time conditioning öğretmek; hız için ayrıca aktif token/context cache tasarlamak. Kullanıcının istediği mekanizma mümkündür, fakat mevcut checkpoint'in tek parametresini değiştirerek doğrulanmış ürün davranışına dönüşmez.
