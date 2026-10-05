# Yerel FLUX.2 Klein outpainting deneyi

30 Eylül 2026; Apple M4 Max, 36 GiB unified memory. Player kaynaklarına müdahale etmez.

## Runtime

`VincentGourbin/flux-2-swift-mlx` commit `315c27e83187909bc5e1f880b4850918342d891b`, Xcode 27A266a, Release arm64. Kaynak build başarılı. Bu sürüm CLI içinde `2.4.0` yazıyor; indirilmiş GitHub release etiketi `v3.1.0` ile aynı şey olarak kaydedilmedi.

Hazır binary ilk gerçek çağrıda eksik `default.metallib` nedeniyle çalışmadı. Bu model kalite sonucu değildir. Xcode'un eksik Metal Toolchain bileşeni indirilip runtime kaynak kodundan derlendi. Binary, yanındaki `mlx-swift_Cmlx.bundle` ile birlikte kullanılır; binary'yi tek başına kopyalamak resource sorununu tekrar yaratabilir. Build logları `logs/` içinde.

```sh
xcodebuild -scheme Flux2CLI -configuration Release \
  -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath /Users/emir/Documents/ChatGPT/ArtWorker/.build/flux2-native \
  -skipPackagePluginValidation -jobs 2 build
```

Bu komut `.build/model-research/flux-2-swift-mlx` içinden çalıştırılır. Package plugin kaynakları incelendi; macOS'ta CUDA build yolu devre dışı. Metal derlemesi Xcode gerektiriyor.

## Ağırlıklar ve yöntem

[model_manifest.json](model_manifest.json) indirilmiş revision, Hub etag ve gerçek dosya boyutlarını kaydeder. Hub etag, bağımsız yerel SHA doğrulaması diye sunulmaz. Distilled ve Base transformer ayrı; Qwen3-4B encoder önceden MLX dört bit; VAE `FLUX.2-small-decoder`.

Runtime'ın güncel `ModelRegistry` klasörleri eski `ModelComponent.localDirectoryName` adlarından farklı. Distilled ve VAE dosyaları canonical klasörlere hardlink ile yerleştirildi; iki kopya ağırlık depolanmadı. İlk yanlış cache yolu denemesi kendi indirme süreci durdurularak kaydedildi.

Ortak çıktı 512×1152. 720 kare source, ortak hazırlıkta 512'ye yeniden boyutlandırılır ve `(0, 320)` konumuna konur. Dış alan deterministik nötr noise; kaynak kenarının içinde 32 px yumuşak maske rampası; source ayrıca reference olarak verilir. Runtime latent blending uygular. Bu, LanPaint'in tam conditional sampler'ı değildir. Final composite, `source_512.png` karesini aynen geri koyar; ham model çıktısı ayrıca saklanır.

```sh
.build/dreamlite-venv/bin/python experiments/flux2/run_trials.py \
  --tracks track1 track2 track3 --model klein-4b \
  --quant int4 --steps 4 --guidance 1 --seed 42 --force
```

Deney script'i her case için command, exit code, log, inference süreleri, toplam process süresi, initialization, MLX peak ve process RSS kaydeder. `--repeat-count 2` aynı process içinde warm karşılaştırma verir; son PNG son tekrarın sonucudur. Mac ölçümleri iPhone benchmark'ı değildir. MLX allocation peak ile RSS toplanmaz.

Gerçek çıktı kalitesi ve son karar [ortak raporda](../../docs/OUTPAINT_TRIAL_RESULTS.md) tutulur.
