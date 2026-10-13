# emergence

**Sıfırdan yazılmış bir transformer, ters-mod (reverse-mode) otomatik türev motoru ve eğitim döngüsü — hiç runtime bağımlılığı yok — tam tarayıcı sekmenizde canlı eğitiliyor.**

[**▶ eğitilirken izle**](https://emirmizrakli.github.io/emergence/) · [komut satırı sürümünü çalıştır](#çalıştır) · [kod haritasını oku](#i̇çinde-tam-olarak-ne-var) · **Türkçe** · [English](README.md)

Sayfayı açtığınız an harf seviyesinde bir dil modeli gözünüzün önünde İngilizce öğrenmeye başlar. Kayıt yok, önceden çizilmiş grafik yok: kayıp eğrisi, attention başları, gömme uzayı, gradyan normları ve modelin yazdığı metin — hepsi ağırlıklar değişirken yeniden hesaplanır.

```
$ node src/cli/train.ts --model gpt --steps 40 --batch 32 --block 48 --embd 64 --layers 4 --heads 4
step    40 | loss 2.9135 | val 2.8843 | lr 3.05e-4 | 1,133 tok/s
  sample @40: COOO ⏎ G ⏎ Xsod euon pe iesice, ands tae y g toy hy knea' here m weuinor f mhliteo'iey
              thvyr s t ths tine  ires   st tthe he ⏎ Tt oke aae ed thh ru
```

Kırk adım — ve İngilizce kelimelerin şekli şimdiden ortada. Üstelik ağırlıkları rastgele gürültü olarak başlamış bir modelde.

---

## Bu proje neden var

Portföydeki her ML aracı, `pip install` kadar uzaklıkta bir sarmalayıcıya (wrapper) dönüşmeye adaydır. `nn.Transformer` dört satırdır ve o dört satırı yazan kişinin attention'ı, geri yayılımı ya da bir eğitimin neden ıraksadığını gerçekten anlayıp anlamadığı hakkında hiçbir şey kanıtlamaz.

Bu yüzden proje, cevabı gizleyecek her soyutlamayı bilinçli olarak kaldırıyor. PyTorch yok, TensorFlow yok, numpy yok, hiçbir ML kütüphanesi yok — grafik kütüphanesi bile yok. Geriye kalanlar:

- `NN`, `NT`, `TN` — önemli olan üç matmul düzenini içeren bir `Float32Array` matris kütüphanesi,
- ilkel işlemlerinden kurduğunuz her şeyi türevleyebilen, kayıt-tabanlı (tape) ters-mod otomatik türev motoru,
- ileri (forward) **ve** geri (backward) geçişi elle yazılmış, decoder-only bir transformer,
- warmup, cosine decay ve gradyan kırpma içeren Adam,
- bir eğitici (trainer), bir checkpoint formatı, bir CLI ve bir worker-thread arayüzü.

İki motor da bilerek var. Hızlı yol elle yazılmış kernel'lar: jenerik bir tape üzerinden tensör akrobasisi 10 kat yavaş. Ama elle yazılmış gradyanlar sessizce yanlış olabilir — bu yüzden bağımsız bir autodiff implementasyonu aynı zamanda bir kâhin (oracle) görevi görüyor ve ikisi de sayısal türevle karşılaştırılıyor (`src/core/gradCheck.ts`). Elle yazılmış bir backward geçişi sonlu farklarla uyuşmazsa test paketi kırılır. Güvenlik argümanı bundan ibaret ve bu argüman README'de iddia edilmiyor, testlerle zorlanıyor.

## İçinde tam olarak ne var

| alan | dosyalar | ne yapıyor |
| --- | --- | --- |
| `src/core/` | `mat.ts`, `autograd.ts`, `rng.ts`, `gradCheck.ts` | matris kernel'ları, tape autodiff, tohumlu RNG (mulberry32 + Box–Muller), merkezi farklarla gradyan kontrolü |
| `src/model/` | `gpt.ts`, `mlp.ts`, `bigram.ts`, `factory.ts`, `param.ts`, `sample.ts`, `flops.ts`, `graph.ts` | üç kademeli model merdiveni, düz parametre/gradyan görünümü, örnekleme, FLOP hesabı, ileri geçiş grafiği tarifi |
| `src/train/` | `trainer.ts`, `optimizer.ts`, `pca.ts`, `viz.ts`, `checkpoint.ts` | eğitim döngüsü, Adam, power-iteration PCA, iç durum anlık görüntüleri, JSON checkpoint'ler |
| `src/ui/` | `worker.ts`, `main.ts`, `charts.ts`, `style.css` | modeli elinde tutan worker, sayfa düzeni ve bağımlılıksız canvas çizici |
| `src/cli/` | `train.ts` | arayüzsüz eğitim; `--compare` ile üç mimariyi aynı bütçeyle çalıştırır |

**4.215 satır** kaynak, **693 satır** test, **0** runtime bağımlılığı. Node 23+ TypeScript'i doğrudan çalıştırır — CLI'ın derleme adımı yok, çünkü Node tipleri kendisi sıyırıyor.

## Ölçülmüş, söz verilmemiş

`node src/cli/train.ts --compare --steps 600` üç kademeyi aynı veri setinde, aynı bütçeyle eğitir. Tek bir dizüstü CPU çekirdeğinden alınmış gerçek çıktı:

| model | parametre | ilk kayıp | son kayıp | doğrulama kaybı | doğrulama perplexity | dakika |
| --- | --- | --- | --- | --- | --- | --- |
| bigram | 4.225 | 4.179 | 3.183 | 3.205 | 24.7 | 0.0 |
| mlp | 151.936 | 4.174 | 2.410 | 2.516 | 12.4 | 6.2 |
| gpt | 106.304 | 4.170 | **2.287** | **2.393** | **11.0** | 4.9 |

Transformer, MLP'den %30 **daha küçük** olduğu halde kazanıyor: 151.936 parametreye karşı 106.304, perplexity 12.4'e karşı 11.0. Sabit pencere, içindeki her karakteri eşit derecede önemli sayar; nedensel maske (causal mask) saymaz. Kurulumun tamamı, her kolonun dürüst okunuşu ve ölçülüp bir kenara bırakılmış bir ıraksama vakası: [`docs/benchmarks.md`](docs/benchmarks.md).

### Demoyla birlikte gelen checkpoint

`gpt` satırını büyütürseniz sayfanın başladığı modeli elde edersiniz — aynı mimari, 2 yerine 4 katman, 207.296 parametre, 2.500 adım:

| | |
| --- | --- |
| parametre | 207.296 (nEmbd 64 · 4 katman · 4 head · blok 48) |
| eğitim kaybı | 2.225 (perplexity 9.3) |
| doğrulama kaybı | **2.309** (perplexity **10.1**) |
| sayfada ne görünüyor | 11. adımda kayıp ≈ 2.30 — rastgele ağırlıkların ≈4.17'si değil |

`npm run train:shakespeare` bunu yeniden üretir, `npm run sync` dosyayı `public/` altına kopyalar, sayfa da ağırlıklardan devam etmeden önce şeklin ve sözlüğün uyduğunu kontrol eder. `--lr 1e-3` süs değil: bu konfigürasyon 3e-3 ile ıraksıyor (benchmark notlarına bakın).

## Okunmaya değer üç şey

### 1. İki motor ve onları anlaşmaya zorlayan bir test

`src/core/autograd.ts` bir tape kurar: her işlem ebeveynlerini ve yerel bir backward closure'ını kaydeder, `backward()` bunu ters yönde gezer. `src/model/gpt.ts` aynı matematiği elle yapar ve iç döngülerde hiç bellek ayırmaz.

Eğitimi yapan elle yazılmış sürüm. Tape ise kâhin. `src/model/gpt.test.ts` gerçek bir batch alır, eğiticinin kullandığı ileri geçişin aynısını çalıştırır, sonra **her** parametrenin gradyanını sonlu farklarla karşılaştırır — artı tape tabanlı motora karşı tam bir karşılaştırma. Çıkış projeksiyonunda ince bir transpoz hatası (bu projenin gerçekten yaptığı hata) bu testten sağ çıkamaz.

### 2. Bir model merdiveni, çünkü asıl mesele başarısızlık

Aynı eğitici ve aynı veri seti üç mimariyi çalıştırır:

| | neyi ifade edebiliyor | nasıl başarısız oluyor |
| --- | --- | --- |
| **bigram** | tek karakterlik bağlam | ~2.4 kaybında plato yapar: harf sıklıklarını öğrenir, o kadar |
| **sabit pencereli MLP** | son 8 karakter, toplamayla sırası bozulmuş | bigram'ı geçer, sonra tıkanır — bir karakterin *nerede* olduğunu bilmez |
| **transformer** | her pozisyon kendinden önceki her pozisyona bakar | düşmeye devam eder ve attention paneli neye bakmayı öğrendiğini gösterir |

Arayüzde aralarında geçiş yapmak — aynı veri seti, aynı adım bütçesi — attention'ın size tam olarak ne kazandırdığının en net göstergesidir.

### 3. Eğitim buffer'larını yeniden kullanan üretim

Örnekleme KV cache olmadan çalışır: her yeni karakter tüm ileri geçişi baştan çalıştırır, böylece çıkarım yolu eğitim yoluyla byte byte aynı kalır ve kayabilecek ikinci bir implementasyon doğmaz. Bunu ucuz tutmak için üretim penceresi **her zaman** tam olarak `blockSize` uzunluğundadır — solda canlı token'lar, sağda dolgu. Nedensellik dolguyu bedava kılar (`n-1`. satır kendinden sonraki token'ları göremez) ve tek bir sabit şekil tutmak, üretimin her karakter için yeni aktivasyon buffer'ı ayırması yerine eğitimdeki aynı buffer'ları kullanmasını sağlar.

Worker ayrıca tick başına 45 ms bütçesi içinde işbirlikçi bir takvimle eğitir ve örnekler; sayfa hiç bloklanmaz, kayıp düşerken metin birkaç karakter halinde akar.

## İddia değil, doğrulama

```bash
npm test        # 35 test, node --test — test framework'ü yok, config yok
```

| test dosyası | neyi sabitliyor |
| --- | --- |
| `src/core/mat.test.ts` | her kernel'i naif üçlü-döngü referansına karşı, `NT`/`TN` devrik düzenleri ve birikim (accumulation) dahil |
| `src/core/autograd.test.ts` | her tape işlemini kendi ileri fonksiyonunun sonlu farklarına karşı |
| `src/model/gpt.test.ts` | üç mimarinin tamamında her parametre gradyanını merkezi farklara ve autodiff kâhinine karşı |
| `src/train/trainer.test.ts` | tokenizer gidiş-dönüşü, warmup/cosine öğrenme hızı takvimi ve checkpoint kaydet → yükle → aynı logit'ler |

Gradyan kontrolleri float32 için ayarlanmış bir adım boyutuyla merkezi farkları kullanır ve yanlış bir transpozun doğrudan patlatacağı bir göreli hata toleransına sahiptir.

## Sayılar, bu makineden

CLI tam olarak aynı veri seti ve adım bütçesi için token/saniye, parametre sayısı, doğrulama kaybı ve perplexity raporlar:

```bash
node src/cli/train.ts --compare --steps 600
```

Demo sayfasının `medium` hazır ayarı 677 bin parametreli bir model çalıştırır; `small` (paketteki checkpoint'in şekli) 207 bin parametre, 4 blok, 4 head, 48 karakterlik bağlam ve tiny shakespeare üzerinde ≈2.31 doğrulama kaybına kadar eğitilmiştir.

Burada hiçbir şey GPU gerektirmiyor ve hiçbir şey GPU kullanmıyor: CPU'nuzda tek iş parçacıklı JavaScript — sayfadaki sayıların inandırıcı olmasının nedeni de bu.

## Çalıştır

```bash
git clone https://github.com/emirmizrakli/emergence
cd emergence
npm install

npm run dev              # canlı sayfa, localhost'ta
npm test                 # tüm test paketi
npm run typecheck        # katı TypeScript, çıktı yok

# arayüzsüz eğitim, sayfadakiyle aynı kod
node src/cli/train.ts --model gpt --steps 4000 --out checkpoints/gpt.json
node src/cli/train.ts --compare --steps 600      # üç mimari, tek tablo

npm run sync             # veri setini indir ve demo dosyalarını hazırla
npm run build            # dist/ içine statik derleme
```

Kendi metninizle deneyin: `npm run dev`, *veri seti*ni **kendi metnin** yapın, bir şeyler yapıştırın — tokenizer ona göre yeniden kurulur ve model gözünüzün önünde, sıfırdan, sizin metninizi öğrenir.

## Demo sayfasını yayına alma

```bash
npm run deploy     # derler, sonra dist/ klasörünü gh-pages branch'ine force-push eder
```

Tek seferlik kurulum: **Settings → Pages → Source: Deploy from a branch → `gh-pages` → `/ (root)`.**
Sayfa statik; serverless fonksiyon, build dakikası ya da runner istemiyor: yayınlamak bir `git push` kadar. (`.github/workflows/` içindeki Actions iş akışı elle çalıştırılan bir typecheck + test koşucusudur, bilinçli olarak deployer değildir — buradaki hiçbir şey yayınlanmak için CI'a bağlı değil.)

## Bu proje ne değil

Sınırları açıkça söylemek işin bir parçası:

- **Sadece CPU.** GPU yok, WebGPU yok, WASM SIMD yok. Batch'ler küçük, modeller küçük. Burası bir laboratuvar, eğitim servisi değil.
- **Harf seviyesi.** BPE yok. Sözlük, veri setindeki farklı karakterlerden oluşur (Shakespeare için 65).
- **KV cache yok.** Üretim, tek bir çıkarım yolunu korumak için bilinçli olarak karakter başına O(n²).
- **Küçük modeller.** 28 bin – 677 bin parametre. İngilizce düzyazı öğrenir, olguları değil.

## Lisans

MIT © Emir Mızraklı
