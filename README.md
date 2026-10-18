# Model Regression Detection System

## CI kapsamı ve doğrulama durumu

GitHub Actions workflow'u, `main` hedefli PR'larda offline testleri ve sabit
`Prompt evaluation gate` kontrolünü tanımlar. Prompt değişikliği yoksa canlı
değerlendirme yapılmaz. Prompt değişikliğinde sabit üç kayıt ve kayıtlı baseline
kullanılır; en fazla 3 Gemini + 3 Groq isteği gönderilir. Gemini çağrıları
arasında en az 60 saniye beklenir. Kritik gerileme ve eksik değerlendirme gate'i
başarısız yapar; birleşmeyi engellemek için GitHub'da gerekli kontrol olarak
tanımlanmalıdır. HTML/JSON artifact, PR yorumu ve isteğe bağlı Slack bildirimi
akışı kodda mevcuttur. Kurulum ve kota rezervasyonu için [ci/SETUP.md](ci/SETUP.md).

Bu davranışlar yerelde mock'larla test edilir; uzaktaki GitHub Actions çalışması,
repo koruma ayarları ve güncel sağlayıcı uyumluluğu bu testlerle doğrulanmaz.
Tam 92 kaydın canlı değerlendirmesi ve otomatik geçmiş deposu henüz yoktur.

## Slow drift: 7 çalışma hareketli ortalaması

`drift_detection.detect_slow_drift` en yeni çalışma dahil son 7 ardışık,
karşılaştırılabilir çalışmanın genel pass oranının aritmetik ortalamasını izler.
Varsayılan asgari ortalama **%90**'dır; tam %90 uyarı üretmez. Bu başlangıç
eşiği ve pencere boyu `DriftConfig` ile değiştirilebilir. Pass tanımı mevcut
doğru kategori ve judge ≥4/5 kuralıdır. Tokenlar bu hesaba girmez.

```python
from drift_detection import DriftConfig, detect_slow_drift

# history: mevcut çalışma hariç (puanlanmış rapor, o çalışmanın veri seti) çiftleri
drift = detect_slow_drift(
    current_scored_report, current_dataset_snapshot,
    history=history,
    config=DriftConfig(window_size=7, minimum_average_pass_rate=0.90),
)
```

Son pencerenin ortalaması eşikten düşükse ve pencere içindeki ardışık
karşılaştırmalarda warning/critical yoksa `status=slow_drift`,
`severity=warning` üretilir. Pencerenin ilk çalışmasından önceki uygun çalışma
mevcutsa onunla geçiş de kontrol edilir. Mevcut tek-çalışma uyarısı varsa
`per_run_alert_present` döner; slow drift adıyla ikinci bir uyarı üretilmez.
Bu kontrol gerçek gönderilmiş Slack mesajlarını değil, yeniden hesaplanan
eşik sonuçlarını inceler. Tekrarlanan bildirimleri kalıcı olarak susturan
bir sistem eklenmemiştir.

Örnek: %92, %91, %90, %89, %88, %87, %86; ortalama %89. Her düşüş 1 yüzde
puan olduğundan tek-çalışma genel uyarısı yoktur; slow drift uyarısı vardır
(kategori doğruluklarında da ayrı uyarı olmadığını varsayarak).

Geçmiş tarih sırasına alınır; saat dilimli `started_at` zorunludur. Aynı
zaman damgası, yinelenen run ID veya mevcut çalışmadan sonraki tarih reddedilir.
Aynı örnek kümesi, girdi, beklenen kategori, referans özet, pass politikası,
judge rubriği ve judge modeli gerekir. Sınıflandırıcı model/prompt değişebilir.
Kısmi ama değişmeyen bir küme ile test mümkündür; bu tüm veri setinin kalitesini
temsil etmez. Eksik puan/API hatası veya uyumsuz geçmiş pencereyi böler;
bu çalışmalar ne sıfır puan sayılır ne de atlanarak eski geçmişe bağlanır.
Yeterli yeni ardışık veri oluşunca hesap yeniden başlar.

Birleştirilmiş canlı subset dosyası gerçek tek çalışma olmadığından drift
penceresine alınmaz. Şu an canlı geçmiş yetersizdir. Bir kez 92 kaydı
çalıştırmak da 7 çalışma oluşturmaz; canlı drift testi için 7 uygun çalışma
gerekir. Tek bir yavaş veya başarısız ağ çağrısı trend puanı değildir.

Sonuç JSON'a uygun `series`, `window_runs`, `moving_average`, dışlanan
çalışmalar ve tek-çalışma uyarılarını içerir. `render_report(..., history=history)`
HTML'de drift durumunu gösterir. `build_slack_message(..., history=history)`
slow drift için WARN mesajı oluşturur; bu fonksiyon mesaj göndermez.
Özel eşiği bu iki fonksiyona `drift_config=DriftConfig(...)` ile verin.
CLI komutları henüz geçmiş dosyalarını otomatik keşfetmez; geçmiş açıkça
programatik olarak sağlanır. Yeni bağımlılık veya ağ çağrısı gerekmez.

## Slack bildirimleri

`slack_alerts.py` kaydedilmiş puanlanmış rapor ve veri setinden Block Kit mesajı
oluşturur. Gemini/Groq çağrısı yapmaz. Webhook `.env` içindeki
`SLACK_WEBHOOK_URL` değerinden okunur; ortam değişkenleri önceliklidir.
Varsayılan komut yalnızca mesajı terminalde önizler:

```powershell
.\.venv\Scripts\python.exe slack_alerts.py --report test_results/combined_subset_live_20261015.json --dataset datasets/golden_v1.json
```

Gerçek kanala tek mesaj göndermek için aynı komuta `--send` ekleyin:

```powershell
.\.venv\Scripts\python.exe slack_alerts.py --report test_results/combined_subset_live_20261015.json --dataset datasets/golden_v1.json --send
```

Karşılaştırmalı bildirimde `--previous` ve `--previous-dataset` birlikte
verilir. Durum eşlemesi: `ok → PASS`, `warning → WARN`, `critical → FAIL`.
Baseline/karşılaştırılabilir metrik yoksa `NOT EVALUATED` gösterilir.
Karşılaştırma kapsamında değerlendirilemeyen veya judge uyumsuz kayıt varsa
`PASS`, `WARN` seviyesine yükseltilir. Mesaj kapsama sayılarını, mevcut pass ve
kategori doğruluğunu, gerileme/iyileşme sayılarını, eski/yeni pass oranını,
kategori doğruluklarını ve yüzde puan farklarını içerir. Ham e-posta ve model
çıktıları gönderilmez. İlk üç kayıt karşılaştırılabilir baseline içermez.

Tam HTML raporu bağlantısı için raporu ekibin erişebileceği bir HTTPS
adresinde barındırıp `HTML_REPORT_URL` ortam/.env değerini ayarlayın veya
`--report-url` verin. Link düğmesi ve düz metin alternatifine URL eklenir;
link içeriği indirilmez, erişilebilirliği otomatik doğrulanmaz. Webhook raporu
yayımlamaz. Yerel dosya/localhost bağlantıları kabul edilmez. URL yoksa test
bildirimi açıkça bağlantının yapılandırılmadığını söyler. Hosting bu aşamada
kurulmamıştır; bağlantılı bildirim için bu yapılandırma halen gereklidir.

Programatik kullanım: `build_slack_message(...)` ardından açıkça
`await send_slack_alert(payload)`. `ThresholdConfig`, mesaj oluşturucusuna
`thresholds=` ile verilebilir. Mevcut classifier endpoint'i ve runner
kendiliğinden Slack mesajı göndermez; değerlendirme akışını çağıran kod bu
işlemi HTML üretiminden sonra çağırmalıdır.

Gönderim tek HTTP POST kullanır; retry/yönlendirme yoktur. Timeout veya
bağlantı hatasında teslim durumu belirsizdir; otomatik tekrar gönderilmez.
Hatalar webhook URL'si/sağlayıcı gövdesi gösterilmeden iletilir. Offline
testler `httpx.MockTransport` kullanır; gerçek Slack'e mesaj göndermez.

## HTML fark raporu

Mevcut üç canlı örneğin raporu:
`test_results/combined_subset_live_20261015.html`. Dosyayı tarayıcıda açın.
Harici grafik kütüphanesi veya internet bağlantısı gerekmez. Bu örnekler iki
oturumdan birleştirilmiştir; raporda bu köken ve önceki 503 denemesi belirtilir.
Önceki karşılaştırılabilir çalışma yoktur; baseline farkları boş ve trend
tek nokta gösterilir. Bu görünüm tüm 92 örneğin başarımı anlamına gelmez.

Kaydedilmiş yeni sonuçlarla tekrar üretmek için (API çağrısı yapmaz):

```powershell
.\.venv\Scripts\python.exe html_report.py --report test_results/combined_subset_live_20261015.json --dataset datasets/golden_v1.json --output test_results/combined_subset_live_20261015.html
```

İleride `--report` parametresine 92 kaydın puanlanmış raporunu vererek
aynı görünümü güncelleyebilirsiniz. Karşılaştırma için `--previous` ile
önceki puanlanmış JSON'u, `--previous-dataset` ile onun veri seti kopyasını
birlikte sağlayın. Gerileyen her kaydın eski/yeni kategori, özet ve judge
puanları yan yana gösterilir. Tüm metrikler mevcut `compare_runs` kurallarını
kullanır; bilinmeyen değerler sıfır olarak sunulmaz.

Programatik kullanımda `html_report.render_report(..., history=[(report,
dataset_snapshot), ...], last_n=10)` geçmişi açıkça alır. Trend, aynı ID
kümesi/referanslar ve karşılaştırılabilir judge yapılandırması olan sonuçları
tarih sırasıyla gösterir. Farklı alt kümeleri birbiriyle trend olarak çizmez.
Trend penceresi seçilmeden önce aynı çalışma ayıklanır: dolu bir `run_id`
önceliklidir. Eski raporlarda deterministik anahtar; `started_at`,
`prompt_version`, `prompt_timestamp`, `model`, `dataset_version`,
`judge_rubric_version` ve `commit_sha` alanlarından oluşur. Aynı anahtarın son
kopyası kullanılır; açıkça verilen previous ve current, trenddeki geçmiş
kopyalarına önceliklidir. Bu metadata aynı olan ayrı çalışmalar için benzersiz
`run_id` sağlayın. HTML oluşturucusuna verilen yinelenen geçmiş ve current
kopyaları drift hesabından önce de ayıklanır; doğrudan drift fonksiyonunun
yinelenen ID/zaman damgası doğrulaması ise korunur.
HTML içindeki e-posta ve model metinleri escape edilir; script çalıştırılmaz.

## Gerileme uyarı eşikleri

Karşılaştırma çıktısındaki `threshold_assessment` varsayılan olarak düşüş
**3 yüzde puanını aşarsa** `warning`, **8 yüzde puanını aşarsa** `critical`
üretir. Tam 3 puan düşüş `ok`, tam 8 puan düşüş `warning` olur. Pozitif
değişimler uyarı üretmez. Bu bir istatistiksel hipotez testi değildir;
p-değeri, güven aralığı veya istatistiksel anlamlılık iddiası üretmez.

```python
from comparison import compare_runs
from thresholds import ThresholdConfig

diff = compare_runs(
    current_scored_report, previous_scored_report,
    current_dataset=current_dataset_snapshot,
    previous_dataset=previous_dataset_snapshot,
    thresholds=ThresholdConfig(
        warning_percentage_points=3.0,
        critical_percentage_points=8.0,
    ),
)
print(diff["threshold_assessment"])
```

Eşikler `0 <= warning < critical <= 100` koşulunu sağlamalıdır. Değişiklik
göreli yüzde yerine yüzde puanla hesaplanır: %90 → %86, -4 puan ve warning'dir.
Genel pass oranına ve her kategorinin doğruluğuna ayrı ayrı uygulanır;
özet seviye değerlendirilebilen metriklerin en ağır seviyesidir. Bir kategori
kritik düşerken genel oran sabit kalsa bile özet `critical` olabilir.

İlk çalışma veya karşılaştırılabilir örneği olmayan metrik `not_evaluated`
olur. Küçük alt kümelerde tek kayıt büyük oran değişimi yaratabilir; her
metrikteki `compared_cases` ve mevcut kapsam bilgileri birlikte incelenmelidir.
Örneğin 80 ortak kayıtta 2 net gerileme -2,5 puandır ve varsayılan genel
eşiği aşmaz; gerileyen ID'ler yine listelenir, kategori eşiği ayrıca aşılabilir.
Gerileme ve iyileşmeler birbirini oran hesabında dengeleyebilir. Eşik aşılmasa
da ID listeleri korunur. Eşik hesabı offline'dır; Slack ve CI katmanları bu
sonuçları bildirim ve gate kararı için kullanır.

## Çalışmaların karşılaştırılması

`comparison.compare_runs` puanlanmış iki raporu API çağrısı yapmadan karşılaştırır.
Pass kuralı `category_match_and_summary_at_least_4_v1`: kategori doğru ve judge
puanı en az 4/5 olmalıdır. Sınıflandırıcı hatası, geçersiz çıktı, eksik judge
ve judge hatası `unevaluable` olarak ayrılır; sessizce başarısız sayılmaz.

```python
from comparison import compare_runs

diff = compare_runs(
    current_scored_report,
    previous_scored_report,
    current_dataset=current_dataset_snapshot,
    previous_dataset=previous_dataset_snapshot,
)
# İlk çalışma:
baseline = compare_runs(current_scored_report, current_dataset=current_dataset_snapshot)
```

Raporlarda model çağrılarından önceki değil, `score_results` sonrası sonuçlar
kullanılmalıdır. Önceki raporu ve o çalışmaya ait veri seti kopyasını çağıran
kod sağlar. Dosya adına/tarihine göre önceki çalışma otomatik seçilmez; veri
setinin güncel kopyası eski çalışma adına kullanılmamalıdır. CI bu fonksiyonu
sabit üç örnekli baseline ile kullanır; genel amaçlı geçmiş deposu veya önceki
çalışmayı otomatik seçen bir mekanizma yoktur.

`pass_policy` alanı bulunmayan eski raporlar yukarıdaki mevcut kuralla
yorumlanır. Açıkça belirtilen policy farklı, boş veya geçersizse (iki raporda
aynı bilinmeyen policy olsa bile) `status=incompatible_pass_policy` döner.
`pass_policy_compatibility` etkili policy değerlerini ve uyumluluğu gösterir;
bu durumda oran farkları hesaplanmaz ve eşik seviyesi `not_evaluated` olur.

ID'ler üzerinden eşleştirme yapılır. Girdi, beklenen kategori veya referans
özet değişmişse kayıt `changed_cases` listesine gider ve oranlardan çıkarılır.
Not/zorluk/inceleme durumu değişiklikleri referans değişikliği sayılmaz.
`added_cases`/`removed_cases`, karşılaştırılan raporların kapsam farklarını da
gösterir; örneğin farklı alt kümeler çalıştırıldığında. Veri seti sürüm adı
aynı olsa bile gerçek içerikler karşılaştırılır.

`overall_pass_rate` iki çalışmada da değerlendirilebilen, referansı aynı
kayıtlardaki pass oranlarını ve yüzde puan farkını içerir. Judge modeli ve
rubriği aynı ve biliniyor olmalıdır; değişenler `judge_incompatible_cases`
listesine alınır. Bu kural kategori doğruluğu hesabını engellemez.
`per_category_accuracy` beklenen kategoriye göre, her iki çalışmada geçerli
sınıflandırıcı çıktısı olan aynı kayıtlarda hesaplanır. Paydalar her metrikte
`compared_cases` ile gösterilir; boş paydada oran/fark `null` olur.

`regressions` pass→fail, `improvements` fail→pass ID listeleridir.
`case_changes` önceki/yeni durumları; `unevaluable_cases` ve `coverage`
karşılaştırmaya giremeyenleri ve kapsama sayılarını gösterir. Hatalı kayıtlar
paydadan çıktığı için yüksek başarı oranı tek başına yeterli değildir;
kapsama bilgisi de incelenmelidir. İlk çalışma `baseline` döner ve fark üretmez.
Çalışma bilgileri mevcut raporlardan alınır; bulunmayan `run_id` gibi alanlar
`null` kalır. Eşikler karşılaştırma sonucuna eklenir; Slack ve CI entegrasyonları
ayrı modüllerdedir.

## Offline çok boyutlu puanlama

`scoring.py`, mevcut runner sonuçlarını yeniden model çağırmadan puanlar:

```python
import asyncio
from scoring import score_results, save_scored_results

# report: daha önce elde edilmiş runner raporu; dataset: yüklenmiş golden JSON.
scored = asyncio.run(score_results(report, dataset))
save_scored_results(scored, "scored_results.json")
```

Her sonuç kaydının `scores` alanında kategori eşleşmesi (`0/1`), özet
uygunluğu (`1–5` veya `null`), sınıflandırma süresi ve token kullanımı tutulur.
Süre runner'dan aynen alınır; thread beklemesi ve doğrulamayı da içerir.
Geçersiz/başarısız sınıflandırmanın puanı `null` olur; hata yanlış kategori
gibi değerlendirilmez. Birleşik puan veya regresyon kararı hesaplanmaz.

İsteğe bağlı async `judge(system_prompt, payload_json)` callback'i bir
`JudgeResult` döndürür. Puan ve gerekçesi Pydantic ile doğrulanır. Judge'ın ham
yanıtı, modeli, süresi, tokenları ve hatası sınıflandırıcıdan ayrı saklanır.
Judge verilmezse durum `not_configured`, özet puanı `null` olur. Hazırlanan
`summary_relevance_v1` rubriği beş puan seviyesini tanımlar. Testlerde yalnızca
sahte judge veya HTTP mock kullanılır. `groq_judge.GroqJudge` adaptörü ve
`openai/gpt-oss-120b` model seçimi mevcuttur; geniş kapsamlı canlı kalibrasyon
tamamlanmamıştır. Offline testler canlı API uyumluluğu kanıtı değildir.

Token ölçümü, sınıflandırıcı yanıtından runner, scoring, comparison ve HTML'e
bağlıdır. Ayrıntılar aşağıdaki token metadata bölümündedir. Bu akışın offline
doğrulanması, güncel bir canlı API yanıtındaki metadata doluluğunu doğrulamaz.

`save_scored_results` UTF-8 JSON yazar ve var olan dosyanın üzerine yazmayı
reddeder. Puanlama veri setini ve özgün runner raporunu değiştirmez.

## Token metadata: ölçüm ve sınırlar

Gemini yanıtının `usage_metadata` alanı aynı SDK çağrısından alınır; token
bilgisi için ek sınıflandırma veya token sayma isteği gönderilmez. Dahili yanıt
metin ve kullanım bilgisini birlikte taşır. `classify_email(email)` yalnızca
kategori/özet sözlüğü, `classify_email_raw()` metin veya `None` döndürmeye devam
eder. Prompt ve kalite skorları token ölçümü nedeniyle değişmez.

Ortak sözleşme `token_usage.py` içindedir. `TokenUsage` ve
`gemini_token_usage()` eski `scoring` import yolundan da kullanılabilir.
Gemini alan eşlemesi:

| Sağlayıcı metadata alanı | Ortak sayaç |
| --- | --- |
| `prompt_token_count` | `input_tokens` |
| `candidates_token_count` | `output_tokens` |
| `total_token_count` | `total_tokens` |

Runner her kayıtta `token_usage` ve ayrı `token_usage_error` saklar. Kullanım,
çıktı doğrulamasından önce kaydedildiğinden bozuk JSON, geçersiz kategori veya
boş yanıt durumunda mevcut sayaçlar korunur. Scoring bu bilgiyi
`scores.token_usage` ve `scores.token_usage_error` alanlarına taşır. Groq judge
sayaçları `scores.judge.token_usage` içindedir; `prompt_tokens`,
`completion_tokens`, `total_tokens` sırasıyla aynı üç ortak alana eşlenir.
Gemini sınıflandırma tüketimi ve Groq değerlendirme tüketimi birleştirilmez.

Eksik metadata veya alan JSON'da `null`, HTML'de **bilinmiyor** olur. Kısmi
metadata'da bilinen sayaçlar korunur; gerçek `0`, bilinmeyen değerden farklıdır.
Sağlayıcının `total_tokens` değeri aynen tutulur; input ve output toplanarak
üretilmez veya düzeltilmez. Toplamın bu iki sayacın toplamına eşit olması
beklenmez. Gemini metadata'sı bozuksa tüm sayaçlar bilinmiyor kalır ve
`invalid_metadata` kodu sınıflandırma hatasından ayrı saklanır; geçerli çıktı
bu nedenle başarısız sayılmaz. Timeout/API hatasında yanıt metadata'sı yoksa
tüketim bilinmiyor kalır; sıfır veya ücretsiz olduğu varsayılmaz.

`compare_runs` çıktısındaki `token_usage_comparisons`, referansı değişmeyen
ortak kayıtlar için `gemini` ve `groq` altında `previous`, `current` ve
`delta_tokens` alanlarını içerir. Fark yeni − önceki şeklindedir ve yalnızca
iki tarafta da ilgili sayaç biliniyorsa hesaplanır; aksi halde `null` kalır.
HTML sağlayıcı bazında input/output/total, eski/yeni farkları ve bilinen sayaç
kapsamını gösterir. Tokenlar pass/fail, kategori doğruluğu, judge skoru,
regresyon, eşik ve drift hesaplarına girmez.

Token alanı olmayan eski raporlar ve baseline'lar desteklenir. Geçmişte
kaydedilmemiş token tüketimi eski e-posta/çıktı metninden veya rapordan yeniden
oluşturulamaz. Mevcut tarihsel artifact'lara tahmini sayaç eklenmez.

**Token sayaçları fatura tutarı veya maliyet tahmini değildir.** Fiyatlandırma,
sağlayıcı/model, fiyat tarihi, cache ve diğer ücretlendirme ayrıntıları olmadan
para tutarı hesaplanmaz. Kaydedilmiş yanıt kullanımı, hesabın tüm tüketimini
veya SDK'nin olası diğer denemelerinin toplamını ölçtüğü iddiasını taşımaz.

`tests/test_token_pipeline.py`, sahte Gemini SDK yanıtlarını ve
`httpx.MockTransport` ile yakalanan Groq yanıtlarını gerçek uygulama zincirinden
geçirir: classifier → runner → scoring → JSON kaydı → comparison → HTML.
Eski baseline, eksik/kısmi/sıfır/bozuk metadata, bozuk model çıktıları ve yedi
çalışmalık sentetik geçmişte kalite kararlarının değişmezliği test edilir.
Tüm testleri aşağıdaki mevcut offline test komutuyla çalıştırın. Bu testler
gerçek Gemini/Groq/Slack isteği göndermez; canlı metadata entegrasyonu ayrıca,
açıkça planlanan ve kota gözetilen bir testle doğrulanmalıdır. Önceki canlı
smoke testleri, bu yeni metadata akışının canlı doğrulaması değildir.

## FastAPI ve Swagger UI

Proje kökünde Windows PowerShell ile:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn main:app --reload
```

API anahtarı mevcut `GEMINI_API_KEY` ortam değişkeninden veya proje kökündeki
`.env` dosyasından okunur. Mevcut ortam değişkenleri `.env` değerlerinden
önceliklidir. İsteğe bağlı `GEMINI_MODEL` modeli seçer; tanımlı değilse mevcut
sınıflandırıcının `gemini-3.6-flash` varsayılanı kullanılır. Prompt mevcut
`PromptConfig` yükleyicisi üzerinden `prompts/v1.yaml` dosyasından gelir.

`http://127.0.0.1:8000/docs` adresinde **POST /classify → Try it out** seçin:

```json
{"email": "I was charged twice this month. Please refund the extra charge."}
```

**Execute** gerçek Gemini isteği gönderir ve kota kullanır. Başarılı yanıt
`category` ve `summary` içerir. Swagger sayfasını açmak API çağrısı yapmaz.
Sunucuyu durdurmak için `Ctrl+C` kullanın.

Hatalar: geçersiz girdi 422, eksik yapılandırma 503, sağlayıcı kota sınırı 429,
timeout 504, diğer sağlayıcı/çıktı hataları 502, beklenmeyen hata 500.
İstemciye iç hata ayrıntıları veya API anahtarı gönderilmez.

Tüm offline testler:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

HTTP testleri FastAPI `TestClient` ve mock sınıflandırıcı kullanır; canlı
Gemini uyumluluğunu doğrulamaz. `tests/test_classifier.py` yalnızca doğrudan
çalıştırıldığında gerçek API çağrısı yapan manuel betiktir.

## Docker

Proje kökünde, Linux container engine açıkken PowerShell ile image oluşturun:

```powershell
docker build -t model-regression-detection-system:local .
```

Image Python 3.12 kullanır ve mevcut `main:app` uygulamasını Uvicorn ile
`0.0.0.0:8000` üzerinde başlatır. Uygulama root olmayan `app` kullanıcısıyla
çalışır. Python modülleri, `prompts/`, `datasets/` ve `ci/` runtime dosyaları
image'a kopyalanır. FastAPI'nin mevcut varsayılan promptu `v1` olarak kalır.

Önce anahtar sağlamadan, yalnızca HTTP başlangıcını kontrol edebilirsiniz:

```powershell
docker run --detach --rm --name model-regression-api -p 127.0.0.1:8000:8000 model-regression-detection-system:local
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/docs
Invoke-RestMethod http://127.0.0.1:8000/openapi.json
docker stop model-regression-api
```

Başlangıç ve bu iki GET isteği Gemini, Groq veya Slack çağrısı yapmaz.
Swagger arayüzünde `POST /classify` çalıştırmak ise gerçek Gemini isteğidir.
Anahtar olmadan sınıflandırma isteği 503 döner.

Gerçek kullanım için yapılandırmayı container oluşturulurken sağlayın:

```powershell
docker run --detach --rm --name model-regression-api --env-file .env -p 127.0.0.1:8000:8000 model-regression-detection-system:local
docker logs model-regression-api
docker stop model-regression-api
```

`--env-file .env` dosyayı image'a kopyalamaz; `KEY=value` satırlarını çalışma
zamanında ortam değişkeni olarak aktarır. Docker env dosyasında değerleri
tırnaksız yazın; dotenv değişken genişletmesine güvenmeyin. Sınıflandırıcı
`GEMINI_API_KEY` ve isteğe bağlı `GEMINI_MODEL` kullanır. Judge için
`GROQ_API_KEY` / `GROQ_MODEL`, Slack için `SLACK_WEBHOOK_URL` ve isteğe bağlı
`HTML_REPORT_URL` mevcut modüllerce okunur; HTTP endpoint'i judge veya Slack'i
otomatik çalıştırmaz. Yalnızca gereken değişkenleri içeren yerel bir env
dosyası sağlayabilirsiniz. Alternatif olarak mevcut PowerShell ortamından
`--env GEMINI_API_KEY --env GEMINI_MODEL` ile aktarın; anahtarı komuta yazmayın.

`.dockerignore` env dosyalarını, sanal ortamı, Git verilerini, cache'leri ve
yerel çıktıları build context dışında tutar. Test dosyaları ve testlerin
kullandığı `test_results/combined_subset_live_20261015.json` fixture'ı context'te
korunur; runtime image'a testler veya yerel raporlar kopyalanmaz.
Tam offline paket mevcut yerel komutla çalıştırılır:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Runtime image test ortamı değildir; tam paket ayrıca CI kota testleri için
Node.js kullanır. Docker aşaması için image'a ek test bağımlılığı eklenmez.
Docker engine erişilebilir değilse build ve container endpoint kontrolleri
doğrulanamaz; yalnızca yerel test başarısı Docker doğrulaması sayılmaz.

## Async test runner

`run_evaluation` yüklenmiş `PromptConfig` ve golden JSON nesnesini alır.
Aşağıdaki örnek çalıştırıldığında tüm veri seti için gerçek API çağrıları yapar.
92 kayıt, 20 RPD sınırını aşar; bu komutu mevcut sınırlı kotayla çalıştırmayın.
Runner'daki concurrency sınırı RPM/RPD sınırlayıcısı değildir. CI akışı ayrı
olarak üç kayıt ve çağrılar arasında bekleme kullanır:

```python
import asyncio
import json
from pathlib import Path
from evaluation_runner import run_evaluation
from prompt_config import load_prompt_config

dataset = json.loads(Path("datasets/golden_v1.json").read_text(encoding="utf-8-sig"))
report = asyncio.run(run_evaluation(load_prompt_config("v1"), dataset, batch_size=4))
```

Her kayıt için `case_id`, özgün `raw_output` metni, doğrulanmış `output`,
`error`, `elapsed_seconds`, `token_usage` ve `token_usage_error` döner.
Geçersiz JSON dahil ham yanıtlar ve mevcut token sayaçları korunur;
bir kaydın hatası diğerlerini durdurmaz. Sonuç sırası veri setiyle aynıdır.
Rapor prompt sürümünü/zamanını, veri seti sürümünü ve model adını içerir.
Beklenen etiketler modele gönderilmez; her kayıt çalıştırılır, inceleme
durumuna göre filtre uygulanmaz.

Senkron sınıflandırıcı çağrıları `asyncio.to_thread` ile en fazla `batch_size`
eşzamanlı çağrı içeren gruplar halinde yürütülür. Runner tekrar deneme yapmaz;
SDK'nin kendi tekrar deneme davranışı geçerlidir. Bu yöntem token fiyatını
düşürmez ve sağlayıcının indirimli Batch API'si değildir. İptal, başlamış
thread çağrılarını durduramaz. Runner puanlama veya dosyaya yazma yapmaz;
puanlama ve puanlanmış JSON kaydı `scoring.py` içinde yapılır.

Runner testlerini API çağrısı olmadan çalıştırmak için:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_evaluation_runner.py -v
```

İlk özellik: Gemini ile müşteri destek e-postalarını sınıflandırma.
Hedef Python sürümü 3.11+'dır; yerel test ortamı 3.12.6 ve CI seçimi 3.12'dir.
3.11 üzerinde kurulum ve test ayrıca doğrulanmamıştır.

## Kurulum (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:GEMINI_API_KEY = "your-api-key"
```

## Kullanım

```python
from email_classifier import DEFAULT_PROMPT, classify_email

result = classify_email("I was charged twice this month. Please refund one charge.")
print(result)
# Örnek çıktı (gerçek model çıktısı değişebilir):
# {"category": "billing", "summary": "The customer requests a refund for a duplicate charge."}

# Prompt, şablon yerine doğrudan sistem talimatı olarak iletilir.
result = classify_email(
    "I cannot reset my password.",
    prompt=DEFAULT_PROMPT + "\nAlways write the summary in Turkish.",
    model="gemini-3.6-flash",
)
```

Fonksiyon `category` ve `summary` alanlarını içeren bir sözlük döndürür.
Kategoriler: `billing`, `technical`, `account`, `general`.
Özetin tek cümle olması prompt ile istenir; dilsel cümle doğrulaması yapılmaz.
Boş girdiler ve geçersiz yanıtlar `ValueError`, eksik API anahtarı
`RuntimeError` üretir. API ve bağlantı hataları çağırana iletilir.
E-posta içeriği işlenmek üzere Gemini API'ye gönderilir.

API entegrasyonu, Google'ın [yapılandırılmış çıktı belgelerini](https://ai.google.dev/gemini-api/docs/generate-content/structured-output) temel alır.

## Prompt sürümleme

Varsayılan prompt `prompts/v1.yaml` dosyasından yüklenir. Her sürüm dosyası
`version_id`, tırnak içinde saat dilimli ISO 8601 `timestamp`, `system_prompt`
ve `few_shot_examples` alanlarını içerir. Örneklerde `input_email` ile
`category` ve `summary` içeren `expected_output` bulunur.

Yeni sürüm için mevcut dosyayı değiştirmek yerine `v2.yaml` gibi yeni bir dosya
ekleyin; `version_id` dosya adıyla eşleşmelidir. `timestamp` sürümün oluşturulma
zamanıdır. Dosyalar çalışma dizininden bağımsız olarak bulunur.

```python
from email_classifier import classify_email
from prompt_config import load_prompt, render_prompt

selected_prompt = render_prompt(load_prompt("v1"))
result = classify_email("Please help me reset my password.", prompt=selected_prompt)
```

`classify_email(email)` ve özel metin prompt kullanımı korunur. Varsayılan
`DEFAULT_PROMPT`, modül yüklenirken YAML talimatı ve örneklerinden oluşturulur.
YAML okuma ve doğrulama `prompt_config.py` içinde yapılır; model çağrısı içermez.

## Testler

Tüm otomatik testleri gerçek API çağrısı yapmadan çalıştırın:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

`tests/test_classifier.py` mevcut manuel API denemesidir; yalnızca doğrudan
çalıştırıldığında `.env` dosyasını yükleyip Gemini çağrısı yapar. Otomatik test
keşfinde çağrı yapmaz. Otomatik testler `tests/` dizinindedir; sınıflandırıcı,
prompt, sözleşme, runner, puanlama, karşılaştırma, eşikler, HTML, drift, Slack
ve CI davranışlarını offline doğrular. Yukarıdaki komutta `-s tests`
gereklidir; kök dizinden varsayılan discovery testleri bulmayabilir.


## Groq judge bağlantısı

`groq_judge.GroqJudge`, mevcut `score_results(..., judge=...)` arayüzünü
kullanır. `.env` dosyasında `GROQ_API_KEY` gerekir. Model varsayılan olarak
`openai/gpt-oss-120b` seçilir; isteğe bağlı `GROQ_MODEL` ile değiştirilebilir.
Ortam değişkenleri `.env` değerlerinden önceliklidir. Anahtar kodda tutulmaz.

```python
import asyncio
from groq_judge import GroqJudge
from scoring import score_results, save_scored_results

# report ve dataset daha önce yüklenmiş olmalıdır.
# Bu satır, her geçerli sonuç için GERÇEK Groq isteği yapar:
scored = asyncio.run(score_results(report, dataset, judge=GroqJudge()))
save_scored_results(scored, "groq_scored_results.json")
```

Adaptör oluşturmak veya modülü import etmek API isteği yapmaz. Çağrıldığında
email, referans özet ve aday özet Groq'a gönderilir. Gemini tekrar çağrılmaz.
Her judge çağrısı tek HTTP isteğidir; otomatik retry ve yönlendirme yoktur.
HTTP timeout 60 saniyedir (ağ işlemi timeout'u, tüm çalışma süresi sınırı değil).
HTTP hataları yanıt gövdesi veya kimlik bilgileri açığa çıkarılmadan kaydedilir.
Puan `1–5` ve gerekçe mevcut Pydantic sözleşmesiyle doğrulanır. Groq tokenları
rapordaki judge bölümünde tutulur; Gemini tokenları ayrı `scores.token_usage`
alanına aktarılır. Metadata akışının canlı doğrulaması henüz yapılmamıştır.
`/classify` endpoint'i yalnızca sınıflandırır; judge'ı otomatik çalıştırmaz.

Uygulama [Groq structured outputs belgelerini](https://console.groq.com/docs/structured-outputs)
izleyerek Chat Completions uç noktasına `json_schema` ve `strict: true` gönderir.
Testler `httpx.MockTransport` kullanır; canlı model uyumluluğunu doğrulamaz.
