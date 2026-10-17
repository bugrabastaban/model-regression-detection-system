# golden_v1 edge case incelemesi

## Güncel durum

`datasets/golden_v1.json` dosyasında **92 kayıt** vardır. İlk 80 kayda eklenen
`email_081`–`email_092` kayıtları da kullanıcı tarafından incelenip onaylanmıştır.
Kayıt bazındaki güncel inceleme durumları:

| review_status | Kayıt sayısı |
| --- | --- |
| approved | 92 |
| pending | 0 |

Her kategoride 23 kayıt vardır: `billing`, `technical`, `account`, `general`.
`expected_difficulty` dağılımı: `easy` 26, `medium` 45, `hard` 21.
ID'ler `email_001`–`email_092` aralığında benzersizdir.

Aşağıdaki dağılım içerik incelemesine dayanır; kategoriler arası belirsizlik,
birden fazla kategoriye işaret eden ipuçlarının baskın kullanıcı talebiyle
ayrılması anlamında kullanılmıştır. Bu, bir sınıflandırma başarımı ölçümü değildir.

| Tür | İlk 80 kayıttaki kapsam | Eklenen 12 onaylı kayıttaki kapsam | Toplam |
| --- | --- | --- | --- |
| Kategori belirsizliği | 18 belirgin sınır örneği; bu aşama için yeterli | 2 | 20 |
| Çok kısa e-posta | 2; yetersiz | 3 | 5 |
| Yazım hatalı e-posta | 0 belirgin örnek; eksik | 2 | 2 |
| Karma dilli e-posta | 0; eksik | 3 | 3 |
| İronik e-posta | 0 belirgin örnek; eksik | 2 | 2 |

Belirsizlik örnekleri: `email_013`, `018`, `020`, `029`, `030`, `031`,
`035`, `044`, `046`, `048`, `050`, `053`, `054`, `061`, `064`, `068`,
`070`, `080` (kısaltılmış numaralarda da `email_` öneki geçerlidir).
Örneğin `email_061`, teknik sorunu anlatsa da iptal ve iade talep eder;
`email_020`, giriş sözcüklerine rağmen uygulama arızasına odaklanır.

Çok kısa tanımı: boşlukla ayrılan en fazla 6 sözcük. Mevcut örnekler
`email_074` ve `email_079`. Küçük harf kullanımı tek başına yazım hatası,
yabancı kişi/ürün adı tek başına karma dil, öfke tek başına ironi sayılmadı.
`email_057` bir soyadı düzeltmesini anlatır; kasıtlı yazım hatalı e-posta
örneği sayılmaz. Mevcut JSON'da Unicode replacement character (`U+FFFD`) yoktur.

Sonradan eklenen 12 onaylı kaydın birincil edge case dağılımı:

- Belirsiz: `email_081–082`; baskın talep kararı kendi `notes` alanlarında.
- Çok kısa: `email_083–085`.
- Yazım hatalı: `email_086–087`.
- Karma dilli: `email_088–090`; Türkçe/İngilizce, İspanyolca/İngilizce,
  Fransızca/İngilizce; özet dili baskın anlatım diline göre seçildi.
- İronik: `email_091–092`; görünürdeki övgü gerçek memnuniyet sayılmadı.

Her kategoriye 3 yeni kayıt eklendi: billing, technical, account, general.
Başlangıçta `pending` olarak eklenen bu 12 kaydın tamamı artık `approved`
durumundadır; inceleme bekleyen taslak yoktur. Toplam sütunu onaylı kapsamı
gösterir. Türler semantik olarak örtüşebilir.

Önceki şema güncellemesinde ilk 80 kaydın yalnızca `difficulty` anahtarı
`expected_difficulty` olarak değiştirildi; değerler ve diğer alanlar korundu.
Bu belge güncellemesinde JSON, etiketler, referans özetler ve prompt dosyaları
değiştirilmemiştir.

Kök `description` alanı hâlâ tüm veriyi insan incelemesi bekleyen LLM taslağı
olarak anlatır. Bu eski açıklama, güncel kayıt bazındaki `review_status`
alanlarıyla çelişir; mevcut durum için kayıt alanları esas alınmıştır. Bu
görevde kök metadata değiştirilmemiştir. Dosya incelemeci kimliği/tarihi gibi
ayrıntılı onay geçmişi tutmaz; `approved` etiketi bağımsız kalite ölçümü değildir.

Çevrimdışı test komutları (proje kökünden):

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_golden_dataset.py -v
```

Veri seti testleri benzersiz ID, zorunlu alan, geçerli kategori/zorluk/onay
değerlerini ve ilk 80 kaydın korunduğunu denetler. Projede runner ve scoring
mevcuttur; offline testler bunlar için mock kullanır. `tests/test_classifier.py`
yalnızca doğrudan çalıştırıldığında canlı Gemini çağrısı yapar; unittest
discovery sırasında çağrı yapmaz. Bu belge tam 92 kayıt için canlı kalite
değerlendirmesi iddiası içermez.
