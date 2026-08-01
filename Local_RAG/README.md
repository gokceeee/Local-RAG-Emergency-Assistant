# Yerel Hayatta Kalma RAG Asistanı

## Projenin Amacı
Bu proje, Microsoft Foundry Local ve Retrieval-Augmented Generation (RAG) mimarisini kullanarak geliştirilmiş **tamamen çevrimdışı (internetsiz)** çalışan bir soru-cevap asistanıdır. Amacımız, hayatta kalma ve acil durum senaryolarına yönelik soruları, önceden hazırlanmış yerel bir veritabanını baz alarak ve halüsinasyon görmeden (uydurmadan), güvenli bir şekilde yanıtlamaktır.

Halüsinasyon engelleme işi yalnızca modele bırakılmaz; bilginin sağlanan belgelerin dışına çıkıp çıkmadığı **kod seviyesinde**, model devreye girmeden önce ve cevabı ürettikten sonra yazılımsal olarak denetlenir. Bu denetim mantığı `rag_core.py` içinde tek kaynak olarak tutulur ve tüm arayüzler oradan çağırır.

## Nasıl Çalışır?
Sistem 3 temel adımdan oluşur:

1. **Veri Alma (Ingestion):** `survival_database.json` içindeki acil durum bilgileri okunur, `qwen3-embedding-0.6b` modeli ile anlamsal vektörlere dönüştürülür ve SQLite veritabanlarına kaydedilir. Bu işlem `main.py` tarafından yürütülür ve konu bazlı veritabanları (genel + afet/acil durum/hayatta kalma) üretir.

2. **Geri Çağırma ve Giriş Kapısı (Retrieval — Katman 1):** Kullanıcının sorusu aynı modelle vektörleştirilir ve SQLite içinde kosinüs benzerliği (cosine similarity) hesaplanarak en alakalı belge parçacıkları (chunk) bulunur. Benzerliği belirlenen eşiğin altında kalan sorularda sistem, **LLM'i hiç çağırmadan** doğrudan "yeterli bilgi yok" yanıtını döner. Bu mantık `database.py` (`search_with_scores`) dosyasındadır.

3. **Üretim ve Doğrulama (Generation — Katman 2):** Bulunan bağlam ve kullanıcının sorusu katı bir prompt ile `phi-3.5-mini` modeline verilir. Model **yalnızca üretim** yapar; ürettiği cevap kullanıcıya gösterilmeden önce kod seviyesinde denetlenir: her cümlenin getirilen kaynaklara dayanıp dayanmadığı (grounding) ve belirtilen kaynağın gerçekten getirilen chunk'lardan biri olup olmadığı `rag_core.py` içinde doğrulanır. Doğrulamayı geçemeyen cevap kullanıcıya gösterilmez, yerine "yeterli bilgi yok" mesajı verilir.

Uygulama iki arayüzle çalışabilir: `gui_tkinter.py` (masaüstü penceresi) veya `app.py` (komut satırı). Her ikisi de aynı `rag_core` doğrulama mantığını kullanır; hangisini çalıştırırsanız çalıştırın halüsinasyon koruması aynıdır.

## Kurulum ve Çalıştırma Talimatları

### 1. Gerekli Kurulumlar
Projenin çalışması için bilgisayarınızda Python yüklü olmalıdır. Terminali açarak gereksinimleri yükleyin:

```bash
pip install -r requirements.txt
```

*(Not: `foundry_local_sdk` Microsoft dökümanlarındaki adımlara göre sisteme önceden kurulmuş/tanıtılmış olmalıdır.)*

### 2. Veritabanını Hazırlama
Asistanı çalıştırmadan önce verilerin vektörleştirilip SQLite'a aktarılması gerekir:
```bash
python main.py
```
*Bu işlem sırasında embedding modeli yerelde yoksa otomatik indirilir, internet hızınıza göre birkaç dakika sürebilir.*

### 3. Asistanı Başlatma
Veriler gömüldükten sonra iki arayüzden birini seçin:

**Masaüstü (Tkinter) arayüzü:**
```bash
python gui_tkinter.py
```
Bir masaüstü penceresi açılır. Üstteki listeden sorgulanacak bilgi tabanını seçip sorunuzu yazarak "Gönder" butonuna basabilirsiniz. Durum çubuğu "Sistem Hazır." olduğunda soru sorabilirsiniz.

**Komut satırı (CLI) arayüzü:**
```bash
python app.py
```
Terminalde "Sormak istediğin soru:" satırına yazıp Enter'a basın. Çıkmak için `çıkış` yazın.

### 4. Otomatik Test Aracı
Sistemin farklı durumlara (doğru bilgi, bulunmayan bilgi ve anlamsız girdi) nasıl tepki verdiğini görmek için:
```bash
python test_rag.py
```

## Tasarım Kararları ve Kısıtlamalar
- **Veritabanı Olarak SQLite:** Projenin taşınabilirliği, sadeliği ve tek bir bilgisayarda dış bağımlılık olmadan çalışabilmesi adına ağır vektör veritabanları yerine SQLite seçildi. Vektörler JSON formatında saklanıp Python ile işlendi.
- **Yerel Donanım Kısıtlamaları:** Modelin cihaz RAM ve VRAM donanımını kullanması sebebiyle, bağlam uzunluğu çok tutulduğunda `OnnxRuntimeGenAIException: Could not allocate the key-value cache buffer` gibi bellek hataları alınabilir. Bu sistemin değil, cihaz kapasitesinin bir limitidir.
- **Kod Seviyesinde Halüsinasyon Kontrolü:** Sağlık ve güvenlik konuları şakaya gelmeyeceği için, "bilgi dışına çıkma" kararı modelin inisiyatifine bırakılmadı. İki yazılımsal kapı uygulandı: **Katman 1** retrieval aşamasında benzerlik eşiğinin altındaki soruları modeli hiç çağırmadan reddeder (`database.py`); **Katman 2** üretilen cevabı, kullanıcıya gösterilmeden önce getirilen kaynaklara karşı grounding ve kaynak doğrulamasından geçirir (`rag_core.py`). Katı sistem prompt'u ise bu iki kod katmanının üstünde tamamlayıcı (ikincil) bir savunma olarak korunur.
- **Ortak Çekirdek (`rag_core.py`):** Sistem prompt'u, fallback mesajı, grounding eşiği ve Katman 2 doğrulama mantığı tek bir modülde tutulur. Böylece masaüstü ve komut satırı arayüzleri arasında bu mantık ayrışmaz (drift olmaz).
- **Grounding Eşiği Kalibrasyonu:** Katman 2'deki `GROUNDING_THRESHOLD` değeri (`rag_core.py`) veriye bağlıdır; çok yüksek tutulursa doğru cevaplar da reddedilebilir, çok düşük tutulursa halüsinasyon sızabilir. Gerçek sorularla ayarlanması gerekir.

---