"""
test_rag.py — RAG Asistanı Otomatik Test Aracı (Yeni Pipeline)

Gerçek uygulama akışını test eder:
  Katman 1: embed_query (talimatlı) → search_with_scores (yapısal, eşik 0.30)
  LLM: cevap üretimi (tamponda, ekrana yazmadan)
  Katman 2: is_answer_grounded (grounding + kaynak doğrulama)

3 farklı senaryo:
  1. Doğru Bilgi  : Veritabanında cevabı bulunan sorular
  2. Bilinmeyen   : Veritabanında cevabı OLMAYAN sorular
  3. Edge Case    : Boş / anlamsız / çok genel girdiler

Kullanım:
    python test_rag.py
"""

from foundry_local_sdk import Configuration, FoundryLocalManager
from database import DatabaseManager
from rag_core import (
    FALLBACK_MESSAGE,
    embed_query,
    prepare_generation,
    is_answer_grounded,
)

# ──────────────────────────────────────────────
# Test Senaryoları
# ──────────────────────────────────────────────

TEST_CASES = [
    # ── Kategori 1: Doğru bilgi (cevap veritabanında var) ──
    {
        "kategori": "Doğru Bilgi",
        "soru": "Depremde evde ne yapmalıyım?",
        "beklenen": "cevaplanmali",
        "aciklama": "Deprem güvenlik rehberinde ev içi bilgi mevcut.",
    },
    {
        "kategori": "Doğru Bilgi",
        "soru": "Mutfakta yağ alev alırsa ne yapmalıyım?",
        "beklenen": "cevaplanmali",
        "aciklama": "Yangın güvenlik rehberinde yağ yangını bilgisi mevcut.",
    },
    {
        "kategori": "Doğru Bilgi",
        "soru": "Sel anında araçtaysam ne yapmalıyım?",
        "beklenen": "cevaplanmali",
        "aciklama": "Su kaynaklı afetler rehberinde araçta sel bilgisi mevcut.",
    },
    {
        "kategori": "Doğru Bilgi",
        "soru": "Yılan ısırığına nasıl müdahale edilir?",
        "beklenen": "cevaplanmali",
        "aciklama": "İlk yardım rehberinde yılan ısırması bilgisi mevcut.",
    },
    {
        "kategori": "Doğru Bilgi",
        "soru": "Kolum kırılırsa ne yapmalıyım?",
        "beklenen": "cevaplanmali",
        "aciklama": "İlk yardım rehberinde kırık kol bilgisi mevcut.",
    },

    # ── Kategori 2: Bilinmeyen (veritabanında cevap yok) ──
    {
        "kategori": "Bilinmeyen",
        "soru": "Mars'a nasıl gidilir?",
        "beklenen": "bilmiyor",
        "aciklama": "Uzay seyahati veritabanında bulunmuyor.",
    },
    {
        "kategori": "Bilinmeyen",
        "soru": "Python'da dictionary nasıl kullanılır?",
        "beklenen": "bilmiyor",
        "aciklama": "Programlama soruları veritabanında bulunmuyor.",
    },

    # ── Kategori 3: Edge Case ──
    {
        "kategori": "Edge Case",
        "soru": "",
        "beklenen": "bos_sorgu",
        "aciklama": "Boş sorgu — sistem hata vermeden yönetmeli.",
    },
    {
        "kategori": "Edge Case",
        "soru": "asdfghjkl",
        "beklenen": "hata_yok",
        "aciklama": "Anlamsız girdi — sistem çökmemeli.",
    },
    {
        "kategori": "Edge Case",
        "soru": "Hayat nedir?",
        "beklenen": "hata_yok",
        "aciklama": "Çok genel/felsefi soru — sistem çökmemeli.",
    },
]


# ──────────────────────────────────────────────
# Ana test akışı
# ──────────────────────────────────────────────

def run_tests():
    print("=" * 60)
    print("  RAG Asistanı — Otomatik Test Aracı (Yeni Pipeline)")
    print("=" * 60)

    config = Configuration(app_name="rag_assistant")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    print("\n[1/3] Sohbet modeli yükleniyor...")
    model = manager.catalog.get_model("phi-3.5-mini")
    model.download()
    model.load()
    chat_client = model.get_chat_client()

    print("[2/3] Embedding modeli yükleniyor...")
    embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    embedding_model.download()
    embedding_model.load()
    embedding_client = embedding_model.get_embedding_client()

    print("[3/3] Veritabanına bağlanılıyor...")
    db = DatabaseManager("general_database.db")
    print("Tüm bileşenler hazır.\n")

    results = []
    total = len(TEST_CASES)

    for idx, tc in enumerate(TEST_CASES, 1):
        kategori = tc["kategori"]
        soru = tc["soru"]
        beklenen = tc["beklenen"]
        aciklama = tc["aciklama"]

        print(f"─── Test {idx}/{total}: [{kategori}] ───")
        print(f"  Soru      : '{soru}'")
        print(f"  Açıklama  : {aciklama}")

        status = "BAŞARILI ✅"
        yanit_ozet = ""
        retrieval_ozet = ""
        katman2_ozet = ""

        try:
            # Boş sorgu edge case
            if not soru.strip():
                retrieval_ozet = "(boş sorgu — retrieval atlandı)"
                yanit_ozet = "(boş sorgu — model çağrılmadı)"
                katman2_ozet = "(atlandı)"
                if beklenen == "bos_sorgu":
                    status = "BAŞARILI ✅"
                print(f"  Sonuç     : {status}")
                results.append({
                    "no": idx, "kategori": kategori, "soru": soru,
                    "status": status, "retrieval": retrieval_ozet,
                    "yanit": yanit_ozet, "katman2": katman2_ozet,
                })
                print()
                continue

            # ─── KATMAN 1: Talimatlı sorgu + yapısal retrieval ───
            query_vector = embed_query(embedding_client, soru)
            search_results = db.search_with_scores(query_vector, top_k=2)

            if not search_results:
                retrieval_ozet = "(hiçbir chunk eşiği geçemedi)"
                yanit_ozet = FALLBACK_MESSAGE
                katman2_ozet = "(atlandı — LLM çağrılmadı)"
                if beklenen == "bilmiyor" or beklenen == "hata_yok":
                    status = "BAŞARILI ✅"
                elif beklenen == "cevaplanmali":
                    status = "HATA ❌ (Geçerli soru reddedildi — Katman 1 çok sıkı)"
                print(f"  Retrieval : {retrieval_ozet}")
                print(f"  Yanıt     : {yanit_ozet}")
                print(f"  Sonuç     : {status}")
                print()
                results.append({
                    "no": idx, "kategori": kategori, "soru": soru,
                    "status": status, "retrieval": retrieval_ozet,
                    "yanit": yanit_ozet, "katman2": katman2_ozet,
                })
                continue

            # Retrieval sonuçlarını özetle
            top_chunk = search_results[0]
            retrieval_ozet = f"skor={top_chunk['score']:.3f} | {top_chunk['content'][:60]}..."

            # ─── LLM: Cevap üretimi (tamponda) ───
            messages, retrieved_vectors, allowed_sources = prepare_generation(soru, search_results)

            full_response = ""
            for chunk in chat_client.complete_streaming_chat(messages):
                if chunk.choices and chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
            full_response = full_response.strip()

            yanit_ozet = full_response[:180].replace("\n", " ")

            # ─── KATMAN 2: Grounding + kaynak doğrulama ───
            grounded = is_answer_grounded(
                full_response, retrieved_vectors, allowed_sources,
                embedding_client, db
            )
            katman2_ozet = "GEÇER ✓" if grounded else "REDDEDİLDİ ✗"

            # Nihai karar (uygulamanın vereceği cevap)
            nihai_cevap = full_response if grounded else FALLBACK_MESSAGE
            bilmiyor = FALLBACK_MESSAGE in nihai_cevap

            # Değerlendirme
            if beklenen == "cevaplanmali":
                if not bilmiyor and len(nihai_cevap.strip()) > 20:
                    status = "BAŞARILI ✅"
                else:
                    status = "HATA ❌ (Geçerli soru cevaplanamadı)"
            elif beklenen == "bilmiyor":
                if bilmiyor:
                    status = "BAŞARILI ✅"
                else:
                    status = "DİKKAT ⚠️ (Cevap üretildi — halüsinasyon olabilir)"
            elif beklenen == "hata_yok":
                status = "BAŞARILI ✅"

        except Exception as e:
            status = "HATA ❌"
            yanit_ozet = str(e)
            katman2_ozet = "(hata)"

        print(f"  Retrieval : {retrieval_ozet}")
        print(f"  Katman 2  : {katman2_ozet}")
        print(f"  Yanıt     : {yanit_ozet}")
        print(f"  Sonuç     : {status}")
        print()

        results.append({
            "no": idx, "kategori": kategori, "soru": soru,
            "status": status, "retrieval": retrieval_ozet,
            "yanit": yanit_ozet, "katman2": katman2_ozet,
        })

    # ── Özet Rapor ──
    print("=" * 60)
    print("  TEST SONUÇ RAPORU")
    print("=" * 60)
    print(f"{'No':<4} {'Kategori':<16} {'Soru':<40} {'Sonuç'}")
    print("-" * 100)

    basarili = dikkat = hata = 0

    for r in results:
        soru_kisa = r["soru"][:37] + "..." if len(r["soru"]) > 37 else r["soru"]
        if not soru_kisa:
            soru_kisa = "(boş)"
        print(f"{r['no']:<4} {r['kategori']:<16} {soru_kisa:<40} {r['status']}")

        if "BAŞARILI" in r["status"]:
            basarili += 1
        elif "DİKKAT" in r["status"]:
            dikkat += 1
        else:
            hata += 1

    print("-" * 100)
    print(f"Toplam: {len(results)} test | ✅ Başarılı: {basarili} | ⚠️ Dikkat: {dikkat} | ❌ Hata: {hata}")
    print("=" * 60)

    # Temizlik
    model.unload()
    embedding_model.unload()
    db.close()


if __name__ == "__main__":
    run_tests()