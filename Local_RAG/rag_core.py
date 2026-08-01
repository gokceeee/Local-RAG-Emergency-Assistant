"""
rag_core.py — RAG çekirdek mantığı (tek kaynak).

Hem gui_tkinter.py (masaüstü) hem app.py (komut satırı) bu modülü kullanır.
Böylece sistem prompt'u, fallback mesajı, grounding eşiği ve Katman 2
doğrulama mantığı tek bir yerde durur; arayüzler arasında ayrışmaz (drift olmaz).

Katman 1 (retrieval giriş kapısı) database.py -> search_with_scores içindedir.
Katman 2 (generation sonrası doğrulama) bu dosyadadır.
"""

import re

# ──────────────────────────────────────────────────────────────
# Sabitler (tek kaynak)
# ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an emergency and disaster safety assistant. "
    "Your task is to answer the user's question in clear, natural Turkish using ONLY the facts provided in the 'Bilgi' section below. "
    "Do not hallucinate, do not make things up, and do not use any external knowledge. "
    "If the 'Bilgi' text does not contain the exact, direct answer to the user's specific query, reply with EXACTLY this sentence and nothing else: "
    "'Üzgünüm, bu soruyu yanıtlamak için veritabanımda yeterli bilgi bulunmuyor.' "
    "Do NOT append any source name if you reply with the fallback message. "
    "\n\nStudy these examples to understand when to answer and when to reject:\n"
    "Example 1 (VALID Match - Answer is present in the text):\n"
    "Bilgi:\n"
    "- Kritik İlk Yardım Müdahaleleri (Boğulma, Kanama, Yanık): Yemek yerken nefes borusu tıkanan (hiç nefes alamayan ve moraran) birine derhal Heimlich manevrası uygulanmalıdır: Hastanın arkasına geçilir, yumruk yapılan el göbek deliğinin üstüne yerleştirilir ve içe-yukarı doğru kuvvetlice baskı yapılır. (İlk Yardım Rehberi)\n"
    "Kullanıcının sorusu: heimlich manevrası nasıl yapılır\n"
    "Response: Yemek yerken nefes borusu tıkanan birine derhal Heimlich manevrası uygulanmalıdır: Hastanın arkasına geçilir, yumruk yapılan el göbek deliğinin üstüne yerleştirilir ve içe-yukarı doğru kuvvetlice baskı yapılır. (İlk Yardım Rehberi)\n\n"
    "Example 2 (VALID Match - Answer is present in the text):\n"
    "Bilgi:\n"
    "- Kritik İlk Yardım Müdahaleleri (Boğulma, Kanama, Yanık): Ciddi ve derin kanamalarda temiz bir bezle kanayan bölgenin üzerine doğrudan, sıkıca baskı uygulanmalı ve bölge kalp seviyesinden yukarıda tutulmalıdır. (İlk Yardım Rehberi)\n"
    "Kullanıcının sorusu: Ciddi ve derin kanamalarda ne yapılmalıdır?\n"
    "Response: Ciddi ve derin kanamalarda temiz bir bezle kanayan bölgenin üzerine doğrudan, sıkıca baskı uygulanmalı ve bölge kalp seviyesinden yukarıda tutulmalıdır. (İlk Yardım Rehberi)\n\n"
    "Example 3 (INVALID Match - Different topic, no answer):\n"
    "Bilgi:\n"
    "- Doğada Barınak Yapımı: Doğada hayatta kalırken rüzgar ve yağmurdan korunmak için geçici barınak yapılmalıdır. (Doğada Hayatta Kalma Rehberi)\n"
    "Kullanıcının sorusu: kesik yaraya nasıl müdahale edilir\n"
    "Response: Üzgünüm, bu soruyu yanıtlamak için veritabanımda yeterli bilgi bulunmuyor.\n\n"
    "Example 4 (INVALID Match - Different topic, no answer):\n"
    "Bilgi:\n"
    "- Yılan Isırmasında İlk Yardım: Yılan ısırmasında hasta sakin tutulmalı ve zehrin yayılmasını engellemek için hareket ettirilmemelidir. (İlk Yardım Rehberi)\n"
    "Kullanıcının sorusu: yanığa ne sürülür\n"
    "Response: Üzgünüm, bu soruyu yanıtlamak için veritabanımda yeterli bilgi bulunmuyor.\n\n"
    "If the 'Bilgi' text has the answer, write a detailed and helpful response in Turkish. "
    "Begin directly with the answer itself. Do NOT open with any preamble or filler such as "
    "'Kullanıcının sorusunu yanıtlamak için', 'Sorunuza göre' or similar; go straight to the content. "
    "At the very end of your response, append the source in parentheses. "
    "Do NOT put the response text in parentheses. Only put the source name in parentheses."
)

# Tek yerden yönetilen fallback mesajı.
FALLBACK_MESSAGE = "Üzgünüm, bu soruyu yanıtlamak için veritabanımda yeterli bilgi bulunmuyor."

# Katman 2 — cevap cümlesi ile kaynak chunk arasında beklenen minimum kosinüs benzerliği.
# ÖNEMLİ: Bu değer veriye bağlıdır ve gerçek sorularla kalibre edilmelidir.
GROUNDING_THRESHOLD = 0.32

# qwen3-embedding ASİMETRİK bir modeldir: SORGULARA talimat öneki eklenir,
# BELGELERE eklenmez (belgeler main.py'da ham gömüldü). Bu önek olmadan doğru
# eşleşmelerin benzerlik skoru düşer ve Katman 1 soruyu yanlışlıkla reddeder.
QUERY_TASK = "Given a user question, retrieve the passage that best answers it"


def embed_query(embedding_client, user_query):
    """Sorguyu qwen3 talimat önekiyle embed eder (retrieval için).
    Belgeler ham gömüldüğünden önek YALNIZCA sorguya uygulanır."""
    instructed = f"Instruct: {QUERY_TASK}\nQuery: {user_query}"
    resp = embedding_client.generate_embeddings([instructed])
    return resp.data[0].embedding


# ──────────────────────────────────────────────────────────────
# Prompt / bağlam hazırlığı
# ──────────────────────────────────────────────────────────────

def prepare_generation(user_query, results):
    """
    Katman 1'den gelen yapısal sonuçları (search_with_scores çıktısı) alıp
    LLM çağrısı ve Katman 2 doğrulaması için gerekenleri hazırlar.

    Dönüş: (messages, retrieved_vectors, allowed_sources)
    """
    context = "\n".join(f"- {r['content']}" for r in results)
    user_prompt = (
        f"Context (Bilgi):\n{context}\n\n"
        f"Question (Soru): {user_query}\n\n"
        f"Instruction: Carefully read the Context. Answer the Question in clear Turkish using ONLY the facts from the Context. "
        f"If the Context does not directly answer the question, you must refuse to answer by replying with EXACTLY: '{FALLBACK_MESSAGE}'"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    retrieved_vectors = [r["vector"] for r in results]
    allowed_sources = {r["source"] for r in results if r["source"]}
    return messages, retrieved_vectors, allowed_sources


# ──────────────────────────────────────────────────────────────
# KATMAN 2 — Kod seviyesinde halüsinasyon denetimi (generation SONRASI)
# ──────────────────────────────────────────────────────────────

def split_sentences(text):
    """Cevabı kaba biçimde cümlelere böler (Türkçe uyumlu)."""
    parts = re.split(r'(?<=[.!?…])\s+|\n+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def verify_grounding(answer, retrieved_vectors, embedding_client, db, threshold=GROUNDING_THRESHOLD):
    """
    KATMAN 2a — Grounding / faithfulness kontrolü.
    Cevabın her cümlesinin, bağlam olarak verilen chunk'lara dayanıp
    dayanmadığını embedding benzerliğiyle kod seviyesinde denetler.

    Dönüş: (is_grounded: bool, unsupported: list[str])
    """
    unsupported = []
    for sentence in split_sentences(answer):
        # Parantez içi kaynak etiketini çıkararak sadece asıl iddiayı ölç.
        clean = re.sub(r'\([^()]*\)', '', sentence).strip()
        if len(clean) < 8:  # neredeyse boş / yalnızca kaynak → atla
            continue

        resp = embedding_client.generate_embeddings([clean])
        sentence_vector = resp.data[0].embedding

        best = max(
            (db.cosine_similarity(sentence_vector, cv) for cv in retrieved_vectors),
            default=0.0,
        )
        if best < threshold:
            unsupported.append(sentence)

    return (len(unsupported) == 0), unsupported


def verify_source(answer, allowed_sources):
    """
    KATMAN 2b — Kaynak doğrulama (deterministik, çok ucuz).
    Model, getirilen chunk'larda OLMAYAN bir kaynak uydurduysa yakalar.

    Kural:
      - Cevapta hiç kaynak (parantez) yoksa → True (2a grounding'e güvenilir).
      - Belirtilen kaynaklardan en az biri getirilen chunk'ların kaynağıysa → True.
      - Belirtilen tüm kaynaklar uydurmaysa → False.
    """
    cited = re.findall(r'\(([^()]+)\)', answer)
    if not cited:
        return True

    allowed_norm = {s.strip().lower() for s in allowed_sources}
    if any(c.strip().lower() in allowed_norm for c in cited):
        return True
    return False


def is_answer_grounded(answer, retrieved_vectors, allowed_sources, embedding_client, db):
    """
    Katman 2'nin birleşik kararı: cevap kullanıcıya gösterilebilir mi?
    Hem kaynak doğrulaması hem grounding geçmeliyse True döner.
    """
    if not answer:
        return False
    if FALLBACK_MESSAGE in answer:
        return True
    if not verify_source(answer, allowed_sources):
        return False
    grounded, _unsupported = verify_grounding(answer, retrieved_vectors, embedding_client, db)
    return grounded