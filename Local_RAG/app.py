from foundry_local_sdk import Configuration, FoundryLocalManager
from database import DatabaseManager
from rag_core import FALLBACK_MESSAGE, embed_query, prepare_generation, is_answer_grounded


def main():
    config = Configuration(app_name="rag_assistant")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    # 1. Sohbet Modeli (Phi-3.5 Mini)
    print("Sohbet modeli kontrol ediliyor...")
    model = manager.catalog.get_model("phi-3.5-mini")

    print("Model indiriliyor/yükleniyor (İlk kez açılıyorsa biraz sürebilir)...")
    model.download()
    model.load()
    chat_client = model.get_chat_client()
    print("Sohbet modeli hazır.")

    # 2. Embedding Modeli ve İstemcisi
    print("Vektör (Embedding) modeli yükleniyor...")
    embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    embedding_model.load()
    embedding_client = embedding_model.get_embedding_client()
    print("Tüm modeller hazır.")

    # 3. Veritabanına bağlan
    db = DatabaseManager("survival_database.db")

    print("\n--- RAG Asistanı Hazır! (Çıkmak için 'çıkış' yazın) ---")

    while True:
        user_query = input("\nSormak istediğin soru: ").strip()

        if not user_query:
            print("Lütfen boş bırakmayın, bir soru sorun.")
            continue

        if user_query.lower() in ["çıkış", "kapat", "exit"]:
            print("Asistan kapatılıyor. Görüşmek üzere!")
            break

        try:
            # Sorguyu vektöre çevir (qwen3 talimat önekiyle)
            query_vector = embed_query(embedding_client, user_query)

            # ─── KATMAN 1: Yapısal retrieval + giriş kapısı ───
            results = db.search_with_scores(query_vector, top_k=2)

            # Eşiği geçen chunk yoksa LLM'i hiç çağırmadan olumsuz cevabı dön.
            if not results:
                print(f"Asistan: {FALLBACK_MESSAGE}")
                continue

            # Ortak çekirdek: prompt + doğrulama için gerekenleri hazırla.
            messages, retrieved_vectors, allowed_sources = prepare_generation(user_query, results)

            # ─── A YOLU: Cevabı EKRANA yazmadan tamponda topla ───
            # (Doğrulama bitmeden kullanıcı yanlış cevabı görmesin.)
            print("Asistan: (yanıt üretiliyor ve kontrol ediliyor...)", end="\r", flush=True)
            full_response = ""
            for chunk in chat_client.complete_streaming_chat(messages, temperature=0.1):
                if chunk.choices and chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
            full_response = full_response.strip()

            # ─── KATMAN 2: Generation SONRASI kod seviyesinde denetim ───
            ok = is_answer_grounded(
                full_response, retrieved_vectors, allowed_sources, embedding_client, db
            )

            # Satırı temizle ve nihai cevabı yaz.
            print(" " * 60, end="\r")
            if ok:
                print(f"Asistan: {full_response}")
            else:
                # Doğrulamayı geçemedi → yanlış cevabı gösterme.
                print(f"Asistan: {FALLBACK_MESSAGE}")

        except Exception as e:
            print(f"\nSorgu işlenirken bir hata oluştu: {e}")

    # Döngü bittiğinde temizlik yap
    model.unload()
    embedding_model.unload()
    db.close()


if __name__ == "__main__":
    main()