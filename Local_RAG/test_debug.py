import json
import re
import sys
from foundry_local_sdk import Configuration, FoundryLocalManager
from database import DatabaseManager
import rag_core

def test_query(user_query):
    config = Configuration(app_name="rag_assistant") # use same config as gui_tkinter
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    model = manager.catalog.get_model("phi-3.5-mini")
    model.download()
    model.load()
    chat_client = model.get_chat_client()

    embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    embedding_model.download()
    embedding_model.load()
    embedding_client = embedding_model.get_embedding_client()

    db = DatabaseManager("general_database.db")

    print(f"\nSorgu: {user_query}")
    query_vector = rag_core.embed_query(embedding_client, user_query)
    
    # Katman 1 retrieval
    results = db.search_with_scores(query_vector, top_k=2)

    if not results:
        print("Katman 1 Red: Hiçbir chunk eşiği geçemedi.")
        return

    print("Katman 1 Kabul: Eşleşen chunklar:")
    for r in results:
        print(f" - {r['content'][:100]}... (Skor: {r['score']:.4f})")

    messages, retrieved_vectors, allowed_sources = rag_core.prepare_generation(user_query, results)

    full_response = ""
    for chunk in chat_client.complete_streaming_chat(messages):
        if chunk.choices and chunk.choices[0].delta.content:
            full_response += chunk.choices[0].delta.content
    full_response = full_response.strip()

    print("\n--- LLM Ham Cevap ---")
    print(full_response)
    print("---------------------")

    # Katman 2 Detayı
    unsupported = []
    for sentence in rag_core.split_sentences(full_response):
        clean = re.sub(r'\([^()]*\)', '', sentence).strip()
        if len(clean) < 8:
            continue
        resp = embedding_client.generate_embeddings([clean])
        sentence_vector = resp.data[0].embedding

        similarities = [db.cosine_similarity(sentence_vector, cv) for cv in retrieved_vectors]
        best = max(similarities, default=0.0)
        print(f"Cümle: \"{clean}\" -> En yüksek benzerlik: {best:.4f}")
        if best < rag_core.GROUNDING_THRESHOLD:
            unsupported.append((sentence, best))

    print(f"\nKatman 2 Sonucu: {'BAŞARILI' if not unsupported else 'REDDEDİLDİ'}")
    if unsupported:
        print("Desteklenmeyen cümleler:")
        for s, score in unsupported:
            print(f" - \"{s}\" (Skor: {score:.4f} < {rag_core.GROUNDING_THRESHOLD})")

if __name__ == "__main__":
    test_query("Deprem/afet çantasında en az 72 saat yetecek neler bulunmalıdır")
    test_query("Doğada hangi yabani otlar güvenle yenilebilir?")
