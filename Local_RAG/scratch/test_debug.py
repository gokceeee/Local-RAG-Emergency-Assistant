import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database import DatabaseManager
from rag_core import embed_query, prepare_generation, is_answer_grounded, verify_grounding, verify_source, split_sentences
from foundry_local_sdk import Configuration, FoundryLocalManager
import re

def debug_query(user_query, db, embedding_client, chat_client):
    print(f"\n--- DEBUGGING QUERY: '{user_query}' ---")
    
    query_vector = embed_query(embedding_client, user_query)
    results = db.search_with_scores(query_vector, top_k=2)

    print(f"Katman 1 Results (threshold=0.42):")
    if not results:
        print("  [ERROR] No results passed threshold!")
        return

    for i, r in enumerate(results):
        print(f"  {i+1}. Score: {r['score']:.4f} | Source: {r['source']} | Content: {r['content'][:100]}...")

    messages, retrieved_vectors, allowed_sources = prepare_generation(user_query, results)

    full_response = ""
    for chunk in chat_client.complete_streaming_chat(messages):
        if chunk.choices and chunk.choices[0].delta.content:
            full_response += chunk.choices[0].delta.content
    full_response = full_response.strip()

    print(f"\nLLM Raw Response:\n{full_response}")

    source_ok = verify_source(full_response, allowed_sources)
    grounded, unsupported = verify_grounding(full_response, retrieved_vectors, embedding_client, db)

    print(f"\nKatman 2 Verification (grounding_threshold=0.50):")
    print(f"  Source OK: {source_ok}")
    print(f"  Grounded: {grounded}")
    if not grounded:
        print(f"  Unsupported sentences: {unsupported}")
        for sentence in split_sentences(full_response):
            clean = re.sub(r'\([^()]*\)', '', sentence).strip()
            if len(clean) < 8:
                continue
            resp = embedding_client.generate_embeddings([clean])
            s_vec = resp.data[0].embedding
            similarities = [db.cosine_similarity(s_vec, cv) for cv in retrieved_vectors]
            print(f"    Sentence: '{clean}'")
            print(f"      Similarities with context: {similarities} (Max: {max(similarities, default=0.0):.4f})")

if __name__ == "__main__":
    config = Configuration(app_name="rag_assistant")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    embedding_model.load()
    embedding_client = embedding_model.get_embedding_client()

    model = manager.catalog.get_model("phi-3.5-mini")
    model.load()
    chat_client = model.get_chat_client()

    db = DatabaseManager("general_database.db")

    queries = [
        "heimlich manevrası nasıl yapılır",
        "Ciddi ve derin kanamalarda ne yapılmalıdır?",
        "Sel uyarısı yapıldığında veya sel başladığında yüksek yerlere mi çıkılmalıdır?",
        "Çığ geliyorsa yamaçta nasıl korunmalıyız?",
        "Nükleer patlama anında sığınakta ne kadar süre kalınmalıdır?"
    ]
    for q in queries:
        debug_query(q, db, embedding_client, chat_client)

    db.close()
    embedding_model.unload()
    model.unload()
