"""
diagnose.py — TEK SEFERLİK retrieval değerlendirme aracı (frontend'lere dokunmaz).

Amaç: Retrieval'ın alakalı/alakasız ayrımını SAYILARLA ölçmek ve iki talimat
varyantını (domain-ağırlıklı vs nötr) yan yana karşılaştırmak.

Her test için doğru chunk'ın kaçıncı sırada geldiğini ve eşiği geçip geçmediğini
gösterir. Böylece hem yanlış reddetme (Arıza A) hem yanlış-chunk (Arıza B)
tek bakışta görülür.

Kullanım:  python diagnose.py
"""

from foundry_local_sdk import Configuration, FoundryLocalManager
from database import DatabaseManager

THRESHOLD = 0.30  # database.search_with_scores ile aynı

# Karşılaştırılacak talimat varyantları
TASK_VARIANTS = {
    "A_domain (mevcut)": "Given a user question about emergency, disaster and survival, retrieve the passage that answers it",
    "B_neutral":         "Given a user question, retrieve the passage that best answers it",
}

# (db dosyası, soru, beklenen_chunk_başlığı | None=reddedilmeli)
CASES = [
    ("emergency_database.db", "kolum kırılırsa napmalıyım",                         "Kırık Kola"),
    ("emergency_database.db", "yaraya nasıl müdahale edilir",                       "Kritik İlk Yardım"),
    ("emergency_database.db", "yılan ısırığına nasıl müdahale edilir",              "Yılan Isırma"),
    ("disaster_database.db",  "selde ne yapmalıyız",                                "Sel"),
    ("disaster_database.db",  "çığ riski bulunan bir yerde ne gibi önlemler alabiliriz", "Çığ"),
    ("disaster_database.db",  "depremde okuldaysak ne yapmalıyız",                  "Okul"),
    ("survival_database.db",  "ormanda nasıl sıcak kalınır",                        "Barınak"),
    ("survival_database.db",  "hangi bitkiler yenilebilir",                         "Yenilebilir"),
    # Reddedilmesi gereken (negatif) durumlar:
    ("survival_database.db",  "ayıyla karşılaşırsak ne yapmalıyız",                 None),
    ("general_database.db",   "nasılsın",                                           None),
    ("general_database.db",   "Mars'a nasıl gidilir",                               None),
]


def instructed(task, query):
    return f"Instruct: {task}\nQuery: {query}"


def ranked(embedding_client, db, query_text):
    resp = embedding_client.generate_embeddings([query_text])
    qvec = resp.data[0].embedding
    db.cursor.execute("SELECT content, embedding FROM documents")
    import json
    scored = []
    for content, emb in db.cursor.fetchall():
        scored.append((db.cosine_similarity(qvec, json.loads(emb)), content))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def title(content):
    return content.split(":")[0][:34]


if __name__ == "__main__":
    config = Configuration(app_name="rag_assistant")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    print("Embedding modeli yükleniyor...")
    embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    embedding_model.download()
    embedding_model.load()
    embedding_client = embedding_model.get_embedding_client()
    print("Hazır.\n")

    db_cache = {}
    def get_db(name):
        if name not in db_cache:
            db_cache[name] = DatabaseManager(name)
        return db_cache[name]

    for vname, task in TASK_VARIANTS.items():
        print("=" * 70)
        print(f"  TALİMAT VARYANTI: {vname}")
        print(f"  \"{task}\"")
        print("=" * 70)

        hit_top1 = 0
        hit_top2 = 0
        pos_total = 0
        neg_false_accept = 0
        neg_total = 0

        for db_name, query, expected in CASES:
            db = get_db(db_name)
            
            # DB'deki satır sayısını al
            db.cursor.execute("SELECT COUNT(*) FROM documents")
            row_count = db.cursor.fetchone()[0]
            
            scored = ranked(embedding_client, db, instructed(task, query))
            top3 = scored[:3]

            print(f"\n[{db_name} (satır sayısı: {row_count})]  \"{query}\"")
            if expected:
                pos_total += 1
                # DB'de gerçekten bu chunk var mı kontrol et
                db.cursor.execute("SELECT content FROM documents")
                db_contents = [r[0] for r in db.cursor.fetchall()]
                exists_in_db = any(expected.lower() in c.lower() for c in db_contents)
                
                exp_pos = next((i for i, (s, c) in enumerate(scored) if expected.lower() in c.lower()), None)
                exp_score = scored[exp_pos][0] if exp_pos is not None else None
                if exp_pos == 0:
                    hit_top1 += 1
                if exp_pos is not None and exp_pos < 2 and exp_score >= THRESHOLD:
                    hit_top2 += 1
                exp_str = f"sıra={exp_pos}  skor={exp_score:.3f}" if exp_pos is not None else "chunk BULUNAMADI"
                print(f"   beklenen '{expected}' (DB'de var mı: {'Evet' if exists_in_db else 'Hayır'}): {exp_str}")
            else:
                neg_total += 1
                if len(scored) > 0:
                    top_score = scored[0][0]
                    if top_score >= THRESHOLD:
                        neg_false_accept += 1
                    print(f"   (reddedilmeli) en yuksek skor={top_score:.3f} "
                          f"{'<- ESIGI GECIYOR (kotu)' if top_score >= THRESHOLD else 'OK altinda'}")
                else:
                    print("   (reddedilmeli) veritabanı boş, skor yok.")

            for s, c in top3:
                mark = "GEÇER" if s >= THRESHOLD else "  -  "
                print(f"      {s:.3f} [{mark}] {title(c)}")

        print(f"\n  --- {vname} OZET ---")
        print(f"    Pozitif (dogru chunk top-1)      : {hit_top1}/{pos_total}")
        print(f"    Pozitif (dogru chunk top-2 & >= {THRESHOLD}) : {hit_top2}/{pos_total}")
        print(f"    Negatif (yanlis kabul, esik gecen): {neg_false_accept}/{neg_total}")
        print()

    for db in db_cache.values():
        db.close()
    embedding_model.unload()
    print("Bitti.")