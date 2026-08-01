import sqlite3
import json
import math
import re

class DatabaseManager:
    def __init__(self, db_name="survival_database.db"):
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.create_table()

    def create_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT,
                embedding TEXT
            )
        ''')
        self.conn.commit()

    def insert_data(self, text, embedding):
        self.cursor.execute('INSERT INTO documents (content, embedding) VALUES (?, ?)', (text, embedding))
        self.conn.commit()

    # ──────────────────────────────────────────────────────────────
    # KATMAN 1 — Yapısal retrieval (giriş kapısı)
    # Tüm arayüzler (gui_tkinter, app, test_rag) bu metodu kullanır.
    # Metin yerine yapısal veri döner, böylece kırılgan string
    # karşılaştırmaları ortadan kalkar ve LLM'e gitmeden önce
    # "cevaplanabilir mi?" kararı kod seviyesinde verilir.
    # ──────────────────────────────────────────────────────────────
    def search_with_scores(self, query_vector, top_k=2, threshold=0.30):
        """
        Kosinüs benzerliğiyle en yakın chunk'ları bulur ve YAPISAL veri döndürür.

        NOT (eşik): 0.30 değeri, nötr talimat önekli (Instruct) sorgu embedding'lerine
        göre belirlenmiştir. Bu düşük eşik, geçerli soruların yanlışlıkla reddedilmesini
        önlemek için bilinçli olarak geniş tutulmuştur; asıl halüsinasyon savunması
        Katman 2 (rag_core.py) tarafından sağlanır.

        Dönüş: eşiği geçen chunk'ların listesi (skora göre azalan sırada):
            [
              {"score": float, "content": str, "source": str, "vector": list[float]},
              ...
            ]
        Hiçbir chunk eşiği geçemezse BOŞ LİSTE döner → çağıran LLM'i hiç çağırmaz.
        'vector' alanı, Katman 2 (grounding) doğrulamasında yeniden hesaplamaya
        gerek kalmadan kullanılır.
        """
        self.cursor.execute('SELECT content, embedding FROM documents')
        rows = self.cursor.fetchall()

        if not rows:
            return []

        scored = []
        for content, embedding_str in rows:
            doc_vector = json.loads(embedding_str)
            similarity = self.cosine_similarity(query_vector, doc_vector)
            if similarity >= threshold:
                scored.append({
                    "score": similarity,
                    "content": content,
                    "source": self.extract_source(content),
                    "vector": doc_vector,
                })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    @staticmethod
    def extract_source(content):
        """Chunk metninin sonundaki '(Kaynak Adı)' etiketini ayıklar.

        Veri, main.py içinde 'metin (kaynak)' biçiminde saklandığı için
        kaynak adı metnin en sonundaki parantez grubudur.
        """
        match = re.search(r'\(([^()]+)\)\s*$', content.strip())
        return match.group(1).strip() if match else ""

    def cosine_similarity(self, v1, v2):
        # Burada 'a' ve 'b' değişkenleri zip içinde tanımlanır
        dot_product = sum(a * b for a, b in zip(v1, v2))

        # Norm hesapları
        norm_a = math.sqrt(sum(x * x for x in v1))
        norm_b = math.sqrt(sum(y * y for y in v2))

        if norm_a == 0 or norm_b == 0:
            return 0
        return dot_product / (norm_a * norm_b)

    def debug_database(self):
        self.cursor.execute('SELECT content FROM documents LIMIT 1')
        row = self.cursor.fetchone()
        if row:
            print(f"DEBUG: Veritabanı boş değil. İlk içerik: {row[0][:50]}")
        else:
            print("DEBUG: HATA! Veritabanı boş görünüyor.")

    def close(self):
        self.conn.close()