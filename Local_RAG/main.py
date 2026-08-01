import os
import json
import traceback
from database import DatabaseManager
from foundry_local_sdk import Configuration, FoundryLocalManager

def main():
    config = Configuration(app_name="Local_RAG")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    dataset_path = "survival_database.json"

    # Oluşturulacak veritabanı dosyaları
    db_configs = {
        "general_database.db": None,  # Hepsini içerir
        "disaster_database.db": [
            "Deprem Güvenlik Rehberi", 
            "Su Kaynaklı Afetler Rehberi", 
            "Doğa Olayları Güvenlik Rehberi"
        ],
        "emergency_database.db": [
            "Yangın Güvenlik Rehberi", 
            "İlk Yardım Rehberi", 
            "Nükleer ve Radyolojik Tehlikeler Rehberi"
        ],
        "survival_database.db": [
            "Doğada Hayatta Kalma Rehberi", 
            "Afet Hazırlık ve Genel Güvenlik Rehberi"
        ]
    }

    print("--- RAG Çoklu Veritabanı Yükleme Aracı Başlatıldı ---")
    
    # Eski dosyaları temizle (mükerrer kayıt olmaması için)
    for db_name in db_configs.keys():
        if os.path.exists(db_name):
            try:
                os.remove(db_name)
                print(f"Eski veritabanı temizlendi: {db_name}")
            except Exception as e:
                print(f"Uyarı: {db_name} temizlenemedi (kilitli olabilir): {e}")

    print("Vektör (Embedding) modeli yükleniyor, lütfen bekleyin...")

    try:
        embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")
        embedding_model.download()
        embedding_model.load()
        client = embedding_model.get_embedding_client()
        print("Model başarıyla yüklendi ve kullanıma hazır!\n")

        # Bağlantıları oluştur
        databases = {name: DatabaseManager(name) for name in db_configs.keys()}

        with open(dataset_path, mode='r', encoding='utf-8') as file:
            data = json.load(file)

            for i, item in enumerate(data):
                try:
                    text_chunk = item['chunk_text']
                    kaynak = item.get('kaynak', 'Bilinmeyen Kaynak')
                    final_chunk = f"{text_chunk} ({kaynak})"
                    
                    print(f"Madde {i + 1} ({kaynak}) işleniyor...")

                    # Vektörü (Embedding) al
                    response = client.generate_embeddings([final_chunk])
                    vector = response.data[0].embedding
                    vector_json = json.dumps(vector)

                    # 1. Genel veritabanına her koşulda kaydet
                    databases["general_database.db"].insert_data(final_chunk, vector_json)

                    # 2. İlgili spesifik veritabanlarına kaydet
                    for db_name, categories in db_configs.items():
                        if categories and kaynak in categories:
                            databases[db_name].insert_data(final_chunk, vector_json)

                except Exception as inner_e:
                    print(f"\nMadde {i + 1} işlenirken hata oluştu: {inner_e}")

            print("\nTüm veriler başarıyla gömüldü ve veritabanlarına ayrıştırıldı:")
            for db_name, db_manager in databases.items():
                # Toplam kayıt sayısını göster
                db_manager.cursor.execute("SELECT COUNT(*) FROM documents")
                count = db_manager.cursor.fetchone()[0]
                print(f" - {db_name}: {count} adet döküman parçası.")

        # Veritabanlarını kapat
        for db_manager in databases.values():
            db_manager.close()

    except Exception as e:
        print("\nKritik bir hata oluştu:")
        traceback.print_exc()
    finally:
        try:
            embedding_model.unload()
        except:
            pass

if __name__ == "__main__":
    main()