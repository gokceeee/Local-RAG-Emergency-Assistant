import customtkinter as ctk
import threading
from foundry_local_sdk import Configuration, FoundryLocalManager
from database import DatabaseManager
from rag_core import (
    SYSTEM_PROMPT,
    FALLBACK_MESSAGE,
    embed_query,
    prepare_generation,
    is_answer_grounded
)

# Tema ve Renk Ayarları (Modern Koyu Tema)
ctk.set_appearance_mode("dark")  # "light", "dark", "system"
ctk.set_default_color_theme("blue")  # "blue", "green", "dark-blue"

# Modeller ve Veritabanı Değişkenleri
config = Configuration(app_name="rag_assistant")
FoundryLocalManager.initialize(config)
manager = FoundryLocalManager.instance

model = None
chat_client = None
embedding_model = None
embedding_client = None
db = None


class ModernRAGGui(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Pencere Özellikleri
        self.title("RAG Acil Durum Asistanı")
        self.geometry("600x700")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Veritabanı Dosya Eşleştirmesi
        self.db_mapping = {
            "Genel Bilgi Tabanı": "general_database.db",
            "Doğal Afetler": "disaster_database.db",
            "Acil Durum & İlk Yardım": "emergency_database.db",
            "Doğada Hayatta Kalma": "survival_database.db"
        }

        # Ana Panel
        self.main_frame = ctk.CTkFrame(self, corner_radius=15)
        self.main_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_rowconfigure(4, weight=1)  # Sohbet geçmişi büyüyecek
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Üst Başlık (Yeniden Tasarlanan Premium Başlık)
        self.title_label = ctk.CTkLabel(
            self.main_frame,
            text="ACİL DURUM VE AFET ASİSTANI",
            font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"),
            text_color="#00ffd2"  # Premium Neon Turkuaz
        )
        self.title_label.grid(row=0, column=0, padx=20, pady=(20, 5), sticky="ew")

        # Veritabanı Seçici Combobox
        self.db_selector_label = ctk.CTkLabel(
            self.main_frame,
            text="Sorgulanacak Bilgi Tabanı Seçin:",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color="#8892b0"
        )
        self.db_selector_label.grid(row=1, column=0, padx=20, pady=(5, 0), sticky="w")

        self.db_selector = ctk.CTkComboBox(
            self.main_frame,
            values=list(self.db_mapping.keys()),
            font=ctk.CTkFont(family="Segoe UI", size=12),
            height=32,
            corner_radius=8,
            command=self.change_database
        )
        self.db_selector.grid(row=2, column=0, padx=20, pady=(2, 10), sticky="ew")
        self.db_selector.set("Genel Bilgi Tabanı")

        # Premium Arayüz Ayırıcı Çizgi (Separator)
        self.separator = ctk.CTkFrame(self.main_frame, height=2, fg_color="#00ffd2")
        self.separator.grid(row=3, column=0, padx=20, pady=(0, 10), sticky="ew")

        # Sohbet Geçmişi Alanı
        self.chat_history = ctk.CTkTextbox(
            self.main_frame,
            font=ctk.CTkFont(family="Segoe UI", size=13),
            corner_radius=10,
            border_width=1,
            border_color="#565b5e"
        )
        self.chat_history.grid(row=4, column=0, padx=20, pady=10, sticky="nsew")

        # Gönderici isimlerini belirginleştirmek için stil tag'leri
        self.chat_history.tag_config("siz", foreground="#60a5fa")
        self.chat_history.tag_config("asistan", foreground="#10b981")

        self.chat_history.insert("end", "Asistan: ", "asistan")
        self.chat_history.insert("end", "Merhaba! Sormak istediğiniz acil durum sorusunu yazıp 'Gönder' butonuna basabilirsiniz.\n\n")
        self.chat_history.configure(state="disabled")

        # Alt Giriş Paneli
        self.input_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.input_frame.grid(row=5, column=0, padx=20, pady=15, sticky="ew")
        self.input_frame.grid_columnconfigure(0, weight=1)

        # Metin Giriş Kutusu
        self.input_entry = ctk.CTkEntry(
            self.input_frame,
            placeholder_text="Sorunuzu buraya yazın...",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            corner_radius=8,
            height=40
        )
        self.input_entry.grid(row=0, column=0, padx=(0, 10), sticky="ew")
        self.input_entry.bind("<Return>", self.send_message)
        self.input_entry.focus_set()

        # Gönder Butonu
        self.send_button = ctk.CTkButton(
            self.input_frame,
            text="Gönder",
            width=100,
            height=40,
            corner_radius=8,
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            command=self.send_message,
            state="disabled"
        )
        self.send_button.grid(row=0, column=1, sticky="e")

        # Alt Durum Çubuğu
        self.status_label = ctk.CTkLabel(
            self.main_frame,
            text="Sistem yükleniyor, lütfen bekleyin...",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color="#8892b0"
        )
        self.status_label.grid(row=6, column=0, padx=20, pady=(0, 10), sticky="ew")

        # Modelleri arka planda yükle
        threading.Thread(target=self.initialize_models, daemon=True).start()

    def initialize_models(self):
        global model, chat_client, embedding_model, embedding_client, db
        try:
            model = manager.catalog.get_model("phi-3.5-mini")
            model.download()
            model.load()
            chat_client = model.get_chat_client()

            embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")
            embedding_model.load()
            embedding_client = embedding_model.get_embedding_client()

            db = DatabaseManager("general_database.db")

            # Arayüzü ana thread üzerinden güvenli şekilde güncelle
            self.after(0, self.on_initialization_success)
        except Exception as e:
            self.after(0, lambda err=e: self.status_label.configure(text=f"Yükleme Hatası: {str(err)}"))

    def change_database(self, choice):
        global db
        try:
            db_file = self.db_mapping[choice]
            if db:
                db.close()
            db = DatabaseManager(db_file)
            self.status_label.configure(text=f"Bilgi Tabanı Değiştirildi: {choice}")
        except Exception as e:
            self.status_label.configure(text=f"Veritabanı değiştirilemedi: {e}")

    def on_initialization_success(self):
        self.status_label.configure(text="Sistem Hazır.")
        self.send_button.configure(state="normal")

    def append_message_safe(self, message, tag=None):
        self.chat_history.configure(state="normal")
        if tag:
            self.chat_history.insert("end", message, tag)
        else:
            self.chat_history.insert("end", message)
        self.chat_history.configure(state="disabled")
        self.chat_history.see("end")

    def send_message(self, event=None):
        # Modeller henüz yüklenmediyse göndermeyi engelle
        if embedding_client is None or chat_client is None:
            return

        message = self.input_entry.get().strip()
        if not message:
            return

        self.input_entry.delete(0, "end")
        self.append_message_safe("Siz: ", "siz")
        self.append_message_safe(f"{message}\n")
        self.status_label.configure(text="Yanıt bekleniyor...")

        threading.Thread(target=self.get_rag_response, args=(message,), daemon=True).start()

    def get_rag_response(self, user_query):
        try:
            # Sorguyu vektöre çevir (qwen3 talimat önekiyle)
            query_vector = embed_query(embedding_client, user_query)

            # ─── KATMAN 1: Yapısal retrieval + giriş kapısı ───
            results = db.search_with_scores(query_vector, top_k=2)

            # Hiçbir chunk eşiği geçemediyse LLM'i HİÇ çağırmadan fallback dön.
            if not results:
                self.after(0, lambda: self.append_message_safe("Asistan: ", "asistan"))
                self.after(0, lambda: self.append_message_safe(FALLBACK_MESSAGE + "\n\n"))
                self.after(0, lambda: self.status_label.configure(text="Sistem Hazır."))
                return

            # Ortak çekirdek: prompt + doğrulama için gerekenleri hazırla.
            messages, retrieved_vectors, allowed_sources = prepare_generation(user_query, results)

            # ─── A YOLU: Cevabı EKRANA yazmadan tamponda topla ───
            # (Doğrulama bitmeden kullanıcı yanlış cevabı görmesin.)
            self.after(0, lambda: self.status_label.configure(text="Yanıt üretiliyor..."))
            full_response = ""
            for chunk in chat_client.complete_streaming_chat(messages):
                if chunk.choices and chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
            full_response = full_response.strip()

            # ─── KATMAN 2: Generation SONRASI kod seviyesinde denetim ───
            self.after(0, lambda: self.status_label.configure(text="Yanıt kontrol ediliyor..."))
            ok = is_answer_grounded(
                full_response, retrieved_vectors, allowed_sources, embedding_client, db
            )

            self.after(0, lambda: self.append_message_safe("Asistan: ", "asistan"))
            if ok:
                # Doğrulamayı geçti → cevabı TEK SEFERDE göster.
                self.after(0, lambda t=full_response: self.append_message_safe(t + "\n\n"))
            else:
                # Doğrulamayı geçemedi → yanlış cevabı GÖSTERME, fallback ver.
                self.after(0, lambda: self.append_message_safe(FALLBACK_MESSAGE + "\n\n"))

            self.after(0, lambda: self.status_label.configure(text="Sistem Hazır."))

        except Exception as e:
            self.after(0, lambda err=e: self.append_message_safe(f"\n[Hata: {str(err)}]\n\n"))
            self.after(0, lambda: self.status_label.configure(text="Sistem Hazır."))


if __name__ == "__main__":
    app = ModernRAGGui()
    app.mainloop()