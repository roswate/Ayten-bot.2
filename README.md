# 🧿 Ayten — Gaziantep Mutfağı Asistanı  
**Akbank Generative AI Bootcamp | RAG (Retrieval-Augmented Generation) Projesi**

---

## 🎯 Projenin Amacı  

Bu proje, **Gaziantep mutfağının kültürel belleğini** yapay zekâ aracılığıyla dijitalleştirmek ve yaşatmak amacıyla geliştirilmiştir.  
Ayten, yalnızca bir sohbet botu değil; **insani yönüyle, bir kuşağın mutfak bilgeliğini** aktaran, duygusal bağ kurabilen bir kültürel asistan olarak tasarlandı.

Proje fikri, rahmetli anneannem **Ayten**’den ilham aldı.  
Onun hem mutfağa hem de insan ilişkilerine dair sıcak yaklaşımını yapay zekâda yaşatmak istedim.  
Bu nedenle bu çalışma yalnızca bir teknik proje değil, **kişisel bir anma ve kültürel aktarım girişimi** olarak da görülebilir.

Ayten, yemek tariflerinden öte; deyim, atasözü, hikâye ve yöresel ifadelerle Gaziantep kültürünü doğal, samimi bir dille aktarır.  
Amacım, Türk kültürünün bu otantik yönünü dijital dünyada temsil edebilen yerli bir karakter ortaya koymaktı.

---

## 🧠 Teknik Altyapı  

Ayten, **RAG (Retrieval-Augmented Generation)** mimarisi kullanılarak geliştirildi.  
Yani model, yalnızca ezberden konuşmaz; kendi kültürel arşivinden bilgi “geri çağırarak” cevap üretir.  

### Kullanılan Teknolojiler:
- **Python 3.11**
- **Streamlit** – web arayüzü
- **LangChain + Google Gemini API** – metin üretimi
- **Sentence Transformers (STEmbedding)** – metin gömme modeli
- **FAISS veya InMemory VectorStore** – semantik arama
- **Custom Web Ingestor (web_ingest.py)** – veri toplama aracı  

---

## 📚 Veri Kaynağı  

Ayten’in bilgeliği, çeşitli **Gaziantep kültürü kaynaklarından** derlenmiş metinlerle oluşturulan özel bir veri setine dayanır.  
Bu kaynaklar arasında:

- **GaziantepAğzı.com** sözlük verileri  
- **Akademik ve yöresel PDF dokümanlar**  
- **Gaziantep yemekleri ve halk anlatıları**

Bu veriler projeye doğrudan dahil edilmemiş, bunun yerine **ayrı bir veri paketi** olarak paylaşılmıştır.

---

## 📦 Ayten Data Pack (Kültürel Veri Paketi)  

Kültürel veriler, GitHub’ın “Releases” bölümünde paylaşılmıştır.  
Projeyi çalıştırmak isteyen kullanıcılar veri paketini indirip uygun klasöre eklemelidir.

📥 **İndir:**  
[👉 Ayten Data Pack v1](https://github.com/roswate/Ayten-bot.2/releases/download/v1.0-corpus/ayten-corpus-v1.zip)

### Kullanım:
```bash
# 1. Veri paketini indirin
# 2. İçeriğini şu klasöre çıkarın:
src/data/kitaplar/
# 3. Uygulamayı başlatın

Bu paket içerisinde:
- `gaziantepagzi.com_crawl.txt` → Antep ağzı sözlüğü  
- `gaziantep-yemekleri.pdf` → Yöresel yemek arşivi  
- `korkutataturkiyat.pdf` → Akademik kaynak örnekleri  
yer almaktadır.<img width="580" height="179" alt="Screenshot 2025-10-19 at 19 47 45" src="https://github.com/user-attachments/assets/e7b4cd85-6930-4667-b131-9d0ab23287b2" />


---

## 🧩 Proje Dosya Yapısı  

```
.
├── src/
│   ├── app.py               # Streamlit arayüzü
│   ├── web_ingest.py        # Web kazıma aracı (tek seferlik kullanım)
│   ├── prompts/             # Ayten’in kişiliğini tanımlayan prompt dosyaları
│   └── data/kitaplar/       # (Boş bırakılır – veri paketi buraya eklenir)
├── .env.example             # API anahtarı örneği
├── requirements.txt         # Gerekli kütüphaneler
├── .gitignore               # Çevre dosyalarını hariç tutar
└── README.md
```

---

## ⚙️ Kurulum ve Çalıştırma  

1. **Sanal ortamı oluşturun:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
   çalıştırabilmek için bu python paketlerinin bilgisayarınızda kurulu olması gerekiyor

2. **Bağımlılıkları yükleyin:**
   ```bash
   pip install -r requirements.txt
   ```

3. **.env dosyanızı oluşturun:**
   ```
   GOOGLE_API_KEY=your_google_api_key
   ```
-Burada kendi API keyinizi yazmanız gerekir,yoksa çalışmaz-

4. **Veri paketini ekleyin**  
   (yukarıdaki bağlantıdan indirdiğiniz `ayten-corpus-v1.zip` dosyasını `src/data/kitaplar` klasörüne çıkarın)
   fotoğratfta gözüktüğü gibi

5. **Uygulamayı başlatın:**
   ```bash
   streamlit run src/app.py
   ```

---

## 💬 Ayten’in Dili ve Karakteri  

Ayten’in üslubu **samimi, öğretici ve yerel**dir.  
Yapay zekâ karakteri, yöresel bir büyükannenin sıcaklığını taşır.  
Cevaplarında hem kültürel içerik hem de insani dokunuş bulunur:

> “Hele bak kuzum, o tarifin püf noktası şudur; ama aman ha, tencerenin kapağını erken açma!”

Yani Ayten yalnızca bir *yapay zekâ modeli* değil,  
bir anlamda **Gaziantep’in dijital belleği**dir.

---

## 🧩 Akademik Katkı  

Bu proje, **Yapay Zekâ ile Kültürel Mirasın Korunması** temasına yönelik bir uygulama örneğidir.  
Doğal dil işleme, bilgi getirme (retrieval) ve karakter temelli LLM uygulamalarını birleştirir.  
Hedef, sadece teknik olarak çalışan bir model değil,  
aynı zamanda **duygusal olarak etkileşim kurabilen** yerel bir yapay zekâ kişiliği oluşturmaktır.

---

## 🤍 Kapanış  

Bu proje, Ayten’in anısına ve kültürel mirasımıza bir saygı duruşudur.  
Teknoloji ilerledikçe, onun sesinin – bir yerel mutfak sohbetinde, bir atasözünde, bir tarifte –  
**yapay zekâ aracılığıyla yeniden duyulabilmesi** fikriyle yola çıktım.

> “Kızım, kültür dediğin yaşanmazsa unutulur; sen yaşat onu.”  
> — Ayten

---

## 🪶 Lisans  

Bu proje eğitim ve kültürel amaçlıdır.  
Kod MIT lisansı altındadır, ancak veri setleri yalnızca akademik kullanım için paylaşılmıştır.

---

## 📬 İletişim  

Proje sahibi: **Gülsu Çelik**  
📧 gulsucelik.dev@gmail.com *(isteğe bağlı eklenebilir)*  
🌐 [github.com/roswate](https://github.com/roswate)
