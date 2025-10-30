# -*- coding: utf-8 -*-
import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from settings import APP_NAME
import google.generativeai as genai

# --- RAG için kütüphaneler ---
import chromadb
from chromadb.utils import embedding_functions
# sentence-transformers çok RAM yiyor, geçici olarak kapatabilirsin
# from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# --- Hafif veri yükleme için cache bloğu ---
@st.cache_data
def load_df():
    import pandas as pd
    # dosyanın konumuna göre yolunu düzenle:
    df = pd.read_csv("data/your.csv")
    return df.head(1000)  # küçük örnek set

df = load_df()

# ==============================
# 0) Ortam & Başlık
# ==============================
load_dotenv()
st.set_page_config(page_title=APP_NAME, page_icon=":shallow_pan_of_food:")
st.title(APP_NAME)
st.markdown("### Gaziantep mutfağından tarifler ve bir büyükten hayata dair ipuçları")

# ==============================
# 1) Kişilik (System Prompt) bu bölüm ayten systmtxt de tanımlaıklarımı içerir
# ==============================
PROMPT_FILE = Path(__file__).parent / "prompts" / "ayten_system.txt"
st.caption(f"Sistem prompt yolu: {PROMPT_FILE}")
if not PROMPT_FILE.exists():
    st.error(f"Kişilik dosyası bulunamadı: {PROMPT_FILE}")
    st.stop()
system_prompt = PROMPT_FILE.read_text(encoding="utf-8")

# ==============================
# 2) Gemini yapılandırma
# ==============================
# ==============================
API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
if not API_KEY:
    st.error("GOOGLE_API_KEY bulunamadı. .env dosyasını kontrol et.")
    st.stop()
genai.configure(api_key=API_KEY)
PREFERRED_MODEL = "gemini-2.5-flash"

# Model & Chat oturumu (tek sefer)
if "gemini_model" not in st.session_state:
    st.session_state.gemini_model = genai.GenerativeModel(
        model_name=PREFERRED_MODEL,
        system_instruction=system_prompt,
    )
if "chat" not in st.session_state:
    st.session_state.chat = st.session_state.gemini_model.start_chat(history=[])

# Sohbet geçmişi (UI)
if "messages" not in st.session_state:
    st.session_state.messages = []  # {"role": "user" | "assistant", "content": str}

# ==============================
# 3) Vektör DB (Chroma) + Embedding
# ==============================
VECTOR_DIR = Path(__file__).resolve().parents[1] / "data" / "vectordb"
VECTOR_DIR.mkdir(parents=True, exist_ok=True)
COLLECTION_NAME = "ayten_docs"
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

if "chroma" not in st.session_state:
    client = chromadb.PersistentClient(path=str(VECTOR_DIR))

    # Bazı ortamlarda torch 3.13 ile sorun olabiliyor; gerekirse CPU fallback dene
    try:
        st_model = SentenceTransformer(EMBED_MODEL_NAME)
    except Exception:
        st_model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")

    class STEmbedding(embedding_functions.EmbeddingFunction):
        def __call__(self, inputs):
            if isinstance(inputs, str):
                inputs = [inputs]
            return st_model.encode(inputs, normalize_embeddings=True).tolist()

    emb_fn = STEmbedding()
    st.session_state.chroma = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"}
    )

# =============== CHUNK & INGEST eklediiğim verileri ve web sitesinden çektiğim sayfalarca oan sözlüğü parçalara ayırarK SİSTEMİN KAFA KARIŞTIRMASINI ÖNLER ===============
from typing import List, Tuple

def chunk_text(text: str, size: int = 800, overlap: int = 150) -> List[str]:
    """Metni sabit büyüklükte parçalara (chunk’lara) ayırır."""
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    i = 0
    while i < len(text):
        j = min(len(text), i + size)
        chunks.append(text[i:j])
        if j == len(text):
            break
        i = max(0, j - overlap)
    return chunks


def add_pdf_to_index(file_obj, fname: str):
    """PDF dosyasını okuyup chunk’lara böler ve Chroma’ya ekler."""
    reader = PdfReader(file_obj)
    docs, metas, ids = [], [], []
    for pno, page in enumerate(reader.pages, start=1):
        txt = page.extract_text() or ""
        for ci, ch in enumerate(chunk_text(txt)):
            ids.append(f"{fname}_p{pno}_c{ci}")
            docs.append(ch)
            metas.append({"source": str(fname), "page": str(pno), "chunk": str(ci)})
    if docs:
        st.session_state.chroma.add(ids=ids, documents=docs, metadatas=metas)


def add_txt_to_index(text: str, fname: str):
    """TXT dosyasını chunk’lara böler ve Chroma’ya ekler."""
    docs, metas, ids = [], [], []
    for ci, ch in enumerate(chunk_text(text)):
        ids.append(f"{fname}_c{ci}")
        docs.append(ch)
        metas.append({"source": str(fname), "page": "", "chunk": str(ci)})
    if docs:
        st.session_state.chroma.add(ids=ids, documents=docs, metadatas=metas)
# =============================================
    
def retrieve(query: str, k=4, max_distance=0.25):
    """Yeterince benzer olmayan parçaları ele (düşük gürültü, sıkı RAG)."""
    res = st.session_state.chroma.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    pairs = []
    for txt, meta, dist in zip(docs, metas, dists):
        if dist is not None and dist <= max_distance:
            meta = dict(meta)
            meta["distance"] = dist
            pairs.append((txt, meta))
    return pairs  # boş dönebilir; “Belge bulunamadı” yazmayacağız.

STYLE_GUARD = (
    "DİKKAT: Aşağıdaki yanıtı kesinlikle Gaziantep ağzıyla ver. " 
    "İstanbul Türkçesine kayma. Kısa, sıcak cümleler; yerel sözler kullan."
)

def build_prompt_with_context(user_msg: str, ctx_pairs):
    if not ctx_pairs:
        return (
            f"{system_prompt}\n\n"
            f"{STYLE_GUARD}\n\n"
            f"Kullanıcı: {user_msg}\n"
            f"Ayten:"
        )

    ctx_lines = [f"- {txt}" for (txt, _meta) in ctx_pairs]
    context = "\n".join(ctx_lines)

    return (
        f"{system_prompt}\n\n"
        f"{STYLE_GUARD}\n"
        f"---\n"
        f"Aşağıdaki BAĞLAMA dayanarak yanıt ver. Bağlamda olmayan bilgi ekleme. "
        f"Belge adı/sayfa yazma.\n"
        f"BAĞLAM:\n{context}\n"
        f"---\n\n"
        f"Kullanıcı: {user_msg}\n"
        f"Ayten:"
    )
# ==============================
# 4) Gömülü PDF/TXT’leri otomatik yükle
# ==============================
st.sidebar.subheader("📚 Ayten’in Arşivi")
DATA_DIR = Path(__file__).resolve().parents[0] / "data" / "kitaplar"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def preload_documents():
    added_files = []
    for file in DATA_DIR.glob("*"):
        if file.suffix.lower() == ".pdf":
            with open(file, "rb") as f:
                add_pdf_to_index(f, file.name)
            added_files.append(file.name)
        elif file.suffix.lower() == ".txt":
            add_txt_to_index(file.read_text(encoding="utf-8"), file.name)
            added_files.append(file.name)
    return added_files

added = preload_documents()
if added:
    st.sidebar.success(f"{len(added)} belge yüklendi:")
    for a in added:
        st.sidebar.caption(f"📘 {a}")
else:
    st.sidebar.warning("Henüz arşivde belge yok. 'src/data/kitaplar' klasörüne PDF veya TXT ekle.")

# ==============================
# 5) Sohbet UI
# ==============================
# Önce geçmişi çiz
for msg in st.session_state.messages:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(msg["content"])

# Girdi (Enter ile gönder; otomatik temizlenir)
user_msg = st.chat_input("Tarif, malzeme ya da pişirme tekniği sor...")

if user_msg:
    # Kullanıcı mesajı
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    # Ayten cevabı
    with st.chat_message("assistant"):
        with st.spinner("Ayten düşünüyor…"):
            ctx_pairs = retrieve(user_msg, k=4)  # boş olabilir; sorun değil
            full_prompt = build_prompt_with_context(user_msg, ctx_pairs)
            try:
                resp = st.session_state.chat.send_message(full_prompt)
                reply_text = (resp.text or "").strip()
            except Exception as e:
                reply_text = f"Hata: {e}"
        st.markdown(reply_text)

    # Geçmişe kaydet
    st.session_state.messages.append({"role": "assistant", "content": reply_text})
