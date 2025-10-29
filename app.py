# -*- coding: utf-8 -*-
import os
from pathlib import Path
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

# RAG
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
try:
    from pypdf import PdfReader
except Exception:
    from PyPDF2 import PdfReader  # lokal fallback

# ==============================
# 0) Ortam & Başlık
# ==============================
load_dotenv()
try:
    from settings import APP_NAME
except Exception:
    APP_NAME = "Ayten Bot"

st.set_page_config(page_title=APP_NAME, page_icon=":shallow_pan_of_food:", layout="wide")
st.title(APP_NAME)
st.markdown("### Gaziantep mutfağından tarifler ve bir büyükten hayata dair ipuçları")

# ==============================
# 1) Kişilik (System Prompt)
# ==============================
PROMPT_FILE = Path(__file__).parent / "prompts" / "ayten_system.txt"
st.caption(f"Sistem prompt yolu: {PROMPT_FILE}")
if not PROMPT_FILE.exists():
    st.error(f"Kişilik dosyası bulunamadı: {PROMPT_FILE}")
    st.stop()
system_prompt = PROMPT_FILE.read_text(encoding="utf-8")

# Meta yanıtları önlemek için ek stil koruması
STYLE_GUARD = (
    "DİKKAT: Yalnızca doğrudan cevabı ver. "
    "‘şöyle yanıt verilir’, ‘taslak’, ‘örnek cevap’ gibi META açıklamalar YASAK. "
    "Ayten adıyla, sıcak ve kısa cümlelerle TÜRKÇE konuş."
)

# ==============================
# 2) Gemini yapılandırma
# ==============================
API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
if not API_KEY:
    st.error("GEMINI_API_KEY / GOOGLE_API_KEY bulunamadı. Secrets veya .env dosyasını kontrol et.")
    st.stop()

genai.configure(api_key=API_KEY)

# 2.5 bazı hesaplarda kapalı olabilir; flash herkes için açık.
PREFERRED_MODEL = st.sidebar.selectbox("Gemini modeli", ["gemini-2,5-flash", "gemini-1.5-pro"], index=0)
TEMPERATURE = st.sidebar.slider("Sıcaklık", 0.0, 1.0, 0.7, 0.05)

@st.cache_resource(show_spinner=False)
def get_model():
    return genai.GenerativeModel(
        model_name=PREFERRED_MODEL,
        system_instruction=f"{system_prompt}\n\n{STYLE_GUARD}",
        generation_config={"temperature": TEMPERATURE}
    )

if "chat" not in st.session_state:
    st.session_state.chat = get_model().start_chat(history=[])
if "messages" not in st.session_state:
    st.session_state.messages = []  # {"role": "user" | "assistant", "content": str}

# ==============================
# 3) Vektör DB (Chroma) + Embedding
# ==============================
# Cloud'da yazılabilir güvenli klasör
CHROMA_DIR = Path("./chroma_store")
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
COLLECTION_NAME = "ayten_docs"

EMBED_MODEL_NAME = st.sidebar.selectbox(
    "Embedding modeli",
    ["sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "all-MiniLM-L6-v2"],
    index=0
)

@st.cache_resource(show_spinner=False)
def get_chroma_and_embedder():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
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
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"}
    )
    return client, collection

client, collection = get_chroma_and_embedder()

# =============== CHUNK & INGEST ===============
def chunk_text(text: str, size: int = 800, overlap: int = 150) -> List[str]:
    text = " ".join((text or "").split())
    if not text:
        return []
    chunks, i, n = [], 0, len(text)
    while i < n:
        j = min(n, i + size)
        chunks.append(text[i:j])
        if j == n:
            break
        i = max(0, j - overlap)
    return chunks

def add_pdf_to_index(file_obj, fname: str):
    reader = PdfReader(file_obj)
    docs, metas, ids = [], [], []
    for pno, page in enumerate(getattr(reader, "pages", []), start=1):
        txt = page.extract_text() or ""
        for ci, ch in enumerate(chunk_text(txt)):
            ids.append(f"{fname}_p{pno}_c{ci}")
            docs.append(ch)
            metas.append({"source": str(fname), "page": str(pno), "chunk": str(ci)})
    if docs:
        collection.add(ids=ids, documents=docs, metadatas=metas)

def add_txt_to_index(text: str, fname: str):
    docs, metas, ids = [], [], []
    for ci, ch in enumerate(chunk_text(text)):
        ids.append(f"{fname}_c{ci}")
        docs.append(ch)
        metas.append({"source": str(fname), "page": "", "chunk": str(ci)})
    if docs:
        collection.add(ids=ids, documents=docs, metadatas=metas)

def retrieve(query: str, k=4, max_distance=0.28):
    """Benzerlik eşiğini biraz gevşettim (.25→.28) ki bağlam kolay düşmesin."""
    res = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    pairs = []
    for txt, meta, dist in zip(docs, metas, dists):
        if dist is None or dist <= max_distance:
            m = dict(meta or {})
            m["distance"] = dist
            pairs.append((txt, m))
    return pairs

def build_prompt_with_context(user_msg: str, ctx_pairs):
    if not ctx_pairs:
        return f"{system_prompt}\n\n{STYLE_GUARD}\n\nKullanıcı: {user_msg}\nAyten:"
    ctx_lines = [f"- {txt}" for (txt, _meta) in ctx_pairs]
    context = "\n".join(ctx_lines)
    return (
        f"{system_prompt}\n\n{STYLE_GUARD}\n"
        f"---\n"
        f"Aşağıdaki BAĞLAMA dayanarak yanıt ver. Bağlamda olmayan bilgi ekleme."
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
DATA_DIR = Path(__file__).parent / "data" / "kitaplar"
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
    st.sidebar.info("‘src/data/kitaplar’ klasörüne PDF veya TXT eklersen otomatik indekslenir.")

# ==============================
# 5) Sohbet UI
# ==============================
for msg in st.session_state.messages:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(msg["content"])

user_msg = st.chat_input("Tarif, malzeme ya da pişirme tekniği sor...")

if user_msg:
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    with st.chat_message("assistant"):
        with st.spinner("Ayten düşünüyor…"):
            ctx_pairs = retrieve(user_msg, k=4)
            full_prompt = build_prompt_with_context(user_msg, ctx_pairs)
            try:
                resp = st.session_state.chat.send_message(full_prompt)
                reply_text = (resp.text or "").strip()
                # Basit meta filtresi
                blockers = ["şu şekilde olacaktır", "taslak", "örnek yanıt", "meta", "şöyle cevap ver"]
                if any(b.lower() in reply_text.lower() for b in blockers):
                    resp2 = st.session_state.chat.send_message(
                        f"{STYLE_GUARD}\n\nKullanıcı: {user_msg}\nAyten:"
                    )
                    reply_text = (resp2.text or "").strip()
            except Exception as e:
                reply_text = f"Hata: {e}"
        st.markdown(reply_text)

    st.session_state.messages.append({"role": "assistant", "content": reply_text})
