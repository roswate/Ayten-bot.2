# -*- coding: utf-8 -*-
import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from settings import APP_NAME
import google.generativeai as genai

# --- RAG için kütüphaneler ---
# AĞIR MODELLERİ DEVRE DIŞI BIRAKIYORUZ
# import chromadb
# from chromadb.utils import embedding_functions
# from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# --- Hafif veri yükleme için cache bloğu ---
@st.cache_data
def load_df():
    import pandas as pd
    df = pd.DataFrame({"Mesaj": ["Ayten Bot hazır 🎉"]})
    return df

df = load_df()

# ==============================
# 0) Ortam & Başlık
# ==============================
load_dotenv()
st.set_page_config(page_title=APP_NAME, page_icon="🥘")
st.title(APP_NAME)
st.markdown("### Gaziantep mutfağından tarifler ve bir büyükten hayata dair ipuçları")

# ==============================
# 1) Kişilik (System Prompt)
# ==============================
PROMPT_FILE = Path(__file__).parent / "prompts" / "ayten_system.txt"
if not PROMPT_FILE.exists():
    st.warning("⚠️ Kişilik dosyası bulunamadı (prompts/ayten_system.txt).")
    system_prompt = "Sen Gaziantepli Ayten'sin, kısa ve içten cevaplar ver."
else:
    system_prompt = PROMPT_FILE.read_text(encoding="utf-8")

# ==============================
# 2) Gemini yapılandırma
# ==============================
API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
if not API_KEY:
    st.error("GOOGLE_API_KEY bulunamadı. .env dosyasını kontrol et.")
    st.stop()

genai.configure(api_key=API_KEY)
PREFERRED_MODEL = "gemini-2.0-flash"  # daha hafif model

if "gemini_model" not in st.session_state:
    st.session_state.gemini_model = genai.GenerativeModel(
        model_name=PREFERRED_MODEL,
        system_instruction=system_prompt,
    )

if "chat" not in st.session_state:
    st.session_state.chat = st.session_state.gemini_model.start_chat(history=[])

if "messages" not in st.session_state:
    st.session_state.messages = []

# ==============================
# 3) (RAG yerine DEMO mod)
# ==============================
USE_RAG = False

def retrieve(query: str, k=4, max_distance=0.25):
    # RAM dostu: Chroma devre dışı, demo amaçlı
    return []

# ==============================
# 4) Belgeleri sadece listele (index yok)
# ==============================
st.sidebar.subheader("📚 Ayten’in Arşivi")
DATA_DIR = Path(__file__).resolve().parents[0] / "data" / "kitaplar"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def preload_documents():
    added_files = []
    for file in DATA_DIR.glob("*"):
        if file.suffix.lower() in [".pdf", ".txt"]:
            added_files.append(file.name)
    return added_files

added = preload_documents()
if added:
    st.sidebar.success(f"{len(added)} belge bulundu:")
    for a in added:
        st.sidebar.caption(f"📘 {a}")
else:
    st.sidebar.warning("Arşiv boş. 'src/data/kitaplar' klasörüne PDF/TXT ekleyebilirsin.")

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
            try:
                response = st.session_state.chat.send_message(user_msg)
                reply_text = (response.text or "").strip()
            except Exception as e:
                reply_text = f"Hata: {e}"
        st.markdown(reply_text)

    st.session_state.messages.append({"role": "assistant", "content": reply_text})
