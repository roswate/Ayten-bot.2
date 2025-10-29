import streamlit as st
import google.generativeai as genai
import os
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

# --- Sayfa Ayarları ---
st.set_page_config(page_title="Ayten Bot", page_icon="🍲", layout="wide")

# --- Başlık ---
st.title("🍲 Ayten Bot")
st.subheader("Gaziantep mutfağından ilham alan kültürel sohbet asistanı")

# --- API Key (Streamlit Secrets üzerinden) ---
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# --- Model Seçimi ---
model = genai.GenerativeModel("gemini-2.5-flash")

# --- Vektör Veritabanı ve Embedder ---
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.Client()
collection = client.get_or_create_collection("ayten_docs")

# --- PDF Yükleme ---
uploaded_file = st.file_uploader("Bir tarif PDF'si yükle (isteğe bağlı)", type=["pdf"])

if uploaded_file is not None:
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    # PDF metinlerini embedding’e dönüştür
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    embeddings = embedding_model.encode(chunks).tolist()

    for i, chunk in enumerate(chunks):
        collection.add(documents=[chunk], embeddings=[embeddings[i]], ids=[f"chunk_{i}"])

    st.success("PDF başarıyla işlendi! Ayten Bot artık bu belgeyi biliyor 🍽️")

# --- Kullanıcı Girişi ---
prompt = st.text_input("Ne merak ediyorsun, kuzum? (örneğin: Gaziantep'te en popüler çorba nedir?)")

if prompt:
    with st.spinner("Ayten düşünüyor..."):
        try:
            # RAG: En yakın içerikleri getir
            query_embedding = embedding_model.encode([prompt]).tolist()[0]
            results = collection.query(query_embeddings=[query_embedding], n_results=3)
            context = " ".join([d for d in results["documents"][0]]) if results["documents"] else ""

            # Gemini'den yanıt al
            full_prompt = f"Kültürel bağlam: {context}\n\nSoru: {prompt}"
            response = model.generate_content(full_prompt)
            st.success(response.text)

        except Exception as e:
            st.error(f"Hata oluştu: {e}")

# --- Alt Bilgi ---
st.caption("© 2025 Ayten Bot | Developed by Gülsu Çelik")
