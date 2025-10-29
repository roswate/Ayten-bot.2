import streamlit as st
import google.generativeai as genai
import os

# --- Sayfa Ayarları ---
st.set_page_config(page_title="Ayten Bot", page_icon="🍲", layout="centered")

# --- Başlık ---
st.title("🍲 Ayten Bot")
st.subheader("Gaziantep mutfağından ilham alan kültürel sohbet asistanı")

# --- API Key (Streamlit Secrets'tan alınır) ---
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# --- Model Seçimi ---
model = genai.GenerativeModel("gemini-1.5-flash")

# --- Kullanıcı Girişi ---
prompt = st.text_input("Ne merak ediyorsun? (örneğin: Ali Nazik tarifi nedir?)")

# --- Yanıt Üretimi ---
if prompt:
    with st.spinner("Ayten düşünüyor..."):
        try:
            response = model.generate_content(prompt)
            st.success(response.text)
        except Exception as e:
            st.error(f"Hata: {e}")

# --- Alt Bilgi ---
st.caption("© 2025 Ayten Bot | Developed by Gülsu Çelik")
