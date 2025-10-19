import os
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

# Uygulama adı
APP_NAME = os.getenv("APP_NAME", "Ayten — Gaziantep Mutfağı Asistanı")

# Google Gemini API anahtarı
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Ayten kişiliği metni
SYSTEM_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "ayten_system.txt")
