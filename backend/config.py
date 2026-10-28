from pathlib import Path
import os
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

APP_NAME = os.getenv("APP_NAME", "TaskPilot AI")

# ── Gemini (legacy / fallback) ────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# ── Ollama (local) ────────────────────────────────────────────────────────────
OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL  = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:1.5b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# AI_PROVIDER selects the active backend: "ollama" | "gemini"
AI_PROVIDER = os.getenv("AI_PROVIDER", "ollama").strip().lower()

# ── Paths ─────────────────────────────────────────────────────────────────────
DATABASE_PATH = ROOT_DIR / os.getenv("DATABASE_PATH", "data/taskpilot.db")
CHROMA_PATH   = ROOT_DIR / os.getenv("CHROMA_PATH",   "data/chroma_db")
UPLOAD_DIR    = ROOT_DIR / os.getenv("UPLOAD_DIR",    "data/uploads")
REPORT_DIR    = ROOT_DIR / os.getenv("REPORT_DIR",    "data/reports")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "15"))

for folder in (DATABASE_PATH.parent, CHROMA_PATH, UPLOAD_DIR, REPORT_DIR):
    folder.mkdir(parents=True, exist_ok=True)
