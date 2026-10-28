"""
RAG helpers for TaskPilot AI.

Embedding back-end is selected by AI_PROVIDER in config:
  - "ollama"  → POST http://localhost:11434/api/embeddings  (nomic-embed-text)
  - "gemini"  → google-genai embed_content  (gemini-embedding-001)

⚠ IMPORTANT – ChromaDB collection compatibility
If you previously indexed documents with Gemini embeddings (768-dim) and are
now switching to Ollama / nomic-embed-text (768-dim, but different vector space),
the cosine distances will be meaningless because the two models produce
incompatible vector spaces.

Resolution: after switching AI_PROVIDER to "ollama", go to the Knowledge Base
page in the app and re-upload / re-index all documents.  The collection will
automatically be cleared of the old provider's vectors when you re-index
(same digest → delete + upsert).  You can also use the Delete button for each
document first to ensure a clean slate.
"""
import hashlib
import re
from pathlib import Path

try:
    import pymupdf as fitz
except ImportError:
    import fitz

import httpx
import chromadb

from backend.config import (
    AI_PROVIDER,
    CHROMA_PATH,
    UPLOAD_DIR,
    GEMINI_API_KEY,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
)

_client = None
_collection = None


def _collection_handle():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _collection = _client.get_or_create_collection(
            name="taskpilot_documents",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ── Text extraction ────────────────────────────────────────────────────────────

def extract_pages(path: Path) -> list[dict]:
    pages = []
    if path.suffix.lower() == ".pdf":
        with fitz.open(path) as doc:
            for i, page in enumerate(doc):
                text = page.get_text("text").strip()
                if text:
                    pages.append({"page": i + 1, "text": text})
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
        pages = [{"page": 1, "text": text}] if text.strip() else []
    return pages


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 140) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    step = max(1, chunk_size - overlap)
    return [text[i : i + chunk_size] for i in range(0, len(text), step)]


# ── Embedding ──────────────────────────────────────────────────────────────────

def _embed_ollama(texts: list[str]) -> list[list[float]]:
    """Call Ollama's local embedding endpoint for each text."""
    vectors = []
    with httpx.Client(timeout=60.0) as client:
        for text in texts:
            resp = client.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={"model": OLLAMA_EMBED_MODEL, "prompt": text[:8000]},
            )
            resp.raise_for_status()
            data = resp.json()
            if "embedding" not in data:
                raise RuntimeError(f"Ollama embedding response missing 'embedding' key: {data}")
            vectors.append(data["embedding"])
    return vectors


def _embed_gemini(texts: list[str]) -> list[list[float]]:
    """Call Gemini embedding API (legacy path)."""
    if not GEMINI_API_KEY:
        raise RuntimeError("Set GEMINI_API_KEY in .env before indexing documents.")
    from google import genai
    gclient = genai.Client(api_key=GEMINI_API_KEY)
    vectors = []
    for text in texts:
        response = gclient.models.embed_content(
            model="gemini-embedding-001", contents=text[:8000]
        )
        vectors.append(response.embeddings[0].values)
    return vectors


def _embed(texts: list[str]) -> list[list[float]]:
    """Dispatch to the configured embedding back-end."""
    if AI_PROVIDER == "gemini":
        return _embed_gemini(texts)
    return _embed_ollama(texts)


# ── Indexing & retrieval ───────────────────────────────────────────────────────

def index_document(path: Path, display_name: str) -> dict:
    pages = extract_pages(path)
    chunks, metadata, ids = [], [], []
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    for page in pages:
        for index, chunk in enumerate(chunk_text(page["text"])):
            chunks.append(chunk)
            metadata.append({"filename": display_name, "page": page["page"], "digest": digest})
            ids.append(f"{digest}-{page['page']}-{index}")
    if not chunks:
        raise ValueError("No readable text found. Scanned PDFs may need OCR, which is not included.")
    collection = _collection_handle()
    # Remove previous vectors for this exact file so we never mix embedding spaces.
    if collection.count():
        collection.delete(where={"digest": digest})
    vectors = _embed(chunks)
    collection.upsert(ids=ids, documents=chunks, metadatas=metadata, embeddings=vectors)
    return {"filename": display_name, "pages_with_text": len(pages), "chunks_indexed": len(chunks)}


def search_documents(query: str, limit: int = 5) -> dict:
    collection = _collection_handle()
    if collection.count() == 0:
        return {"matches": [], "message": "No documents indexed yet. Upload a PDF or TXT file first."}
    vectors = _embed([query[:2000]])
    result = collection.query(
        query_embeddings=vectors,
        n_results=max(1, min(limit, 8)),
        include=["documents", "metadatas", "distances"],
    )
    matches = []
    docs      = (result.get("documents") or [[]])[0]
    metas     = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    for doc, meta, distance in zip(docs, metas, distances):
        matches.append({
            "text":     doc,
            "filename": meta.get("filename"),
            "page":     meta.get("page"),
            "distance": round(float(distance), 4),
        })
    return {"matches": matches}


def list_indexed_documents() -> list[dict]:
    collection = _collection_handle()
    if not collection.count():
        return []
    data = collection.get(include=["metadatas"])
    unique = {}
    for meta in data.get("metadatas") or []:
        if meta:
            unique[meta.get("filename", "document")] = {"filename": meta.get("filename", "document")}
    return list(unique.values())


def delete_document(filename: str) -> int:
    collection = _collection_handle()
    data = collection.get(where={"filename": filename}, include=["metadatas"])
    ids = data.get("ids") or []
    if ids:
        collection.delete(ids=ids)
    safe_name = Path(filename).name
    path = UPLOAD_DIR / safe_name
    if path.exists():
        path.unlink()
    return len(ids)
