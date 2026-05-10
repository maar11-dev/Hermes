"""
RAG Engine — con soporte de streaming para respuestas en tiempo real
"""

import os
import json
import hashlib
import fitz
import chromadb
import httpx
from pathlib import Path
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from typing import AsyncGenerator

load_dotenv()

LLM_PROVIDER    = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.2")
ANTHROPIC_KEY   = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
EMBED_MODEL     = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", "60"))
TOP_K           = int(os.getenv("TOP_K", "5"))

DATA_DIR = Path(__file__).parent.parent / "data"

SYSTEM_PROMPT = (
    "Eres un asistente académico inteligente y versátil. "
    "Tienes acceso al contenido de los apuntes del usuario. "
    "Puedes responder preguntas, hacer resúmenes, generar apuntes estructurados, "
    "crear tests de repaso, explicar conceptos, comparar ideas o cualquier otra tarea que el usuario necesite. "
    "Responde siempre de forma clara y bien organizada, usando Markdown cuando ayude. "
    "Si la información no está en los apuntes proporcionados, indícalo con honestidad. "
    "Responde siempre en el mismo idioma que el usuario."
)


class RAGEngine:
    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.meta_path = DATA_DIR / "documents.json"

        print(f"[RAG] Cargando embeddings: {EMBED_MODEL}")
        self.embedder = SentenceTransformer(EMBED_MODEL)

        print("[RAG] Iniciando ChromaDB...")
        self.chroma = chromadb.PersistentClient(path=str(DATA_DIR / "chroma_db"))
        self.col = self.chroma.get_or_create_collection(
            name="notas",
            metadata={"hnsw:space": "cosine"},
        )

        self.docs: dict = self._load_meta()
        print(f"[RAG] Listo. {len(self.docs)} documentos indexados.")

    def _load_meta(self) -> dict:
        if self.meta_path.exists():
            return json.loads(self.meta_path.read_text(encoding="utf-8"))
        return {}

    def _save_meta(self):
        self.meta_path.write_text(
            json.dumps(self.docs, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _extract_text(self, content: bytes) -> str:
        doc = fitz.open(stream=content, filetype="pdf")
        return "\n\n".join(page.get_text("text") for page in doc)

    def _chunk_text(self, text: str) -> list[str]:
        words = text.split()
        step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
        return [
            " ".join(words[i: i + CHUNK_SIZE])
            for i in range(0, len(words), step)
            if len(" ".join(words[i: i + CHUNK_SIZE]).strip()) > 30
        ]

    # ── API pública ───────────────────────────────────────────────────────────

    def add_document(self, content: bytes, filename: str) -> dict:
        doc_id = hashlib.sha1(content).hexdigest()[:16]
        if doc_id in self.docs:
            return self.docs[doc_id]

        text   = self._extract_text(content)
        chunks = self._chunk_text(text)
        if not chunks:
            raise ValueError("No se pudo extraer texto del PDF.")

        embeddings = self.embedder.encode(chunks, show_progress_bar=False).tolist()
        ids        = [f"{doc_id}_c{i}" for i in range(len(chunks))]
        metadatas  = [{"doc_id": doc_id, "filename": filename, "chunk": i} for i in range(len(chunks))]

        self.col.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)

        pages = len(fitz.open(stream=content, filetype="pdf"))
        entry = {"id": doc_id, "filename": filename, "chunks": len(chunks), "chars": len(text), "pages": pages}
        self.docs[doc_id] = entry
        self._save_meta()
        return entry

    def list_documents(self) -> list[dict]:
        return list(self.docs.values())

    def delete_document(self, doc_id: str):
        results = self.col.get(where={"doc_id": doc_id})
        if results["ids"]:
            self.col.delete(ids=results["ids"])
        self.docs.pop(doc_id, None)
        self._save_meta()

    def _retrieve(self, query: str, doc_ids: list[str]) -> list[str]:
        total = self.col.count()
        if total == 0:
            return []
        qe    = self.embedder.encode([query]).tolist()
        where = None
        if doc_ids:
            where = {"doc_id": doc_ids[0]} if len(doc_ids) == 1 else {"doc_id": {"$in": doc_ids}}
        results = self.col.query(query_embeddings=qe, n_results=min(TOP_K, total), where=where)
        return results["documents"][0] if results["documents"] else []

    def _build_prompt(self, message: str, doc_ids: list[str]) -> tuple[str, str]:
        chunks  = self._retrieve(message, doc_ids)
        context = "\n\n---\n\n".join(chunks) if chunks else "(No hay documentos seleccionados)"
        prompt  = f"CONTENIDO DE LOS APUNTES:\n\n{context}\n\n{'─'*60}\n\nSOLICITUD:\n{message}"
        return SYSTEM_PROMPT, prompt

    # ── Streaming ─────────────────────────────────────────────────────────────

    async def stream(self, message: str, doc_ids: list[str]) -> AsyncGenerator[str, None]:
        system, prompt = self._build_prompt(message, doc_ids)
        if LLM_PROVIDER == "anthropic":
            async for token in self._stream_anthropic(system, prompt):
                yield token
        else:
            async for token in self._stream_ollama(system, prompt):
                yield token

    async def _stream_ollama(self, system: str, user: str) -> AsyncGenerator[str, None]:
        async with httpx.AsyncClient(timeout=180) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model":    OLLAMA_MODEL,
                    "stream":   True,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                },
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        data  = json.loads(line)
                        token = data.get("message", {}).get("content", "")
                        if token:
                            yield token
                        if data.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue

    async def _stream_anthropic(self, system: str, user: str) -> AsyncGenerator[str, None]:
        async with httpx.AsyncClient(timeout=180) as client:
            async with client.stream(
                "POST",
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key":         ANTHROPIC_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type":      "application/json",
                },
                json={
                    "model":      ANTHROPIC_MODEL,
                    "max_tokens": 4096,
                    "stream":     True,
                    "system":     system,
                    "messages":   [{"role": "user", "content": user}],
                },
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if raw == "[DONE]":
                        break
                    try:
                        data = json.loads(raw)
                        if data.get("type") == "content_block_delta":
                            token = data.get("delta", {}).get("text", "")
                            if token:
                                yield token
                    except json.JSONDecodeError:
                        continue
