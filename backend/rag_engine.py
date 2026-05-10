"""
RAG Engine Optimizado — Hermes v2
Mejoras: Control de contexto, citación de fuentes (página) y limpieza de texto.
"""

import os
import json
import hashlib
import fitz  # PyMuPDF
import chromadb
import httpx
from pathlib import Path
from sentence_transformers import CrossEncoder, SentenceTransformer
from dotenv import load_dotenv
from typing import AsyncGenerator

load_dotenv()

LLM_PROVIDER    = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.2")
ANTHROPIC_KEY   = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")
EMBED_MODEL     = os.getenv("EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K           = int(os.getenv("TOP_K", "5"))
RERANK_MULTIPLIER = int(os.getenv("RERANK_MULTIPLIER", "3"))

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

HistoryMessage = dict[str, str]

class RAGEngine:
    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.meta_path = DATA_DIR / "documents.json"
        self.reranker = None

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

    def _get_reranker(self):
        if self.reranker is None:
            print(f"[RAG] Cargando reranker: {CROSS_ENCODER_MODEL}")
            self.reranker = CrossEncoder(CROSS_ENCODER_MODEL)
        return self.reranker

    def _extract_text_with_pages(self, content: bytes) -> list[dict]:
        pages_data = []
        doc = fitz.open(stream=content, filetype="pdf")
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text:
                pages_data.append({"page_num": i + 1, "text": text})
        return pages_data

    def _chunk_text(self, pages_data: list[dict]) -> list[dict]:
        chunks_with_meta = []
        for page in pages_data:
            words = page["text"].split()
            step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
            for i in range(0, len(words), step):
                chunk_text = " ".join(words[i: i + CHUNK_SIZE])
                if len(chunk_text.strip()) > 40:
                    chunks_with_meta.append({
                        "text": chunk_text,
                        "page": page["page_num"]
                    })
        return chunks_with_meta

    # ── API pública ───────────────────────────────────────────────────────────

    def add_document(self, content: bytes, filename: str) -> dict:
        doc_id = hashlib.sha1(content).hexdigest()[:16]
        if doc_id in self.docs:
            return self.docs[doc_id]

        pages_data  = self._extract_text_with_pages(content)
        chunks_data = self._chunk_text(pages_data)

        if not chunks_data:
            raise ValueError("No se pudo extraer texto del PDF (podría ser una imagen que necesita OCR).")

        texts      = [c["text"] for c in chunks_data]
        embeddings = self.embedder.encode(texts, show_progress_bar=False).tolist()
        ids        = [f"{doc_id}_c{i}" for i in range(len(chunks_data))]
        metadatas  = [
            {"doc_id": doc_id, "filename": filename, "page": c["page"]}
            for c in chunks_data
        ]

        self.col.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

        # FIX 1: incluir 'chars' para que el frontend pueda calcular el tamaño en KB
        total_chars = sum(len(c["text"]) for c in chunks_data)
        entry = {
            "id":       doc_id,
            "filename": filename,
            "chunks":   len(chunks_data),
            "pages":    len(pages_data),
            "chars":    total_chars,
        }
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

    def _retrieve(self, query: str, doc_ids: list[str]) -> str:
        total = self.col.count()
        if total == 0:
            return ""

        qe    = self.embedder.encode([query]).tolist()
        where = None
        if doc_ids:
            where = {"doc_id": doc_ids[0]} if len(doc_ids) == 1 else {"doc_id": {"$in": doc_ids}}

        fetch_k = min(max(TOP_K * RERANK_MULTIPLIER, TOP_K), total)
        results = self.col.query(query_embeddings=qe, n_results=fetch_k, where=where)

        candidates = []
        if results["documents"] and results["metadatas"]:
            for doc_text, meta in zip(results["documents"][0], results["metadatas"][0]):
                # FIX 2: usar .get() para ser compatibles con chunks indexados
                # por versiones anteriores del engine (que usaban 'chunk' en vez de 'page')
                filename = meta.get("filename", "Desconocido")
                page     = meta.get("page", "?")
                source_info = f"[Fuente: {filename}, Pág: {page}]"
                candidates.append({
                    "text": doc_text,
                    "meta": meta,
                    "source_info": source_info,
                })

        if not candidates:
            return ""

        try:
            reranker = self._get_reranker()
            pair_inputs = [(query, candidate["text"]) for candidate in candidates]
            scores = reranker.predict(pair_inputs)
            ranked_candidates = [candidate for _, candidate in sorted(zip(scores, candidates), key=lambda item: item[0], reverse=True)]
        except Exception as e:
            print(f"[RAG] Aviso: no se pudo aplicar reranking ({e}). Se usa el orden original.")
            ranked_candidates = candidates

        context_parts = [f"{candidate['source_info']}\n{candidate['text']}" for candidate in ranked_candidates[:TOP_K]]

        return "\n\n---\n\n".join(context_parts)

    def _build_prompt(self, message: str, doc_ids: list[str]) -> tuple[str, str]:
        context = self._retrieve(message, doc_ids)
        if not context:
            context = "(No hay documentos seleccionados o la base de datos está vacía)"
        prompt = f"CONTENIDO DE LOS APUNTES:\n\n{context}\n\n{'─'*60}\n\nSOLICITUD:\n{message}"
        return SYSTEM_PROMPT, prompt

    def _normalize_history(self, history: list[HistoryMessage] | None) -> list[HistoryMessage]:
        if not history:
            return []

        normalized = []
        for item in history[-6:]:
            role = item.get("role", "")
            content = item.get("content", "")
            if role not in {"user", "assistant"}:
                continue
            if not isinstance(content, str) or not content.strip():
                continue
            normalized.append({"role": role, "content": content.strip()})
        return normalized

    # ── Streaming ─────────────────────────────────────────────────────────────

    async def stream(self, message: str, doc_ids: list[str], history: list[HistoryMessage] | None = None) -> AsyncGenerator[str, None]:
        system, prompt = self._build_prompt(message, doc_ids)
        normalized_history = self._normalize_history(history)
        if LLM_PROVIDER == "anthropic":
            async for token in self._stream_anthropic(system, prompt, normalized_history):
                yield token
        else:
            async for token in self._stream_ollama(system, prompt, normalized_history):
                yield token

    async def _stream_ollama(self, system: str, user: str, history: list[HistoryMessage] | None = None) -> AsyncGenerator[str, None]:
        messages = [{"role": "system", "content": system}]
        messages.extend(history or [])
        messages.append({"role": "user", "content": user})

        async with httpx.AsyncClient(timeout=180) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model":    OLLAMA_MODEL,
                    "stream":   True,
                    "messages": messages,
                    "options": {"num_ctx": 8192},
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

    async def _stream_anthropic(self, system: str, user: str, history: list[HistoryMessage] | None = None) -> AsyncGenerator[str, None]:
        messages = list(history or [])
        messages.append({"role": "user", "content": user})

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
                    "messages":   messages,
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