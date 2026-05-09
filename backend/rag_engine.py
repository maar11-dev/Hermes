"""
RAG Engine — Núcleo del agente de apuntes
Soporta Ollama (local) o Anthropic API como LLM backend.
"""

import os
import hashlib
import json
import fitz  # PyMuPDF
import chromadb
import httpx
from pathlib import Path
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ─── Configuración ──────────────────────────────────────────────────────────
LLM_PROVIDER    = os.getenv("LLM_PROVIDER", "ollama")          # "ollama" | "anthropic"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "gemma2:2b")
ANTHROPIC_KEY   = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
EMBED_MODEL     = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", "300"))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", "60"))
TOP_K           = int(os.getenv("TOP_K", "2"))

DATA_DIR = Path(__file__).parent.parent / "data"

# ─── System prompts por modo ─────────────────────────────────────────────────
SYSTEM_PROMPTS = {
    "chat": (
        "Eres un asistente académico experto y preciso. "
        "Utiliza el contexto proporcionado de los apuntes del usuario para responder "
        "de forma clara y estructurada. Si la información no está en el contexto, indícalo honestamente. "
        "Responde siempre en el mismo idioma que la pregunta del usuario."
    ),
    "summarize": (
        "Eres un experto en síntesis académica. "
        "Crea resúmenes claros, completos y bien estructurados del material proporcionado. "
        "Usa títulos (##), listas de puntos clave (- ) y resalta los conceptos más importantes. "
        "Incluye una sección de 'Ideas clave' al final. "
        "Responde en el mismo idioma que el texto de los apuntes."
    ),
    "notes": (
        "Eres un experto en toma de apuntes académicos al estilo Cornell o Zettelkasten. "
        "Transforma el contenido en apuntes estructurados con: "
        "1) Título y fecha, 2) Conceptos clave con definiciones, 3) Esquema numerado, "
        "4) Ejemplos relevantes, 5) Preguntas de repaso. "
        "Usa formato Markdown con emojis de sección (📌, 💡, 🔑, ❓). "
        "Responde en el mismo idioma que el texto."
    ),
    "quiz": (
        "Eres un profesor experto en evaluación. "
        "Genera preguntas de repaso tipo test (4 opciones A/B/C/D) y preguntas de desarrollo "
        "basadas en el contenido de los apuntes. Incluye las respuestas correctas al final. "
        "Responde en el mismo idioma que el texto de los apuntes."
    ),
}


class RAGEngine:
    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.meta_path = DATA_DIR / "documents.json"

        print(f"[RAG] Cargando modelo de embeddings: {EMBED_MODEL}")
        self.embedder = SentenceTransformer(EMBED_MODEL)

        print("[RAG] Iniciando ChromaDB...")
        self.chroma = chromadb.PersistentClient(path=str(DATA_DIR / "chroma_db"))
        self.col = self.chroma.get_or_create_collection(
            name="notas",
            metadata={"hnsw:space": "cosine"},
        )

        self.docs: dict = self._load_meta()
        print(f"[RAG] Listo. {len(self.docs)} documentos en la base de datos.")

    # ── Persistencia de metadatos ─────────────────────────────────────────────

    def _load_meta(self) -> dict:
        if self.meta_path.exists():
            return json.loads(self.meta_path.read_text(encoding="utf-8"))
        return {}

    def _save_meta(self):
        self.meta_path.write_text(
            json.dumps(self.docs, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # ── Procesamiento de PDFs ─────────────────────────────────────────────────

    def _extract_text(self, content: bytes) -> str:
        doc = fitz.open(stream=content, filetype="pdf")
        pages = []
        for page in doc:
            pages.append(page.get_text("text"))
        return "\n\n".join(pages)

    def _chunk_text(self, text: str) -> list[str]:
        """Chunking por palabras con solapamiento."""
        words = text.split()
        chunks = []
        step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
        for i in range(0, len(words), step):
            chunk = " ".join(words[i : i + CHUNK_SIZE])
            if len(chunk.strip()) > 30:  # descartar trozos demasiado cortos
                chunks.append(chunk.strip())
        return chunks

    # ── API pública ───────────────────────────────────────────────────────────

    def add_document(self, content: bytes, filename: str) -> dict:
        doc_id = hashlib.sha1(content).hexdigest()[:16]

        if doc_id in self.docs:
            return self.docs[doc_id]

        text = self._extract_text(content)
        chunks = self._chunk_text(text)

        if not chunks:
            raise ValueError("No se pudo extraer texto del PDF.")

        embeddings = self.embedder.encode(chunks, show_progress_bar=False).tolist()
        ids        = [f"{doc_id}_c{i}" for i in range(len(chunks))]
        metadatas  = [
            {"doc_id": doc_id, "filename": filename, "chunk": i}
            for i in range(len(chunks))
        ]

        self.col.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)

        entry = {
            "id":       doc_id,
            "filename": filename,
            "chunks":   len(chunks),
            "chars":    len(text),
            "pages":    len(fitz.open(stream=content, filetype="pdf")),
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

    def _retrieve(self, query: str, doc_ids: list[str]) -> list[str]:
        total = self.col.count()
        if total == 0:
            return []

        qe    = self.embedder.encode([query]).tolist()
        where = None
        if doc_ids:
            where = {"doc_id": doc_ids[0]} if len(doc_ids) == 1 else {"doc_id": {"$in": doc_ids}}

        results = self.col.query(
            query_embeddings=qe,
            n_results=min(TOP_K, total),
            where=where,
        )
        return results["documents"][0] if results["documents"] else []

    async def query(self, message: str, doc_ids: list[str], mode: str) -> str:
        chunks  = self._retrieve(message, doc_ids)
        context = "\n\n---\n\n".join(chunks) if chunks else "(Sin documentos seleccionados)"

        system = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["chat"])
        prompt = f"CONTEXTO DE LOS APUNTES:\n\n{context}\n\n{'─'*60}\n\nSOLICITUD DEL USUARIO:\n{message}"

        if LLM_PROVIDER == "anthropic":
            return await self._anthropic(system, prompt)
        return await self._ollama(system, prompt)

    # ── LLM backends ─────────────────────────────────────────────────────────

    async def _ollama(self, system: str, user: str) -> str:
        async with httpx.AsyncClient(timeout=180) as client:
            r = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model":    OLLAMA_MODEL,
                    "stream":   False,
                    "messages": [
                        {"role": "system",  "content": system},
                        {"role": "user",    "content": user},
                    ],
                },
            )
            r.raise_for_status()
            return r.json()["message"]["content"]

    async def _anthropic(self, system: str, user: str) -> str:
        async with httpx.AsyncClient(timeout=180) as client:
            r = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key":         ANTHROPIC_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type":      "application/json",
                },
                json={
                    "model":      ANTHROPIC_MODEL,
                    "max_tokens": 4096,
                    "system":     system,
                    "messages":   [{"role": "user", "content": user}],
                },
            )
            r.raise_for_status()
            return r.json()["content"][0]["text"]
