"""
RAG Engine — Orquestador ligero.
Este módulo crea e integra los componentes especializados:
`embedder.py`, `vector_store.py`, `indexer.py`, `retriever.py`.
Mantiene la lógica de streaming hacia proveedores LLM (Ollama/Anthropic).
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
import httpx
from typing import AsyncGenerator

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")

from RAG.embedder import Embedder
from RAG.vector_store import VectorStore
from RAG.indexer import Indexer
from RAG.retriever import Retriever

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
        print("[RAG] Iniciando componentes: embedder, vector_store, indexer, retriever")
        self.embedder = Embedder()
        self.vs = VectorStore()
        self.indexer = Indexer(self.embedder, self.vs)
        self.retriever = Retriever(self.embedder, self.vs, self.indexer)

    # — Document management (delegado a indexer)
    def add_document(self, content: bytes, filename: str) -> dict:
        entry = self.indexer.add_document(content, filename)
        # rebuild bm25 corpus in retriever via indexer state
        self.indexer._build_bm25_corpus()
        return entry

    def list_documents(self) -> list[dict]:
        return self.indexer.list_documents()

    def delete_document(self, doc_id: str):
        self.indexer.delete_document(doc_id)
        self.indexer._build_bm25_corpus()

    # — Prompt & history helpers
    def _build_prompt(self, message: str, doc_ids: list[str]) -> tuple[str, str]:
        context = self.retriever.retrieve(message, doc_ids)
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

    # — Streaming hacia LLMs (igual comportamiento que antes)
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
                "POST", f"{OLLAMA_BASE_URL}/api/chat",
                json={"model": OLLAMA_MODEL, "stream": True, "messages": messages, "options": {"num_ctx": 8192}},
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
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
                "POST", "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": os.getenv("ANTHROPIC_API_KEY", ""), "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json={"model": ANTHROPIC_MODEL, "max_tokens": 4096, "stream": True, "system": system, "messages": messages},
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
