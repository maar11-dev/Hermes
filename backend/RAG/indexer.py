import hashlib
import json
from pathlib import Path
from typing import List

import fitz
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent.parent.parent / "data"


class Indexer:
    def __init__(self, embedder, vector_store):
        self.embedder = embedder
        self.vs = vector_store
        self.meta_path = DATA_DIR / "documents.json"
        self.docs = self._load_meta()
        self.bm25_corpus: list[tuple[str, dict]] = []
        self._build_bm25_corpus()

    def _load_meta(self) -> dict:
        if self.meta_path.exists():
            return json.loads(self.meta_path.read_text(encoding="utf-8"))
        return {}

    def _save_meta(self):
        self.meta_path.write_text(json.dumps(self.docs, indent=2, ensure_ascii=False), encoding="utf-8")

    def _build_bm25_corpus(self):
        if self.vs.count() == 0:
            self.bm25_corpus = []
            return
        all_data = self.vs.get(include=["documents", "metadatas"])
        if all_data.get("documents"):
            self.bm25_corpus = list(zip(all_data["documents"], all_data["metadatas"]))

    def _extract_text_with_pages(self, content: bytes) -> List[dict]:
        pages_data = []
        doc = fitz.open(stream=content, filetype="pdf")
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text:
                pages_data.append({"page_num": i + 1, "text": text})
        return pages_data

    def _chunk_text(self, pages_data: List[dict], chunk_size: int = 600, overlap: int = 100) -> List[dict]:
        chunks_with_meta = []
        for page in pages_data:
            words = page["text"].split()
            step = max(1, chunk_size - overlap)
            for i in range(0, len(words), step):
                chunk_text = " ".join(words[i: i + chunk_size])
                if len(chunk_text.strip()) > 40:
                    chunks_with_meta.append({"text": chunk_text, "page": page["page_num"]})
        return chunks_with_meta

    def add_document(self, content: bytes, filename: str) -> dict:
        doc_id = hashlib.sha1(content).hexdigest()[:16]
        if doc_id in self.docs:
            return self.docs[doc_id]

        pages_data = self._extract_text_with_pages(content)
        chunks_data = self._chunk_text(pages_data)

        if not chunks_data:
            raise ValueError("No se pudo extraer texto del PDF (podría ser una imagen que necesita OCR).")

        texts = [c["text"] for c in chunks_data]
        embeddings = self.embedder.encode(texts, show_progress_bar=False)
        ids = [f"{doc_id}_c{i}" for i in range(len(chunks_data))]
        metadatas = [{"doc_id": doc_id, "filename": filename, "page": c["page"]} for c in chunks_data]

        self.vs.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

        self.bm25_corpus.extend(zip(texts, metadatas))

        total_chars = sum(len(c["text"]) for c in chunks_data)
        entry = {"id": doc_id, "filename": filename, "chunks": len(chunks_data), "pages": len(pages_data), "chars": total_chars}
        self.docs[doc_id] = entry
        self._save_meta()
        return entry

    def list_documents(self) -> list:
        return list(self.docs.values())

    def delete_document(self, doc_id: str):
        results = self.vs.get_where(where={"doc_id": doc_id})
        if results.get("ids"):
            self.vs.delete(ids=results["ids"])
        self.docs.pop(doc_id, None)
        self._save_meta()
        self.bm25_corpus = [(t, m) for t, m in self.bm25_corpus if m.get("doc_id") != doc_id]
