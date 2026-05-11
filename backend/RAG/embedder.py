import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
CACHE_SIZE = int(os.getenv("CACHE_SIZE", "256"))


class Embedder:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or EMBED_MODEL
        print(f"[Embedder] Cargando modelo de embeddings: {self.model_name}")
        self._enc = SentenceTransformer(self.model_name)
        self._cache: dict[str, list[float]] = {}

    def encode(self, texts: list[str], show_progress_bar: bool = False) -> list[list[float]]:
        arr = self._enc.encode(texts, show_progress_bar=show_progress_bar)
        try:
            return arr.tolist()
        except Exception:
            return list(map(list, arr))

    def encode_query(self, query: str) -> list[float]:
        key = " ".join(query.strip().lower().split())
        if key in self._cache:
            return self._cache[key]
        emb = self._enc.encode([query]).tolist()[0]
        if len(self._cache) >= CACHE_SIZE:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = emb
        return emb
