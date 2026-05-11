from pathlib import Path
import chromadb
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent.parent.parent / "data"


class VectorStore:
    def __init__(self, collection_name: str = "notas"):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        print("[VectorStore] Inicializando ChromaDB client")
        self.client = chromadb.PersistentClient(path=str(DATA_DIR / "chroma_db"))
        self.col = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, ids, embeddings, documents, metadatas):
        return self.col.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def query(self, query_embeddings, n_results=5, where=None):
        return self.col.query(query_embeddings=query_embeddings, n_results=n_results, where=where)

    def get(self, include=None):
        include = include or ["documents", "metadatas", "ids"]
        return self.col.get(include=include)

    def get_where(self, where, include=None):
        include = include or ["documents", "metadatas", "ids"]
        return self.col.get(where=where, include=include)

    def delete(self, ids):
        return self.col.delete(ids=ids)

    def count(self):
        return self.col.count()
