import os
from typing import List

from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

load_dotenv()

CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
TOP_K = int(os.getenv("TOP_K", "5"))
RERANK_MULTIPLIER = int(os.getenv("RERANK_MULTIPLIER", "3"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "5000"))
CACHE_SIZE = int(os.getenv("CACHE_SIZE", "256"))


class Retriever:
    def __init__(self, embedder, vector_store, indexer):
        self.embedder = embedder
        self.vs = vector_store
        self.indexer = indexer
        self.reranker = None
        self.retrieve_cache: dict[str, str] = {}

    def _bm25_search(self, query: str, corpus: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
        from rank_bm25 import BM25Okapi

        if not corpus:
            return []
        texts = [t for t, _ in corpus]
        tokenized = [t.lower().split() for t in texts]
        scores = BM25Okapi(tokenized).get_scores(query.lower().split())
        n = min(TOP_K * RERANK_MULTIPLIER, len(corpus))
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [(texts[i], corpus[i][1]) for i in top_idx]

    def _rrf(self, *rankings: list[tuple[str, dict]], k: int = 60) -> list[tuple[str, dict]]:
        scores = {}
        item_map = {}
        for ranking in rankings:
            for rank, (text, meta) in enumerate(ranking):
                key = text[:120]
                scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
                item_map[key] = (text, meta)
        return [item_map[k_] for k_ in sorted(scores, key=lambda k_: scores[k_], reverse=True)]

    def _get_reranker(self):
        if self.reranker is None:
            print(f"[Retriever] Cargando reranker: {CROSS_ENCODER_MODEL}")
            self.reranker = CrossEncoder(CROSS_ENCODER_MODEL)
        return self.reranker

    def retrieve(self, query: str, doc_ids: List[str] | None = None) -> str:
        total = self.vs.count()
        if total == 0:
            return ""

        normalized_query = " ".join(query.strip().lower().split())
        doc_key = ",".join(sorted(doc_ids)) if doc_ids else "all"
        cache_key = f"{normalized_query}|{doc_key}|{total}|{TOP_K}|{RERANK_MULTIPLIER}"
        if cache_key in self.retrieve_cache:
            return self.retrieve_cache[cache_key]

        where = None
        if doc_ids:
            where = {"doc_id": doc_ids[0]} if len(doc_ids) == 1 else {"doc_id": {"$in": doc_ids}}

        qe = [self.embedder.encode_query(query)]
        fetch_k = min(TOP_K * RERANK_MULTIPLIER, total)
        vec_res = self.vs.query(query_embeddings=qe, n_results=fetch_k, where=where)
        vec_list = list(zip(
            vec_res.get("documents", [[]])[0] if vec_res.get("documents") else [],
            vec_res.get("metadatas", [[]])[0] if vec_res.get("metadatas") else [],
        ))

        corpus = [(t, m) for t, m in self.indexer.bm25_corpus if not doc_ids or m.get("doc_id") in (doc_ids or [])]
        bm25_list = self._bm25_search(query, corpus)

        fused = self._rrf(vec_list, bm25_list)

        candidates = []
        for text, meta in fused:
            filename = meta.get("filename", "Desconocido")
            page = meta.get("page", "?")
            source_info = f"[Fuente: {filename}, Pág: {page}]"
            candidates.append({"text": text, "meta": meta, "source_info": source_info})

        if not candidates:
            return ""

        try:
            reranker = self._get_reranker()
            pair_inputs = [(query, c["text"]) for c in candidates]
            scores = reranker.predict(pair_inputs)
            candidates = [c for _, c in sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)]
        except Exception as e:
            print(f"[Retriever] Aviso: reranking fallido ({e}). Usando orden RRF.")

        context_parts = []
        seen_texts = set()
        used_chars = 0
        for candidate in candidates:
            normalized_text = " ".join(candidate["text"].split())
            if normalized_text in seen_texts:
                continue
            context_piece = f"{candidate['source_info']}\n{candidate['text']}"
            if context_parts and used_chars + len(context_piece) > MAX_CONTEXT_CHARS:
                break
            seen_texts.add(normalized_text)
            context_parts.append(context_piece)
            used_chars += len(context_piece)
            if len(context_parts) >= TOP_K:
                break

        context = "\n\n---\n\n".join(context_parts)

        if len(self.retrieve_cache) >= CACHE_SIZE:
            self.retrieve_cache.pop(next(iter(self.retrieve_cache)))
        self.retrieve_cache[cache_key] = context
        return context
