"""
Hybrid retrieval pipeline:

  1. Dense search  -> langchain Chroma vectorstore, local HuggingFace embeddings
  2. Sparse search -> BM25 over tokenized corpus
  3. Fuse both ranked lists with Reciprocal Rank Fusion (RRF) — robust,
     no score-scale normalization needed between embedding distance and BM25 scores.
  4. Rerank the fused candidates with a local cross-encoder for a final precision pass
     (replaces the old Cohere rerank call — no external API needed).

RRF: score(doc) = sum over each ranker of 1 / (k + rank_in_that_list)
"""
import json
import pickle
from dataclasses import dataclass

from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder

from config import (
    CHROMA_DIR,
    BM25_PATH,
    DOCSTORE_PATH,
    COLLECTION_NAME,
    RERANK_MODEL_NAME,
    BGE_QUERY_INSTRUCTION,
    DENSE_TOP_K,
    SPARSE_TOP_K,
    RRF_K,
    FUSED_TOP_K,
    FINAL_TOP_K,
    RRF_ONLY,
)
from indexer import tokenize, get_embedder


@dataclass
class RetrievedChunk:
    id: str
    text: str
    source: str
    page_start: int
    page_end: int
    score: float
    rank_dense: int | None = None
    rank_sparse: int | None = None


class HybridRetriever:
    def __init__(self):
        self.embedder = get_embedder()
        self.vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embedder,
            persist_directory=str(CHROMA_DIR),
        )

        with open(BM25_PATH, "rb") as f:
            bm25_data = pickle.load(f)
        self.bm25 = bm25_data["bm25"]
        self.bm25_ids = bm25_data["ids"]

        self.docstore: dict[str, dict] = {}
        with open(DOCSTORE_PATH) as f:
            for line in f:
                row = json.loads(line)
                self.docstore[row["id"]] = row

        self._cross_encoder = None  # lazy-loaded, only needed if reranking

    @property
    def cross_encoder(self) -> CrossEncoder:
        if self._cross_encoder is None:
            self._cross_encoder = CrossEncoder(RERANK_MODEL_NAME)
        return self._cross_encoder

    # -- individual retrievers --

    def _dense_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        # bge models expect an instruction prefix on the query side only
        prefixed_query = BGE_QUERY_INSTRUCTION + query
        results = self.vectorstore.similarity_search_with_score(prefixed_query, k=top_k)
        # metadata["id"] is our chunk id; Chroma's internal distance is returned as score
        return [(doc.metadata["id"], score) for doc, score in results]

    def _sparse_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(zip(self.bm25_ids, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    # -- fusion --

    def _rrf_fuse(
        self, dense: list[tuple[str, float]], sparse: list[tuple[str, float]]
    ) -> list[RetrievedChunk]:
        rrf_scores: dict[str, float] = {}
        rank_dense: dict[str, int] = {}
        rank_sparse: dict[str, int] = {}

        for rank, (cid, _) in enumerate(dense, start=1):
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (RRF_K + rank)
            rank_dense[cid] = rank

        for rank, (cid, _) in enumerate(sparse, start=1):
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (RRF_K + rank)
            rank_sparse[cid] = rank

        fused_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
        results = []
        for cid in fused_ids[:FUSED_TOP_K]:
            row = self.docstore.get(cid)
            if row is None:
                continue
            results.append(
                RetrievedChunk(
                    id=cid,
                    text=row["text"],
                    source=row["source"],
                    page_start=row["page_start"],
                    page_end=row["page_end"],
                    score=rrf_scores[cid],
                    rank_dense=rank_dense.get(cid),
                    rank_sparse=rank_sparse.get(cid),
                )
            )
        return results

    # -- rerank (local cross-encoder, no external API) --

    def _rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not candidates:
            return candidates
        pairs = [(query, c.text) for c in candidates]
        scores = self.cross_encoder.predict(pairs)
        for c, s in zip(candidates, scores):
            c.score = float(s)
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:FINAL_TOP_K]

    # -- public API --

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        dense = self._dense_search(query, DENSE_TOP_K)
        sparse = self._sparse_search(query, SPARSE_TOP_K)
        fused = self._rrf_fuse(dense, sparse)
        if RRF_ONLY:
            return fused[:FINAL_TOP_K]
        return self._rerank(query, fused)


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "What is the main contribution of this paper?"
    retriever = HybridRetriever()
    results = retriever.retrieve(q)
    for r in results:
        print(f"[{r.score:.4f}] {r.source} p.{r.page_start}-{r.page_end} "
              f"(dense_rank={r.rank_dense}, sparse_rank={r.rank_sparse})")
        print(f"  {r.text[:180]}...\n")
