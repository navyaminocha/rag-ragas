"""
Build both halves of the hybrid index:
  - Dense: local HuggingFace embeddings (via langchain) stored in a persistent
    langchain Chroma vectorstore
  - Sparse: BM25Okapi over tokenized chunk text, pickled to disk

Both indexes are keyed by the same chunk `id` so results can be fused later.
"""
import json
import pickle
import re

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from config import (
    EMBED_MODEL_NAME,
    EMBED_DEVICE,
    CHROMA_DIR,
    BM25_PATH,
    DOCSTORE_PATH,
    COLLECTION_NAME,
    PDF_DIR,
)
from ingest import load_and_chunk_all, Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def get_embedder() -> HuggingFaceEmbeddings:
    """
    Shared embedder used at both index-build time and query time, so the two
    stay consistent. bge models are trained for cosine similarity over
    normalized vectors, hence normalize_embeddings=True.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={"device": EMBED_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_indexes(chunks: list[Chunk]) -> None:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    BM25_PATH.parent.mkdir(parents=True, exist_ok=True)

    embedder = get_embedder()

    print(f"Embedding {len(chunks)} chunks locally with {EMBED_MODEL_NAME}...")
    documents = [
        Document(
            page_content=c.text,
            metadata={
                "id": c.id,
                "source": c.source,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "chunk_index": c.chunk_index,
            },
        )
        for c in chunks
    ]
    ids = [c.id for c in chunks]

    # Fresh collection on every full rebuild
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedder,
        persist_directory=str(CHROMA_DIR),
    )
    try:
        vectorstore.delete_collection()
    except Exception:
        pass
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedder,
        ids=ids,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )
    vectorstore.persist()
    print(f"Dense index: {len(ids)} vectors persisted to {CHROMA_DIR}")

    # --- BM25 sparse index ---
    print("Building BM25 sparse index...")
    tokenized_corpus = [tokenize(c.text) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "ids": ids}, f)
    print(f"Sparse index persisted to {BM25_PATH}")

    # --- Docstore (id -> text/metadata) used at query time to hydrate BM25 hits ---
    with open(DOCSTORE_PATH, "w") as f:
        for c in chunks:
            f.write(json.dumps({
                "id": c.id,
                "text": c.text,
                "source": c.source,
                "page_start": c.page_start,
                "page_end": c.page_end,
            }) + "\n")
    print(f"Docstore persisted to {DOCSTORE_PATH}")


if __name__ == "__main__":
    print(f"Loading + chunking PDFs from {PDF_DIR} ...")
    chunks = load_and_chunk_all()
    build_indexes(chunks)
    print("\nDone. Run `python cli.py query \"your question\"` or `streamlit run app.py`.")
