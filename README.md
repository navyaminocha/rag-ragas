# Hybrid Search RAG for Research Papers

Dense (local HuggingFace embeddings via langchain + Chroma) + sparse (BM25)
retrieval, fused with Reciprocal Rank Fusion, reranked with a local
cross-encoder, answered with Gemini. Streamlit frontend included.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in GOOGLE_API_KEY (from Google AI Studio)
```

Drop your PDFs into `papers/`, or upload them through the Streamlit sidebar.

## Usage

**Streamlit app (recommended):**
```bash
streamlit run app.py
```
Upload PDFs in the sidebar, click **Build / Rebuild index**, then ask questions
in the chat box. Sources with page numbers are shown per answer.

**CLI:**
```bash
python cli.py build                       # chunk PDFs, build dense + sparse indexes
python cli.py query "your question here"  # hybrid retrieve + generate answer
```

## How it works

1. **`ingest.py`** — PyMuPDF extracts text page-by-page (de-hyphenates
   line-wrapped words), then a sliding char window chunks the whole document
   (not page-by-page, so chunks aren't artificially cut at page breaks).
2. **`indexer.py`** — embeds each chunk locally with `BAAI/bge-base-en-v1.5`
   via langchain's `HuggingFaceEmbeddings`, stored in a persistent langchain
   `Chroma` vectorstore. Separately builds a BM25Okapi index over tokenized
   chunk text. Both are keyed by the same chunk id (stored in Chroma metadata).
3. **`retriever.py`** — at query time, runs dense search (Chroma, embeddings
   normalized for cosine similarity, with the bge query-instruction prefix
   prepended) and sparse search (BM25) in parallel, fuses the two ranked
   lists with RRF (`score = Σ 1/(k + rank)`, no score normalization needed
   since RRF only uses rank position), then reranks the fused top candidates
   with a local `cross-encoder/ms-marco-MiniLM-L-6-v2` model — no external
   rerank API needed.
4. **`generator.py`** — builds a context block with per-chunk source/page
   tags and asks Gemini (`gemini-2.5-flash` by default) to answer strictly
   from those excerpts, citing `(source, p.X)` inline.
5. **`pipeline.py`** — wraps retrieval + generation into one
   `RAGPipeline.query()` call, used by both `cli.py` and `app.py`.
6. **`app.py`** — Streamlit UI: upload PDFs, build/rebuild/clear the index,
   and chat against the corpus with an expandable sources panel per answer.

## Tuning

All knobs (chunk size, top-k at each stage, RRF constant, embedding/rerank
model names, whether to skip reranking) live in `config.py`.

## Notes

- Rebuilding (`build` / the sidebar button) wipes and recreates the Chroma
  collection — it's a full rebuild, not incremental upsert. For incremental
  adds on a large corpus, extend `indexer.py` to check existing ids first.
- First run downloads the embedding model (~440MB) and cross-encoder
  (~90MB) from HuggingFace; they're cached locally afterward.
- Everything except generation runs locally/offline — only the final answer
  call hits the Gemini API.
- Swap `GENERATION_MODEL` or `EMBED_MODEL_NAME` in `config.py` to try
  different models; set `EMBED_DEVICE = "cuda"` if you have a GPU.
