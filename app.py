"""
Streamlit frontend for the hybrid search RAG pipeline.

Run with: streamlit run app.py
"""
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from config import PDF_DIR, CHROMA_DIR, BM25_PATH, DOCSTORE_PATH, OPENROUTER_API_KEY, EVAL_LOG_PATH

st.set_page_config(page_title="Research Paper Hybrid RAG", page_icon="📄", layout="wide")


def index_exists() -> bool:
    return CHROMA_DIR.exists() and BM25_PATH.exists() and DOCSTORE_PATH.exists()


def log_for_eval(question: str, answer: str, sources: list) -> None:
    """Append a real Q&A exchange to the raw eval log (curate_eval_set.py reads this
    later to draft candidate ground_truth — this button does NOT modify eval_questions.json
    directly, that would skip the review step)."""
    EVAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "question": question,
        "answer": answer,
        "contexts": [s.text for s in sources],
        "sources": [{"source": s.source, "page_start": s.page_start, "page_end": s.page_end} for s in sources],
        "logged_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(EVAL_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


@st.cache_resource(show_spinner=False)
def load_pipeline():
    from pipeline import RAGPipeline
    return RAGPipeline()


def run_build():
    from ingest import load_and_chunk_all
    from indexer import build_indexes

    with st.status("Building index...", expanded=True) as status:
        st.write("Chunking PDFs...")
        chunks = load_and_chunk_all()
        st.write(f"{len(chunks)} chunks extracted.")
        st.write("Embedding + indexing (dense + BM25)... this can take a while on first run.")
        build_indexes(chunks)
        status.update(label="Index built.", state="complete")

    load_pipeline.clear()  # force reload with the fresh index
    st.session_state["index_ready"] = True


# --- Sidebar: corpus management ---

with st.sidebar:
    st.header("📄 Corpus")

    if not OPENROUTER_API_KEY:
        st.warning("OPENROUTER_API_KEY is not set. Add it to your .env file before querying.")

    uploaded = st.file_uploader("Upload research PDFs", type="pdf", accept_multiple_files=True)
    if uploaded:
        PDF_DIR.mkdir(parents=True, exist_ok=True)
        for f in uploaded:
            (PDF_DIR / f.name).write_bytes(f.getbuffer())
        st.success(f"Saved {len(uploaded)} file(s) to {PDF_DIR.name}/")

    existing = sorted(PDF_DIR.glob("*.pdf")) if PDF_DIR.exists() else []
    st.caption(f"{len(existing)} PDF(s) currently in the corpus")
    with st.expander("View files"):
        for p in existing:
            st.text(p.name)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔨 Build / Rebuild index", disabled=not existing, use_container_width=True):
            run_build()
    with col2:
        if st.button("🗑️ Clear index", use_container_width=True):
            for path in [CHROMA_DIR, BM25_PATH, DOCSTORE_PATH]:
                if path.exists():
                    shutil.rmtree(path) if path.is_dir() else path.unlink()
            load_pipeline.clear()
            st.session_state["index_ready"] = False
            st.success("Index cleared.")

    if "index_ready" not in st.session_state:
        st.session_state["index_ready"] = index_exists()

    st.divider()
    with st.expander("⚙️ Retrieval settings (read-only, edit config.py)"):
        import config
        st.text(f"Embedding model: {config.EMBED_MODEL_NAME}")
        st.text(f"Rerank model: {config.RERANK_MODEL_NAME}")
        st.text(f"Generation model: {config.GENERATION_MODEL}")
        st.text(f"Dense/Sparse top-k: {config.DENSE_TOP_K}/{config.SPARSE_TOP_K}")
        st.text(f"Final top-k after rerank: {config.FINAL_TOP_K}")

# --- Main: chat ---

st.title("Hybrid Search RAG over your research papers")
st.caption("Dense (local embeddings) + sparse (BM25) retrieval, fused with RRF, "
           "reranked with a local cross-encoder, answered with Gemini.")

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "logged_eval_ids" not in st.session_state:
    st.session_state["logged_eval_ids"] = set()

for i, msg in enumerate(st.session_state["messages"]):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    pages = f"p.{s.page_start}" if s.page_start == s.page_end else f"p.{s.page_start}-{s.page_end}"
                    st.markdown(f"**{s.source}** — {pages} (score: {s.score:.4f})")
                    st.caption(s.text[:300] + "...")
            if i in st.session_state["logged_eval_ids"]:
                st.caption("📌 Saved for evaluation")
            elif st.button("📌 Save this Q&A for evaluation", key=f"log_{i}"):
                log_for_eval(msg["question"], msg["content"], msg["sources"])
                st.session_state["logged_eval_ids"].add(i)
                st.rerun()

if not st.session_state["index_ready"]:
    st.info("Upload PDFs and click **Build / Rebuild index** in the sidebar to get started.")
else:
    question = st.chat_input("Ask a question about your papers...")
    if question:
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving + generating..."):
                pipeline = load_pipeline()
                result = pipeline.query(question)
            st.markdown(result.answer)
            if result.sources:
                with st.expander("Sources"):
                    for s in result.sources:
                        pages = f"p.{s.page_start}" if s.page_start == s.page_end else f"p.{s.page_start}-{s.page_end}"
                        st.markdown(f"**{s.source}** — {pages} (score: {s.score:.4f})")
                        st.caption(s.text[:300] + "...")

        st.session_state["messages"].append({
            "role": "assistant", "content": result.answer, "sources": result.sources, "question": question
        })
        st.rerun()  # rerun so the "Save for evaluation" button renders via the replay loop above
