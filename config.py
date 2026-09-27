"""
Central configuration for the hybrid RAG pipeline.
All tunables live here so nothing is buried in code.
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # falls back to whatever is already in the environment

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent
PDF_DIR = PROJECT_ROOT / "papers"          # drop your research PDFs here
INDEX_DIR = PROJECT_ROOT / "index_store"   # chroma + bm25 persisted here
CHROMA_DIR = INDEX_DIR / "chroma"
BM25_PATH = INDEX_DIR / "bm25_index.pkl"
DOCSTORE_PATH = INDEX_DIR / "docstore.jsonl"  # chunk_id -> text/metadata, for BM25 lookups
EVAL_LOG_PATH = Path("eval_logs/query_log.jsonl")

# --- API keys ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# --- Embeddings (local, via langchain HuggingFaceEmbeddings) ---
# bge-base is a strong general-purpose retrieval model; swap for a domain-specific
# one if you find something better tuned for scientific text (e.g. allenai/specter2).
EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"
EMBED_DEVICE = "cuda"          # set to "cuda" if you have a GPU available
# bge models expect a query-side instruction prefix for asymmetric search
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# --- Gemini models ---
GENERATION_MODEL = "meta-llama/llama-3.1-8b-instruct"
TESTSET_GENERATOR_MODEL = "meta-llama/llama-3.1-8b-instruct"
TESTSET_CRITIC_MODEL = "meta-llama/llama-3.1-8b-instruct"
# --- Local cross-encoder reranker (replaces Cohere rerank, runs on CPU) ---
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- Generation ---
MAX_ANSWER_TOKENS = 800

# --- Chunking ---
CHUNK_SIZE_CHARS = 8000 #1400       # ~300-350 tokens, good for dense academic text
CHUNK_OVERLAP_CHARS =100 #200

# --- Retrieval ---
DENSE_TOP_K = 20        # candidates pulled from Chroma before fusion
SPARSE_TOP_K = 20       # candidates pulled from BM25 before fusion
RRF_K = 60              # standard reciprocal-rank-fusion constant
FUSED_TOP_K = 10        # how many survive fusion into reranking
FINAL_TOP_K = 5         # how many go into the LLM context after rerank
RRF_ONLY = False        # if True, skip cross-encoder rerank and use fused ranking directly

COLLECTION_NAME = "research_papers"
