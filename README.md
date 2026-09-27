# rag-ragas

Evaluation harness for the [Hybrid Search RAG for Research Papers](https://github.com/navyaminocha/rag-ragas) pipeline : a dense (embeddings) + sparse (BM25) hybrid retriever, reranked with a cross-encoder, evaluated end-to-end with **RAGAS** and tracked with **MLflow**.

The core idea: don't just build a RAG pipeline and eyeball a few answers : build a repeatable loop that (1) generates a synthetic eval set straight from your own PDFs, (2) lets you curate/trust that set before scoring anything, (3) runs the four standard RAGAS metrics against it, and (4) logs every run's config + metrics + per-question breakdown to MLflow so runs are comparable over time.

---

## Architecture

```
PDFs (papers/)
   │
   ▼
ingest.py          PyMuPDF text extraction, page-by-page, de-hyphenated,
                    then sliding-window chunking across page boundaries
   │
   ▼
indexer.py          ├─ dense:  BAAI/bge-base-en-v1.5 embeddings → Chroma (persistent)
                     └─ sparse: BM25Okapi over tokenized chunk text
                     both keyed by the same chunk_id
   │
   ▼
retriever.py        dense top-k ∥ sparse top-k  →  Reciprocal Rank Fusion (RRF)
                     →  cross-encoder/ms-marco-MiniLM-L-6-v2 reranks fused top-k
                     →  final_top_k chunks
   │
   ▼
generator.py         builds a source/page-tagged context block, prompts an LLM
                      (via OpenRouter) to answer strictly from those excerpts
                      with inline (source, p.X) citations
   │
   ▼
pipeline.py           RAGPipeline.query() glues retrieval + generation together,
                       used by both the CLI and the Streamlit app
```

`app.py` (Streamlit) and `cli.py` are the two ways to actually *use* the pipeline. Everything below this line is the **evaluation** layer built on top of it.

---

## Evaluation pipeline

The eval side of the repo has three stages, each its own script, designed so you never score against ground-truth you haven't actually checked:

### 1. Generate candidate questions : `generate_testset.py`
Uses RAGAS's `TestsetGenerator` to synthesize question / ground-truth pairs directly from the indexed PDFs (it does its own document chunking internally, independent of the retrieval index). Produces a mix of:
- **simple** questions (default 50%)
- **reasoning** questions (default 25%)
- **multi-context** questions (default 25%)

```
python generate_testset.py --num-questions 20 --simple 0.5 --reasoning 0.25 --multi-context 0.25
```

Every generated pair is written to `eval_questions.json` tagged `"reviewed": false, "source": "synthetic"` : nothing generated here is trusted automatically. Full generation detail (contexts, evolution type) is also dumped to `eval_questions_full.csv` for inspection.

### 2. Curate from real usage : `curate_eval_set.py`
An alternative/complementary source of eval questions: real queries logged from the Streamlit app (via the "save this Q&A for evaluation" button) get a **draft reference answer** written for them, grounded only in the passages that were actually retrieved for that query at the time, using a stronger judge LLM. These also land in `eval_questions.json` as `"reviewed": false, "source": "logged"`.

```
python curate_eval_set.py --limit 10
```

**Either way, a human has to open `eval_questions.json` and flip `"reviewed": true` on each entry they've checked** : `ragas_eval.py` skips everything else. This is the guardrail against silently scoring against shallow or hallucinated synthetic ground truth.

### 3. Score + log : `ragas_eval.py`
Runs the hybrid pipeline against every reviewed question, times retrieval and generation separately, scores the results with RAGAS, and logs everything to MLflow.

```
python ragas_eval.py --questions eval_questions.json --experiment "Rag pipeline basic"
mlflow ui   # view results
```

---

## Metrics (RAGAS)

Four standard RAGAS metrics are computed per question and averaged into the run's aggregate score:

| Metric | What it measures | Needs ground truth? |
|---|---|---|
| **Faithfulness** | Whether every claim in the generated answer is actually supported by the retrieved context (checks for hallucination) | No |
| **Answer Relevancy** | How well the generated answer actually addresses the question asked (penalizes vague/incomplete/off-topic answers) | No |
| **Context Precision** | Of the chunks retrieved, how many were actually relevant/useful for answering the question | Yes |
| **Context Recall** | Whether the retrieved context covers everything needed to produce the ground-truth answer (a retrieval-quality metric) | Yes |

Because `context_precision` and `context_recall` need a ground-truth answer to compare against, their scores are only meaningful for rows where `ground_truth` in `eval_questions.json` is a real, reviewed answer : not a placeholder.

Scoring itself is done by an **LLM-as-judge** (`config.GENERATION_MODEL`, via OpenRouter) plus the same local embedding model used for retrieval (`BAAI/bge-base-en-v1.5`), so no extra API keys are needed beyond `OPENROUTER_API_KEY`.

### Alongside the RAGAS scores, each run also logs:
- `retrieval_time` / `generation_time` : mean seconds per question, timed separately
- `retrieved_chunks` : mean number of chunks that made it into the final context
- `question_words` / `answer_words` : rough verbosity signal

### And as run parameters (for comparing configs across runs):
`embed_model`, `rerank_model`, `generation_model`, `chunk_size_chars`, `chunk_overlap_chars`, `final_top_k`, `num_questions`

Every run also writes a **per-question CSV** (`eval_results.csv`) : question, answer, ground truth, contexts, timings, and each RAGAS metric individually : and logs it as an MLflow artifact, so you can drill into which specific questions dragged an average down instead of only seeing the aggregate.

MLflow run names auto-increment (`Evaluation_1`, `Evaluation_2`, …) per experiment, so repeated tuning runs stay organized without manual naming.

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in OPENROUTER_API_KEY
```

Drop PDFs into `papers/`, or upload them through the Streamlit sidebar.

## Usage

**Streamlit app (recommended for interactive use):**
```bash
streamlit run app.py
```
Upload PDFs → **Build / Rebuild index** → ask questions in the chat box. Answers show sources with page numbers, and there's a button to log a Q&A pair for later curation into the eval set.

**CLI:**
```bash
python cli.py build                       # chunk PDFs, build dense + sparse indexes
python cli.py query "your question here"  # hybrid retrieve + generate answer
```

**Evaluation loop, end to end:**
```bash
python cli.py build
python generate_testset.py --num-questions 20
# open eval_questions.json, check the drafted ground_truth answers, set "reviewed": true
python ragas_eval.py --experiment "Rag pipeline basic"
mlflow ui
```

---

## Tuning

All knobs live in `config.py` : nothing is buried in code:

| Setting | Purpose |
|---|---|
| `EMBED_MODEL_NAME` | dense embedding model (default `BAAI/bge-base-en-v1.5`) |
| `RERANK_MODEL_NAME` | local cross-encoder reranker (default `cross-encoder/ms-marco-MiniLM-L-6-v2`) |
| `GENERATION_MODEL` / `TESTSET_GENERATOR_MODEL` / `TESTSET_CRITIC_MODEL` | LLMs used for answering, testset generation, and testset critique (via OpenRouter) |
| `CHUNK_SIZE_CHARS` / `CHUNK_OVERLAP_CHARS` | chunking window over extracted PDF text |
| `DENSE_TOP_K` / `SPARSE_TOP_K` | candidates pulled from each retriever before fusion |
| `RRF_K` | Reciprocal Rank Fusion constant (standard default: 60) |
| `FUSED_TOP_K` | how many survive RRF fusion into reranking |
| `FINAL_TOP_K` | how many chunks actually go into the LLM's context |
| `RRF_ONLY` | skip the cross-encoder rerank and use fused ranking directly (ablation switch) |

Because these are all logged as MLflow run parameters, you can flip one knob (e.g. `RRF_ONLY` or `FINAL_TOP_K`), rerun `ragas_eval.py`, and compare the metric deltas directly in the MLflow UI.

## Notes

- Rebuilding the index (`cli.py build` / the Streamlit sidebar button) wipes and recreates the Chroma collection : it's a full rebuild, not an incremental upsert.
- First run downloads the embedding model (~440MB) and cross-encoder (~90MB) from Hugging Face; cached locally afterward.
- Retrieval, reranking, and embeddings all run locally/offline : only generation and RAGAS's judge calls hit an external API (OpenRouter).
- Synthetic ground truth from `generate_testset.py` can be shallow or occasionally wrong : it's meant as a draft to review, not a finished benchmark.
