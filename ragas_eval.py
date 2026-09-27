"""
Run a RAGAS evaluation over a test question set against the hybrid RAG
pipeline, and log everything to MLflow so you get the same dashboard as
the Evaluation_N screenshot: an Overview tab with per-metric values, plus
the per-question breakdown as a logged artifact.

Usage:
    python ragas_eval.py                          # uses eval_questions.json
    python ragas_eval.py --questions my_set.json
    python ragas_eval.py --experiment Neil_RAG

Then view results with:
    mlflow ui
"""
import argparse
import json
import time
from pathlib import Path

import mlflow
import pandas as pd
from datasets import Dataset
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

import config
from generator import generate_answer
from indexer import get_embedder
from retriever import HybridRetriever

DEFAULT_QUESTIONS_PATH = config.PROJECT_ROOT / "eval_questions.json"
DEFAULT_EXPERIMENT = "Rag pipeline basic"


def load_questions(path: Path) -> list[dict]:
    with open(path) as f:
        rows = json.load(f)
    if not rows:
        raise ValueError(f"{path} is empty — add at least one reviewed question/ground_truth pair.")

    reviewed = [r for r in rows if r.get("reviewed", True)]  # entries with no flag = assume hand-written/trusted
    skipped = len(rows) - len(reviewed)
    if skipped:
        print(f"Skipping {skipped} unreviewed entr{'y' if skipped == 1 else 'ies'} in {path} "
              f"(set \"reviewed\": true once you've checked the draft ground_truth).")
    if not reviewed:
        raise ValueError(
            f"All {len(rows)} entries in {path} are unreviewed. Open the file, check each "
            f"draft ground_truth against the retrieved contexts, and set \"reviewed\": true "
            f"before running eval — see curate_eval_set.py."
        )
    return reviewed


def run_pipeline_with_timing(retriever: HybridRetriever, question: str) -> dict:
    """Run retrieval + generation separately so each stage can be timed,
    matching retrieval_time / generation_time in the target dashboard."""
    t0 = time.perf_counter()
    chunks = retriever.retrieve(question)
    retrieval_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    answer = generate_answer(question, chunks)
    generation_time = time.perf_counter() - t0

    return {
        "answer": answer,
        "contexts": [c.text for c in chunks],
        "retrieved_chunks": len(chunks),
        "retrieval_time": retrieval_time,
        "generation_time": generation_time,
        "question_words": len(question.split()),
        "answer_words": len(answer.split()),
    }


def build_eval_records(questions: list[dict]) -> list[dict]:
    retriever = HybridRetriever()
    records = []
    for row in questions:
        q = row["question"]
        gt = row.get("ground_truth", "")
        print(f"Running: {q}")
        result = run_pipeline_with_timing(retriever, q)
        records.append({
            "question": q,
            "ground_truth": gt,
            **result,
        })
    return records


def run_ragas(records: list[dict]):
    """Score answer_relevancy / context_precision / context_recall / faithfulness.
    context_precision and context_recall need ground_truth; if it's missing/placeholder
    for a row, ragas will still run but that row's precision/recall may be unreliable —
    fill in real ground_truth answers in eval_questions.json for meaningful numbers."""
    ds = Dataset.from_list([
        {
            "question": r["question"],
            "answer": r["answer"],
            "contexts": r["contexts"],
            "ground_truth": r["ground_truth"],
        }
        for r in records
    ])

    judge_llm = LangchainLLMWrapper(
    ChatOpenAI(
        model=config.GENERATION_MODEL,
        api_key=config.OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )
    )

    judge_embeddings = LangchainEmbeddingsWrapper(
    get_embedder()
    )

    result = evaluate(
    ds,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ],
    llm=judge_llm,
    embeddings=judge_embeddings,
    )
    return result

def next_run_name(experiment_name: str, prefix: str = "Evaluation_") -> str:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return f"{prefix}1"
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    if runs.empty:
        return f"{prefix}1"
    existing = [
        name for name in runs.get("tags.mlflow.runName", pd.Series(dtype=str)).dropna()
        if name.startswith(prefix)
    ]
    nums = [int(n[len(prefix):]) for n in existing if n[len(prefix):].isdigit()]
    return f"{prefix}{(max(nums) + 1) if nums else 1}"


def log_to_mlflow(records: list[dict], ragas_result, experiment_name: str):
    mlflow.set_experiment(experiment_name)
    run_name = next_run_name(experiment_name)

    per_question_df = pd.DataFrame(records)
    ragas_df = ragas_result.to_pandas()
    # merge ragas per-row metric scores back onto our timing/count columns
    merged = per_question_df.reset_index(drop=True).join(
        ragas_df[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]]
    )

    aggregate_metrics = {
        "faithfulness": merged["faithfulness"].mean(),
        "answer_relevancy": merged["answer_relevancy"].mean(),
        "context_precision": merged["context_precision"].mean(),
        "context_recall": merged["context_recall"].mean(),
        "answer_words": merged["answer_words"].mean(),
        "question_words": merged["question_words"].mean(),
        "retrieval_time": merged["retrieval_time"].mean(),
        "generation_time": merged["generation_time"].mean(),
        "retrieved_chunks": merged["retrieved_chunks"].mean(),
    }

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "embed_model": config.EMBED_MODEL_NAME,
            "rerank_model": config.RERANK_MODEL_NAME,
            "generation_model": config.GENERATION_MODEL,
            "chunk_size_chars": config.CHUNK_SIZE_CHARS,
            "chunk_overlap_chars": config.CHUNK_OVERLAP_CHARS,
            "final_top_k": config.FINAL_TOP_K,
            "num_questions": len(records),
        })
        mlflow.log_metrics(aggregate_metrics)

        out_path = config.PROJECT_ROOT / "eval_results.csv"
        merged.to_csv(out_path, index=False)
        mlflow.log_artifact(str(out_path))

        print(f"\nLogged run '{run_name}' to experiment '{experiment_name}'.")
        print("Aggregate metrics:")
        for k, v in aggregate_metrics.items():
            print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser(description="RAGAS evaluation logged to MLflow")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--experiment", type=str, default=DEFAULT_EXPERIMENT)
    args = parser.parse_args()

    questions = load_questions(args.questions)
    records = build_eval_records(questions)
    ragas_result = run_ragas(records)
    log_to_mlflow(records, ragas_result, args.experiment)


if __name__ == "__main__":
    main()
