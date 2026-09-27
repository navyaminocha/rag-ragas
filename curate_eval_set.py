"""
Turn real logged Streamlit queries into REVIEWABLE candidate eval questions.

This does NOT produce a trustworthy eval set by itself — it drafts a
candidate reference answer for each logged question, grounded in the
contexts that were *actually retrieved* for that question, using a stronger
judge model. Every candidate is written with "reviewed": false.

ragas_eval.py refuses to score unreviewed entries. You must open
eval_questions.json, read each draft, fix it if it's wrong or shallow, and
flip "reviewed" to true before it counts.

Usage:
    python curate_eval_set.py                  # draft candidates for all new logged questions
    python curate_eval_set.py --limit 10        # cap how many to draft this run (cost control)
"""
import argparse
import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from openai import OpenAI
from indexer import get_embedder
import config

DEFAULT_OUTPUT = config.PROJECT_ROOT / "eval_questions.json"

DRAFT_PROMPT = """You are drafting a gold-standard reference answer for a RAG evaluation benchmark. You will be shown a question and the source passages retrieved for it.

Write a high-quality reference answer based only on the provided passages. Explain the concept clearly and naturally instead of simply quoting or pointing to where it is mentioned. Synthesize information from multiple passages when appropriate to produce a single coherent answer.

The answer should be:
- Factually accurate and fully supported by the provided passages.
- Complete enough to answer the question while remaining concise.
- Easy to understand, with brief explanations of important concepts or methods when needed.
- Free of unsupported assumptions or outside knowledge.

If the passages do not contain enough information to answer the question confidently, explicitly state that the available context is insufficient instead of inventing information. Such cases indicate that the question may need to be removed from the benchmark or that retrieval should be improved.

Question: {question}

Retrieved passages:
{contexts}

Write only the reference answer, nothing else."""


def load_logged_queries() -> list[dict]:
    if not config.EVAL_LOG_PATH.exists():
        raise FileNotFoundError(
            f"No query log at {config.EVAL_LOG_PATH}. Use the 'Save this Q&A for evaluation' "
            f"button in the Streamlit app first."
        )
    records = []
    with open(config.EVAL_LOG_PATH) as f:
        for line in f:
            records.append(json.loads(line))
    # dedupe keeping the last occurrence per question (freshest retrieval wins)
    by_question = {r["question"]: r for r in records}
    return list(by_question.values())


def load_existing_eval_set() -> list[dict]:
    if not DEFAULT_OUTPUT.exists():
        return []
    with open(DEFAULT_OUTPUT) as f:
        return json.load(f)


def draft_ground_truth(client: OpenAI, question: str, contexts: list[str]) -> str:
    context_block = "\n\n---\n\n".join(contexts)
    prompt = DRAFT_PROMPT.format(question=question, contexts=context_block)
    response = client.chat.completions.create(
    model=config.GENERATION_MODEL,
    messages=[
        {"role": "user", "content": prompt}
    ],
    max_tokens=800,
    )

    return response.choices[0].message.content.strip()


def main():
    parser = argparse.ArgumentParser(description="Draft reviewable eval candidates from logged Streamlit queries")
    parser.add_argument("--limit", type=int, default=None, help="Cap how many new candidates to draft this run")
    args = parser.parse_args()

    if not config.OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY env var not set.")

    logged = load_logged_queries()
    existing = load_existing_eval_set()
    embedder = get_embedder()

    existing_questions = [row["question"] for row in existing]

    existing_embeddings = []

    if existing_questions:
        existing_embeddings = embedder.embed_documents(existing_questions)

    new_candidates = [r for r in logged if r["question"] not in existing_questions]
    if args.limit:
        new_candidates = new_candidates[:args.limit]

    if not new_candidates:
        print("No new logged questions to draft — eval_questions.json is already up to date "
              "with everything logged so far.")
        return

    client = OpenAI(
    api_key=config.OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    )
    print(f"Drafting {len(new_candidates)} candidate reference answer(s) with {config.GENERATION_MODEL}...")

    for record in new_candidates:
        print(f"  {record['question']}")
        draft = draft_ground_truth(client, record["question"], record["contexts"])
        existing.append({
            "question": record["question"],
            "ground_truth": draft,
            "reviewed": False,
            "source": "logged",
        })

    with open(DEFAULT_OUTPUT, "w") as f:
        json.dump(existing, f, indent=2)

    unreviewed_count = sum(1 for r in existing if not r.get("reviewed", False))
    print(f"\nWrote {len(new_candidates)} new candidate(s) to {DEFAULT_OUTPUT}.")
    print(f"{unreviewed_count} total entries are unreviewed and will be SKIPPED by ragas_eval.py.")
    print("Open eval_questions.json, fix any drafts that are wrong/shallow, "
          "set \"reviewed\": true on each you trust, then run ragas_eval.py.")


if __name__ == "__main__":
    main()
