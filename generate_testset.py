"""
Generate a synthetic eval set — questions + ground-truth answers — directly
from your indexed PDFs, using RAGAS's TestsetGenerator. This is what fills in
context_precision / context_recall's ground_truth requirement without you
hand-writing reference answers.

RAGAS reads full documents (not our small retrieval chunks — it does its own
chunking internally to build simple/reasoning/multi-context questions), so
this loads each PDF page as a separate document via the same PyMuPDF
extraction ingest.py uses, independent of the retrieval index.

Usage:
    python generate_testset.py                          # 10 questions, writes eval_questions.json
    python generate_testset.py --num-questions 20
    python generate_testset.py --simple 0.5 --reasoning 0.25 --multi-context 0.25
    python generate_testset.py --max-source-docs 40      # cap PDF pages fed in (cost control)
"""
import argparse
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context

import config
from indexer import get_embedder
from ingest import extract_pages

DEFAULT_OUTPUT = config.PROJECT_ROOT / "eval_questions.json"
DEFAULT_FULL_OUTPUT = config.PROJECT_ROOT / "eval_questions_full.csv"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def load_source_documents(max_docs: int | None) -> list[Document]:
    pdf_files = sorted(config.PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {config.PDF_DIR}.")

    documents = []
    for pdf_path in pdf_files:
        pages = extract_pages(pdf_path)
        for page_num, text in enumerate(pages, start=1):
            if not text.strip():
                continue
            documents.append(Document(
                page_content=text,
                metadata={"source": pdf_path.name, "page": page_num},
            ))

    if max_docs and len(documents) > max_docs:
        print(f"Capping {len(documents)} pages down to {max_docs} (--max-source-docs).")
        documents = documents[:max_docs]

    print(f"Loaded {len(documents)} source pages from {len(pdf_files)} PDF(s).")
    return documents


def build_generator() -> TestsetGenerator:
    generator_llm = ChatOpenAI(
        model=config.TESTSET_GENERATOR_MODEL,
        api_key=config.OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )
    critic_llm = ChatOpenAI(
        model=config.TESTSET_CRITIC_MODEL,
        api_key=config.OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )
    embeddings = LangchainEmbeddingsWrapper(get_embedder())
    return TestsetGenerator.from_langchain(generator_llm, critic_llm, embeddings)


def main():
    parser = argparse.ArgumentParser(description="Generate a RAGAS testset from papers/")
    parser.add_argument("--num-questions", type=int, default=10)
    parser.add_argument("--max-source-docs", type=int, default=40,
                         help="Cap on PDF pages fed to the generator, to bound LLM cost.")
    parser.add_argument("--simple", type=float, default=0.5)
    parser.add_argument("--reasoning", type=float, default=0.25)
    parser.add_argument("--multi-context", type=float, default=0.25)
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    if not config.OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY env var not set.")

    documents = load_source_documents(args.max_source_docs)
    generator = build_generator()

    distributions = {
        simple: args.simple,
        reasoning: args.reasoning,
        multi_context: args.multi_context,
    }

    print(f"Generating {args.num_questions} questions "
          f"(simple={args.simple}, reasoning={args.reasoning}, multi_context={args.multi_context})...")
    testset = generator.generate_with_langchain_docs(
        documents,
        test_size=args.num_questions,
        distributions=distributions,
        raise_exceptions=False,
    )

    df = testset.to_pandas()
    if df.empty:
        raise RuntimeError(
            "RAGAS generated 0 usable questions. Try more/larger source pages "
            "(--max-source-docs) or fewer questions."
        )

    # eval_questions.json — merge with whatever's already there (curated entries from
    # curate_eval_set.py, previous runs, etc.) rather than clobbering it
    existing = []
    output_path = Path(args.output)
    if output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
    existing_questions = {row["question"] for row in existing}

    new_pairs = [
        {
            "question": row["question"],
            "ground_truth": row["ground_truth"],
            "reviewed": False,
            "source": "synthetic",
        }
        for _, row in df.iterrows()
        if row["question"] not in existing_questions
    ]
    merged = existing + new_pairs
    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\nAdded {len(new_pairs)} new question/ground_truth pair(s) to {output_path} "
          f"({len(merged)} total).")
    print("New entries are marked \"reviewed\": false — ragas_eval.py will skip them until you "
          "check each one and flip it to true.")

    # full RAGAS output (contexts, evolution_type, etc.) kept for inspection/debugging
    df.to_csv(DEFAULT_FULL_OUTPUT, index=False)
    print(f"Full generation detail (contexts, evolution_type) saved to {DEFAULT_FULL_OUTPUT}")

    print("\nSpot-check these before trusting them — synthetic ground truth can be shallow "
          "or occasionally wrong; skim eval_questions_full.csv and fix anything off.")


if __name__ == "__main__":
    main()
