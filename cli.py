"""
CLI:
  python cli.py build              -> chunk PDFs in papers/ and build both indexes
  python cli.py query "question"   -> hybrid retrieve + generate an answer
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description="Hybrid search RAG over research PDFs")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("build", help="Chunk PDFs and build dense+sparse indexes")

    query_parser = sub.add_parser("query", help="Ask a question")
    query_parser.add_argument("question", type=str)

    args = parser.parse_args()

    if args.command == "build":
        from ingest import load_and_chunk_all
        from indexer import build_indexes
        chunks = load_and_chunk_all()
        build_indexes(chunks)

    elif args.command == "query":
        from pipeline import RAGPipeline
        pipeline = RAGPipeline()
        result = pipeline.query(args.question)
        print(f"\n{result.answer}\n")
        print("Sources:")
        for s in result.sources:
            print(f"  - {s.source} p.{s.page_start}-{s.page_end} (score={s.score:.4f})")


if __name__ == "__main__":
    main()
