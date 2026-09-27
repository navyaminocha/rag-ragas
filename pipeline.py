"""
End-to-end RAG pipeline object: query -> hybrid retrieve -> generate.
"""
from dataclasses import dataclass

from retriever import HybridRetriever, RetrievedChunk
from generator import generate_answer


@dataclass
class RAGResult:
    query: str
    answer: str
    sources: list[RetrievedChunk]


class RAGPipeline:
    def __init__(self):
        self.retriever = HybridRetriever()

    def query(self, question: str) -> RAGResult:
        chunks = self.retriever.retrieve(question)
        answer = generate_answer(question, chunks)
        return RAGResult(query=question, answer=answer, sources=chunks)


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "Summarize the main contributions across these papers."
    pipeline = RAGPipeline()
    result = pipeline.query(q)
    print(f"Q: {result.query}\n")
    print(f"A: {result.answer}\n")
    print("Sources:")
    for s in result.sources:
        print(f"  - {s.source} p.{s.page_start}-{s.page_end} (score={s.score:.4f})")
