"""
Take retrieved chunks + a user question, build a grounded prompt, and call
an OpenRouter-hosted model to produce a cited answer.

Uses the `openai` SDK's OpenAI-compatible client pointed at OpenRouter
(https://openrouter.ai/api/v1) rather than Google's `google.genai` SDK.
OpenRouter exposes an OpenAI-compatible /chat/completions endpoint, so the
same client class works for any model available on OpenRouter — only the
base_url, api_key, and model string change.
"""
from openai import OpenAI

from config import OPENROUTER_API_KEY, GENERATION_MODEL, MAX_ANSWER_TOKENS
from retriever import RetrievedChunk

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

SYSTEM_PROMPT = """You are a research assistant answering questions strictly from the
provided paper excerpts. Rules:
- Only use information present in the excerpts below.
- Cite every claim inline as (source, p.X).
- If the excerpts don't contain the answer, say so plainly instead of guessing.
- Be precise and technical; the reader is a researcher, not a layperson."""


def build_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for c in chunks:
        pages = f"p.{c.page_start}" if c.page_start == c.page_end else f"p.{c.page_start}-{c.page_end}"
        blocks.append(f"[Source: {c.source}, {pages}]\n{c.text}")
    return "\n\n---\n\n".join(blocks)


def generate_answer(query: str, chunks: list[RetrievedChunk]) -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY env var not set.")
    if not chunks:
        return "No relevant passages were retrieved for this question."

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)
    context = build_context(chunks)
    prompt = f"Excerpts:\n\n{context}\n\n---\n\nQuestion: {query}"

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        max_tokens=MAX_ANSWER_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content
