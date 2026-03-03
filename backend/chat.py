from __future__ import annotations

import os
import re
from dataclasses import asdict
from typing import Any

from backend.llm import generate_text
from backend.rag import RetrievedChunk, retrieve
from storage.notebook_store import NotebookStore


SYSTEM_PROMPT = (
    "You are an academic research assistant. Answer only with support from retrieved context. "
    "Write naturally in your own words, not as copied source text. "
    "If context is insufficient, say what is missing. Keep answers concise and factual."
)
MAX_DISTANCE = float(os.getenv("RAG_MAX_DISTANCE", "0.7"))


def _format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"[Source {i}] name={chunk.source_name}; type={chunk.source_type}; chunk={chunk.chunk_index}\n{chunk.text}"
        )
    return "\n\n".join(blocks)


def _fallback_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "I couldn't find relevant notebook content yet. Add sources first, then ask again."

    # PDF extraction can insert hard line breaks between every word; normalize for readability.
    normalized = re.sub(r"\s+", " ", chunks[0].text).strip()
    preview = normalized[:700]
    if len(normalized) > 700:
        preview += "..."

    source = chunks[0].source_name
    return (
        "### Quick Answer\n"
        f"Based on your uploaded material, here is the most relevant part for: **{question}**\n\n"
        "### Key Evidence\n"
        f"{preview}\n\n"
        f"**Source:** {source}"
    )


def _citations(chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for chunk in chunks:
        key = (chunk.source_name, chunk.chunk_index)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "source_name": chunk.source_name,
                "source_type": chunk.source_type,
                "chunk_index": chunk.chunk_index,
                "score": round(chunk.score, 4),
            }
        )
    return out


def _format_chat_answer(answer: str) -> str:
    text = (answer or "").strip()
    if not text:
        return "### Quick Answer\n\nI could not generate an answer from the current context."

    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    has_headings = bool(re.search(r"(?m)^###\s+", text))
    if has_headings:
        # Keep model-provided sections, just normalize spacing.
        text = re.sub(r"(?m)^(###\s+)", r"\n\1", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text

    # Convert dense paragraph into readable sections.
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return f"### Quick Answer\n\n{text}"

    quick = " ".join(sentences[:2]).strip()
    key_points = sentences[2:6]

    if not key_points:
        # Fallback to clause-based bullets when we only got one long sentence.
        clauses = [c.strip() for c in re.split(r";\s+|,\s+(?=[A-Z])", text) if c.strip()]
        key_points = clauses[:4]

    if key_points:
        bullets = "\n".join([f"- {point}" for point in key_points])
    else:
        bullets = "- No additional details were available."

    return (
        "### Quick Answer\n\n"
        f"{quick}\n\n"
        "### Key Points\n\n"
        f"{bullets}\n"
    )


def chat_with_notebook(
    store: NotebookStore,
    username: str,
    notebook_id: str,
    user_message: str,
    top_k: int = 4,
) -> dict[str, Any]:
    paths = store.notebook_paths(username, notebook_id)
    retrieved_all = retrieve(paths.chroma, user_message, top_k=top_k)
    retrieved = [chunk for chunk in retrieved_all if chunk.score <= MAX_DISTANCE]
    # If threshold is too strict for this corpus/query, keep best available chunks instead of empty context.
    if not retrieved and retrieved_all:
        retrieved = retrieved_all[: min(2, len(retrieved_all))]

    context = _format_context(retrieved)
    prompt = (
        "Question:\n"
        f"{user_message}\n\n"
        "Retrieved context:\n"
        f"{context}\n\n"
        "Instructions:\n"
        "1) Answer using only retrieved context.\n"
        "2) Paraphrase and synthesize in natural language; do not copy long phrases from sources.\n"
        "3) Use clean markdown with these sections: '### Quick Answer' and '### Key Points'.\n"
        "4) If unsure, explicitly state uncertainty.\n"
        "5) End with a short 'Sources used' line listing source names."
    )
    answer = generate_text(prompt, SYSTEM_PROMPT, fallback_text=_fallback_answer(user_message, retrieved))
    answer = _format_chat_answer(answer)

    citation_rows = _citations(retrieved)
    store.save_message(username, notebook_id, "user", user_message)
    store.save_message(username, notebook_id, "assistant", answer, citations=citation_rows)

    return {
        "answer": answer,
        "citations": citation_rows,
        "retrieved": [asdict(r) for r in retrieved],
    }
