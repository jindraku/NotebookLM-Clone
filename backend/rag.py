from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb

COLLECTION_NAME = "sources"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class RetrievedChunk:
    text: str
    source_name: str
    source_type: str
    chunk_index: int
    score: float


def _collection(chroma_dir: Path):
    # Lazy import to avoid loading heavy embedding stack during app startup.
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    client = chromadb.PersistentClient(path=str(chroma_dir))
    emb = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    return client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=emb)


def retrieve(chroma_dir: str | Path, query: str, top_k: int = 4) -> list[RetrievedChunk]:
    collection = _collection(Path(chroma_dir))
    response: dict[str, Any] = collection.query(query_texts=[query], n_results=top_k)

    docs = response.get("documents", [[]])[0]
    metas = response.get("metadatas", [[]])[0]
    distances = response.get("distances", [[]])[0]

    out: list[RetrievedChunk] = []
    for doc, meta, dist in zip(docs, metas, distances):
        source_name = (meta or {}).get("source_name", "unknown")
        source_type = (meta or {}).get("source_type", "unknown")
        chunk_index = int((meta or {}).get("chunk_index", -1))
        out.append(
            RetrievedChunk(
                text=doc,
                source_name=source_name,
                source_type=source_type,
                chunk_index=chunk_index,
                score=float(dist) if dist is not None else 0.0,
            )
        )
    return out
