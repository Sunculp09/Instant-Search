
"""Search the Qdrant playlist index and answer with timestamped sources."""

from __future__ import annotations

import argparse
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

from vector_store import COLLECTION, get_qdrant_client

ROOT = Path(__file__).resolve().parent
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Load the embedding model once per process."""
    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")


def retrieve(question: str, limit: int = 5) -> list[dict[str, Any]]:
    """Retrieve relevant transcript chunks directly from Qdrant."""
    db = get_qdrant_client()
    try:
        if not db.collection_exists(COLLECTION):
            raise RuntimeError(
                "Qdrant collection not found. Create the vector index first."
            )

        query_vector = get_embedder().encode(
            question, normalize_embeddings=True
        ).tolist()

        points = db.query_points(
            collection_name=COLLECTION,
            query=query_vector,
            limit=max(limit * 5, 25),
            with_payload=True,
        ).points

        return [
            {**point.payload, "score": point.score}
            for point in points
            if point.payload
        ][:limit]
    finally:
        db.close()


def answer_with_sources(question: str, limit: int = 5) -> dict[str, Any]:
    matches = retrieve(question, limit)

    if not matches:
        return {
            "answer": "I could not find a matching topic in the indexed captions.",
            "sources": [],
        }

    load_dotenv(ROOT / ".env")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing.")

    context = "\n\n".join(
        f"[{i}] Video: {item.get('video_title', 'Unknown')}\n"
        f"Time: {item.get('start_timestamp', 'Unknown')}\n"
        f"Caption: {item.get('text', '')}"
        for i, item in enumerate(matches, start=1)
    )

    response = Groq(api_key=api_key).chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer using only the caption context. Cite chunks by bracket "
                    "number, such as [1]. Distinguish a passing mention from a full "
                    "explanation. Do not invent details."
                ),
            },
            {
                "role": "user",
                "content=f"Caption context:\n{context}\n\nUser question: {question}",
            },
        ],
    )

    answer = response.choices[0].message.content or (
        "I could not find a supported answer."
    )

    sources = [
        {
            "video_title": item.get("video_title", "Unknown"),
            "timestamp": item.get("start_timestamp", "Unknown"),
            "start_seconds": item.get("start_seconds", 0),
            "video_id": item.get("video_id"),
            "url": item.get("youtube_url", ""),
            "score": item.get("score", 0.0),
        }
        for item in matches
    ]

    return {"answer": answer, "sources": sources}


def answer_question(question: str, limit: int = 5) -> str:
    result = answer_with_sources(question, limit)
    citations = "\n".join(
        f"[{i}] {source['video_title']} ({source['timestamp']}) — {source['url']}"
        for i, source in enumerate(result["sources"], start=1)
    )
    return (
        f"{result['answer']}\n\nVideos and timestamps:\n{citations}"
        if citations
        else result["answer"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search playlist captions in Qdrant."
    )
    parser.add_argument("statement", nargs="?")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    statement = args.statement or input("Enter your question: ").strip()
    if not statement:
        parser.error("statement cannot be empty")

    print(answer_question(statement, args.top_k))


if __name__ == "__main__":
    main()
