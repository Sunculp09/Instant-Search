"""Search the existing playlist vector DB and answer with timestamped sources."""

from __future__ import annotations

import argparse
import json
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
QUERY_STOP_WORDS = {
    "a", "an", "and", "are", "at", "does", "find", "for", "from", "how", "in",
    "is", "me", "of", "on", "or", "please", "show", "tell", "that", "the", "this",
    "to", "video", "was", "were", "what", "when", "where", "which", "who", "why",
    "with", "speaker", "explain", "explains", "talk", "talks", "about",
}


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Load the embedding model once per process."""
    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def load_transcript_chunks() -> tuple[dict[str, Any], ...]:
    """Load the compact transcript file once for exact phrase matching."""
    path = ROOT / "knowledge.json"
    if not path.exists():
        return ()
    with path.open(encoding="utf-8") as transcript_file:
        return tuple(json.load(transcript_file))


def exact_query_phrases(question: str) -> list[str]:
    words = [
        word
        for word in re.findall(r"[\w]+", question.casefold(), flags=re.UNICODE)
        if word not in QUERY_STOP_WORDS and len(word) > 1
    ]
    return [" ".join(words[index:index + 2]) for index in range(len(words) - 1)]


def retrieve(question: str, limit: int = 5) -> list[dict[str, Any]]:
    """Combine exact transcript phrases with nearest vector matches."""
    db = get_qdrant_client()
    if not db.collection_exists(COLLECTION):
        db.close()
        raise RuntimeError("Vector collection not found. Run create_vector_db.py first.")

    query_vector = get_embedder().encode(question, normalize_embeddings=True).tolist()
    results = db.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=max(limit * 5, 25),
        with_payload=True,
    ).points
    db.close()
    semantic_matches = [
        {**(point.payload or {}), "score": point.score}
        for point in results
        if point.payload
    ]

    # Vector similarity can miss a rare but important exact topic phrase. Search
    # the local transcript source for exact two-word phrases and prefer those
    # clips when present, while keeping vector search for paraphrases.
    phrases = exact_query_phrases(question)
    if not phrases:
        return semantic_matches[:limit]

    semantic_by_key = {
        (item.get("video_id"), item.get("start_timestamp"), item.get("text")): item
        for item in semantic_matches
    }
    phrase_matches: list[dict[str, Any]] = []
    for chunk in load_transcript_chunks():
        text = str(chunk.get("text", "")).casefold()
        matching_phrases = [phrase for phrase in phrases if phrase in text]
        if not matching_phrases:
            continue
        key = (chunk.get("video_id"), chunk.get("start_timestamp"), chunk.get("text"))
        semantic_item = semantic_by_key.get(key, {})
        phrase_matches.append(
            {
                **chunk,
                "score": semantic_item.get("score", 0.0),
                "_matched_phrase": max(matching_phrases, key=len),
            }
        )

    if phrase_matches:
        phrase_matches.sort(key=lambda item: (len(item["_matched_phrase"]), item["score"]), reverse=True)
        return [{key: value for key, value in item.items() if key != "_matched_phrase"} for item in phrase_matches[:limit]]
    return semantic_matches[:limit]


def answer_with_sources(question: str, limit: int = 5) -> dict[str, Any]:
    matches = retrieve(question, limit)
    if not matches:
        return {"answer": "I could not find a matching topic in the indexed captions.", "sources": []}

    load_dotenv(ROOT / ".env")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to week3/proj_day/.env.")

    context = "\n\n".join(
        f"[{i}] Video: {item['video_title']}\n"
        f"Time: {item['start_timestamp']}\n"
        f"Caption: {item['text']}"
        for i, item in enumerate(matches, start=1)
    )
    response = Groq(api_key=api_key).chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the user's statement or question using only the caption context. "
                    "Cite useful transcript chunks by bracket number, such as [1]. Distinguish "
                    "a passing mention of a topic from an explanation of it. If the captions only "
                    "mention the topic, say that they mention it but do not explain it. If no "
                    "caption is relevant, say you could not find it. Do not invent details."
                ),
            },
            {"role": "user", "content": f"Caption context:\n{context}\n\nUser statement: {question}"},
        ],
    )
    answer = response.choices[0].message.content or "I could not find a supported answer."
    sources = [
        {
            "video_title": item["video_title"],
            "timestamp": item["start_timestamp"],
            "start_seconds": item.get("start_seconds", 0),
            "video_id": item.get("video_id"),
            "url": item["youtube_url"],
            "score": item["score"],
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
    if not citations:
        return result["answer"]
    return f"{result['answer']}\n\nVideos and timestamps:\n{citations}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Search playlist captions in the existing vector DB.")
    parser.add_argument("statement", nargs="?", help="Topic or question to find in the playlist")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    statement = args.statement or input("Enter your statement or question: ").strip()
    if not statement:
        parser.error("statement cannot be empty")
    print(answer_question(statement, args.top_k))


if __name__ == "__main__":
    main()
