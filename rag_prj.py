"""Search the existing playlist vector DB and answer with timestamped sources."""

from __future__ import annotations

import argparse
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dotenv import load_dotenv
from groq import Groq
from qdrant_client import models

from vector_store import COLLECTION, get_qdrant_client


ROOT = Path(__file__).resolve().parent
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
OUT_OF_SCOPE_RESPONSE = "Please ask anything about DSA playlist."
QUERY_STOP_WORDS = {
    "a", "an", "and", "are", "at", "does", "find", "for", "from", "how", "in",
    "is", "me", "of", "on", "or", "please", "show", "tell", "that", "the", "this",
    "to", "video", "was", "were", "what", "when", "where", "which", "who", "why",
    "with", "speaker", "explain", "explains", "talk", "talks", "about",
}


def uses_qdrant_cloud_inference() -> bool:
    load_dotenv(ROOT / ".env")
    if os.getenv("QDRANT_MODE", "auto").lower() == "local":
        return False
    host = urlparse(os.getenv("QDRANT_URL", "")).hostname or ""
    return host.endswith(".cloud.qdrant.io")


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

    if uses_qdrant_cloud_inference():
        query_vector = models.Document(text=f"query: {question}", model=EMBEDDING_MODEL)
    else:
        from fastembed import TextEmbedding

        embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
        query_vector = next(embedder.embed([f"query: {question}"])).tolist()
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

    # Narrow to an episode only when the user explicitly names one.
    episode = re.search(r"\b(?:episode|ep)\s*#?\s*(\d+)\b", question, re.IGNORECASE)
    episode_title = (
        re.compile(rf"\bepisode\s*{episode.group(1)}\b", re.IGNORECASE)
        if episode else None
    )
    if episode_title:
        semantic_matches = [
            item for item in semantic_matches
            if episode_title.search(str(item.get("video_title", "")))
        ]

    query_terms = [
        word for word in re.findall(r"[\w]+", question.casefold())
        if word not in QUERY_STOP_WORDS and len(word) > 1
    ]
    broad_topic_query = len(query_terms) <= 2

    # Search captions and titles for exact phrases as well as using vector
    # matches. Titles let users target a specific episode directly.
    phrases = exact_query_phrases(question)
    if episode_title and not phrases:
        phrases = [f"episode {episode.group(1)}"]
    if not phrases:
        if broad_topic_query and query_terms:
            semantic_matches.sort(
                key=lambda item: (
                    all(term in str(item.get("video_title", "")).casefold() for term in query_terms),
                    item["score"],
                ),
                reverse=True,
            )
        return semantic_matches[:limit]

    semantic_by_key = {
        (item.get("video_id"), item.get("start_timestamp"), item.get("text")): item
        for item in semantic_matches
    }
    teaching_cues = ("intro", "introduction", "easy", "algorithm", "explanation", "explained", "basics", "course", "tutorial")
    phrase_matches: list[dict[str, Any]] = []
    for chunk in load_transcript_chunks():
        text = str(chunk.get("text", "")).casefold()
        title = str(chunk.get("video_title", "")).casefold()
        if episode_title and not episode_title.search(str(chunk.get("video_title", ""))):
            continue
        matching_phrases = [phrase for phrase in phrases if phrase in text or phrase in title]
        if not matching_phrases:
            continue
        key = (chunk.get("video_id"), chunk.get("start_timestamp"), chunk.get("text"))
        semantic_item = semantic_by_key.get(key, {})
        phrase_matches.append(
            {
                **chunk,
                "score": semantic_item.get("score", 0.0),
                "_matched_phrase": max(matching_phrases, key=len),
                "_body_match": any(phrase in text for phrase in phrases),
                "_title_match": any(phrase in title for phrase in phrases),
                "_teaching_title": broad_topic_query and any(cue in title for cue in teaching_cues),
            }
        )

    if phrase_matches:
        phrase_matches.sort(
            key=lambda item: (
                item["_title_match"], item["_teaching_title"], item["_body_match"],
                len(item["_matched_phrase"]), item["score"],
            ),
            reverse=True,
        )
        selected: list[dict[str, Any]] = []
        per_video: dict[str, int] = {}
        for item in phrase_matches:
            video_id = str(item.get("video_id", item.get("video_title", "")))
            if per_video.get(video_id, 0) >= 2:
                continue
            selected.append({key: value for key, value in item.items() if not key.startswith("_")})
            per_video[video_id] = per_video.get(video_id, 0) + 1
            if len(selected) >= limit:
                break
        if episode_title or any(item["_title_match"] or item["score"] > 0 for item in phrase_matches):
            return selected
    if episode_title:
        return semantic_matches[:limit]
    return semantic_matches[:limit]


def answer_with_sources(question: str, limit: int = 5) -> dict[str, Any]:
    matches = retrieve(question, limit)
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
                    "First decide whether the user's request is about data structures and algorithms (DSA), "
                    "programming, coding interview problems, or finding where one of those is discussed in "
                    "this DSA playlist. Use a permissive scope threshold: return the out-of-scope response only "
                    "when the request is clearly unrelated (for example, asking for an animal species or a "
                    "weather forecast) or is obvious keyboard-spam gibberish. Do not reject a short or single-word "
                    "topic query. Allow misspellings, phonetic spellings, transliterations, abbreviations, and "
                    "unfamiliar algorithm or data-structure names; for example, 'ballmenford' may mean "
                    "'Bellman-Ford'. When a query could plausibly be a DSA topic, treat it as in scope and search. "
                    "This applies to questions in any language. Do not treat a coincidental word in a caption as "
                    "proof that an otherwise clearly unrelated request is in scope. "
                    "For an in-scope request, answer briefly in the same language as the user's question, using "
                    "only the caption context. "
                    "Cite useful transcript chunks by bracket number, such as [1]. Distinguish "
                    "a passing mention of a topic from an explanation of it. If the captions only "
                    "mention the topic, say so in one short sentence. If they explain it, state "
                    "what they explain in one short sentence. If no caption is relevant, say you "
                    "could not find it. Do not invent details or add a long narrative."
                ),
            },
            {"role": "user", "content": f"Caption context:\n{context}\n\nUser statement: {question}"},
        ],
    )
    answer = response.choices[0].message.content or "I could not find a supported answer."
    if answer.strip().casefold().rstrip(".!? ") == OUT_OF_SCOPE_RESPONSE.casefold().rstrip(".!? "):
        return {"answer": OUT_OF_SCOPE_RESPONSE, "sources": []}
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
