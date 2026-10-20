"""Fetch playlist captions and create timestamped transcript chunks in knowledge.json."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi


ROOT = Path(__file__).resolve().parent
KNOWLEDGE_FILE = ROOT / "knowledge.json"
CHUNK_SECONDS = 45
CHUNK_CHAR_LIMIT = 900
DEFAULT_LANGUAGES = ["en-US", "en", "hi"]


@dataclass
class TranscriptChunk:
    text: str
    video_id: str
    title: str
    start: float
    end: float

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}&t={int(self.start)}s"


def format_timestamp(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def get_playlist_videos(playlist_url: str) -> list[dict[str, str]]:
    options: dict[str, Any] = {"extract_flat": True, "quiet": True, "ignoreerrors": True}
    with yt_dlp.YoutubeDL(options) as ydl:
        playlist = ydl.extract_info(playlist_url, download=False)
    if not playlist:
        raise RuntimeError("Could not read the playlist. Check that the URL is correct and accessible.")

    videos = []
    for entry in playlist.get("entries") or []:
        if entry and entry.get("id"):
            videos.append({"video_id": entry["id"], "title": entry.get("title") or entry["id"]})
    return videos


def build_chunks(snippets: list[dict[str, Any]], video_id: str, title: str) -> list[TranscriptChunk]:
    chunks: list[TranscriptChunk] = []
    lines: list[str] = []
    chunk_start: float | None = None
    chunk_end = 0.0

    def save_chunk() -> None:
        nonlocal lines, chunk_start, chunk_end
        text = re.sub(r"\s+", " ", " ".join(lines)).strip()
        if text and chunk_start is not None:
            chunks.append(TranscriptChunk(text, video_id, title, chunk_start, chunk_end))
        lines = []
        chunk_start = None

    for snippet in snippets:
        text = str(snippet.get("text", "")).replace("\n", " ").strip()
        if not text:
            continue
        start = float(snippet.get("start", 0.0))
        end = start + float(snippet.get("duration", 0.0))
        current_text = " ".join(lines)
        if lines and (
            end - (chunk_start if chunk_start is not None else start) > CHUNK_SECONDS
            or len(current_text) + len(text) > CHUNK_CHAR_LIMIT
        ):
            save_chunk()
        if chunk_start is None:
            chunk_start = start
        lines.append(text)
        chunk_end = end
    save_chunk()
    return chunks


def create_knowledge(playlist_url: str, languages: list[str], reset: bool = False) -> None:
    videos = get_playlist_videos(playlist_url)
    if not videos:
        raise RuntimeError("No videos were found in the playlist.")

    playlist_video_ids = {video["video_id"] for video in videos}
    previous: list[dict[str, Any]] = []
    if not reset and KNOWLEDGE_FILE.exists():
        try:
            previous = json.loads(KNOWLEDGE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            previous = []
    previous_video_ids = {item["video_id"] for item in previous if "video_id" in item}
    same_playlist = bool(previous_video_ids) and previous_video_ids.issubset(playlist_video_ids)
    if not same_playlist:
        previous = []
        previous_video_ids = set()

    knowledge = list(previous)
    print(f"Playlist has {len(videos)} videos; {len(previous_video_ids)} already have captions indexed.")
    skipped = 0
    for number, video in enumerate(videos, start=1):
        if video["video_id"] in previous_video_ids:
            continue
        print(f"[{number}/{len(videos)}] Captions: {video['title']}")
        try:
            transcript = YouTubeTranscriptApi().fetch(video["video_id"], languages=languages)
            chunks = build_chunks(transcript.to_raw_data(), video["video_id"], video["title"])
        except Exception as exc:
            skipped += 1
            reason = " ".join(str(exc).split())
            print(f"  Skipped: {reason[:220] or type(exc).__name__}")
            if "YouTube is blocking requests from your IP" in str(exc):
                print("  Stopping caption requests; run this command later to retry remaining videos.")
                break
            continue

        for chunk_index, chunk in enumerate(chunks):
            knowledge.append({
                "chunk_id": f"{chunk.video_id}:{chunk_index}",
                "video_id": chunk.video_id,
                "video_title": chunk.title,
                "youtube_url": chunk.url,
                "start_seconds": chunk.start,
                "end_seconds": chunk.end,
                "start_timestamp": format_timestamp(chunk.start),
                "end_timestamp": format_timestamp(chunk.end),
                "text": chunk.text,
            })

    with KNOWLEDGE_FILE.open("w", encoding="utf-8") as file:
        json.dump(knowledge, file, ensure_ascii=False, indent=2)
    video_count = len({item["video_id"] for item in knowledge})
    print(f"Saved {len(knowledge)} chunks from {video_count} videos to {KNOWLEDGE_FILE.name}; skipped {skipped} this run.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create timestamped chunks from YouTube playlist captions.")
    parser.add_argument("playlist_url", help="YouTube playlist URL")
    parser.add_argument("--languages", nargs="+", default=DEFAULT_LANGUAGES)
    parser.add_argument("--reset", action="store_true", help="Replace knowledge.json and start from the beginning")
    args = parser.parse_args()
    create_knowledge(args.playlist_url, args.languages, args.reset)


if __name__ == "__main__":
    main()
