"""Shared Qdrant connection settings for local development and cloud deployment."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from qdrant_client import QdrantClient


ROOT = Path(__file__).resolve().parent
QDRANT_DIR = ROOT / "data" / "qdrant"
# The cloud deployment uses this collection for all-MiniLM-L6-v2 vectors.
COLLECTION = "playlist_transcripts_minilm_v1"


def get_qdrant_client() -> QdrantClient:
    """Use Qdrant Cloud when configured; otherwise use the local persistent DB."""
    load_dotenv(ROOT / ".env")
    mode = os.getenv("QDRANT_MODE", "auto").lower()
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    if mode == "local":
        QDRANT_DIR.parent.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(QDRANT_DIR))
    if mode not in {"auto", "cloud"}:
        raise RuntimeError("QDRANT_MODE must be 'local', 'cloud', or 'auto'.")
    if qdrant_url:
        if not qdrant_api_key:
            raise RuntimeError("QDRANT_API_KEY is required when QDRANT_URL is set.")
        host = urlparse(qdrant_url).hostname or ""
        return QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=60,
            cloud_inference=host.endswith(".cloud.qdrant.io"),
        )
    if qdrant_api_key:
        raise RuntimeError("Set QDRANT_URL together with QDRANT_API_KEY.")
    if mode == "cloud":
        raise RuntimeError("QDRANT_URL and QDRANT_API_KEY are required when QDRANT_MODE=cloud.")
    QDRANT_DIR.parent.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(QDRANT_DIR))
