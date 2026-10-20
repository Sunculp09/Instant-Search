"""Index knowledge.json in Qdrant Cloud or a local Qdrant database."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

from vector_store import COLLECTION, QDRANT_DIR, get_qdrant_client


ROOT = Path(__file__).resolve().parent
KNOWLEDGE_FILE = ROOT / "knowledge.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_SIZE = 384
BATCH_SIZE = 64


def uses_qdrant_cloud_inference() -> bool:
    load_dotenv(ROOT / ".env")
    if os.getenv("QDRANT_MODE", "auto").lower() == "local":
        return False
    host = urlparse(os.getenv("QDRANT_URL", "")).hostname or ""
    return host.endswith(".cloud.qdrant.io")


def read_knowledge() -> list[dict[str, Any]]:
    if not KNOWLEDGE_FILE.exists():
        raise FileNotFoundError(f"{KNOWLEDGE_FILE.name} not found. Create it before building the vector DB.")
    records = json.loads(KNOWLEDGE_FILE.read_text(encoding="utf-8"))
    if not records:
        raise ValueError("knowledge.json is empty; there are no chunks to embed.")
    for number, record in enumerate(records, start=1):
        if not record.get("chunk_id") or not record.get("text"):
            raise ValueError(f"Record {number} needs both 'chunk_id' and 'text'.")
    return records


def stored_chunk_ids(db: QdrantClient) -> set[str]:
    if not db.collection_exists(COLLECTION):
        return set()
    found: set[str] = set()
    offset: Any = None
    while True:
        points, offset = db.scroll(
            collection_name=COLLECTION,
            limit=512,
            offset=offset,
            with_payload=["chunk_id"],
            with_vectors=False,
        )
        found.update(
            str(point.payload["chunk_id"])
            for point in points
            if point.payload and point.payload.get("chunk_id")
        )
        if offset is None:
            return found


def create_vector_db(rebuild: bool = False) -> None:
    records = read_knowledge()
    record_ids = {str(record["chunk_id"]) for record in records}
    db = get_qdrant_client()

    collection_exists = db.collection_exists(COLLECTION)
    existing_ids = stored_chunk_ids(db) if collection_exists else set()
    existing_count = db.count(collection_name=COLLECTION, exact=True).count if collection_exists else 0
    if rebuild or (collection_exists and (
        existing_count != len(existing_ids) or not existing_ids.issubset(record_ids)
    )):
        db.delete_collection(COLLECTION)
        collection_exists = False
        existing_ids = set()

    if not collection_exists:
        db.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(size=EMBEDDING_SIZE, distance=models.Distance.COSINE),
        )

    records_to_add = [record for record in records if str(record["chunk_id"]) not in existing_ids]
    if not records_to_add:
        db.close()
        destination = os.getenv("QDRANT_URL") or str(QDRANT_DIR)
        print(f"Vector DB is up to date: {len(records)} chunks in {destination}.")
        return

    print(f"Embedding and indexing {len(records_to_add)} chunks...")
    use_cloud_inference = uses_qdrant_cloud_inference()
    embedder = None
    if not use_cloud_inference:
        from fastembed import TextEmbedding

        embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
    next_point_id = db.count(collection_name=COLLECTION, exact=True).count
    for offset in range(0, len(records_to_add), BATCH_SIZE):
        batch = records_to_add[offset : offset + BATCH_SIZE]
        if use_cloud_inference:
            vectors = [
                models.Document(text=record["text"], model=EMBEDDING_MODEL)
                for record in batch
            ]
        else:
            vectors = list(embedder.embed([record["text"] for record in batch]))
        points = [
            models.PointStruct(
                id=next_point_id + offset + index,
                vector=vector if use_cloud_inference else vector.tolist(),
                payload=record,
            )
            for index, (record, vector) in enumerate(zip(batch, vectors, strict=True))
        ]
        db.upsert(collection_name=COLLECTION, points=points)

    total = db.count(collection_name=COLLECTION, exact=True).count
    db.close()
    destination = os.getenv("QDRANT_URL") or str(QDRANT_DIR)
    print(f"Indexed {total} chunks in {destination}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed knowledge.json chunks and store them in Qdrant.")
    parser.add_argument("--rebuild", action="store_true", help="Delete and recreate the vector collection")
    args = parser.parse_args()
    create_vector_db(args.rebuild)


if __name__ == "__main__":
    main()
