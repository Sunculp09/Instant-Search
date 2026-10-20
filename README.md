# Playlist transcript search

The workflow is split into three steps: create timestamped caption chunks, create their vector index, then search that index and answer with timestamped video links.

## Setup

From this folder, install dependencies:

```powershell
uv sync
```

Create a `.env` file with your Groq key for answer generation:

```text
GROQ_API_KEY=your_key_here
```

The embedding model is `paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions). Qdrant stores vectors locally under `data/qdrant`.

## 1. Create or update knowledge.json

```powershell
uv run python create_knowledge.py "https://www.youtube.com/playlist?list=YOUR_PLAYLIST_ID"
```

Captions are requested in `en-US`, then `en`, then `hi` by default. The script writes one object per chunk with `chunk_id`, `video_id`, `video_title`, timestamped `youtube_url`, start/end seconds and formatted timestamps, and the caption `text`. It resumes by skipping videos already present in `knowledge.json`; rerun it later to retry skipped videos. Pass `--reset` to replace `knowledge.json` and start over.

## 2. Create or update the vector database

```powershell
uv run python create_vector_db.py
```

This reads `knowledge.json`, embeds each chunk, and stores the vectors and metadata in Qdrant. It adds missing chunks and rebuilds if it detects stale chunks. Use `--rebuild` to recreate the collection.

## 3. Search and answer

```powershell
uv run python rag_prj.py "Where does the speaker explain binary search?"
```

`rag_prj.py` embeds the query, searches Qdrant using cosine similarity, sends the closest captions to Groq, and prints the answer with video links and timestamps.

Videos without accessible captions are skipped; this project does not transcribe audio when captions are missing.

## Website deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for the Render API, Vercel frontend, and Qdrant Cloud setup.
