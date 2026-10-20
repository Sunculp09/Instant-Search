# Deploy PlaylistIndex

This project has three parts:

| Part | Runs on | Purpose |
| --- | --- | --- |
| `frontend/` | Vercel | Search page shown in the browser |
| `api.py` | Render | Receives searches, looks up transcript chunks, and asks Groq to answer |
| Qdrant Cloud | Qdrant | Stores the transcript vectors used for similarity search |

The existing `data/qdrant` folder is a local development database. Render cannot use that folder as its permanent production database, so production needs Qdrant Cloud.

## Before deployment

You need:

- A Git repository connected to both Render and Vercel.
- A Qdrant Cloud cluster URL and API key.
- A Groq API key.
- The project's transcript chunks in `knowledge.json`.

Keep secret keys in service environment settings or the local `.env` file. Never put them in frontend files or commit them to Git.

## 1. Upload vectors to Qdrant Cloud

Add the Qdrant Cloud URL and key to the project's local `.env`:

```text
QDRANT_URL=https://your-cluster-url.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key
```

From this project folder, run:

```powershell
uv run python create_vector_db.py --rebuild
```

This reads `knowledge.json` and uploads the embeddings to the configured Qdrant collection. It does not delete your local transcript JSON.

## 2. Deploy the backend to Render

Push this project to your Git provider. In Render, create a **Blueprint** from that repository; it will read `render.yaml`.

Set these values in the Render service environment:

| Name | Value |
| --- | --- |
| `GROQ_API_KEY` | Your Groq API key |
| `QDRANT_URL` | Your Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | Your Qdrant Cloud key |
| `FRONTEND_ORIGINS` | Your Vercel site URL, for example `https://your-site.vercel.app` |

When the service is live, open its `/health` address. It should return `{"status":"ok"}`. `/docs` shows the API documentation.

## 3. Deploy the frontend to Vercel

Import the same repository into Vercel and set **Root Directory** to `frontend`. The included `frontend/vercel.json` selects the Vite build and output folder.

Add this Vercel environment variable:

| Name | Value |
| --- | --- |
| `VITE_API_URL` | Your Render service URL, with no trailing slash |

Deploy the site. Copy its public URL into Render's `FRONTEND_ORIGINS`, then redeploy the Render service so its browser access list includes Vercel.

## Local run

From this project folder, start the API:

```powershell
uv run uvicorn api:app --reload
```

In a second terminal, start the website:

```powershell
cd frontend
npm install
npm run dev
```

Open the local address printed by Vite (normally `http://localhost:5173`). The frontend uses `http://127.0.0.1:8000` for the API unless `VITE_API_URL` is set.

## Current transcript coverage

The current local knowledge file contains 2,442 chunks from 53 videos. YouTube blocked caption requests for many playlist entries during collection, so confirm transcript coverage before relying on the search results for the full playlist.
