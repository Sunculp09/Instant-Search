"""FastAPI backend for the playlist transcript search website."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag_prj import answer_with_sources


load_dotenv()

app = FastAPI(
    title="Playlist Transcript Search API",
    description="Find relevant video moments in a playlist of transcripts.",
    version="1.0.0",
)

origins = [
    origin.strip()
    for origin in os.getenv("FRONTEND_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class AskRequest(BaseModel):
    statement: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10)


@app.get("/")
def root() -> dict[str, str]:
    return {"name": "Playlist Transcript Search API", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask")
def ask(request: AskRequest) -> dict[str, object]:
    try:
        return answer_with_sources(request.statement, request.top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Search failed; check the Render service logs.") from exc
