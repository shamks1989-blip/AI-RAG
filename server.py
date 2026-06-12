"""
server.py — FastAPI backend for the Multimodal RAG Knowledge Base Agent
Replaces the Gradio app.py with a pure HTML/CSS/JS single-page interface.
query.py and ingest.py are unchanged — only the frontend changes.
"""

import os
import base64
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn

load_dotenv()

from query import run_query, get_indexed_documents
from ingest import ingest_single_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
DOCS_DIR   = Path(os.getenv("DOCS_DIR", "./docs"))
APP_PORT   = int(os.getenv("APP_PORT", "7860"))

app = FastAPI(title="RAG Knowledge Base Agent")

# Serve brain image and other loader assets
_loader_dir = Path(__file__).parent / "loader"
if _loader_dir.exists():
    app.mount("/loader", StaticFiles(directory=str(_loader_dir)), name="loader")


# ── Pydantic models ───────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    image: Optional[str] = None  # base64-encoded image bytes


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = Path(__file__).parent / "index.html"
    return html_path.read_text(encoding="utf-8")


@app.post("/api/chat")
async def chat(req: ChatRequest):
    image_bytes = None
    if req.image:
        try:
            image_bytes = base64.b64decode(req.image)
        except Exception:
            pass

    history = [{"role": m.role, "content": m.content} for m in req.history]

    try:
        result = run_query(
            user_query=req.message,
            chat_history=history,
            uploaded_image_bytes=image_bytes,
            chroma_dir=CHROMA_DIR,
        )
        log.info(f"Answer generated — intent={result.get('intent')}, sources={len(result.get('sources', []))}")
        return result
    except Exception as e:
        log.exception("Chat error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = DOCS_DIR / file.filename
    content = await file.read()
    save_path.write_bytes(content)

    try:
        result = ingest_single_file(str(save_path), CHROMA_DIR)
        docs = [{"source": s, "title": t} for s, t in get_indexed_documents(CHROMA_DIR)]
        return {"status": "ok", "result": result, "documents": docs}
    except Exception as e:
        log.exception("Upload error")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents")
async def documents():
    docs = [{"source": s, "title": t} for s, t in get_indexed_documents(CHROMA_DIR)]
    return {"documents": docs}


if __name__ == "__main__":
    log.info(f"Starting RAG Agent → http://localhost:{APP_PORT}")
    uvicorn.run("server:app", host="0.0.0.0", port=APP_PORT, reload=False)
