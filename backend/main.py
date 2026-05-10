"""
Hermes — Servidor FastAPI con streaming
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import os
import webbrowser
import uvicorn

from rag_engine import RAGEngine

app = FastAPI(title="Hermes RAG API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAGEngine()

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
BROWSER_FLAG = Path(__file__).parent.parent / "data" / ".browser_opened"


class ChatRequest(BaseModel):
    message: str
    doc_ids: list[str] = []
    history: list[dict[str, str]] = []


@app.on_event("startup")
async def open_browser_on_startup():
    if os.getenv("OPEN_BROWSER_ON_START", "1") != "1":
        return
    if BROWSER_FLAG.exists():
        return

    BROWSER_FLAG.parent.mkdir(parents=True, exist_ok=True)
    BROWSER_FLAG.write_text("opened", encoding="utf-8")
    webbrowser.open_new_tab("http://localhost:8000")


@app.get("/api/health")
async def health():
    return {"status": "ok", "docs": len(rag.list_documents())}


@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Solo se admiten archivos PDF.")
    try:
        content = await file.read()
        return rag.add_document(content, file.filename)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(500, f"Error procesando PDF: {e}")


@app.get("/api/documents")
async def list_documents():
    return rag.list_documents()


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    rag.delete_document(doc_id)
    return {"status": "deleted"}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "El mensaje no puede estar vacío.")

    async def generate():
        try:
            async for token in rag.stream(req.message, req.doc_ids, req.history):
                yield token
        except Exception as e:
            yield f"\n\n⚠️ Error: {e}"

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")


@app.get("/")
async def serve_frontend():
    index = FRONTEND_DIR / "index.html"
    if not index.exists():
        return {"error": "Frontend no encontrado"}
    return FileResponse(str(index))

@app.get("/{file_path:path}")
async def serve_static(file_path: str):
    """Servir archivos estáticos del frontend (CSS, JS, etc.)"""
    if file_path.startswith("api/"):
        raise HTTPException(404)
    file_full_path = FRONTEND_DIR / file_path
    if file_full_path.exists() and file_full_path.is_file():
        return FileResponse(str(file_full_path))
    raise HTTPException(404)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
