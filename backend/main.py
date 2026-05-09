"""
Hermes — Servidor FastAPI
Sirve la API REST y el frontend estático.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import uvicorn

from rag_engine import RAGEngine

app = FastAPI(title="Hermes RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAGEngine()

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


# ─── Modelos ──────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message:      str
    doc_ids:      list[str] = []
    mode:         str       = "chat"  # chat | summarize | notes | quiz


class DeleteResponse(BaseModel):
    status: str


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "docs": len(rag.list_documents())}


@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se admiten archivos PDF.")
    try:
        content = await file.read()
        doc = rag.add_document(content, file.filename)
        return doc
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando PDF: {e}")


@app.get("/api/documents")
async def list_documents():
    return rag.list_documents()


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str) -> DeleteResponse:
    rag.delete_document(doc_id)
    return DeleteResponse(status="deleted")


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="El mensaje no puede estar vacío.")
    try:
        response = await rag.query(req.message, req.doc_ids, req.mode)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error del LLM: {e}")


# ─── Sirve el frontend ────────────────────────────────────────────────────────

@app.get("/")
async def serve_frontend():
    index = FRONTEND_DIR / "index.html"
    if not index.exists():
        return {"error": "Frontend no encontrado. Asegúrate de que existe frontend/index.html"}
    return FileResponse(str(index))

# Archivos estáticos adicionales (si los hubiera)
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ─── Arranque ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
