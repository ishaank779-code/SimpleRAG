import os
import uuid
import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db, init_db
from app.models import Document, ChatHistory
from app.schemas import (
    DocumentOut,
    QueryRequest,
    QueryResponse,
    ChatHistoryOut,
    HealthResponse,
    SourceChunk,
)
from app.rag_pipeline import get_rag

settings = get_settings()

app = FastAPI(
    title="SimpleRAG – Document Q&A System",
    description="Full RAG pipeline with FastAPI + MySQL + FAISS + Sentence-Transformers",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.faiss_index_dir).mkdir(parents=True, exist_ok=True)
    init_db()
    # Pre-load models
    get_rag()
    print("[App] SimpleRAG started successfully")


@app.get("/health", response_model=HealthResponse)
def health(db: Session = Depends(get_db)):
    mysql_ok = "ok"
    try:
        db.execute(__import__("sqlalchemy").text("SELECT 1"))
    except Exception as e:
        mysql_ok = f"error: {e}"

    rag = get_rag()
    faiss_status = f"ok ({rag.index.ntotal} vectors)" if rag.index else "not loaded"

    return HealthResponse(
        status="ok",
        mysql=mysql_ok,
        faiss=faiss_status,
        embedding_model=settings.embedding_model,
    )


@app.post("/documents/upload", response_model=DocumentOut)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    allowed = {"pdf", "txt", "md"}
    ext = file.filename.rsplit(".", 1)[-1].lower() if file.filename else ""
    if ext not in allowed:
        raise HTTPException(400, f"Only {allowed} files are supported")

    # Save file
    stored_name = f"{uuid.uuid4().hex}.{ext}"
    file_path = os.path.join(settings.upload_dir, stored_name)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    rag = get_rag()
    try:
        doc = rag.ingest_document(
            db=db,
            file_path=file_path,
            original_name=file.filename,
            stored_filename=stored_name,
            file_type=ext,
        )
        return doc
    except Exception as e:
        # Clean up file on failure
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(500, f"Ingestion failed: {str(e)}")


@app.get("/documents", response_model=List[DocumentOut])
def list_documents(db: Session = Depends(get_db)):
    docs = db.query(Document).order_by(Document.created_at.desc()).all()
    return docs


@app.get("/documents/{doc_id}", response_model=DocumentOut)
def get_document(doc_id: int, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(404, "Document not found")
    return doc


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: int, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(404, "Document not found")

    # Remove physical file
    file_path = os.path.join(settings.upload_dir, doc.filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    # Note: We do not remove vectors from FAISS for simplicity in this short project.
    # In production you would rebuild the index or use a more advanced index.
    db.delete(doc)
    db.commit()
    return {"message": f"Document {doc_id} deleted"}


@app.post("/query", response_model=QueryResponse)
def query_rag(payload: QueryRequest, db: Session = Depends(get_db)):
    if payload.document_id is not None:
        doc = db.query(Document).filter(Document.id == payload.document_id).first()
        if not doc:
            raise HTTPException(404, "Document not found")
        if doc.status != "ready":
            raise HTTPException(400, f"Document status is '{doc.status}', not ready")

    rag = get_rag()
    result = rag.query(
        db=db,
        question=payload.question,
        document_id=payload.document_id,
        top_k=payload.top_k,
    )
    return QueryResponse(
        answer=result["answer"],
        sources=[SourceChunk(**s) for s in result["sources"]],
        question=result["question"],
    )


@app.get("/history", response_model=List[ChatHistoryOut])
def get_history(
    limit: int = Query(20, ge=1, le=100),
    document_id: Optional[int] = None,
    db: Session = Depends(get_db),
):
    q = db.query(ChatHistory).order_by(ChatHistory.created_at.desc())
    if document_id is not None:
        q = q.filter(ChatHistory.document_id == document_id)
    return q.limit(limit).all()


@app.get("/")
def root():
    return {
        "message": "SimpleRAG API is running",
        "docs": "/docs",
        "health": "/health",
    }
