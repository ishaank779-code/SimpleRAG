from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class DocumentOut(BaseModel):
    id: int
    filename: str
    original_name: str
    file_type: str
    num_chunks: int
    status: str
    created_at: datetime

    class Config:
        from_attributes = True


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    document_id: Optional[int] = None  # None = search all documents
    top_k: Optional[int] = Field(default=None, ge=1, le=10)


class SourceChunk(BaseModel):
    chunk_id: int
    document_id: int
    document_name: str
    content: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    question: str


class ChatHistoryOut(BaseModel):
    id: int
    document_id: Optional[int]
    question: str
    answer: str
    sources: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class HealthResponse(BaseModel):
    status: str
    mysql: str
    faiss: str
    embedding_model: str
