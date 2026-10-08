import os
import json
import pickle
from pathlib import Path
from typing import List, Tuple, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from openai import OpenAI
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import Document, Chunk, ChatHistory

settings = get_settings()


class RAGPipeline:
    def __init__(self):
        self.settings = settings
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.index: Optional[faiss.IndexFlatIP] = None  # Inner Product (cosine after normalize)
        self.id_counter = 0  # global faiss_id counter
        self._load_or_create_index()

        self.llm_client = None
        if settings.openai_api_key:
            self.llm_client = OpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
            )

    def _index_path(self) -> Path:
        return Path(self.settings.faiss_index_dir) / "index.faiss"

    def _meta_path(self) -> Path:
        return Path(self.settings.faiss_index_dir) / "meta.pkl"

    def _load_or_create_index(self):
        Path(self.settings.faiss_index_dir).mkdir(parents=True, exist_ok=True)
        index_file = self._index_path()
        meta_file = self._meta_path()

        if index_file.exists() and meta_file.exists():
            self.index = faiss.read_index(str(index_file))
            with open(meta_file, "rb") as f:
                meta = pickle.load(f)
                self.id_counter = meta.get("id_counter", 0)
            print(f"[RAG] Loaded FAISS index with {self.index.ntotal} vectors")
        else:
            # Inner product + normalized vectors = cosine similarity
            self.index = faiss.IndexFlatIP(self.dimension)
            self.id_counter = 0
            print("[RAG] Created new FAISS index")

    def _save_index(self):
        faiss.write_index(self.index, str(self._index_path()))
        with open(self._meta_path(), "wb") as f:
            pickle.dump({"id_counter": self.id_counter}, f)

    def extract_text(self, file_path: str, file_type: str) -> str:
        if file_type == "pdf":
            reader = PdfReader(file_path)
            pages = []
            for page in reader.pages:
                text = page.extract_text() or ""
                pages.append(text)
            return "\n".join(pages)
        elif file_type in ("txt", "md"):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    def chunk_text(self, text: str) -> List[str]:
        chunk_size = self.settings.chunk_size
        overlap = self.settings.chunk_overlap
        chunks = []
        start = 0
        text = text.strip()
        if not text:
            return []

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    def embed(self, texts: List[str]) -> np.ndarray:
        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # important for IndexFlatIP
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)

    def ingest_document(
        self,
        db: Session,
        file_path: str,
        original_name: str,
        stored_filename: str,
        file_type: str,
    ) -> Document:
        # 1. Create DB record
        doc = Document(
            filename=stored_filename,
            original_name=original_name,
            file_type=file_type,
            status="processing",
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)

        try:
            # 2. Extract + chunk
            text = self.extract_text(file_path, file_type)
            chunks = self.chunk_text(text)
            if not chunks:
                doc.status = "failed"
                db.commit()
                raise ValueError("No text could be extracted from the document")

            # 3. Embed
            embeddings = self.embed(chunks)

            # 4. Add to FAISS + MySQL
            start_id = self.id_counter
            for i, (chunk_text, emb) in enumerate(zip(chunks, embeddings)):
                faiss_id = self.id_counter
                self.index.add(emb.reshape(1, -1))
                self.id_counter += 1

                chunk_row = Chunk(
                    document_id=doc.id,
                    chunk_index=i,
                    content=chunk_text,
                    faiss_id=faiss_id,
                )
                db.add(chunk_row)

            doc.num_chunks = len(chunks)
            doc.status = "ready"
            db.commit()
            self._save_index()
            print(f"[RAG] Ingested document {doc.id} with {len(chunks)} chunks")
            return doc

        except Exception as e:
            doc.status = "failed"
            db.commit()
            raise e

    def retrieve(
        self,
        db: Session,
        question: str,
        document_id: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> List[Tuple[Chunk, float]]:
        if self.index is None or self.index.ntotal == 0:
            return []

        k = top_k or self.settings.top_k
        query_emb = self.embed([question])

        # Search more candidates then filter by document_id if needed
        search_k = min(k * 5 if document_id else k, self.index.ntotal)
        scores, indices = self.index.search(query_emb, search_k)

        results = []
        for score, faiss_idx in zip(scores[0], indices[0]):
            if faiss_idx < 0:
                continue
            chunk = (
                db.query(Chunk)
                .filter(Chunk.faiss_id == int(faiss_idx))
                .first()
            )
            if not chunk:
                continue
            if document_id is not None and chunk.document_id != document_id:
                continue
            results.append((chunk, float(score)))
            if len(results) >= k:
                break
        return results

    def generate_answer(self, question: str, contexts: List[str]) -> str:
        if not contexts:
            return "I could not find any relevant information in the uploaded documents to answer this question."

        context_block = "\n\n---\n\n".join(contexts)

        system_prompt = (
            "You are a helpful assistant that answers questions strictly based on the provided context. "
            "If the context does not contain enough information, say so clearly. "
            "Do not invent facts. Keep answers concise and accurate."
        )
        user_prompt = f"""Context:
{context_block}

Question: {question}

Answer based only on the context above:"""

        if self.llm_client is None:
            # Fallback when no API key is set – return the retrieved context
            return (
                "[LLM API key not configured – showing retrieved context only]\n\n"
                + context_block[:1500]
            )

        try:
            response = self.llm_client.chat.completions.create(
                model=self.settings.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=800,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error calling LLM: {str(e)}\n\nRetrieved context:\n{context_block[:1000]}"

    def query(
        self,
        db: Session,
        question: str,
        document_id: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> dict:
        retrieved = self.retrieve(db, question, document_id, top_k)
        contexts = [c.content for c, _ in retrieved]
        answer = self.generate_answer(question, contexts)

        sources = []
        for chunk, score in retrieved:
            sources.append(
                {
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "document_name": chunk.document.original_name if chunk.document else "Unknown",
                    "content": chunk.content[:300] + ("..." if len(chunk.content) > 300 else ""),
                    "score": round(score, 4),
                }
            )

        # Save chat history
        history = ChatHistory(
            document_id=document_id,
            question=question,
            answer=answer,
            sources=json.dumps(sources),
        )
        db.add(history)
        db.commit()

        return {
            "answer": answer,
            "sources": sources,
            "question": question,
        }


# Singleton
_rag: Optional[RAGPipeline] = None


def get_rag() -> RAGPipeline:
    global _rag
    if _rag is None:
        _rag = RAGPipeline()
    return _rag
