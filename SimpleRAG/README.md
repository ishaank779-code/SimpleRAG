# SimpleRAG – Full RAG Project (Python + MySQL)

A complete, short but production-style **Retrieval-Augmented Generation (RAG)** system.

**Stack**
- **Backend**: FastAPI (Python)
- **Database**: MySQL (documents metadata + chunks + chat history)
- **Vector Store**: FAISS (CPU)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (local, free)
- **LLM**: OpenAI-compatible API (OpenAI / Groq / Together etc.)

---

## Features

| Feature | Description |
|---------|-------------|
| Document Upload | PDF, TXT, MD |
| Chunking | Configurable size + overlap |
| Embeddings | Local MiniLM model |
| Vector Search | FAISS (cosine similarity) |
| MySQL | Stores documents, chunks, chat history |
| Query | Ask questions, get answer + sources |
| Chat History | All Q&A saved in MySQL |
| Health Check | `/health` endpoint |

---

## Project Structure

```
SimpleRAG/
├── app/
│   ├── main.py            # FastAPI routes
│   ├── config.py          # Settings from .env
│   ├── database.py        # SQLAlchemy engine
│   ├── models.py          # Document, Chunk, ChatHistory tables
│   ├── schemas.py         # Pydantic models
│   └── rag_pipeline.py    # Core RAG logic
├── uploads/               # Uploaded files
├── faiss_index/           # Persisted FAISS index
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup (Step by Step)

### 1. Create MySQL Database

```sql
CREATE DATABASE simple_rag CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

(Tables are created automatically by SQLAlchemy on first run.)

### 2. Python Environment

```bash
cd SimpleRAG
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=simple_rag

# Free option – Groq (https://console.groq.com)
OPENAI_API_KEY=gsk_xxxxxxxx
OPENAI_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=llama-3.1-8b-instant

# Or OpenAI
# OPENAI_API_KEY=sk-xxxx
# OPENAI_BASE_URL=https://api.openai.com/v1
# LLM_MODEL=gpt-4o-mini
```

> Without API key the system still works – it returns the retrieved context instead of LLM answer.

### 4. Run the Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open: **http://localhost:8000/docs** (Swagger UI)

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/documents/upload` | Upload PDF/TXT/MD |
| GET | `/documents` | List all documents |
| GET | `/documents/{id}` | Get one document |
| DELETE | `/documents/{id}` | Delete document |
| POST | `/query` | Ask a question (RAG) |
| GET | `/history` | Chat history |

### Example: Upload

```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -F "file=@sample.pdf"
```

### Example: Query

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the document?", "top_k": 3}'
```

Response:

```json
{
  "answer": "...",
  "sources": [
    {
      "chunk_id": 1,
      "document_id": 1,
      "document_name": "sample.pdf",
      "content": "...",
      "score": 0.82
    }
  ],
  "question": "What is the main topic..."
}
```

---

## How RAG Works (Pipeline)

```
1. Upload PDF/TXT
      ↓
2. Extract text (pypdf)
      ↓
3. Chunk (500 chars, 50 overlap)
      ↓
4. Embed with MiniLM → 384-dim vectors
      ↓
5. Store vectors in FAISS + metadata in MySQL
      ↓
6. User asks question
      ↓
7. Embed question → FAISS top-k search
      ↓
8. Retrieve matching chunks from MySQL
      ↓
9. Build prompt with context → LLM → Answer + Sources
      ↓
10. Save Q&A in chat_history table
```

---

## Why MySQL + FAISS?

- **MySQL** → relational data (documents, chat history, future user auth)
- **FAISS** → fast vector similarity search (perfect for short/medium projects)
- Easy to understand and extend (add user login, multi-tenant, etc.)

---

## Possible Extensions (for portfolio)

1. User authentication (JWT)
2. Streaming answers (SSE)
3. Hybrid search (BM25 + vector)
4. Docker + docker-compose
5. Simple Streamlit / React frontend
6. Rebuild FAISS index on document delete

---

## License

MIT – free to use for learning & portfolio.
