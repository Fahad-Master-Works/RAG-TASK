
# RAG Pipeline (Arabic) — FastAPI + FAISS + Groq

> **Features**
> - Arabic-friendly embeddings: `intfloat/multilingual-e5-base` (supports Arabic).
> - Vector store: FAISS (cosine similarity).
> - API: FastAPI `GET /ask?question=...` returning `answer` + `sources`.
> - Token safety: context cap + short answers.
> - Dockerized, CPU-ready by default.

## 1) Quick Start (Local, without Docker)

### Requirements
- Python 3.10+
- A Groq API Key in your environment (`GROQ_API_KEY`).
- Your PDF file (default path is `./data/AI.pdf`).

### Steps
```bash
# 1) Create a venv (optional but recommended)
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

# 2) Install deps
pip install -r requirements.txt

# 3) Prepare folders
mkdir -p storage data

# 4) Put your PDF in ./data
# (Example: AI.pdf)
# cp /path/to/AI.pdf ./data/AI.pdf

# 5) Set your API key (Linux/macOS)
export GROQ_API_KEY=xxxxxxxxxxxxxxxx

#    On Windows PowerShell:
#    $env:GROQ_API_KEY="xxxxxxxxxxxxxxxx"

# 6) Run the API
uvicorn app.main:app --reload --port 8000
```

Open http://127.0.0.1:8000/health — you should see `{"status":"ok"}`.

### Ask questions
```
GET http://127.0.0.1:8000/ask?question=ماهو العنوان الاساسي
GET http://127.0.0.1:8000/ask?question=انواع الذكاء الاصطناعي
```

## 2) Run with Docker (recommended)

```bash
# 1) Build the image
docker build -t rag-fastapi-groq .

# 2) Create data + storage folders
mkdir -p data storage
# Copy your PDF into ./data (e.g., AI.pdf)

# 3) Run the container
docker run --rm -p 8000:8000 \
  -e GROQ_API_KEY=$GROQ_API_KEY \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/storage:/app/storage \
  rag-fastapi-groq
```

Now call:
```
curl "http://localhost:8000/ask?question=ماهو العنوان الاساسي"
curl "http://localhost:8000/ask?question=انواع الذكاء الاصطناعي"
```

## 3) Configuration

You can override defaults via ENV:
- `PDF_PATH` (default `/app/data/AI.pdf` in Docker, `data/AI.pdf` locally)
- `INDEX_DIR` (default `storage`)
- `EMBEDDING_MODEL` (default `intfloat/multilingual-e5-base`)
- `GROQ_MODEL` (default `llama-3.1-8b-instant`) — change to `llama-3.3-70b-versatile` for higher quality
- `TOP_K` (default `4`)
- `CHUNK_SIZE` (default `900`)
- `CHUNK_OVERLAP` (default `120`)
- `MAX_CONTEXT_CHARS` (default `12000`)
- `REBUILD_INDEX` (set to `1` to force re-index at next start)

## 4) Notes on Arabic Embeddings
- This project uses **Multilingual-E5** which **supports Arabic**.
- Use the proper prefixes:
  - Documents: `passage: <text>`
  - Queries: `query: <question>`

## 5) How it works
1. On first run, the app extracts text from the PDF, chunks it, embeds with E5, stores vectors in FAISS.
2. When you call `/ask`, it embeds your question, retrieves top `k` chunks, builds an Arabic prompt, and asks a Groq LLM.
3. The API returns a concise Arabic answer + the list of sourced chunks (page numbers + scores).

## 6) Troubleshooting
- **`GROQ_API_KEY is not set`** → export the key or create `.env` and run via `dotenv`.
- **Empty answers** → try larger `TOP_K` or increase `CHUNK_SIZE`.
- **Index too old** → `REBUILD_INDEX=1` then restart.
- **Windows PowerShell env** → `$env:GROQ_API_KEY="..."`

## 7) Example `.env`
```
# copy to .env and fill
GROQ_API_KEY=xxxxxxxxxxxxxxxx
PDF_PATH=./data/AI.pdf
```

---

**Good luck!** 🚀
