
import os
import io
import math
import json
from typing import List, Dict, Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import faiss
from dotenv import load_dotenv

# ========== Config ==========
load_dotenv()
PDF_PATH = os.getenv("PDF_PATH", "data/AI.pdf")  # default relative path; can be overridden
INDEX_DIR = os.getenv("INDEX_DIR", "storage")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
TOP_K_DEFAULT = int(os.getenv("TOP_K", "4"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))          # characters
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))    # characters
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))  # safety cap for prompt context
REBUILD_INDEX = os.getenv("REBUILD_INDEX", "0") == "1"

# Ensure index dir exists
os.makedirs(INDEX_DIR, exist_ok=True)

# ========== Utilities ==========
def _approx_tokens_from_chars(s: str) -> int:
    # Rough 4 chars/token heuristic
    return max(1, math.ceil(len(s) / 4))

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / n

def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    # Clean and normalize whitespace
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

def _load_pdf_chunks(pdf_path: str) -> List[Dict[str, Any]]:
    reader = PdfReader(pdf_path)
    docs = []
    for i, page in enumerate(reader.pages, start=1):
        txt = page.extract_text() or ""
        for ch in _chunk_text(txt):
            docs.append({"page": i, "text": ch})
    return docs

def _save_index(index: faiss.Index, docs: List[Dict[str, Any]]):
    faiss.write_index(index, os.path.join(INDEX_DIR, "faiss.index"))
    with open(os.path.join(INDEX_DIR, "docs.json"), "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False)

def _load_index():
    index_path = os.path.join(INDEX_DIR, "faiss.index")
    docs_path = os.path.join(INDEX_DIR, "docs.json")
    if os.path.exists(index_path) and os.path.exists(docs_path) and not REBUILD_INDEX:
        index = faiss.read_index(index_path)
        with open(docs_path, "r", encoding="utf-8") as f:
            docs = json.load(f)
        return index, docs
    return None, None

def _build_index(pdf_path: str, model: SentenceTransformer):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at {pdf_path}. Set PDF_PATH environment variable.")
    docs = _load_pdf_chunks(pdf_path)
    if not docs:
        raise ValueError("No text could be extracted from the PDF.")
    passages = [f"passage: {d['text']}" for d in docs]
    embs = model.encode(passages, batch_size=32, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    # Cosine sim via dot product on normalized vectors
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs.astype(np.float32))
    _save_index(index, docs)
    return index, docs

def _ensure_index(model: SentenceTransformer):
    idx, docs = _load_index()
    if idx is not None:
        return idx, docs
    return _build_index(PDF_PATH, model)

def _search(index: faiss.Index, model: SentenceTransformer, query: str, k: int = TOP_K_DEFAULT):
    q_emb = model.encode([f"query: {query}"], convert_to_numpy=True, normalize_embeddings=True)
    q_emb = q_emb.astype(np.float32)
    scores, idxs = index.search(q_emb, k)
    return scores[0], idxs[0]

def _compose_prompt(question: str, hits: List[Dict[str, Any]]) -> str:
    # Build Arabic-first RAG prompt with sources
    context_blocks = []
    total_chars = 0
    for h in hits:
        snippet = h["text"]
        # limit snippet size if needed
        if len(snippet) > 2000:
            snippet = snippet[:2000] + "…"
        block = f"[المصدر صفحة {h['page']}]\n{snippet}"
        if total_chars + len(block) > MAX_CONTEXT_CHARS:
            break
        context_blocks.append(block)
        total_chars += len(block)

    context = "\n\n---\n\n".join(context_blocks) if context_blocks else "لا توجد مقاطع مناسبة مسترجعة."
    system = (
    "أنت مساعد ذكي يقدّم إجابات موجزة وواضحة باللغة العربية اعتمادًا على المقاطع المزوّدة. "
    "إذا كان السؤال عن العنوان الأساسي أو ملخص، فاعتمد على العنوان أو أول نصوص الوثيقة. "
    "إذا كان السؤال عن أنواع أو تفاصيل، لخّص ما تجده حتى لو النص غير مكتمل. "
    "لا تقل 'لا توجد معلومات' إلا إذا لم يكن هناك أي ذكر أو تلميح للموضوع إطلاقًا."
    )

    user = (
        f"السؤال: {question}\n\n"
        f"المقاطع المسترجعة من الوثيقة:\n{context}\n\n"
        "أعطني إجابة مباشرة ومختصرة مع الاستناد للمقاطع."
    )
    prompt = {"system": system, "user": user}
    return prompt

# ========== App Setup ==========
app = FastAPI(title="RAG API with FAISS + Groq (Arabic)")

class AskResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

# Lazy globals
_model: Optional[SentenceTransformer] = None
_index: Optional[faiss.Index] = None
_docs: Optional[List[Dict[str, Any]]] = None
_groq_client = None

def _init_all():
    global _model, _index, _docs, _groq_client
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    if _index is None or _docs is None or REBUILD_INDEX:
        _index, _docs = _ensure_index(_model)
    if _groq_client is None:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set. Please configure your API key.")
        from groq import Groq
        _groq_client = Groq(api_key=GROQ_API_KEY)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ask", response_model=AskResponse)
def ask(question: str = Query(..., description="Your question in Arabic (or English)."),
        top_k: int = Query(TOP_K_DEFAULT, ge=1, le=10)):
    try:
        _init_all()
        scores, idxs = _search(_index, _model, question, k=top_k)
        hits = []
        for score, idx in zip(scores.tolist(), idxs.tolist()):
            if idx == -1:
                continue
            d = _docs[idx]
            hits.append({"page": d["page"], "text": d["text"], "score": float(score)})
        prompt = _compose_prompt(question, hits)

        # Build messages for Groq Chat Completions
        messages = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ]

        # Call Groq
        completion = _groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.2,
        )
        answer = completion.choices[0].message.content.strip()

        # Trim overly long answers (safety)
        if _approx_tokens_from_chars(answer) > 1500:
            answer = answer[:6000] + "…"

        return AskResponse(answer=answer, sources=hits)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
