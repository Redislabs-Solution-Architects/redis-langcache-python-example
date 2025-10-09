#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import uuid
from typing import List, Tuple

import numpy as np
from dotenv import load_dotenv
from pypdf import PdfReader
from openai import OpenAI
import redis
from redis.exceptions import ResponseError

from langcache import LangCache
from langcache.models import SearchStrategy

load_dotenv()

# =========================
# Config (env or defaults)
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

REDIS_URL = os.getenv("REDIS_URL")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") or None
REDIS_TLS = os.getenv("REDIS_TLS", "false").lower() in {"1", "true", "yes"}

INDEX_NAME = os.getenv("REDIS_INDEX_NAME", "docs_idx_v1")
DOC_PREFIX = os.getenv("DOC_PREFIX", "doc")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1536"))

LANGCACHE_API_KEY = os.getenv("LANGCACHE_API_KEY") or os.getenv("LANGCACHE_SERVICE_KEY")
LANGCACHE_CACHE_ID = os.getenv("LANGCACHE_CACHE_ID")
LANGCACHE_BASE_URL = os.getenv("LANGCACHE_BASE_URL", "https://gcp-us-east4.langcache.redis.io")

PDF_PATH = os.getenv("PDF_PATH", "regulamento-de-truco-2019.pdf")

# =========================
# Clients
# =========================
if not OPENAI_API_KEY:
    raise SystemExit("Missing OPENAI_API_KEY in env.")

openai_client = OpenAI(api_key=OPENAI_API_KEY)


def connect_redis_from_env():
    """Connect to Redis using REDIS_URL if provided; else host/port/TLS vars."""
    if REDIS_URL:
        return redis.Redis.from_url(REDIS_URL, decode_responses=False)
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        password=REDIS_PASSWORD,
        ssl=REDIS_TLS,
        decode_responses=False,
    )


r = connect_redis_from_env()
try:
    r.ping()
except Exception as e:
    raise SystemExit(f"[Redis connection failed] {e}")


# =========================
# Vector Index Helpers
# =========================
def ensure_index():
    """Create an HNSW vector index on HASH if it doesn't exist."""
    try:
        r.execute_command("FT.INFO", INDEX_NAME)
        return
    except ResponseError:
        pass

    cmd = [
        "FT.CREATE", INDEX_NAME,
        "ON", "HASH",
        "PREFIX", "1", f"{DOC_PREFIX}:",
        "SCHEMA",
        "text", "TEXT",
        "source", "TAG",
        "embedding", "VECTOR", "HNSW", "6",
        "TYPE", "FLOAT32",
        "DIM", str(EMBED_DIM),
        "DISTANCE_METRIC", "COSINE",
    ]
    r.execute_command(*cmd)


def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)
    resp = openai_client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = [np.array(item.embedding, dtype=np.float32) for item in resp.data]
    return np.vstack(vecs)


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> List[str]:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(text[start:end])
        start += max(1, chunk_size - overlap)
    return [c.strip() for c in chunks if c.strip()]


def load_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    return "\n\n".join((p.extract_text() or "").strip() for p in reader.pages)


def upsert_pdf_to_redis(path: str) -> int:
    ensure_index()
    text = load_pdf_text(path)
    chunks = chunk_text(text)
    if not chunks:
        return 0

    vecs = embed_texts(chunks)
    source = os.path.basename(path)
    pipe = r.pipeline()
    for chunk, vec in zip(chunks, vecs):
        key = f"{DOC_PREFIX}:{uuid.uuid4()}"
        pipe.hset(key, mapping={
            b"text": chunk.encode("utf-8"),
            b"source": source.encode("utf-8"),
            b"embedding": vec.tobytes(),
        })
    pipe.execute()
    return len(chunks)


def knn_search(query: str, k: int = 5) -> List[dict]:
    qvecs = embed_texts([query])
    if qvecs.shape[0] == 0:
        return []
    qvec = qvecs[0]

    query_expr = f'*=>[KNN {k} @embedding $vec AS score]'
    args = [
        "FT.SEARCH", INDEX_NAME, query_expr,
        "PARAMS", "2", "vec", qvec.tobytes(),
        "SORTBY", "score", "ASC",
        "RETURN", "3", "text", "source", "score",
        "DIALECT", "2",
    ]
    res = r.execute_command(*args)

    out = []
    if not res or res[0] == 0:
        return out

    i = 1
    while i < len(res):
        key = res[i]; i += 1
        fields = res[i]; i += 1
        row = {"key": key.decode() if isinstance(key, (bytes, bytearray)) else key}

        def _to_str(x):
            return x.decode("utf-8", "ignore") if isinstance(x, (bytes, bytearray)) else str(x)

        if isinstance(fields, (list, tuple)):
            for j in range(0, len(fields), 2):
                field = _to_str(fields[j])
                value = fields[j + 1]
                if field == "text":
                    row["text"] = _to_str(value)
                elif field == "source":
                    row["source"] = _to_str(value)
                elif field == "score":
                    try:
                        row["score"] = float(_to_str(value))
                    except ValueError:
                        row["score"] = None
        out.append(row)
    return out


def build_context(hits: List[dict]) -> Tuple[str, List[str]]:
    parts, sources = [], []
    for h in hits:
        if "text" in h:
            parts.append(h["text"])
        if "source" in h:
            sources.append(h["source"])
    ctx = "\n\n---\n\n".join(parts)

    seen = set()
    uniq_sources = []
    for s in sources:
        if s not in seen:
            seen.add(s)
            uniq_sources.append(s)
    return ctx, uniq_sources


def answer_with_context(question: str, context: str) -> str:
    msgs = [
        {"role": "system", "content": "Answer using only the provided context. If unsure, say you don't know."},
        {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"},
    ]
    resp = openai_client.chat.completions.create(model=OPENAI_MODEL, messages=msgs, temperature=0.2)
    return resp.choices[0].message.content.strip()


def chat():
    if not (LANGCACHE_BASE_URL and LANGCACHE_CACHE_ID and LANGCACHE_API_KEY):
        print("[WARN] LangCache env vars missing; caching will fail.")
    print("RAG over Redis (redis-py) + LangCache — type 'exit' to quit.\n")

    with LangCache(
        server_url=LANGCACHE_BASE_URL,
        cache_id=LANGCACHE_CACHE_ID,
        api_key=LANGCACHE_API_KEY,
    ) as lc:

        while True:
            q = input("Ask: ").strip()
            if q.lower() in {"exit", "quit"}:
                break

            t0 = time.perf_counter()
            cached = None
            try:
                cached = lc.search(
                    prompt=q,
                    similarity_threshold=0.7,
                    attributes={"index": INDEX_NAME},
                )
            except Exception as e:
                print(f"[LangCache search error] {e}")

            t_cache = time.perf_counter() - t0

            if (not cached) or (not getattr(cached, "data", None)):
                try:
                    cached = lc.search(
                        prompt=q,
                        attributes={"index": INDEX_NAME},
                        search_strategies=[SearchStrategy.EXACT, SearchStrategy.SEMANTIC],
                    )
                except Exception as e:
                    print(f"[LangCache exact search error] {e}")

            if cached and getattr(cached, "data", None):
                print(f"[CACHE HIT] {t_cache:.3f}s")
                print(cached.data[0].response)
                print("-" * 60)
                continue

            print(f"[CACHE MISS] {t_cache:.3f}s → retrieving...")
            hits = knn_search(q, k=5)
            if not hits:
                print("No relevant context found in the vector index.")
                print("-" * 60)
                continue

            ctx, sources = build_context(hits)
            ans = answer_with_context(q, ctx)

            try:
                lc.set(
                    prompt=q,
                    response=ans,
                    attributes={"index": INDEX_NAME, "sources": ",".join(sources[:5])},
                )
            except Exception as e:
                print(f"[LangCache set error] {e}")

            print(ans)
            print("Sources:", sources)
            print("-" * 60)


if __name__ == "__main__":
    if os.path.exists(PDF_PATH):
        try:
            n = upsert_pdf_to_redis(PDF_PATH)
            print(f"Ingested {n} chunks from: {PDF_PATH}")
        except Exception as e:
            print(f"[Ingestion error] {e}")
    else:
        print(f"[WARN] PDF not found at {PDF_PATH}; skipping ingestion.")

    chat()