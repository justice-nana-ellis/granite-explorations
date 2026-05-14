"""RAG pipeline: CSV/Excel ingestion → pgvector → Claude analysis.

Database backend (auto-selected from env vars):
  DATABASE_URL set            → direct PostgreSQL via psycopg2
                                (Neon, Render, Railway, AWS RDS, Aurora, etc.)
  SUPABASE_URL + SUPABASE_KEY → Supabase REST client
  Both set                    → DATABASE_URL takes priority

Embedding backend (auto-selected from env vars):
  HUGGINGFACE_API_KEY set     → Hugging Face Inference API  (free, cloud-friendly)
  otherwise                   → local sentence-transformers  (free, no key needed)
  SENTENCE_TRANSFORMER_MODEL  → override the local model name (optional)
"""
from __future__ import annotations

import asyncio
import json
import logging

import numpy as np
import pandas as pd

from app.config import settings

logger = logging.getLogger(__name__)

EMBED_MODEL_HF = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIMS     = 384
EMBED_BATCH    = 256
INSERT_BATCH   = 500

_pg_conn        = None
_sb_client      = None
_local_model    = None


# ── Backend selector ───────────────────────────────────────────────────────────

def _backend() -> str:
    if settings.database_url:
        return "postgres"
    if settings.supabase_url and settings.supabase_key:
        return "supabase"
    raise RuntimeError(
        "No RAG database configured. "
        "Set DATABASE_URL  (direct Postgres)  or  "
        "SUPABASE_URL + SUPABASE_KEY  in your .env file."
    )


# ── PostgreSQL direct (psycopg2) ───────────────────────────────────────────────

def _get_pg_conn():
    global _pg_conn
    import psycopg2
    from pgvector.psycopg2 import register_vector

    if _pg_conn is None or _pg_conn.closed:
        _pg_conn = psycopg2.connect(settings.database_url)
        register_vector(_pg_conn)
        _pg_conn.autocommit = False
    else:
        try:
            _pg_conn.cursor().execute("SELECT 1")
        except Exception:
            _pg_conn = psycopg2.connect(settings.database_url)
            register_vector(_pg_conn)
            _pg_conn.autocommit = False
    return _pg_conn


async def _pg_ingest(session_id: str, filename: str,
                     contents: list[str], metadatas: list[dict],
                     embeddings: list[list[float]]) -> None:
    from psycopg2.extras import execute_values

    def _run():
        conn = _get_pg_conn()
        cur  = conn.cursor()
        execute_values(
            cur,
            """
            INSERT INTO rag_documents
                (session_id, filename, content, metadata, embedding)
            VALUES %s
            """,
            [
                (session_id, filename, contents[i],
                 json.dumps(metadatas[i]), np.array(embeddings[i]))
                for i in range(len(contents))
            ],
        )
        conn.commit()

    await asyncio.to_thread(_run)


async def _pg_retrieve(session_id: str, qvec: list[float], top_k: int) -> list[dict]:
    def _run():
        conn = _get_pg_conn()
        cur  = conn.cursor()
        cur.execute(
            """
            SELECT id, content, metadata,
                   1 - (embedding <=> %s) AS similarity
            FROM   rag_documents
            WHERE  session_id = %s
            ORDER  BY embedding <=> %s
            LIMIT  %s
            """,
            (np.array(qvec), session_id, np.array(qvec), top_k),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    return await asyncio.to_thread(_run)


async def _pg_delete(session_id: str) -> None:
    def _run():
        conn = _get_pg_conn()
        cur  = conn.cursor()
        cur.execute("DELETE FROM rag_documents WHERE session_id = %s", (session_id,))
        conn.commit()

    await asyncio.to_thread(_run)


# ── Supabase REST client ───────────────────────────────────────────────────────

def _get_sb_client():
    global _sb_client
    if _sb_client is None:
        from supabase import create_client
        _sb_client = create_client(settings.supabase_url, settings.supabase_key)
    return _sb_client


async def _sb_ingest(session_id: str, filename: str,
                     contents: list[str], metadatas: list[dict],
                     embeddings: list[list[float]]) -> None:
    sb = _get_sb_client()
    records = [
        {
            "session_id": session_id,
            "filename":   filename,
            "content":    contents[i],
            "metadata":   metadatas[i],
            "embedding":  embeddings[i],
        }
        for i in range(len(contents))
    ]
    for i in range(0, len(records), INSERT_BATCH):
        batch = records[i:i + INSERT_BATCH]
        await asyncio.to_thread(
            lambda b=batch: sb.table("rag_documents").insert(b).execute()
        )


async def _sb_retrieve(session_id: str, qvec: list[float], top_k: int) -> list[dict]:
    sb = _get_sb_client()
    result = await asyncio.to_thread(
        lambda: sb.rpc("match_rag_documents", {
            "query_embedding": qvec,
            "p_session_id":    session_id,
            "match_count":     top_k,
        }).execute()
    )
    return result.data or []


async def _sb_delete(session_id: str) -> None:
    sb = _get_sb_client()
    await asyncio.to_thread(
        lambda: sb.table("rag_documents").delete().eq("session_id", session_id).execute()
    )


# ── Embeddings ─────────────────────────────────────────────────────────────────

def _get_local_model():
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        name = settings.sentence_transformer_model or "all-MiniLM-L6-v2"
        logger.info("Loading local embedding model '%s' (first run may download ~90 MB)…", name)
        _local_model = SentenceTransformer(name)
        logger.info("Embedding model ready.")
    return _local_model


async def _embed_local(texts: list[str]) -> list[list[float]]:
    model   = _get_local_model()
    vectors = await asyncio.to_thread(
        model.encode, texts, show_progress_bar=False, convert_to_numpy=True
    )
    return vectors.tolist()


async def _embed_hf(texts: list[str]) -> list[list[float]]:
    import requests as _req
    HF_BATCH = 32  # HF free API payload limit
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), HF_BATCH):
        chunk = texts[i:i + HF_BATCH]
        resp = await asyncio.to_thread(
            lambda c=chunk: _req.post(
                f"https://api-inference.huggingface.co/pipeline/feature-extraction/{EMBED_MODEL_HF}",
                headers={"Authorization": f"Bearer {settings.huggingface_api_key}"},
                json={"inputs": c, "options": {"wait_for_model": True}},
                timeout=60,
            )
        )
        resp.raise_for_status()
        all_embeddings.extend(resp.json())

    return all_embeddings


async def _embed(texts: list[str]) -> list[list[float]]:
    # SENTENCE_TRANSFORMER_MODEL set → always use local (overrides HF key)
    # HUGGINGFACE_API_KEY set        → use HF Inference API
    # neither                        → local with default model
    if settings.sentence_transformer_model:
        return await _embed_local(texts)
    if settings.huggingface_api_key:
        return await _embed_hf(texts)
    return await _embed_local(texts)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _to_native(v):
    try:
        if isinstance(v, np.integer):  return int(v)
        if isinstance(v, np.floating): return None if np.isnan(v) else float(v)
        if isinstance(v, np.bool_):    return bool(v)
    except Exception:
        pass
    if isinstance(v, pd.Timestamp):
        return v.isoformat() if not pd.isna(v) else None
    try:
        if pd.isna(v): return None
    except (TypeError, ValueError):
        pass
    return v


# ── Public API ─────────────────────────────────────────────────────────────────

async def ingest_df(df: pd.DataFrame, session_id: str, filename: str) -> int:
    """Embed and store every row. Streams in batches — safe for 800K+ rows."""
    backend     = _backend()
    rows_iter   = df.iterrows()
    exhausted   = False
    total       = 0

    while not exhausted:
        contents:  list[str]  = []
        metadatas: list[dict] = []

        for _ in range(EMBED_BATCH):
            try:
                _, row = next(rows_iter)
                content = "; ".join(f"{k}={v}" for k, v in row.items() if pd.notna(v))
                contents.append(content)
                metadatas.append({k: _to_native(v) for k, v in row.items()})
            except StopIteration:
                exhausted = True
                break

        if not contents:
            break

        embeddings = await _embed(contents)

        if backend == "postgres":
            await _pg_ingest(session_id, filename, contents, metadatas, embeddings)
        else:
            await _sb_ingest(session_id, filename, contents, metadatas, embeddings)

        total += len(contents)
        logger.info("RAG ingestion: %d / %d rows…", total, len(df))

    logger.info("RAG ingestion complete: %d rows — session=%s backend=%s", total, session_id, backend)
    return total


async def retrieve(question: str, session_id: str, top_k: int = 15) -> list[dict]:
    """Embed question, return top-K most similar rows."""
    [qvec]  = await _embed([question])
    backend = _backend()

    if backend == "postgres":
        return await _pg_retrieve(session_id, qvec, top_k)
    return await _sb_retrieve(session_id, qvec, top_k)


async def rag_query(
    question: str,
    session_id: str,
    top_k: int = 15,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    system_prompt: str | None = None,
) -> tuple[str, list[dict]]:
    """Full RAG pipeline: vector search → context → Claude analysis."""
    rows    = await retrieve(question, session_id, top_k=top_k)
    context = "\n".join(r["content"] for r in rows) if rows else "No relevant data found."

    from app.services.claude_service import claude_service

    system = system_prompt or (
        "You are a precise data analyst. Answer using only the provided data rows. "
        "Be specific, cite exact numbers, and highlight key trends."
    )
    messages = [{
        "role": "user",
        "content": f"[Relevant data rows]\n{context}\n\n[Question]\n{question}",
    }]
    analysis = await claude_service.complete(
        messages,
        system=system,
        model=settings.claude_chat_model,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return analysis, rows


async def delete_session_data(session_id: str) -> None:
    """Remove all ingested rows for a session."""
    backend = _backend()
    if backend == "postgres":
        await _pg_delete(session_id)
    else:
        await _sb_delete(session_id)
    logger.info("RAG: deleted all rows for session=%s backend=%s", session_id, backend)
