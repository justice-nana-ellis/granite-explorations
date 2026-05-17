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


async def _pg_ingest(rag_id: str, filename: str,
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
                (rag_id, filename, contents[i],
                 json.dumps(metadatas[i]), np.array(embeddings[i]))
                for i in range(len(contents))
            ],
        )
        conn.commit()

    await asyncio.to_thread(_run)


async def _pg_retrieve(rag_id: str, qvec: list[float], top_k: int) -> list[dict]:
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
            (np.array(qvec), rag_id, np.array(qvec), top_k),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    return await asyncio.to_thread(_run)


async def _pg_delete(rag_id: str) -> None:
    def _run():
        conn = _get_pg_conn()
        cur  = conn.cursor()
        try:
            cur.execute("DELETE FROM rag_documents WHERE session_id = %s", (rag_id,))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()

    await asyncio.to_thread(_run)


# ── Supabase REST client ───────────────────────────────────────────────────────

def _get_sb_client():
    # Always create a fresh client — the httpx session inside supabase-py is not
    # thread-safe, so sharing one instance across asyncio.to_thread calls causes
    # [Errno 35] ReadErrors when concurrent requests hit the same HTTP/2 connection.
    from supabase import create_client
    return create_client(settings.supabase_url, settings.supabase_key)


async def _sb_ingest(rag_id: str, filename: str,
                     contents: list[str], metadatas: list[dict],
                     embeddings: list[list[float]]) -> None:
    sb = _get_sb_client()
    records = [
        {
            "session_id": rag_id,   # DB column is session_id
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


async def _sb_retrieve(rag_id: str, qvec: list[float], top_k: int) -> list[dict]:
    sb = _get_sb_client()
    result = await asyncio.to_thread(
        lambda: sb.rpc("match_rag_documents", {
            "query_embedding": qvec,
            "p_session_id":    rag_id,
            "match_count":     top_k,
        }).execute()
    )
    return result.data or []


async def _sb_batch_delete(
    filters: dict,
    batch_size: int = 200,
    progress_callback=None,  # optional async callable(deleted_so_far: int, message: str)
) -> int:
    """Delete rows matching filters in small batches.

    Each batch is a separate asyncio.to_thread call so the async progress_callback
    can be awaited between batches — safe for any dataset size.
    """
    deleted = 0
    while True:
        def _fetch(f=filters):
            sb    = _get_sb_client()
            query = sb.table("rag_documents").select("id").limit(batch_size)
            for col, val in f.items():
                query = query.eq(col, val)
            return [r["id"] for r in (query.execute().data or [])]

        ids = await asyncio.to_thread(_fetch)
        if not ids:
            break

        def _delete(chunk=ids):
            sb = _get_sb_client()
            sb.table("rag_documents").delete().in_("id", chunk).execute()

        await asyncio.to_thread(_delete)
        deleted += len(ids)

        msg = f"Deleted {deleted:,} rows so far…"
        logger.info("RAG batch delete — %s", msg)
        if progress_callback:
            await progress_callback(deleted, msg)

    return deleted


async def _sb_delete(rag_id: str, progress_callback=None) -> None:
    await _sb_batch_delete({"session_id": rag_id}, progress_callback=progress_callback)


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

async def ingest_df(
    df: pd.DataFrame,
    rag_id: str,
    filename: str,
    progress_callback=None,  # optional async callable(msg: str)
) -> int:
    """Embed and store every row. Streams in batches — safe for 800K+ rows.

    If progress_callback is provided it is awaited after every batch so the
    caller can relay live progress (e.g. SSE stream) while the terminal logs
    continue as normal.
    """
    backend     = _backend()
    rows_iter   = df.iterrows()
    exhausted   = False
    total       = 0
    total_rows  = len(df)

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
            await _pg_ingest(rag_id, filename, contents, metadatas, embeddings)
        else:
            await _sb_ingest(rag_id, filename, contents, metadatas, embeddings)

        total += len(contents)
        msg = f"[{filename}] {total:,} / {total_rows:,} rows ingested…"
        logger.info("RAG ingestion progress — %s", msg)
        if progress_callback:
            await progress_callback(msg)

    logger.info("RAG ingestion complete: %d rows — rag_id=%s backend=%s", total, rag_id, backend)
    invalidate_metadata_cache(rag_id)
    from app.services.fund_forecast_service import invalidate_analytics_cache
    invalidate_analytics_cache(rag_id)
    return total


async def retrieve(question: str, rag_id: str, top_k: int = 15) -> list[dict]:
    """Embed question, return top-K most similar rows."""
    [qvec]  = await _embed([question])
    backend = _backend()

    if backend == "postgres":
        return await _pg_retrieve(rag_id, qvec, top_k)
    return await _sb_retrieve(rag_id, qvec, top_k)


async def rag_query(
    question: str,
    rag_id: str,
    top_k: int = 15,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    system_prompt: str | None = None,
) -> tuple[str, list[dict]]:
    """Full RAG pipeline: vector search → context → Claude analysis."""
    rows    = await retrieve(question, rag_id, top_k=top_k)
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


async def rag_retrieve_multi(
    question: str,
    rag_ids: list[str],
    top_k: int = 10,
) -> tuple[list[dict], str]:
    """Retrieve rows from multiple rag_ids in parallel.

    Returns (all_rows, context_string) without calling Claude — lets the
    caller stream the LLM response independently.
    """
    results = await asyncio.gather(
        *[retrieve(question, rid, top_k=top_k) for rid in rag_ids],
        return_exceptions=True,
    )
    all_rows: list[dict] = []
    for rid, result in zip(rag_ids, results):
        if isinstance(result, Exception):
            logger.warning("RAG retrieve failed for rag_id=%s: %s", rid, result)
            continue
        for row in result:
            row["rag_id"] = rid
        all_rows.extend(result)
    all_rows.sort(key=lambda r: r.get("similarity") or 0, reverse=True)
    context = "\n".join(r["content"] for r in all_rows) if all_rows else "No relevant data found."
    return all_rows, context


async def rag_chat_multi(
    question: str,
    rag_ids: list[str],
    top_k: int = 10,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    system_prompt: str | None = None,
) -> tuple[str, list[dict]]:
    """Retrieve from multiple rag_ids in parallel, combine, then send to Claude."""
    # Embed question once, retrieve from all rag_ids concurrently
    results = await asyncio.gather(
        *[retrieve(question, rid, top_k=top_k) for rid in rag_ids],
        return_exceptions=True,
    )

    all_rows: list[dict] = []
    for rid, result in zip(rag_ids, results):
        if isinstance(result, Exception):
            logger.warning("RAG retrieve failed for rag_id=%s: %s", rid, result)
            continue
        for row in result:
            row["rag_id"] = rid   # tag each row with its source
        all_rows.extend(result)

    # Sort combined rows by similarity descending
    all_rows.sort(key=lambda r: r.get("similarity") or 0, reverse=True)

    context = "\n".join(r["content"] for r in all_rows) if all_rows else "No relevant data found."

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
    return analysis, all_rows


async def delete_rag_data(rag_id: str, progress_callback=None) -> None:
    """Remove all ingested rows for a rag_id."""
    backend = _backend()
    if backend == "postgres":
        await _pg_delete(rag_id)
    else:
        await _sb_delete(rag_id, progress_callback=progress_callback)
    logger.info("RAG: deleted all rows for rag_id=%s backend=%s", rag_id, backend)
    invalidate_metadata_cache(rag_id)
    from app.services.fund_forecast_service import invalidate_analytics_cache
    invalidate_analytics_cache(rag_id)


async def list_rag_files(rag_id: str) -> list[dict]:
    """List all files ingested under a rag_id with row counts."""
    backend = _backend()

    if backend == "postgres":
        def _run():
            conn = _get_pg_conn()
            cur  = conn.cursor()
            cur.execute(
                """
                SELECT filename,
                       COUNT(*)        AS row_count,
                       MIN(created_at) AS ingested_at
                FROM   rag_documents
                WHERE  session_id = %s
                GROUP  BY filename
                ORDER  BY MIN(created_at)
                """,
                (rag_id,),
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
        return await asyncio.to_thread(_run)

    else:
        sb = _get_sb_client()
        result = await asyncio.to_thread(
            lambda: sb.table("rag_documents")
                .select("filename, created_at")
                .eq("session_id", rag_id)
                .execute()
        )
        rows = result.data or []
        counts: dict[str, dict] = {}
        for r in rows:
            fname = r["filename"]
            if fname not in counts:
                counts[fname] = {"filename": fname, "row_count": 0, "ingested_at": r["created_at"]}
            counts[fname]["row_count"] += 1
        return list(counts.values())


async def delete_rag_file(rag_id: str, filename: str, progress_callback=None) -> int:
    """Delete all rows for a specific file within a rag_id. Returns deleted count."""
    backend = _backend()

    if backend == "postgres":
        def _run():
            conn = _get_pg_conn()
            cur  = conn.cursor()
            try:
                cur.execute(
                    "DELETE FROM rag_documents WHERE session_id = %s AND filename = %s",
                    (rag_id, filename),
                )
                count = cur.rowcount
                conn.commit()
                return count
            except Exception:
                conn.rollback()
                raise
            finally:
                cur.close()
        count = await asyncio.to_thread(_run)

    else:
        count = await _sb_batch_delete(
            {"session_id": rag_id, "filename": filename},
            progress_callback=progress_callback,
        )

    logger.info("RAG: deleted %d rows for file=%s rag_id=%s", count, filename, rag_id)
    invalidate_metadata_cache(rag_id)
    from app.services.fund_forecast_service import invalidate_analytics_cache
    invalidate_analytics_cache(rag_id)
    return count


# Cache: (rag_id, limit) → (rows, fetch_time). Avoids re-fetching the same
# dataset on every forecast call within the TTL window.
_metadata_cache: dict[tuple, tuple[list, float]] = {}
_METADATA_CACHE_TTL = 300  # seconds — 5 minutes


def invalidate_metadata_cache(rag_id: str) -> None:
    """Drop all cache entries for a rag_id (call after ingest/delete)."""
    keys = [k for k in _metadata_cache if k[0] == rag_id]
    for k in keys:
        del _metadata_cache[k]


async def fetch_all_metadata(rag_id: str, limit: int = 16000) -> list[dict]:
    """Return raw metadata dicts for every row stored under rag_id.

    Used by non-RAG consumers (e.g. TimesFM) that need the full tabular data
    rather than a vector search result.  Capped at `limit` rows so callers
    stay within model context windows.

    Results are cached in-memory for _METADATA_CACHE_TTL seconds so repeated
    forecast calls against the same rag_id skip the database round-trip.
    """
    import time

    key = (rag_id, limit)
    cached = _metadata_cache.get(key)
    if cached:
        rows, ts = cached
        if time.time() - ts < _METADATA_CACHE_TTL:
            logger.info("fetch_all_metadata: cache hit rag_id=%s (%d rows)", rag_id, len(rows))
            return rows

    backend = _backend()

    if backend == "postgres":
        def _run():
            conn = _get_pg_conn()
            cur  = conn.cursor()
            cur.execute(
                """
                SELECT metadata
                FROM   rag_documents
                WHERE  session_id = %s
                ORDER  BY id
                LIMIT  %s
                """,
                (rag_id, limit),
            )
            rows = []
            for (meta,) in cur.fetchall():
                if isinstance(meta, str):
                    import json as _json
                    meta = _json.loads(meta)
                rows.append(meta or {})
            return rows
        rows = await asyncio.to_thread(_run)

    else:
        sb = _get_sb_client()
        result = await asyncio.to_thread(
            lambda: sb.table("rag_documents")
                .select("metadata")
                .eq("session_id", rag_id)
                .limit(limit)
                .execute()
        )
        rows = []
        for r in (result.data or []):
            meta = r.get("metadata") or {}
            if isinstance(meta, str):
                import json as _json
                meta = _json.loads(meta)
            rows.append(meta)

    _metadata_cache[key] = (rows, time.time())
    return rows


async def list_all_rags() -> list[dict]:
    """List every rag_id with its files and total row count."""
    backend = _backend()

    if backend == "postgres":
        def _run():
            conn = _get_pg_conn()
            cur  = conn.cursor()
            cur.execute(
                """
                SELECT session_id          AS rag_id,
                       COUNT(DISTINCT filename) AS file_count,
                       COUNT(*)                 AS total_rows,
                       MIN(created_at)          AS created_at,
                       MAX(created_at)          AS last_updated,
                       array_agg(DISTINCT filename) AS filenames
                FROM   rag_documents
                GROUP  BY session_id
                ORDER  BY MIN(created_at) DESC
                """
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
        return await asyncio.to_thread(_run)

    else:
        # Try the fast server-side RPC first (requires list_all_rag_sessions() SQL function).
        # If the function doesn't exist yet, fall back to paginated client-side aggregation.
        try:
            result = await asyncio.to_thread(
                lambda: _get_sb_client().rpc("list_all_rag_sessions", {}).execute()
            )
            return result.data or []
        except Exception as rpc_err:
            if "PGRST202" not in str(rpc_err):
                raise
            logger.info("list_all_rag_sessions RPC not found — falling back to paginated aggregation")

        # Fallback: page through all rows in chunks of 1000 so every RAG is included
        PAGE = 1000
        rags: dict[str, dict] = {}
        offset = 0
        while True:
            def _fetch(o=offset):
                sb = _get_sb_client()
                return (
                    sb.table("rag_documents")
                    .select("session_id, filename, created_at")
                    .range(o, o + PAGE - 1)
                    .execute()
                    .data or []
                )
            rows = await asyncio.to_thread(_fetch)
            if not rows:
                break
            for r in rows:
                rid = r["session_id"]
                if rid not in rags:
                    rags[rid] = {
                        "rag_id":       rid,
                        "filenames":    set(),
                        "total_rows":   0,
                        "created_at":   r["created_at"],
                        "last_updated": r["created_at"],
                    }
                rags[rid]["filenames"].add(r["filename"])
                rags[rid]["total_rows"] += 1
                if r["created_at"] > rags[rid]["last_updated"]:
                    rags[rid]["last_updated"] = r["created_at"]
            if len(rows) < PAGE:
                break
            offset += PAGE
        return [
            {**s, "filenames": list(s["filenames"]), "file_count": len(s["filenames"])}
            for s in rags.values()
        ]
