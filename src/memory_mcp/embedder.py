"""Ollama embedding client.

Embeddings are stored on a best-effort basis. If Ollama is unreachable or
returns an error, the observation lands with `embedding=NULL` and search
falls back to keyword mode for that row.
"""

from __future__ import annotations

import logging
import time

import httpx

from memory_mcp.config import get_settings
from memory_mcp.metrics import EMBED_CALL_DURATION_SECONDS, EMBED_CALLS_TOTAL

logger = logging.getLogger(__name__)


async def embed(content: str) -> list[float] | None:
    """Return an embedding for `content`, or None on failure.

    Uses the modern Ollama `/api/embed` endpoint (the legacy
    `/api/embeddings` was removed in recent Ollama releases). Response
    shape: `{"embeddings": [[...]]}` (plural, one row per input).
    """
    if not content.strip():
        return None

    settings = get_settings()
    url = f"{settings.ollama_base_url.rstrip('/')}/api/embed"
    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=settings.embed_timeout) as client:
            r = await client.post(
                url,
                json={"model": settings.embed_model, "input": content},
            )
            r.raise_for_status()
            data = r.json()
            rows = data.get("embeddings") or []
            if not rows or not rows[0]:
                logger.warning("Ollama returned no embeddings for input")
                EMBED_CALLS_TOTAL.labels(status="empty").inc()
                EMBED_CALL_DURATION_SECONDS.observe(time.perf_counter() - start)
                return None
            EMBED_CALLS_TOTAL.labels(status="success").inc()
            EMBED_CALL_DURATION_SECONDS.observe(time.perf_counter() - start)
            return rows[0]
    except Exception:
        logger.exception("embed failed; storing NULL")
        EMBED_CALLS_TOTAL.labels(status="error").inc()
        EMBED_CALL_DURATION_SECONDS.observe(time.perf_counter() - start)
        return None
