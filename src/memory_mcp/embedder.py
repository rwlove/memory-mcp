"""Ollama embedding client.

Embeddings are stored on a best-effort basis. If Ollama is unreachable or
returns an error, the observation lands with `embedding=NULL` and search
falls back to keyword mode for that row.
"""

from __future__ import annotations

import logging

import httpx

from memory_mcp.config import get_settings

logger = logging.getLogger(__name__)


async def embed(content: str) -> list[float] | None:
    """Return an embedding for `content`, or None on failure."""
    if not content.strip():
        return None

    settings = get_settings()
    url = f"{settings.ollama_base_url.rstrip('/')}/api/embeddings"
    try:
        async with httpx.AsyncClient(timeout=settings.embed_timeout) as client:
            r = await client.post(
                url,
                json={"model": settings.embed_model, "prompt": content},
            )
            r.raise_for_status()
            data = r.json()
            embedding = data.get("embedding")
            if not embedding:
                logger.warning("Ollama returned empty embedding for prompt")
                return None
            return embedding
    except Exception:
        logger.exception("embed failed; storing NULL")
        return None
