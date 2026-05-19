"""Postgres pool, schema verification, and source-field utilities."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator

import asyncpg

from memory_mcp.config import get_settings

logger = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None


async def _on_connect(conn: asyncpg.Connection) -> None:
    """Per-connection setup. Registers JSONB codec so `source` reads back as dict."""
    await conn.set_type_codec(
        "jsonb",
        encoder=json.dumps,
        decoder=json.loads,
        schema="pg_catalog",
    )


async def init_pool() -> asyncpg.Pool:
    """Create the connection pool and verify the kg schema. Idempotent."""
    global _pool
    if _pool is not None:
        return _pool

    settings = get_settings()
    _pool = await asyncpg.create_pool(
        settings.database_url,
        min_size=settings.pool_min_size,
        max_size=settings.pool_max_size,
        init=_on_connect,
    )
    await _verify_schema(_pool)
    logger.info("Postgres pool initialised (max=%d)", settings.pool_max_size)
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


@asynccontextmanager
async def get_conn() -> AsyncIterator[asyncpg.Connection]:
    pool = await init_pool()
    async with pool.acquire() as conn:
        yield conn


async def _verify_schema(pool: asyncpg.Pool) -> None:
    """Fail fast if kg tables are missing. Schema is applied out-of-band."""
    async with pool.acquire() as conn:
        for table in ("entities", "observations", "relations"):
            row = await conn.fetchrow(
                """
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'kg' AND table_name = $1
                """,
                table,
            )
            if row is None:
                raise RuntimeError(
                    f"kg.{table} missing - apply schema.sql before starting the server"
                )


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_source(source: dict[str, Any] | None) -> dict[str, Any]:
    """Stamp caller-provided source with a UTC timestamp if absent.

    The server never overwrites caller-provided `at`. Always provides one
    if the caller omitted it.
    """
    src = dict(source or {})
    src.setdefault("at", now_iso())
    return src


def encode_vector(values: list[float]) -> str:
    """Encode a vector for asyncpg.

    asyncpg doesn't know the pgvector type natively. The `vector` type's
    text input format is `'[v1,v2,...]'`, which Postgres accepts as a
    text literal that the type's input function parses.
    """
    return "[" + ",".join(repr(float(v)) for v in values) + "]"
