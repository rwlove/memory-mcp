"""Search tools."""

from __future__ import annotations

import logging
from typing import Annotated, Any, Literal

from fastmcp import FastMCP
from pydantic import Field

from memory_mcp.db import encode_vector, get_conn
from memory_mcp.embedder import embed

logger = logging.getLogger(__name__)


SearchMode = Literal["keyword", "semantic", "hybrid"]


async def _keyword(
    query: str,
    type_filter: str | None,
    namespace_filter: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    args: list[Any] = [f"%{query}%"]
    clauses = ["o.content ILIKE $1", "o.deleted_at IS NULL"]
    if type_filter:
        args.append(type_filter)
        clauses.append(f"e.type = ${len(args)}")
    if namespace_filter:
        args.append(namespace_filter)
        clauses.append(f"e.namespace = ${len(args)}")
    args.append(limit)
    sql = f"""
        SELECT o.id AS observation_id, o.content, o.source, o.created_at,
               e.id AS entity_id, e.name AS entity_name, e.type AS entity_type,
               e.namespace
        FROM kg.observations o
        JOIN kg.entities e ON e.id = o.entity_id
        WHERE {' AND '.join(clauses)}
        ORDER BY o.created_at DESC
        LIMIT ${len(args)}
    """
    async with get_conn() as conn:
        rows = await conn.fetch(sql, *args)
    return [dict(r) | {"match": "keyword"} for r in rows]


async def _semantic(
    query: str,
    type_filter: str | None,
    namespace_filter: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    vector = await embed(query)
    if vector is None:
        logger.warning("semantic search requested but embedding unavailable")
        return []

    args: list[Any] = [encode_vector(vector)]
    clauses = ["o.deleted_at IS NULL", "o.embedding IS NOT NULL"]
    if type_filter:
        args.append(type_filter)
        clauses.append(f"e.type = ${len(args)}")
    if namespace_filter:
        args.append(namespace_filter)
        clauses.append(f"e.namespace = ${len(args)}")
    args.append(limit)
    sql = f"""
        SELECT o.id AS observation_id, o.content, o.source, o.created_at,
               e.id AS entity_id, e.name AS entity_name, e.type AS entity_type,
               e.namespace,
               (o.embedding <=> $1::vector) AS distance
        FROM kg.observations o
        JOIN kg.entities e ON e.id = o.entity_id
        WHERE {' AND '.join(clauses)}
        ORDER BY o.embedding <=> $1::vector
        LIMIT ${len(args)}
    """
    async with get_conn() as conn:
        rows = await conn.fetch(sql, *args)
    return [dict(r) | {"match": "semantic"} for r in rows]


def register_search_tools(mcp: FastMCP) -> None:
    """Register the search tool on the MCP server."""

    @mcp.tool()
    async def search(
        query: Annotated[
            str, Field(description="Search query text.")
        ],
        mode: Annotated[
            SearchMode,
            Field(
                default="hybrid",
                description=(
                    "keyword: ILIKE match. "
                    "semantic: vchord HNSW cosine distance over embeddings. "
                    "hybrid: union of both, semantic first (de-duplicated by observation_id)."
                ),
            ),
        ] = "hybrid",
        type_filter: Annotated[
            str | None,
            Field(default=None, description="Restrict to entities of this type."),
        ] = None,
        namespace_filter: Annotated[
            str | None,
            Field(default=None, description="Restrict to entities in this namespace."),
        ] = None,
        limit: Annotated[
            int, Field(default=20, ge=1, le=100, description="Max rows.")
        ] = 20,
    ) -> dict[str, Any]:
        """Find observations matching `query`.

        Use `mode=hybrid` (default) unless you have a specific reason. Hybrid
        runs semantic and keyword in parallel and merges results, so it works
        even when an embedding wasn't generated for a given observation.
        """
        if mode == "keyword":
            results = await _keyword(query, type_filter, namespace_filter, limit)
        elif mode == "semantic":
            results = await _semantic(query, type_filter, namespace_filter, limit)
        else:  # hybrid
            sem = await _semantic(query, type_filter, namespace_filter, limit)
            kw = await _keyword(query, type_filter, namespace_filter, limit)
            seen: set[int] = set()
            results = []
            for row in (*sem, *kw):
                obs_id = row["observation_id"]
                if obs_id in seen:
                    continue
                seen.add(obs_id)
                results.append(row)
                if len(results) >= limit:
                    break

        return {"success": True, "mode": mode, "count": len(results), "results": results}
