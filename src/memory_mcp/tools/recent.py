"""Recent-activity tool: most recently created observations."""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastmcp import FastMCP
from pydantic import Field

from memory_mcp.db import get_conn
from memory_mcp.metrics import track_tool

logger = logging.getLogger(__name__)


def register_recent_tools(mcp: FastMCP) -> None:
    """Register recent-activity tools on the MCP server."""

    @mcp.tool()
    @track_tool("recent")
    async def recent(
        limit: Annotated[
            int,
            Field(default=20, ge=1, le=200, description="Max observations to return."),
        ] = 20,
        agent_filter: Annotated[
            str | None,
            Field(
                default=None,
                description="If set, only observations whose source.agent matches.",
            ),
        ] = None,
        since: Annotated[
            str | None,
            Field(
                default=None,
                description="ISO 8601 timestamp. Only observations created at-or-after this time.",
            ),
        ] = None,
        entity_name: Annotated[
            str | None,
            Field(
                default=None,
                description="If set, restrict to observations on this entity.",
            ),
        ] = None,
    ) -> dict[str, Any]:
        """Return the most-recent observations (live only), newest first.

        Useful for "what's been added/changed lately" catch-up queries.
        Combine with `agent_filter` to see one agent's recent writes, or
        `since` to scope to a session.
        """
        clauses = ["o.deleted_at IS NULL"]
        args: list[Any] = []

        if agent_filter:
            args.append(agent_filter)
            clauses.append(f"o.source ->> 'agent' = ${len(args)}")
        if since:
            args.append(since)
            clauses.append(f"o.created_at >= ${len(args)}::timestamptz")
        if entity_name:
            args.append(entity_name)
            clauses.append(f"e.name = ${len(args)}")

        args.append(limit)
        sql = f"""
            SELECT o.id AS observation_id, o.content, o.source, o.created_at,
                   e.id AS entity_id, e.name AS entity_name,
                   e.type AS entity_type, e.namespace
            FROM kg.observations o
            JOIN kg.entities e ON e.id = o.entity_id
            WHERE {' AND '.join(clauses)}
            ORDER BY o.created_at DESC
            LIMIT ${len(args)}
        """
        async with get_conn() as conn:
            rows = await conn.fetch(sql, *args)

        return {
            "success": True,
            "count": len(rows),
            "observations": [dict(r) for r in rows],
        }
