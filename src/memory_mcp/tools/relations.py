"""Relation tools."""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastmcp import FastMCP
from pydantic import Field

from memory_mcp.db import get_conn, normalize_source
from memory_mcp.metrics import track_tool

logger = logging.getLogger(__name__)


async def _create_relation(
    from_name: str, to_name: str, rel_type: str, source: dict[str, Any]
) -> dict[str, Any]:
    async with get_conn() as conn:
        f = await conn.fetchrow(
            "SELECT id FROM kg.entities WHERE name = $1", from_name
        )
        t = await conn.fetchrow(
            "SELECT id FROM kg.entities WHERE name = $1", to_name
        )
        if f is None:
            return {"success": False, "error": f"entity '{from_name}' not found"}
        if t is None:
            return {"success": False, "error": f"entity '{to_name}' not found"}

        try:
            row = await conn.fetchrow(
                """
                INSERT INTO kg.relations (from_entity, to_entity, type, source)
                VALUES ($1, $2, $3, $4)
                RETURNING id, created_at
                """,
                f["id"],
                t["id"],
                rel_type,
                source,
            )
        except Exception as exc:
            # UNIQUE (from_entity, to_entity, type) violation lands here.
            return {"success": False, "error": str(exc)}

    return {
        "success": True,
        "relation_id": row["id"],
        "from": from_name,
        "to": to_name,
        "type": rel_type,
    }


def register_relation_tools(mcp: FastMCP) -> None:
    """Register relation tools on the MCP server."""

    @mcp.tool()
    @track_tool("create_relation")
    async def create_relation(
        from_entity: Annotated[
            str, Field(description="Source entity name.")
        ],
        to_entity: Annotated[
            str, Field(description="Target entity name.")
        ],
        type: Annotated[
            str,
            Field(description="Edge type (e.g. 'located_at', 'depends_on', 'owns')."),
        ],
        source: Annotated[
            dict[str, Any] | None,
            Field(default=None, description="Provenance metadata."),
        ] = None,
    ) -> dict[str, Any]:
        """Create a typed directed edge between two existing entities.

        Errors if either entity is missing or if the same (from, to, type)
        triple already exists.
        """
        return await _create_relation(
            from_entity, to_entity, type, normalize_source(source)
        )

    @mcp.tool()
    @track_tool("link")
    async def link(
        from_entity: Annotated[
            str, Field(description="Source entity name.")
        ],
        to_entity: Annotated[
            str, Field(description="Target entity name.")
        ],
        type: Annotated[
            str,
            Field(
                default="related_to",
                description="Edge type. Defaults to 'related_to' for [[name]]-style linking.",
            ),
        ] = "related_to",
        source: Annotated[
            dict[str, Any] | None,
            Field(default=None, description="Provenance metadata."),
        ] = None,
    ) -> dict[str, Any]:
        """Sugar for `create_relation` with a default type.

        Designed for the `[[name]]`-style cross-linking pattern common in
        markdown memory systems.
        """
        return await _create_relation(
            from_entity, to_entity, type, normalize_source(source)
        )
