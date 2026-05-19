"""Batch entity/observation creation tools.

All batch operations are transactional — partial success is impossible.
If any single entry fails, the whole batch rolls back and no rows persist.
This is intentional: bulk-seeding is a "ship the whole shape or nothing"
operation, and dealing with partial state is worse than re-running.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastmcp import FastMCP
from pydantic import Field

from memory_mcp.db import encode_vector, get_conn, normalize_source
from memory_mcp.embedder import embed
from memory_mcp.metrics import track_tool

logger = logging.getLogger(__name__)


async def _insert_observation_in_tx(
    conn: Any,
    entity_id: int,
    content: str,
    source: dict[str, Any],
) -> int:
    vector = await embed(content)
    if vector is None:
        row = await conn.fetchrow(
            """
            INSERT INTO kg.observations (entity_id, content, embedding, source)
            VALUES ($1, $2, NULL, $3)
            RETURNING id
            """,
            entity_id,
            content,
            source,
        )
    else:
        row = await conn.fetchrow(
            """
            INSERT INTO kg.observations (entity_id, content, embedding, source)
            VALUES ($1, $2, $3::vector, $4)
            RETURNING id
            """,
            entity_id,
            content,
            encode_vector(vector),
            source,
        )
    return row["id"]


def register_batch_tools(mcp: FastMCP) -> None:
    """Register batch operations."""

    @mcp.tool()
    @track_tool("create_entities")
    async def create_entities(
        entities: Annotated[
            list[dict[str, Any]],
            Field(
                description=(
                    "List of entity specs. Each dict supports: name (str, required), "
                    "type (str, required), observations (list[str], optional), "
                    "namespace (str, optional)."
                ),
            ),
        ],
        source: Annotated[
            dict[str, Any] | None,
            Field(
                default=None,
                description="Shared provenance for all entities + their initial observations.",
            ),
        ] = None,
    ) -> dict[str, Any]:
        """Bulk-create entities + their initial observations. Transactional.

        Returns a parallel list of `{name, entity_id, observation_ids}` so
        the caller can correlate. If any single entry fails (duplicate name,
        missing required field, etc.) the whole batch rolls back.
        """
        if not entities:
            return {"success": False, "error": "empty entities list"}

        src = normalize_source(source)
        results: list[dict[str, Any]] = []
        async with get_conn() as conn:
            async with conn.transaction():
                for spec in entities:
                    name = spec.get("name")
                    type_ = spec.get("type")
                    if not name or not type_:
                        raise ValueError(
                            f"entity spec missing required name/type: {spec}"
                        )
                    obs_list = spec.get("observations") or []
                    namespace = spec.get("namespace")

                    row = await conn.fetchrow(
                        """
                        INSERT INTO kg.entities (name, type, namespace, source)
                        VALUES ($1, $2, $3, $4)
                        RETURNING id
                        """,
                        name,
                        type_,
                        namespace,
                        src,
                    )
                    entity_id = row["id"]
                    obs_ids: list[int] = []
                    for content in obs_list:
                        obs_id = await _insert_observation_in_tx(
                            conn, entity_id, content, src
                        )
                        obs_ids.append(obs_id)
                    results.append(
                        {
                            "name": name,
                            "entity_id": entity_id,
                            "observation_ids": obs_ids,
                        }
                    )

        return {"success": True, "count": len(results), "results": results}

    @mcp.tool()
    @track_tool("add_observations")
    async def add_observations(
        items: Annotated[
            list[dict[str, Any]],
            Field(
                description=(
                    "List of {entity_name, content} dicts. Each entity must "
                    "already exist."
                ),
            ),
        ],
        source: Annotated[
            dict[str, Any] | None,
            Field(
                default=None,
                description="Shared provenance for all observations.",
            ),
        ] = None,
    ) -> dict[str, Any]:
        """Bulk-append observations to existing entities. Transactional.

        Returns a parallel list of `{entity_name, observation_id}`. If any
        entity is missing or any insert fails, the whole batch rolls back.
        """
        if not items:
            return {"success": False, "error": "empty items list"}

        src = normalize_source(source)
        results: list[dict[str, Any]] = []
        async with get_conn() as conn:
            async with conn.transaction():
                for item in items:
                    entity_name = item.get("entity_name")
                    content = item.get("content")
                    if not entity_name or not content:
                        raise ValueError(
                            f"item missing entity_name/content: {item}"
                        )
                    entity = await conn.fetchrow(
                        "SELECT id FROM kg.entities WHERE name = $1",
                        entity_name,
                    )
                    if entity is None:
                        raise ValueError(f"entity '{entity_name}' not found")
                    obs_id = await _insert_observation_in_tx(
                        conn, entity["id"], content, src
                    )
                    await conn.execute(
                        "UPDATE kg.entities SET updated_at = now() WHERE id = $1",
                        entity["id"],
                    )
                    results.append(
                        {
                            "entity_name": entity_name,
                            "observation_id": obs_id,
                        }
                    )

        return {"success": True, "count": len(results), "results": results}
