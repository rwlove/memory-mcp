"""Entity + observation tools."""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastmcp import FastMCP
from pydantic import Field

from memory_mcp.db import encode_vector, get_conn, normalize_source
from memory_mcp.embedder import embed

logger = logging.getLogger(__name__)


async def _insert_observation(
    conn: Any,
    entity_id: int,
    content: str,
    source: dict[str, Any],
) -> int:
    """Insert one observation, embedding best-effort. Returns the new id."""
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


def register_entity_tools(mcp: FastMCP) -> None:
    """Register entity/observation tools on the MCP server."""

    @mcp.tool()
    async def create_entity(
        name: Annotated[
            str,
            Field(description="Unique entity name (e.g. 'beast', 'postgres-langgraph-memory')."),
        ],
        type: Annotated[
            str,
            Field(description="Entity type (e.g. 'host', 'cnpg-cluster', 'person')."),
        ],
        observations: Annotated[
            list[str] | None,
            Field(default=None, description="Initial observation strings."),
        ] = None,
        namespace: Annotated[
            str | None,
            Field(default=None, description="Optional grouping (e.g. 'infra')."),
        ] = None,
        source: Annotated[
            dict[str, Any] | None,
            Field(
                default=None,
                description="Provenance metadata. Server fills 'at' if absent. Minimum: {'agent': 'your-agent-name'}.",
            ),
        ] = None,
    ) -> dict[str, Any]:
        """Create an entity with optional initial observations.

        Errors if `name` already exists. Use `add_observation` to extend
        an existing entity.
        """
        src = normalize_source(source)
        async with get_conn() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    INSERT INTO kg.entities (name, type, namespace, source)
                    VALUES ($1, $2, $3, $4)
                    RETURNING id, created_at
                    """,
                    name,
                    type,
                    namespace,
                    src,
                )
                entity_id = row["id"]
                obs_ids: list[int] = []
                for content in observations or []:
                    obs_id = await _insert_observation(conn, entity_id, content, src)
                    obs_ids.append(obs_id)
        return {
            "success": True,
            "entity_id": entity_id,
            "observation_ids": obs_ids,
        }

    @mcp.tool()
    async def add_observation(
        entity_name: Annotated[
            str, Field(description="Name of an existing entity.")
        ],
        content: Annotated[
            str, Field(description="Observation text.")
        ],
        source: Annotated[
            dict[str, Any] | None,
            Field(default=None, description="Provenance metadata."),
        ] = None,
    ) -> dict[str, Any]:
        """Append an observation to an existing entity."""
        src = normalize_source(source)
        async with get_conn() as conn:
            entity = await conn.fetchrow(
                "SELECT id FROM kg.entities WHERE name = $1", entity_name
            )
            if entity is None:
                return {
                    "success": False,
                    "error": f"entity '{entity_name}' not found",
                }
            obs_id = await _insert_observation(conn, entity["id"], content, src)
            await conn.execute(
                "UPDATE kg.entities SET updated_at = now() WHERE id = $1",
                entity["id"],
            )
        return {"success": True, "observation_id": obs_id}

    @mcp.tool()
    async def get_entity(
        name: Annotated[str, Field(description="Entity name.")],
        expand_relations: Annotated[
            bool,
            Field(
                default=False,
                description="If true, include one-hop relations (both directions).",
            ),
        ] = False,
    ) -> dict[str, Any]:
        """Fetch an entity by name, with all live observations and optional one-hop relations."""
        async with get_conn() as conn:
            entity = await conn.fetchrow(
                """
                SELECT id, name, type, namespace, source, created_at, updated_at
                FROM kg.entities WHERE name = $1
                """,
                name,
            )
            if entity is None:
                return {"success": False, "error": f"entity '{name}' not found"}

            obs = await conn.fetch(
                """
                SELECT id, content, source, created_at
                FROM kg.observations
                WHERE entity_id = $1 AND deleted_at IS NULL
                ORDER BY created_at
                """,
                entity["id"],
            )

            relations: list[dict[str, Any]] = []
            if expand_relations:
                rows = await conn.fetch(
                    """
                    SELECT r.id, r.type, r.source, r.created_at,
                           ef.name AS from_name, et.name AS to_name
                    FROM kg.relations r
                    JOIN kg.entities ef ON ef.id = r.from_entity
                    JOIN kg.entities et ON et.id = r.to_entity
                    WHERE r.from_entity = $1 OR r.to_entity = $1
                    ORDER BY r.created_at
                    """,
                    entity["id"],
                )
                relations = [dict(r) for r in rows]

        return {
            "success": True,
            "entity": dict(entity),
            "observations": [dict(o) for o in obs],
            "relations": relations,
        }

    @mcp.tool()
    async def list_entities(
        type_filter: Annotated[
            str | None,
            Field(default=None, description="Restrict to entities of this type."),
        ] = None,
        namespace_filter: Annotated[
            str | None,
            Field(default=None, description="Restrict to entities in this namespace."),
        ] = None,
        limit: Annotated[
            int, Field(default=50, ge=1, le=500, description="Max rows to return.")
        ] = 50,
    ) -> dict[str, Any]:
        """List entities with optional filters. Newest first."""
        clauses: list[str] = []
        args: list[Any] = []
        if type_filter:
            args.append(type_filter)
            clauses.append(f"type = ${len(args)}")
        if namespace_filter:
            args.append(namespace_filter)
            clauses.append(f"namespace = ${len(args)}")

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        args.append(limit)
        sql = f"""
            SELECT id, name, type, namespace, source, created_at, updated_at
            FROM kg.entities
            {where}
            ORDER BY created_at DESC
            LIMIT ${len(args)}
        """
        async with get_conn() as conn:
            rows = await conn.fetch(sql, *args)
        return {
            "success": True,
            "count": len(rows),
            "entities": [dict(r) for r in rows],
        }

    @mcp.tool()
    async def delete_observation(
        observation_id: Annotated[
            int, Field(description="Observation id to soft-delete.")
        ],
    ) -> dict[str, Any]:
        """Soft-delete an observation. Sets `deleted_at = now()`; never DROPs."""
        async with get_conn() as conn:
            row = await conn.fetchrow(
                """
                UPDATE kg.observations
                SET deleted_at = now()
                WHERE id = $1 AND deleted_at IS NULL
                RETURNING id
                """,
                observation_id,
            )
        if row is None:
            return {
                "success": False,
                "error": f"observation {observation_id} not found or already deleted",
            }
        return {"success": True, "observation_id": observation_id}

    @mcp.tool()
    async def update_entity(
        name: Annotated[
            str, Field(description="Current entity name (used to look it up).")
        ],
        new_name: Annotated[
            str | None,
            Field(
                default=None,
                description="New name. Must be unique. Relations re-resolve by id so existing edges survive a rename.",
            ),
        ] = None,
        type: Annotated[
            str | None,
            Field(default=None, description="New entity type. None = no change."),
        ] = None,
        namespace: Annotated[
            str | None,
            Field(
                default=None,
                description="New namespace grouping. Pass an empty string '' to clear (set NULL).",
            ),
        ] = None,
        source: Annotated[
            dict[str, Any] | None,
            Field(default=None, description="Provenance metadata for the update."),
        ] = None,
    ) -> dict[str, Any]:
        """Update mutable fields on an existing entity. Bumps `updated_at`.

        `source` overwrites the entity's source field with the new provenance
        — past provenance is in observation rows. Pass `namespace=""` (empty
        string) to set NULL.
        """
        sets: list[str] = []
        args: list[Any] = []
        if new_name is not None:
            args.append(new_name)
            sets.append(f"name = ${len(args)}")
        if type is not None:
            args.append(type)
            sets.append(f"type = ${len(args)}")
        if namespace is not None:
            args.append(namespace if namespace else None)
            sets.append(f"namespace = ${len(args)}")
        if source is not None:
            args.append(normalize_source(source))
            sets.append(f"source = ${len(args)}")
        if not sets:
            return {"success": False, "error": "no fields to update"}

        sets.append("updated_at = now()")
        args.append(name)
        sql = f"""
            UPDATE kg.entities
            SET {', '.join(sets)}
            WHERE name = ${len(args)}
            RETURNING id, name, type, namespace, updated_at
        """
        async with get_conn() as conn:
            try:
                row = await conn.fetchrow(sql, *args)
            except Exception as exc:
                return {"success": False, "error": str(exc)}
        if row is None:
            return {"success": False, "error": f"entity '{name}' not found"}
        return {"success": True, "entity": dict(row)}

    @mcp.tool()
    async def update_observation(
        observation_id: Annotated[
            int, Field(description="Observation id to update.")
        ],
        content: Annotated[
            str, Field(description="New content. Will be re-embedded.")
        ],
        source: Annotated[
            dict[str, Any] | None,
            Field(default=None, description="Provenance for the update."),
        ] = None,
    ) -> dict[str, Any]:
        """Replace an observation's content and re-embed.

        Errors if the observation is soft-deleted. Use `add_observation` + an
        explicit `delete_observation` if you want history-preserving "the
        fact changed" semantics. This tool is for correcting typos / outdated
        wording in-place.
        """
        src = normalize_source(source)
        vector = await embed(content)
        async with get_conn() as conn:
            if vector is None:
                row = await conn.fetchrow(
                    """
                    UPDATE kg.observations
                    SET content = $1, embedding = NULL, source = $2
                    WHERE id = $3 AND deleted_at IS NULL
                    RETURNING id, entity_id
                    """,
                    content,
                    src,
                    observation_id,
                )
            else:
                row = await conn.fetchrow(
                    """
                    UPDATE kg.observations
                    SET content = $1, embedding = $2::vector, source = $3
                    WHERE id = $4 AND deleted_at IS NULL
                    RETURNING id, entity_id
                    """,
                    content,
                    encode_vector(vector),
                    src,
                    observation_id,
                )
            if row is None:
                return {
                    "success": False,
                    "error": f"observation {observation_id} not found or already deleted",
                }
            await conn.execute(
                "UPDATE kg.entities SET updated_at = now() WHERE id = $1",
                row["entity_id"],
            )
        return {"success": True, "observation_id": observation_id}
