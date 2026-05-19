"""Graph-walk tool: BFS traversal from a start entity."""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastmcp import FastMCP
from pydantic import Field

from memory_mcp.db import get_conn
from memory_mcp.metrics import track_tool

logger = logging.getLogger(__name__)


def register_graph_tools(mcp: FastMCP) -> None:
    """Register graph-walk tools on the MCP server."""

    @mcp.tool()
    @track_tool("graph_walk")
    async def graph_walk(
        start: Annotated[
            str, Field(description="Entity name to start the walk from.")
        ],
        depth: Annotated[
            int,
            Field(
                default=2,
                ge=1,
                le=4,
                description="Maximum hops. 1 = immediate neighbors. Capped at 4 to bound DB load.",
            ),
        ] = 2,
        relation_type_filter: Annotated[
            list[str] | None,
            Field(
                default=None,
                description="If set, only follow edges of these relation types.",
            ),
        ] = None,
        include_observations: Annotated[
            bool,
            Field(
                default=False,
                description="If true, include each entity's live observations in the result. Verbose.",
            ),
        ] = False,
    ) -> dict[str, Any]:
        """BFS from `start` up to `depth` hops, return reachable entities + edges.

        Returns:
        - entities: list of entity dicts in BFS order (start first)
        - relations: list of edges traversed
        - frontier_hit: True if the walk stopped at the depth cap instead
          of exhausting the connected component (signal that more exists
          out there if you increase depth).
        """
        async with get_conn() as conn:
            start_row = await conn.fetchrow(
                """
                SELECT id, name, type, namespace, source, created_at, updated_at
                FROM kg.entities WHERE name = $1
                """,
                start,
            )
            if start_row is None:
                return {"success": False, "error": f"entity '{start}' not found"}

            visited_ids: set[int] = {start_row["id"]}
            entities: list[dict[str, Any]] = [dict(start_row)]
            relations: list[dict[str, Any]] = []
            current_layer: list[int] = [start_row["id"]]
            frontier_hit = False

            for _hop in range(depth):
                if not current_layer:
                    break

                params: list[Any] = [current_layer]
                rel_filter_sql = ""
                if relation_type_filter:
                    params.append(relation_type_filter)
                    rel_filter_sql = f"AND r.type = ANY(${len(params)})"

                rows = await conn.fetch(
                    f"""
                    SELECT r.id, r.from_entity, r.to_entity, r.type,
                           r.source, r.created_at,
                           ef.name AS from_name, et.name AS to_name
                    FROM kg.relations r
                    JOIN kg.entities ef ON ef.id = r.from_entity
                    JOIN kg.entities et ON et.id = r.to_entity
                    WHERE (r.from_entity = ANY($1) OR r.to_entity = ANY($1))
                    {rel_filter_sql}
                    """,
                    *params,
                )

                next_layer: list[int] = []
                for r in rows:
                    rel = dict(r)
                    relations.append(rel)
                    for nid in (r["from_entity"], r["to_entity"]):
                        if nid not in visited_ids:
                            visited_ids.add(nid)
                            next_layer.append(nid)

                if next_layer:
                    ent_rows = await conn.fetch(
                        """
                        SELECT id, name, type, namespace, source,
                               created_at, updated_at
                        FROM kg.entities WHERE id = ANY($1)
                        ORDER BY name
                        """,
                        next_layer,
                    )
                    entities.extend(dict(e) for e in ent_rows)

                current_layer = next_layer

            # If anything still connects to current_layer outside visited_ids,
            # the walk would have continued.
            if current_layer:
                more = await conn.fetchrow(
                    """
                    SELECT 1
                    FROM kg.relations r
                    WHERE (r.from_entity = ANY($1) OR r.to_entity = ANY($1))
                      AND r.from_entity != ALL($2)
                      AND r.to_entity != ALL($2)
                    LIMIT 1
                    """,
                    current_layer,
                    list(visited_ids),
                )
                frontier_hit = more is not None

            if include_observations and entities:
                obs_rows = await conn.fetch(
                    """
                    SELECT id, entity_id, content, source, created_at
                    FROM kg.observations
                    WHERE entity_id = ANY($1) AND deleted_at IS NULL
                    ORDER BY entity_id, created_at
                    """,
                    [e["id"] for e in entities],
                )
                by_entity: dict[int, list[dict[str, Any]]] = {}
                for o in obs_rows:
                    by_entity.setdefault(o["entity_id"], []).append(dict(o))
                for e in entities:
                    e["observations"] = by_entity.get(e["id"], [])

        return {
            "success": True,
            "start": start,
            "depth": depth,
            "entity_count": len(entities),
            "relation_count": len(relations),
            "frontier_hit": frontier_hit,
            "entities": entities,
            "relations": relations,
        }
