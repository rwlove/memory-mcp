"""memory-mcp - core server factory."""

from __future__ import annotations

import logging

import httpx
from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from memory_mcp import __version__
from memory_mcp.config import get_settings
from memory_mcp.db import get_conn

logger = logging.getLogger(__name__)

SERVER_INSTRUCTIONS = """
You are connected to a knowledge-graph memory server.

The graph is shared across agents (LangGraph, Claude Code, Open WebUI,
HolmesGPT, etc.). Writes carry provenance via the `source` parameter -
always provide at minimum `{"agent": "<your-agent-name>"}` and any
session/namespace context you have.

Use `search` (mode=hybrid) for general "what do we know about X" queries.
Use `get_entity` when you have a specific name. Use `list_entities` with
type or namespace filters to enumerate.

Soft delete only: `delete_observation` sets `deleted_at`, it never DROPs
rows. Do not assume something has been hard-deleted.
"""


def create_server() -> FastMCP:
    """Build a configured FastMCP server with all tools registered."""
    settings = get_settings()

    mcp = FastMCP(
        name="memory-mcp",
        version=__version__,
        instructions=SERVER_INSTRUCTIONS,
    )

    from memory_mcp.tools.entities import register_entity_tools
    from memory_mcp.tools.relations import register_relation_tools
    from memory_mcp.tools.search import register_search_tools

    register_entity_tools(mcp)
    register_relation_tools(mcp)
    register_search_tools(mcp)

    _register_health_routes(mcp)

    logger.info(
        "memory-mcp initialised (embed=%s @ %s)",
        settings.embed_model,
        settings.ollama_base_url,
    )
    return mcp


def _register_health_routes(mcp: FastMCP) -> None:
    """Register /healthz and /readyz HTTP probes alongside the MCP endpoint.

    /healthz — process is up and the Postgres pool can serve a SELECT 1.
    /readyz  — /healthz plus Ollama responds to /api/tags (so embeddings
               aren't going to silently 404 forever). Returns 503 on either
               failure; readiness is the gate the kubelet uses to drop us
               out of the Service while Ollama recovers.
    """

    @mcp.custom_route("/healthz", methods=["GET"])
    async def healthz(_: Request) -> JSONResponse:
        try:
            async with get_conn() as conn:
                await conn.fetchval("SELECT 1")
        except Exception as exc:
            logger.exception("healthz: db ping failed")
            return JSONResponse(
                {"status": "error", "component": "db", "detail": str(exc)},
                status_code=503,
            )
        return JSONResponse({"status": "ok"})

    @mcp.custom_route("/readyz", methods=["GET"])
    async def readyz(_: Request) -> JSONResponse:
        settings = get_settings()
        try:
            async with get_conn() as conn:
                await conn.fetchval("SELECT 1")
        except Exception as exc:
            logger.exception("readyz: db ping failed")
            return JSONResponse(
                {"status": "error", "component": "db", "detail": str(exc)},
                status_code=503,
            )

        ollama_url = f"{settings.ollama_base_url.rstrip('/')}/api/tags"
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                r = await client.get(ollama_url)
                r.raise_for_status()
        except Exception as exc:
            logger.warning("readyz: ollama probe failed: %s", exc)
            return JSONResponse(
                {"status": "error", "component": "ollama", "detail": str(exc)},
                status_code=503,
            )

        return JSONResponse({"status": "ok"})
