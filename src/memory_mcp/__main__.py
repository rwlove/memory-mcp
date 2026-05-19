"""Entry points for memory-mcp.

Provides `main()` (stdio) and `main_web()` (streamable-http) transports.
"""

from __future__ import annotations

import asyncio
import logging
import sys

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )


async def _run_stdio() -> None:
    from memory_mcp.db import close_pool, init_pool
    from memory_mcp.server import create_server

    await init_pool()
    mcp = create_server()
    try:
        await mcp.run_async(transport="stdio")
    finally:
        await close_pool()


async def _run_web() -> None:
    from memory_mcp.config import get_settings
    from memory_mcp.db import close_pool, init_pool
    from memory_mcp.server import create_server

    settings = get_settings()
    await init_pool()
    mcp = create_server()
    try:
        await mcp.run_async(
            transport="streamable-http",
            host=settings.host,
            port=settings.port,
        )
    finally:
        await close_pool()


def main() -> None:
    """stdio transport entry point."""
    _setup_logging()
    logger.info("Starting memory-mcp (stdio)")
    asyncio.run(_run_stdio())


def main_web() -> None:
    """Streamable HTTP transport entry point."""
    _setup_logging()
    from memory_mcp.config import get_settings

    settings = get_settings()
    logger.info(
        "Starting memory-mcp (streamable-http) on %s:%s",
        settings.host,
        settings.port,
    )
    asyncio.run(_run_web())


if __name__ == "__main__":
    main()
