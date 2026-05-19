"""Prometheus instrumentation for memory-mcp.

Per-tool counters + histograms + embedder metrics. Exposed via
`/metrics` (registered in server.py). Default process + GC collectors
already cover CPU / memory / fd-count / GC pauses.

Pattern: `@track_tool("tool_name")` wraps the tool function. Counter
labels: `(tool, status)` where status is "success" or "error". Histogram
labeled by tool. Wrap once at registration time; do NOT add per-call
wrappers (would double-count via re-registration).
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any, TypeVar

from prometheus_client import Counter, Histogram

# ----------------------------------------------------------------------
# Metric definitions
# ----------------------------------------------------------------------

TOOL_CALLS_TOTAL = Counter(
    "memory_mcp_tool_calls_total",
    "Total tool calls into memory-mcp.",
    labelnames=("tool", "status"),
)

TOOL_CALL_DURATION_SECONDS = Histogram(
    "memory_mcp_tool_call_duration_seconds",
    "Tool call latency in seconds.",
    labelnames=("tool",),
    # Buckets tuned for "DB-touching tool with optional embed". p99 should
    # land in 1-3s territory unless Ollama is degraded.
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

EMBED_CALLS_TOTAL = Counter(
    "memory_mcp_embed_calls_total",
    "Embedding requests sent to Ollama.",
    labelnames=("status",),  # success | empty | error
)

EMBED_CALL_DURATION_SECONDS = Histogram(
    "memory_mcp_embed_call_duration_seconds",
    "Embedding request latency in seconds.",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)


# ----------------------------------------------------------------------
# Tool wrapper
# ----------------------------------------------------------------------

F = TypeVar("F", bound=Callable[..., Awaitable[Any]])


def track_tool(name: str) -> Callable[[F], F]:
    """Decorator wrapping an async tool with Prometheus counters + histogram.

    Status label:
      - `success`: function returned normally AND the result dict (if any)
        does not have `success: False`. Tool-level "logical failure"
        (e.g. "entity not found") still counts as success here — those
        aren't operational errors, they're well-formed negative responses.
      - `error`: function raised an exception.
    """

    def decorator(fn: F) -> F:
        @wraps(fn)
        async def wrapped(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                result = await fn(*args, **kwargs)
            except Exception:
                TOOL_CALLS_TOTAL.labels(tool=name, status="error").inc()
                TOOL_CALL_DURATION_SECONDS.labels(tool=name).observe(
                    time.perf_counter() - start
                )
                raise
            TOOL_CALLS_TOTAL.labels(tool=name, status="success").inc()
            TOOL_CALL_DURATION_SECONDS.labels(tool=name).observe(
                time.perf_counter() - start
            )
            return result

        return wrapped  # type: ignore[return-value]

    return decorator
