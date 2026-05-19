# memory-mcp

Knowledge-graph MCP server backed by Postgres + pgvector (vchord HNSW), exposing entity / observation / relation tools to any MCP-compatible client.

Built with [FastMCP](https://github.com/jlowin/fastmcp). Designed to be deployed as a backend behind an MCP aggregating gateway (e.g. mcp-gateway under the [Kuadrant MCP CRDs](https://kuadrant.io)), so multiple agent runtimes (LangGraph, Claude Code, Open WebUI, HolmesGPT) can share a single memory substrate.

## Concepts

| Object       | Description |
|--------------|-------------|
| `entity`     | A node in the graph (`name` UNIQUE, `type`, optional `namespace`). E.g. `("beast", "host", "infra")`. |
| `observation`| A free-text fact attached to an entity. Embedded for semantic search. Soft-deletable. |
| `relation`   | A typed directed edge between two entities. E.g. `("beast", "located_at", "office")`. |
| `source`     | Provenance JSONB attached to every write — `{agent, claude_namespace?, session_id?, at}`. Server fills `at` if omitted. |

Soft delete is intentional: observations get `deleted_at` set, never DROPped. This matches the design rule of "don't delete memories silently across namespaces."

## Tools (prefix `memory_` when fronted by an aggregating gateway)

| Tool | Purpose |
|------|---------|
| `create_entity` | Create a node + initial observations. |
| `add_observation` | Append an observation to an existing entity. |
| `get_entity` | Fetch by name, optionally expand one-hop relations. |
| `list_entities` | Enumerate (paged), filterable by type / namespace. |
| `search` | Keyword / semantic / hybrid search over observations. Hybrid combines ILIKE matching with vchord HNSW cosine distance. |
| `create_relation` | Create a typed edge between two entities. |
| `link` | Convenience alias for `create_relation`. |
| `delete_observation` | Soft-delete (sets `deleted_at`). |

## Configuration

Environment variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `DATABASE_URL` | — (required) | Postgres connection URI to a database with the `kg` schema applied (see [Schema](#schema)). |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint for embeddings. |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model. Must produce 768-dim vectors. |
| `EMBED_TIMEOUT` | `30` | Seconds. |
| `POOL_MIN_SIZE` | `1` | asyncpg pool min. |
| `POOL_MAX_SIZE` | `5` | asyncpg pool max. |
| `HOST` | `0.0.0.0` | Streamable HTTP bind host. |
| `PORT` | `8070` | Streamable HTTP bind port. |

## Schema

The server **does not** apply schema — it verifies on startup and fails fast if the `kg.entities` / `kg.observations` / `kg.relations` tables are missing. Apply schema via the bundled `schema.sql` (or an out-of-band migration tool) before the server starts.

```sql
CREATE SCHEMA IF NOT EXISTS kg;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS vchord;

-- See sql/schema.sql for full DDL.
```

Embeddings use `vector(768)`. Semantic search uses the `vchordrq` index with `vector_cosine_ops` (`<=>` operator).

## Transports

```bash
# Streamable HTTP (production / k8s)
DATABASE_URL=postgres://... memory-mcp-web

# stdio (local dev / Claude Desktop)
DATABASE_URL=postgres://... memory-mcp
```

## License

MIT.
