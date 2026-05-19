-- memory-mcp knowledge-graph schema.
--
-- Apply this once to the target database (the server does NOT apply
-- schema — it verifies on startup and fails fast if missing). All
-- statements are idempotent, so re-running is safe.
--
-- Prerequisite extensions:
--   CREATE EXTENSION IF NOT EXISTS vector;
--   CREATE EXTENSION IF NOT EXISTS vchord;
--
-- The home-ops cluster has these pre-enabled at CNPG bootstrap (see
-- kubernetes/apps/databases/cloudnative-pg/config/langgraph-memory/
-- cluster.yaml postInitApplicationSQL). Mirror lives at
-- kubernetes/apps/mcp-system/memory-mcp/app/resources/schema.sql and
-- MUST stay byte-identical with this file.

CREATE SCHEMA IF NOT EXISTS kg;

CREATE TABLE IF NOT EXISTS kg.entities (
  id          BIGSERIAL PRIMARY KEY,
  name        TEXT NOT NULL UNIQUE,
  type        TEXT NOT NULL,
  namespace   TEXT,
  source      JSONB NOT NULL,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS kg.observations (
  id          BIGSERIAL PRIMARY KEY,
  entity_id   BIGINT NOT NULL REFERENCES kg.entities(id) ON DELETE CASCADE,
  content     TEXT NOT NULL,
  embedding   vector(768),
  source      JSONB NOT NULL,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  deleted_at  TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS kg.relations (
  id           BIGSERIAL PRIMARY KEY,
  from_entity  BIGINT NOT NULL REFERENCES kg.entities(id) ON DELETE CASCADE,
  to_entity    BIGINT NOT NULL REFERENCES kg.entities(id) ON DELETE CASCADE,
  type         TEXT NOT NULL,
  source       JSONB NOT NULL,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (from_entity, to_entity, type)
);

CREATE INDEX IF NOT EXISTS kg_obs_entity_idx
  ON kg.observations(entity_id)
  WHERE deleted_at IS NULL;

CREATE INDEX IF NOT EXISTS kg_ent_type_idx ON kg.entities(type);
CREATE INDEX IF NOT EXISTS kg_ent_namespace_idx ON kg.entities(namespace);

CREATE INDEX IF NOT EXISTS kg_obs_embedding_hnsw
  ON kg.observations
  USING vchordrq (embedding vector_cosine_ops);
