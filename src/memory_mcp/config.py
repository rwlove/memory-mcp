"""Configuration for memory-mcp."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Server settings loaded from environment variables."""

    database_url: str = Field(
        alias="DATABASE_URL",
        description="Postgres connection URI. Must point at a database with the `kg` schema applied.",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        alias="OLLAMA_BASE_URL",
    )
    embed_model: str = Field(
        default="nomic-embed-text",
        alias="EMBED_MODEL",
    )
    embed_timeout: int = Field(
        default=30,
        alias="EMBED_TIMEOUT",
    )
    pool_min_size: int = Field(default=1, alias="POOL_MIN_SIZE")
    pool_max_size: int = Field(default=5, alias="POOL_MAX_SIZE")
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8070, alias="PORT")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()
