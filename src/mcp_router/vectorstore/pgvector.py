"""pgvector-backed VectorIndex (optional, real-mode).

Mirrors the offline MemoryIndex semantics: exact cosine similarity with ties
broken by item_id ascending, so recall@k ordering matches the stdlib default
path exactly. The score returned is (1 - cosine_distance), i.e. cosine
similarity, identical to what MemoryIndex reports.

All heavy deps (psycopg, pgvector) are imported inside __init__ so that merely
importing this module never breaks the pure-stdlib default path.
"""
from __future__ import annotations

import os
from typing import List, Tuple

from .base import VectorIndex


class PgVectorIndex(VectorIndex):
    """Vector index backed by Postgres + the pgvector extension.

    Uses an HNSW index with vector_cosine_ops and orders by the cosine
    distance operator (<=>). Ties are broken by item_id ascending to match
    MemoryIndex, keeping recall@k reproducible across backends.
    """

    def __init__(self, dim: int = 512):
        # Imported here so importing the module never requires psycopg/pgvector.
        import psycopg
        from pgvector.psycopg import register_vector

        self.dim = dim

        database_url = os.environ.get("DATABASE_URL")
        if not database_url:
            raise RuntimeError(
                "DATABASE_URL environment variable is required for PgVectorIndex"
            )

        self._conn = psycopg.connect(database_url, autocommit=True)

        # Ensure the extension exists before registering the vector type.
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self._conn)

        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS mcp_router_tool_embedding (
                item_id INT PRIMARY KEY,
                embedding vector({dim})
            )
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS mcp_router_tool_embedding_hnsw
            ON mcp_router_tool_embedding
            USING hnsw (embedding vector_cosine_ops)
            """
        )

    def add(self, item_id: int, vector: List[float]) -> None:
        self._conn.execute(
            """
            INSERT INTO mcp_router_tool_embedding (item_id, embedding)
            VALUES (%s, %s)
            ON CONFLICT (item_id) DO UPDATE SET embedding = EXCLUDED.embedding
            """,
            (item_id, vector),
        )

    def search(self, vector: List[float], k: int,
               allowed: set[int] | None = None) -> List[Tuple[int, float]]:
        # ORDER BY distance asc == cosine similarity desc; secondary key item_id
        # asc mirrors MemoryIndex tie-breaking. Score is (1 - distance).
        if allowed is None:
            rows = self._conn.execute(
                """
                SELECT item_id, embedding <=> %s AS distance
                FROM mcp_router_tool_embedding
                ORDER BY distance ASC, item_id ASC
                LIMIT %s
                """,
                (vector, k),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT item_id, embedding <=> %s AS distance
                FROM mcp_router_tool_embedding
                WHERE item_id = ANY(%s)
                ORDER BY distance ASC, item_id ASC
                LIMIT %s
                """,
                (vector, list(allowed), k),
            ).fetchall()

        return [(int(item_id), 1.0 - float(distance)) for item_id, distance in rows]
