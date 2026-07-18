-- Initialize the pgvector extension for the mcp_router benchmark database.
-- Runs automatically on first container start via docker-entrypoint-initdb.d.
CREATE EXTENSION IF NOT EXISTS vector;
