# Minimal image for the mcp-router pgvector/production path.
# The default `make bench` (memory backend) needs NO Docker at all; this
# image exists only for the real providers path (local embed + pgvector).
FROM python:3.12-slim

# Build tools are needed for some optional wheels (e.g. psycopg[binary] fallbacks).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project metadata first so the editable install layer caches well.
COPY pyproject.toml ./
COPY src ./src
COPY Makefile ./

# Install the package with the pgvector + local-embedding extras.
RUN pip install --no-cache-dir -e .[pg,local]

# Default: run the benchmark. Overridden by docker-compose to `make bench`.
CMD ["make", "bench"]
