"""Real (online) LLM adapter backed by the Anthropic Messages API tool-use surface.

ClaudeLLM implements the same ``choose_tools`` contract as the offline ``MockLLM``:
given a natural-language query and a list of candidate :class:`~mcp_router.models.Tool`
objects, it returns up to ``n`` chosen tool ids. It does so by exposing each
candidate to Claude as a *tool definition* and asking the model — with
``tool_choice`` forced to ``any`` — which tool(s) it would call for the query. The
tool names Claude selects are mapped back to the original tool ids.

VCR determinism
---------------
Every real API call is expensive, rate-limited, and non-deterministic, which is
poison for a reproducible benchmark. To make repeat runs deterministic *and* free,
this adapter records a simple on-disk "VCR cassette" cache:

* The cache key is ``sha256`` of the tuple that fully determines the request:
  ``(model_id, query, sorted(candidate namespaced_names), n)``. Sorting the
  candidate names makes the key invariant to the order candidates are passed in —
  the same logical question always maps to the same key.
* The value stored on disk is the returned list of tool ids, as JSON.
* On a cache hit the recorded ids are replayed verbatim and no network call is made.
  On a miss the API is queried once and the result is written back.

Because the key is content-addressed, the first run "records" against the live API
and every subsequent run "replays" identically — the benchmark produces the same
numbers offline, with no API key required after the initial recording. Deleting a
cassette file (or the whole ``cassettes/`` directory) forces a re-record.

All heavy/optional imports (``anthropic``) happen *inside* methods so that importing
this module never breaks the pure-stdlib offline default path.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import List

# Model ids (see BenchConfig / task spec). Default real-mode runner is Sonnet 5.
MODEL_ID = "claude-sonnet-5"

# Where recorded cassettes live. Sibling of the package, so it survives across runs
# and is easy to commit for reproducibility.
_CASSETTE_DIR = Path(__file__).resolve().parent.parent / "cassettes"

# Anthropic tool names must match ^[a-zA-Z0-9_-]{1,64}$ — namespaced names like
# "email.send_smtp" contain dots, so they need sanitizing.
_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]")


def _sanitize(name: str) -> str:
    """Coerce a namespaced tool name into a valid Anthropic tool name."""
    cleaned = _NAME_RE.sub("_", name)
    if not cleaned:
        cleaned = "tool"
    return cleaned[:64]


class ClaudeLLM:
    """LLMProvider backed by Claude tool-use, with an on-disk VCR cache.

    Implements ``choose_tools(query, candidates, n) -> list[int]``.
    """

    def __init__(self, model_id: str = MODEL_ID, cassette_dir: Path | str | None = None):
        self.model_id = model_id
        self._cassette_dir = Path(cassette_dir) if cassette_dir is not None else _CASSETTE_DIR
        self._client = None  # lazily constructed on first cache miss

    # -- cassette helpers ---------------------------------------------------

    def _cache_key(self, query: str, candidates: list, n: int) -> str:
        names = sorted(t.namespaced_name for t in candidates)
        payload = json.dumps(
            {"model_id": self.model_id, "query": query, "candidates": names, "n": n},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cassette_path(self, key: str) -> Path:
        return self._cassette_dir / f"{key}.json"

    def _read_cassette(self, key: str) -> List[int] | None:
        path = self._cassette_path(key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        ids = data.get("ids") if isinstance(data, dict) else data
        if not isinstance(ids, list):
            return None
        return [int(i) for i in ids]

    def _write_cassette(self, key: str, query: str, candidates: list, n: int, ids: List[int]) -> None:
        self._cassette_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "model_id": self.model_id,
            "query": query,
            "candidates": sorted(t.namespaced_name for t in candidates),
            "n": n,
            "ids": ids,
        }
        tmp = self._cassette_path(key).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self._cassette_path(key))

    # -- provider surface ---------------------------------------------------

    def choose_tools(self, query: str, candidates: list, n: int) -> List[int]:
        """Return up to ``n`` tool ids Claude would call for ``query``.

        Replays from a cassette when one exists; otherwise queries the API once
        (via tool-use with ``tool_choice=any``) and records the result.
        """
        if not candidates or n <= 0:
            return []

        key = self._cache_key(query, candidates, n)
        cached = self._read_cassette(key)
        if cached is not None:
            # Guard the replay against a changed candidate set: only ids that are
            # still present are returned.
            valid = {t.id for t in candidates}
            return [i for i in cached if i in valid][:n]

        ids = self._query_api(query, candidates, n)
        self._write_cassette(key, query, candidates, n, ids)
        return ids

    # -- API call -----------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            import anthropic  # imported lazily so the stdlib path never needs it

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            # anthropic.Anthropic() also resolves ANTHROPIC_API_KEY / auth profiles
            # itself; passing it explicitly when present keeps behavior obvious.
            self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        return self._client

    def _query_api(self, query: str, candidates: list, n: int) -> List[int]:
        client = self._get_client()

        # Build tool definitions and a sanitized-name -> tool id map. Sanitizing
        # can collide (two names differing only by a dot); disambiguate with a
        # numeric suffix so the reverse map stays exact.
        tools = []
        name_to_id: dict[str, int] = {}
        used: set[str] = set()
        for tool in candidates:
            base = _sanitize(tool.namespaced_name)
            name = base
            suffix = 1
            while name in used:
                tag = f"_{suffix}"
                name = base[: 64 - len(tag)] + tag
                suffix += 1
            used.add(name)
            name_to_id[name] = tool.id
            tools.append(
                {
                    "name": name,
                    "description": tool.description,
                    "input_schema": {"type": "object", "properties": {}},
                }
            )

        system = (
            "You are a tool router for an MCP gateway. Given a user request and a set "
            "of available tools, call the tool (or tools) you would use to satisfy the "
            f"request. Call at most {n} tool(s), most relevant first. Call only tools "
            "that are genuinely relevant."
        )

        response = client.messages.create(
            model=self.model_id,
            max_tokens=1024,
            system=system,
            tools=tools,
            tool_choice={"type": "any"},
            # Forced tool_choice is incompatible with adaptive thinking on this
            # model; disable it so the request is accepted and deterministic.
            thinking={"type": "disabled"},
            messages=[{"role": "user", "content": query}],
        )

        # Collect chosen tool names in order, dedup, map back to ids, cap at n.
        chosen: List[int] = []
        seen: set[int] = set()
        for block in response.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            tool_id = name_to_id.get(block.name)
            if tool_id is None or tool_id in seen:
                continue
            seen.add(tool_id)
            chosen.append(tool_id)
            if len(chosen) >= n:
                break
        return chosen
