"""Deterministic primitives. No Math.random / no wall-clock in logic paths.

All randomness derives from an explicit seed via stdlib `random.Random`, and all
hashing uses a stable digest (not Python's salted hash()) so results are stable
across processes and machines.
"""
from __future__ import annotations

import hashlib
import random
import subprocess
from typing import Iterable


def stable_hash(text: str) -> int:
    """Process-stable 64-bit hash (Python's builtin hash() is salted per-process)."""
    d = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(d, "big")


def rng(*parts: object) -> random.Random:
    """Return a seeded RNG bound to a namespace so independent draws don't
    interfere. e.g. rng(SEED, "catalog", size) is reproducible and isolated."""
    key = "|".join(str(p) for p in parts)
    return random.Random(stable_hash(key))


def git_sha(default: str = "unknown") -> str:
    """Best-effort current commit sha; falls back gracefully outside a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return default


def tokenize(text: str) -> list[str]:
    """Lowercase alnum tokenizer shared by embeddings and lexical scoring."""
    tok, cur = [], []
    for ch in text.lower():
        if ch.isalnum():
            cur.append(ch)
        elif cur:
            tok.append("".join(cur))
            cur = []
    if cur:
        tok.append("".join(cur))
    return tok


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0
