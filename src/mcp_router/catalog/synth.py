"""Deterministic synthetic catalog with a *nested* staircase of near-duplicate
distractors — the mechanism that produces a genuine recall cliff.

Design (and its honest limits):
  * catalog(100) ⊂ catalog(200) ⊂ catalog(300): master list fixed, catalog(N) =
    master[:N]. Same gold tools at every size; only the distractor pool grows.
  * A query for gold g shares `core_share` topic tokens + a rare keyword with g.
    Each distractor shares Xd ~ Uniform{4..12} of the query tokens. Under pure
    semantic (bag-of-words) similarity a distractor out-ranks gold iff it shares
    more, so as the distractor pool grows recall@k collapses. This slope depends
    on `core_share` and the embedding geometry — hence `bench sweep` varies
    core_share and `--embed mock_char` re-checks under a non-BoW geometry.
  * Keyword collisions: a `kw_collision_ratio` fraction of distractors carry the
    gold's rare keyword too. For those the lexical signal no longer uniquely
    identifies gold, so the hybrid router can genuinely lose *recall* (e.g. at k=1
    hierarchical can beat hybrid). The routing McNemar is therefore computed on
    recall_hit — not the degenerate "hybrid always ≥ semantic" construction.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

from ..config import DEFAULT
from ..determinism import rng, tokenize
from ..models import Catalog, Query, Tool, ToolGroup

GROUPS: List[Tuple[str, str, str]] = [
    ("email", "email", "message"), ("file", "file", "document"),
    ("calendar", "calendar", "event"), ("database", "database", "record"),
    ("search", "search", "result"), ("http", "http", "request"),
    ("payment", "payment", "transaction"), ("chat", "chat", "message"),
    ("storage", "storage", "object"), ("auth", "auth", "credential"),
]
ACTIONS = ["send", "read", "delete", "list", "update", "create"]
N_BASE = len(GROUPS) * len(ACTIONS)              # 60
HOT_TARGETS = list(range(30))                    # queried + crowded golds

_TOPIC_BANK: Dict[str, List[str]] = {
    "email": ["inbox", "recipient", "subject", "attachment", "draft", "thread", "reply", "forward", "signature", "spam", "cc", "bcc", "mailbox", "compose", "deliver", "bounce"],
    "file": ["folder", "path", "directory", "upload", "download", "rename", "copy", "move", "permission", "metadata", "chunk", "binary", "archive", "extension", "checksum", "symlink"],
    "calendar": ["meeting", "invite", "reminder", "schedule", "slot", "recurrence", "attendee", "timezone", "availability", "agenda", "booking", "duration", "allday", "rsvp", "conflict", "calendarid"],
    "database": ["row", "column", "index", "schema", "table", "transaction", "migration", "primary", "foreign", "constraint", "rollback", "commit", "cursor", "shard", "replica", "vacuum"],
    "search": ["ranking", "relevance", "facet", "filter", "highlight", "snippet", "recall", "precision", "tokenizer", "synonym", "boost", "pagination", "suggest", "score", "analyzer", "shingle"],
    "http": ["response", "header", "status", "endpoint", "method", "retry", "timeout", "redirect", "cookie", "gateway", "proxy", "throttle", "route", "tls", "keepalive", "compression"],
    "payment": ["charge", "refund", "invoice", "settlement", "authorization", "capture", "chargeback", "currency", "ledger", "payout", "fee", "reconcile", "dispute", "receipt", "wallet", "escrow"],
    "chat": ["channel", "mention", "reaction", "presence", "typing", "dm", "emoji", "pin", "unread", "member", "broadcast", "history", "notify", "roster", "muting", "slashcommand"],
    "storage": ["bucket", "blob", "lifecycle", "versioning", "replication", "tier", "prefix", "multipart", "acl", "encrypt", "presign", "quota", "manifest", "region", "coldline", "immutable"],
    "auth": ["token", "session", "scope", "grant", "refresh", "claim", "role", "policy", "identity", "revoke", "issuer", "audience", "consent", "mfa", "principal", "assertion"],
}
_GENERIC = ["data", "operation", "service", "resource", "item", "target", "source", "payload", "option", "config", "handler", "context"]
_FILLER = [f"attr{i}" for i in range(400)]
_KW_BASE = ["smtp", "imap", "grpc", "rest", "graphql", "s3", "gcs", "azure", "oauth", "saml", "jwt", "kafka", "sql", "webhook", "sftp", "ftp", "amqp", "redis", "mqtt", "soap"]


@dataclass(frozen=True)
class Spec:
    """Every knob that shapes the catalog. Varying it drives sensitivity sweeps."""
    seed: int = DEFAULT.seed
    core_share: int = DEFAULT.core_share
    kw_collision: float = DEFAULT.kw_collision_ratio


DEFAULT_SPEC = Spec()


def _kw(seq: int) -> str:
    return f"{_KW_BASE[seq % len(_KW_BASE)]}{seq}"


def _group_of_base(base_id: int) -> Tuple[str, str, str]:
    return GROUPS[base_id // len(ACTIONS)]


def query_tokens(base_id: int, spec: Spec = DEFAULT_SPEC):
    """Pure function of (gold, spec): (query_tokens, gold_core, kw, gid, action)."""
    gid, noun, obj = _group_of_base(base_id)
    action = ACTIONS[base_id % len(ACTIONS)]
    kw = _kw(base_id)
    r = rng(spec.seed, "qtok", spec.core_share, base_id)
    core = r.sample(_TOPIC_BANK[gid], spec.core_share)
    q = list(core) + [noun, obj] + r.sample(_GENERIC, 2)
    return q, core, kw, gid, action


def _schema_json(name: str, description: str) -> str:
    return json.dumps({
        "name": name, "description": description,
        "input_schema": {"type": "object", "properties": {
            "target": {"type": "string", "description": "target resource identifier"},
            "options": {"type": "object", "description": "operation options and flags"},
        }, "required": ["target"]},
    })


def estimate_tokens(name: str, description: str) -> int:
    """Heuristic tool-schema token estimate: ~4 chars/token over the JSON schema a
    gateway actually puts in the model's context. A heuristic, NOT a real
    tokenizer — the production path uses anthropic/tiktoken (see roadmap)."""
    return max(1, len(_schema_json(name, description)) // 4)


def _tool_text(tokens: List[str], kw: str, action: str, noun: str, obj: str) -> str:
    return f"{action} a {noun} {obj} : " + " ".join(tokens) + f" using {kw} protocol"


def _base_tool(base_id: int, spec: Spec) -> Tool:
    q, core, kw, gid, action = query_tokens(base_id, spec)
    _, noun, obj = _group_of_base(base_id)
    r = rng(spec.seed, "goldfill", base_id)
    desc = _tool_text(core + r.sample(_FILLER, 4), kw, action, noun, obj)
    name = f"{gid}.{action}_{kw}"
    return Tool(id=base_id, namespaced_name=name, group=gid, description=desc,
                keywords=[kw], is_distractor=False, token_cost=estimate_tokens(name, desc))


def _distractor(dist_seq: int, spec: Spec) -> Tool:
    target = HOT_TARGETS[dist_seq % len(HOT_TARGETS)]
    dist_id = N_BASE + dist_seq
    q, _core, target_kw, gid, action = query_tokens(target, spec)
    _, noun, obj = _group_of_base(target)
    r = rng(spec.seed, "distractor", spec.core_share, dist_id)
    # share up to the full query length (which is core_share + 4); for the default
    # core_share=8 this is randint(4,12), keeping default results byte-identical.
    xd = r.randint(4, len(q))
    shared = r.sample(q, xd)
    pad = r.sample(_FILLER, max(0, 12 - xd))
    # keyword collision: a fraction carry the gold's own rare keyword, so lexical
    # no longer uniquely pins gold -> hybrid can lose.
    collide = r.random() < spec.kw_collision
    own_kw = target_kw if collide else f"var{dist_id}"
    cross = (dist_seq % 2 == 1)
    if cross:
        idx = (GROUPS.index(_group_of_base(target)) + 3) % len(GROUPS)
        dgid, dnoun, dobj = GROUPS[idx]
    else:
        dgid, dnoun, dobj = gid, noun, obj
    desc = _tool_text(shared + pad, own_kw, action, dnoun, dobj)
    name = f"{dgid}.{action}_{own_kw}"
    return Tool(id=dist_id, namespaced_name=name, group=dgid, description=desc,
                keywords=[own_kw], is_distractor=True, token_cost=estimate_tokens(name, desc))


def _group_objects(tools: List[Tool]) -> List[ToolGroup]:
    by_group: Dict[str, List[int]] = {}
    for t in tools:
        by_group.setdefault(t.group, []).append(t.id)
    out = []
    for gid, noun, obj in GROUPS:
        if gid in by_group:
            desc = f"{gid} {noun} {obj} operations : " + " ".join(_TOPIC_BANK[gid])
            out.append(ToolGroup(name=gid, description=desc, tool_ids=sorted(by_group[gid])))
    return out


_MASTER_CACHE: Dict[Spec, List[Tool]] = {}


def _master(spec: Spec) -> List[Tool]:
    if spec not in _MASTER_CACHE:
        _MASTER_CACHE[spec] = ([_base_tool(i, spec) for i in range(N_BASE)]
                               + [_distractor(j, spec) for j in range(300 - N_BASE)])
    return _MASTER_CACHE[spec]


def build_catalog(size: int, spec: Spec = DEFAULT_SPEC) -> Catalog:
    master = _master(spec)
    if size > len(master):
        raise ValueError(f"size {size} exceeds master pool {len(master)}")
    tools = master[:size]
    return Catalog(size=size, tools=tools, groups=_group_objects(tools))


def _query_text(base_id: int, ambiguous: bool, spec: Spec) -> str:
    q, _core, kw, gid, action = query_tokens(base_id, spec)
    _, noun, obj = _group_of_base(base_id)
    body = f"please {action} a {noun} {obj} with " + " ".join(q)
    return body if ambiguous else body + f" using {kw}"


def generate_queries(cfg=DEFAULT, spec: Spec = DEFAULT_SPEC) -> List[Query]:
    out: List[Query] = []
    max_size = max(cfg.catalog_sizes)
    for i in range(cfg.n_queries):
        r = rng(spec.seed, "query", i)
        roll = r.random()
        diff = "multi" if roll < cfg.multi_tool_ratio else (
            "ambiguous" if roll < cfg.multi_tool_ratio + 0.15 else "single")
        if diff == "multi":
            n_gold = r.choice([2, 3])
            groups_order = r.sample(range(len(GROUPS)), n_gold)
            golds = [g * len(ACTIONS) + r.randrange(len(ACTIONS)) for g in groups_order]
            golds = [g for g in golds if g in HOT_TARGETS] or [HOT_TARGETS[i % 30]]
            text = " ; ".join(_query_text(g, False, spec) for g in golds)
            gold_ids = golds
        elif diff == "ambiguous":
            g = HOT_TARGETS[r.randrange(len(HOT_TARGETS))]
            text, gold_ids = _query_text(g, True, spec), [g]
        else:
            g = HOT_TARGETS[r.randrange(len(HOT_TARGETS))]
            text, gold_ids = _query_text(g, False, spec), [g]
        _, _, _, gid, _ = query_tokens(gold_ids[0], spec)
        out.append(Query(id=i, text=text, gold_tool_ids=sorted(set(gold_ids)),
                         group=gid, difficulty=diff, distractor_pool_size=max_size))
    return out
