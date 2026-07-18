"""Deterministic synthetic catalog with a *nested* staircase of near-duplicate
distractors — the mechanism that produces a genuine recall cliff.

Why the cliff is real (a property of the geometry, not a scripted number):

  * catalog(100) ⊂ catalog(200) ⊂ catalog(300): the master list is fixed and
    catalog(N) = master[:N]. The same gold tools exist at every size; only the
    distractor pool grows.

  * A query for gold g carries 12 tokens Q (+ a rare keyword). The gold tool
    shares exactly 8 of Q plus the keyword  =>  gold overlaps the query on 9
    tokens under bag-of-words similarity.

  * Each distractor of g shares Xd ~ Uniform{4..12} of Q (no keyword). Under pure
    semantic similarity a distractor out-ranks gold iff Xd > 9 (i.e. it shares 10+
    of the query's tokens). Distractors are round-robin assigned to HOT_TARGETS,
    so each hot gold accrues ≈1 distractor at N=100 and ≈8 at N=300. More
    distractors => higher chance ≥k of them out-share gold => gold drops out of
    top-k => recall@k collapses. That is the cliff.

  * The keyword is rare (query + only gold). Lexical/BM25 matching (hybrid
    strategy) keys off it and pulls gold back to the top — which is why hybrid
    recovers the recall that semantic-topk loses.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from ..config import DEFAULT
from ..determinism import rng, tokenize
from ..models import Catalog, Query, Tool, ToolGroup

GROUPS: List[Tuple[str, str, str]] = [
    ("email", "email", "message"),
    ("file", "file", "document"),
    ("calendar", "calendar", "event"),
    ("database", "database", "record"),
    ("search", "search", "result"),
    ("http", "http", "request"),
    ("payment", "payment", "transaction"),
    ("chat", "chat", "message"),
    ("storage", "storage", "object"),
    ("auth", "auth", "credential"),
]
ACTIONS = ["send", "read", "delete", "list", "update", "create"]

N_BASE = len(GROUPS) * len(ACTIONS)              # 60 base/gold-eligible tools
HOT_TARGETS = list(range(30))                    # queried + crowded golds
CORE_SHARE = 8                                   # gold shares 8 topic tokens + kw

_TOPIC_BANK: Dict[str, List[str]] = {
    "email": ["inbox", "recipient", "subject", "attachment", "draft", "thread",
              "reply", "forward", "signature", "spam", "cc", "bcc", "mailbox",
              "compose", "deliver", "bounce"],
    "file": ["folder", "path", "directory", "upload", "download", "rename",
             "copy", "move", "permission", "metadata", "chunk", "binary",
             "archive", "extension", "checksum", "symlink"],
    "calendar": ["meeting", "invite", "reminder", "schedule", "slot",
                 "recurrence", "attendee", "timezone", "availability", "agenda",
                 "booking", "duration", "allday", "rsvp", "conflict", "calendarid"],
    "database": ["row", "column", "index", "schema", "table", "transaction",
                 "migration", "primary", "foreign", "constraint", "rollback",
                 "commit", "cursor", "shard", "replica", "vacuum"],
    "search": ["ranking", "relevance", "facet", "filter", "highlight", "snippet",
               "recall", "precision", "tokenizer", "synonym", "boost",
               "pagination", "suggest", "score", "analyzer", "shingle"],
    "http": ["response", "header", "status", "endpoint", "method", "retry",
             "timeout", "redirect", "cookie", "gateway", "proxy", "throttle",
             "route", "tls", "keepalive", "compression"],
    "payment": ["charge", "refund", "invoice", "settlement", "authorization",
                "capture", "chargeback", "currency", "ledger", "payout", "fee",
                "reconcile", "dispute", "receipt", "wallet", "escrow"],
    "chat": ["channel", "mention", "reaction", "presence", "typing", "dm",
             "emoji", "pin", "unread", "member", "broadcast", "history",
             "notify", "roster", "muting", "slashcommand"],
    "storage": ["bucket", "blob", "lifecycle", "versioning", "replication",
                "tier", "prefix", "multipart", "acl", "encrypt", "presign",
                "quota", "manifest", "region", "coldline", "immutable"],
    "auth": ["token", "session", "scope", "grant", "refresh", "claim", "role",
             "policy", "identity", "revoke", "issuer", "audience", "consent",
             "mfa", "principal", "assertion"],
}
_GENERIC = ["data", "operation", "service", "resource", "item", "target",
            "source", "payload", "option", "config", "handler", "context"]
_FILLER = [f"attr{i}" for i in range(400)]
_KW_BASE = ["smtp", "imap", "grpc", "rest", "graphql", "s3", "gcs", "azure",
            "oauth", "saml", "jwt", "kafka", "sql", "webhook", "sftp", "ftp",
            "amqp", "redis", "mqtt", "soap"]


def _kw(seq: int) -> str:
    return f"{_KW_BASE[seq % len(_KW_BASE)]}{seq}"


def _group_of_base(base_id: int) -> Tuple[str, str, str]:
    return GROUPS[base_id // len(ACTIONS)]


def query_tokens(base_id: int) -> Tuple[List[str], List[str], str, str, str]:
    """Pure function of a gold tool. Returns:
      (query_tokens[12], gold_core[8], kw, group_id, action)
    query_tokens = 8 core group-topic words + [noun, obj] + 2 generic.
    gold_core     = the 8 core words the gold tool actually contains.
    """
    gid, noun, obj = _group_of_base(base_id)
    action = ACTIONS[base_id % len(ACTIONS)]
    kw = _kw(base_id)
    r = rng(DEFAULT.seed, "qtok", base_id)
    core = r.sample(_TOPIC_BANK[gid], CORE_SHARE)        # 8 shared with gold
    q = list(core) + [noun, obj] + r.sample(_GENERIC, 2)  # 12 query tokens
    return q, core, kw, gid, action


def _tool_text(tokens: List[str], kw: str, action: str, noun: str, obj: str) -> str:
    return f"{action} a {noun} {obj} : " + " ".join(tokens) + f" using {kw} protocol"


def _base_tool(base_id: int) -> Tool:
    q, core, kw, gid, action = query_tokens(base_id)
    _, noun, obj = _group_of_base(base_id)
    r = rng(DEFAULT.seed, "goldfill", base_id)
    unique = r.sample(_FILLER, 4)                          # gold-only tokens
    desc = _tool_text(core + unique, kw, action, noun, obj)  # shares 8 of Q + kw
    return Tool(
        id=base_id, namespaced_name=f"{gid}.{action}_{kw}", group=gid,
        description=desc, keywords=[kw], is_distractor=False,
        token_cost=48 + 2 * len(tokenize(desc)),
    )


def _distractor(dist_seq: int) -> Tool:
    target = HOT_TARGETS[dist_seq % len(HOT_TARGETS)]
    dist_id = N_BASE + dist_seq
    q, _core, _kw2, gid, action = query_tokens(target)
    _, noun, obj = _group_of_base(target)
    r = rng(DEFAULT.seed, "distractor", dist_id)
    xd = r.randint(4, 12)                                  # shared tokens with Q
    shared = r.sample(q, xd)
    pad = r.sample(_FILLER, 12 - xd)
    own_kw = f"var{dist_id}"
    cross = (dist_seq % 2 == 1)                            # half cross-group
    if cross:
        idx = (GROUPS.index(_group_of_base(target)) + 3) % len(GROUPS)
        dgid, dnoun, dobj = GROUPS[idx]
    else:
        dgid, dnoun, dobj = gid, noun, obj
    desc = _tool_text(shared + pad, own_kw, action, dnoun, dobj)
    return Tool(
        id=dist_id, namespaced_name=f"{dgid}.{action}_{own_kw}", group=dgid,
        description=desc, keywords=[own_kw], is_distractor=True,
        token_cost=48 + 2 * len(tokenize(desc)),
    )


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


_MASTER: List[Tool] = (
    [_base_tool(i) for i in range(N_BASE)]
    + [_distractor(j) for j in range(300 - N_BASE)]
)


def build_catalog(size: int) -> Catalog:
    if size > len(_MASTER):
        raise ValueError(f"size {size} exceeds master pool {len(_MASTER)}")
    tools = _MASTER[:size]
    return Catalog(size=size, tools=tools, groups=_group_objects(tools))


def _query_text(base_id: int, ambiguous: bool) -> str:
    q, _core, kw, gid, action = query_tokens(base_id)
    _, noun, obj = _group_of_base(base_id)
    body = f"please {action} a {noun} {obj} with " + " ".join(q)
    return body if ambiguous else body + f" using {kw}"


def generate_queries(cfg=DEFAULT) -> List[Query]:
    """Labeled query set. single (kw present), multi (2-3 golds across groups),
    ambiguous (kw omitted -> lexical can't help, harder)."""
    out: List[Query] = []
    max_size = max(cfg.catalog_sizes)
    for i in range(cfg.n_queries):
        r = rng(cfg.seed, "query", i)
        roll = r.random()
        if roll < cfg.multi_tool_ratio:
            diff = "multi"
        elif roll < cfg.multi_tool_ratio + 0.15:
            diff = "ambiguous"
        else:
            diff = "single"

        if diff == "multi":
            n_gold = r.choice([2, 3])
            # pick golds from distinct groups so hierarchical top-2 is stressed
            groups_order = r.sample(range(len(GROUPS)), n_gold)
            golds = [g * len(ACTIONS) + r.randrange(len(ACTIONS)) for g in groups_order]
            golds = [g for g in golds if g in HOT_TARGETS] or [HOT_TARGETS[i % 30]]
            text = " ; ".join(_query_text(g, ambiguous=False) for g in golds)
            gold_ids = golds
        elif diff == "ambiguous":
            g = HOT_TARGETS[r.randrange(len(HOT_TARGETS))]
            text = _query_text(g, ambiguous=True)
            gold_ids = [g]
        else:
            g = HOT_TARGETS[r.randrange(len(HOT_TARGETS))]
            text = _query_text(g, ambiguous=False)
            gold_ids = [g]

        _, _, _, gid, _ = query_tokens(gold_ids[0])
        out.append(Query(id=i, text=text, gold_tool_ids=sorted(set(gold_ids)),
                         group=gid, difficulty=diff, distractor_pool_size=max_size))
    return out
