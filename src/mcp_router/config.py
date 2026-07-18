"""Central configuration. Everything that affects benchmark output is here and
is recorded into each BenchRun so results are reproducible and auditable."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import List


# ---- global determinism knobs -------------------------------------------------
# Changing any of these changes results; they are stamped into every BenchRun.
SEED = int(os.environ.get("MCPR_SEED", "1234"))
EMBED_DIM = int(os.environ.get("MCPR_EMBED_DIM", "512"))

# Catalog staircase sizes (distractor pool grows -> recall cliff appears).
CATALOG_SIZES: List[int] = [100, 200, 300]

# k values swept for recall@k / exposure.
K_VALUES: List[int] = [1, 3, 5, 10]

# Routing strategies compared (must match routing.strategies registry keys).
STRATEGIES: List[str] = ["passthrough", "semantic_topk", "hierarchical", "hybrid"]

# Bootstrap resamples for confidence intervals.
BOOTSTRAP_N = int(os.environ.get("MCPR_BOOTSTRAP_N", "1000"))

# Size of the label self-consistency subset (NOT human-verified; see labeler.py).
SELF_CHECK_N = 50

# Gold tokens shared with the query (controls the cliff geometry; sensitivity-swept).
CORE_SHARE = int(os.environ.get("MCPR_CORE_SHARE", "8"))

# Fraction of distractors that COLLIDE on the gold's rare keyword, so the hybrid
# router can genuinely lose (breaks the degenerate McNemar b=0).
KW_COLLISION_RATIO = float(os.environ.get("MCPR_KW_COLLISION", "0.2"))


@dataclass
class BenchConfig:
    """Snapshot of every parameter that influences a run. Serialized into results."""
    seed: int = SEED
    embed_dim: int = EMBED_DIM
    catalog_sizes: List[int] = field(default_factory=lambda: list(CATALOG_SIZES))
    k_values: List[int] = field(default_factory=lambda: list(K_VALUES))
    strategies: List[str] = field(default_factory=lambda: list(STRATEGIES))
    bootstrap_n: int = BOOTSTRAP_N
    self_check_n: int = SELF_CHECK_N
    core_share: int = CORE_SHARE
    kw_collision_ratio: float = KW_COLLISION_RATIO
    lex_weight: float = 2.0
    embed_provider: str = "mock"       # mock | mock_char | local
    llm_provider: str = "mock"         # mock | claude
    vector_backend: str = "memory"     # memory | pgvector
    n_queries: int = 180               # synthetic labeled queries
    multi_tool_ratio: float = 0.25     # fraction of queries needing 2-3 gold tools

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT = BenchConfig()
