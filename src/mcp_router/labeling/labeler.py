"""Label-quality report.

Pipeline modeled: an LLM proposes gold tools for each query (auto-labeling); a
50-query human-verified golden subset checks those labels. We report Cohen's
kappa between the LLM labels and the (ground-truth) human labels.

Offline, the "human" labels are the synthesized ground truth and the "LLM"
labeler is MockLLM ranking the whole catalog — it sometimes picks a near-dup
distractor instead of the true tool, so kappa is high but < 1 (realistic).
"""
from __future__ import annotations

from typing import Dict, List

from ..config import DEFAULT
from ..determinism import rng, tokenize
from ..models import Catalog, Query
from ..vectorstore.base import cosine
from .kappa import agreement, cohen_kappa


def _label_predict(query: str, tools: list, n: int, embedder) -> List[int]:
    """Offline stand-in for an LLM reading tool docs and proposing gold tools.
    Uses semantic + exact-keyword (lexical) scoring: it labels keyword-bearing
    queries correctly, but ambiguous queries (no keyword) still trip it — so the
    resulting kappa is high but < 1, as real auto-labeling is."""
    q = embedder.embed(query)
    qtokens = set(tokenize(query))
    scored = []
    for t in tools:
        lex = sum(1 for w in (t.keywords or []) if w in qtokens)
        scored.append((t.id, cosine(q, embedder.embed(t.embed_text)) + 2.0 * lex))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return [tid for tid, _ in scored[:n]]


def label_quality_report(catalog: Catalog, queries: List[Query], llm, cfg=DEFAULT) -> Dict:
    subset = queries[: cfg.human_verified_n]
    by_id = catalog.by_id()
    all_tools = catalog.tools
    embedder = getattr(llm, "embedder", None)

    human_labels: List[int] = []
    llm_labels: List[int] = []
    n_disagree = 0

    for q in subset:
        true_gold = set(q.gold_tool_ids)
        if embedder is not None:
            pred = set(_label_predict(q.text, all_tools, len(true_gold), embedder))
        else:  # pragma: no cover - real LLM labeler path
            pred = set(llm.choose_tools(q.text, all_tools, len(true_gold)))
        # candidate items to label: true gold ∪ llm prediction ∪ a few sampled
        # distractors (negatives) so kappa isn't trivially inflated by all-zeros.
        r = rng(cfg.seed, "labelneg", q.id)
        distractor_ids = [t.id for t in all_tools if t.is_distractor]
        negatives = r.sample(distractor_ids, min(4, len(distractor_ids)))
        candidates = sorted(true_gold | pred | set(negatives))
        for cid in candidates:
            h = 1 if cid in true_gold else 0
            l = 1 if cid in pred else 0
            human_labels.append(h)
            llm_labels.append(l)
            if h != l:
                n_disagree += 1

    return {
        "human_verified_n": len(subset),
        "labeled_pairs": len(human_labels),
        "disagreements": n_disagree,
        "raw_agreement": round(agreement(human_labels, llm_labels), 4),
        "cohen_kappa": round(cohen_kappa(human_labels, llm_labels), 4),
        "labeler_model": getattr(llm, "model_id", "unknown"),
    }
