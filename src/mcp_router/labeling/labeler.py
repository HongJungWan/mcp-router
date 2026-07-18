"""Label self-consistency report.

HONESTY NOTE: there is no human annotator in the offline path. This does NOT
report human-verified label quality. It reports the *self-consistency* of an
automatic labeler against the synthesized ground truth — i.e. how often a
semantic+lexical labeler recovers the tool the generator planted. It is a sanity
check on the labeling mechanism, NOT evidence that the labels match human
judgement. Real inter-annotator agreement would require a human pass (roadmap).

Two independence properties vs. the router/agent:
  * the labeler is not told the gold cardinality; it decides how many tools to
    label via a score-gap threshold (offline path). The opt-in real-LLM path uses
    a small fixed budget (also not len(gold)) — see `_LABEL_BUDGET`.
  * it uses semantic+lexical scoring (a different signal than the Jaccard agent).
"""
from __future__ import annotations

from typing import Dict, List

from ..config import DEFAULT
from ..determinism import rng, tokenize
from ..models import Catalog, Query
from ..vectorstore.base import cosine
from .kappa import agreement, cohen_kappa

# A candidate is labeled "gold" if its score is within this fraction of the top
# score. This lets the labeler choose its own cardinality (1..few) instead of
# being told the true count.
_REL_GAP = 0.15
# Fixed budget for the opt-in real-LLM path (NOT len(gold) — no cardinality leak).
_LABEL_BUDGET = 3


def _label_predict(query: str, tools: list, embedder) -> List[int]:
    """Auto-labeler: rank by semantic + exact-keyword score, then keep every
    candidate within _REL_GAP of the top (so cardinality is inferred, not given).
    Capped at 3 to avoid degenerate all-positive labels."""
    q = embedder.embed(query)
    qtokens = set(tokenize(query))
    scored = []
    for t in tools:
        lex = sum(1 for w in (t.keywords or []) if w in qtokens)
        scored.append((t.id, cosine(q, embedder.embed(t.embed_text)) + 2.0 * lex))
    scored.sort(key=lambda x: (-x[1], x[0]))
    if not scored:
        return []
    top = scored[0][1]
    thresh = top - _REL_GAP * (abs(top) + 1e-9)
    return [tid for tid, s in scored[:3] if s >= thresh]


def label_quality_report(catalog: Catalog, queries: List[Query], llm, cfg=DEFAULT) -> Dict:
    subset = queries[: cfg.self_check_n]
    all_tools = catalog.tools
    embedder = getattr(llm, "embedder", None)

    truth_labels: List[int] = []
    auto_labels: List[int] = []
    n_disagree = 0

    for q in subset:
        true_gold = set(q.gold_tool_ids)
        if embedder is not None:
            pred = set(_label_predict(q.text, all_tools, embedder))
        else:  # pragma: no cover - real LLM labeler path (unrun)
            pred = set(llm.choose_tools(q.text, all_tools, _LABEL_BUDGET))
        # label set = truth ∪ prediction ∪ sampled distractor negatives (varied
        # count so kappa's prevalence sensitivity is exercised, not fixed).
        r = rng(cfg.seed, "labelneg", q.id)
        distractor_ids = [t.id for t in all_tools if t.is_distractor]
        k_neg = r.choice([3, 4, 5])
        negatives = r.sample(distractor_ids, min(k_neg, len(distractor_ids)))
        for cid in sorted(true_gold | pred | set(negatives)):
            h = 1 if cid in true_gold else 0
            a = 1 if cid in pred else 0
            truth_labels.append(h)
            auto_labels.append(a)
            n_disagree += (h != a)

    return {
        "note": "self-consistency vs synthesized ground truth; NOT human-verified",
        "self_check_n": len(subset),
        "labeled_pairs": len(truth_labels),
        "disagreements": n_disagree,
        "raw_agreement": round(agreement(truth_labels, auto_labels), 4),
        "self_consistency_kappa": round(cohen_kappa(truth_labels, auto_labels), 4),
        "labeler_model": getattr(llm, "model_id", "unknown"),
    }
