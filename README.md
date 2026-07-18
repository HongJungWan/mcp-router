# MCP Gateway — tool-routing benchmark

A benchmark harness that answers one question with reproducible data:

> When an agent's MCP tool catalog grows to hundreds of tools, **does exposing
> only a semantic top-k of them silently drop the right tool?**

Short answer, measured here: **yes — recall@1 collapses from ~0.70 to ~0.13 as
the catalog grows 100→300**, and a hybrid (vector + lexical) router recovers most
of the lost recall. Task-success and cost tell a second, more nuanced story
(below): exposing *everything* is worse on almost every axis except raw recall.

> Status: **M3 (the offline benchmark)** — the evaluation half of a larger MCP
> Gateway project. The live serving path (MCP server/client federation, RBAC,
> circuit breakers) is a documented roadmap item, **not** built here. Every
> headline number below is produced by the offline, deterministic mock path;
> the real Claude / pgvector / bge adapters are wired but **have not been run**
> (no cassettes committed). See "What is real vs mock" before drawing conclusions.

---

## The recall cliff (primary result — the clean one)

`semantic_topk` fractional recall@1 as the near-duplicate distractor pool grows
(seed=1234, 180 labeled queries, cluster-bootstrap 95% CI):

| catalog size | semantic_topk recall@1 | 95% CI (gold-cluster bootstrap) |
|---|---|---|
| 100 | **0.705** | 0.512–0.871 |
| 200 | **0.278** | 0.113–0.463 |
| 300 | **0.128** | 0.017–0.268 |

**Fractional recall@k at catalog=300** (|gold ∩ top-k| / |gold|):

| strategy | recall@1 | recall@3 | recall@5 | tokens@3 (p95, est.) |
|---|---|---|---|---|
| passthrough (expose all) | 1.00 | 1.00 | 1.00 | **31,084** |
| semantic_topk | 0.13 | 0.62 | 0.93 | 316 |
| hierarchical (top-2 groups) | 0.52 | 0.70 | 0.95 | 316 |
| **hybrid (vector+lexical)** | 0.45 | **0.92** | **1.00** | 316 |

Hybrid recovers recall@3 from 0.62 → **0.92**. Note hybrid is *not* uniformly
best: at k=1 hierarchical (0.52) beats hybrid (0.45), because ~20% of distractors
deliberately **collide on the gold keyword** — so the lexical signal no longer
uniquely pins gold and hybrid can lose. (This is intentional; see below.)

## The second axis: task-success ≠ recall (the nuanced one)

A **deliberately weak, decoupled** ReAct agent (Jaccard over tool descriptions —
a *different* signal than the router; see "closed-loop" note) then picks tools
from only the exposed set. Task-success@3 at catalog=300:

| strategy | recall@3 | task-success@3 | tokens@3 |
|---|---|---|---|
| passthrough | 1.00 | **0.05** | 31,084 |
| semantic_topk | 0.62 | 0.05 | 316 |
| hierarchical | 0.70 | **0.50** | 316 |
| hybrid | 0.92 | 0.16 | 316 |

The interesting finding: **exposing every tool (passthrough) gives perfect recall
but the worst task-success (0.05)** — 300 look-alike tools overwhelm selection —
*and* costs ~100× the tokens. Recall and selection-accuracy are different axes:
the highest-recall strategy (hybrid) is not the highest task-success one
(hierarchical, whose tight per-group exposed set is easiest to choose from).

> **Read this honestly.** The absolute task-success numbers are low because the
> offline agent is a weak Jaccard proxy on purpose. What is meaningful is the
> *relative ordering* and the structural fact that a recall miss forces a failure
> (the agent never sees a dropped tool). Real agent quality is the opt-in Claude
> path, which has not been run here.

McNemar on paired per-query task-success (BH-corrected across the family):

| size | k | A | B | b | c | χ² | p | q |
|---|---|---|---|---|---|---|---|---|
| 300 | 1 | semantic_topk | hierarchical | 0 | 69 | 67.0 | 2.7e-16 | 1.6e-15 |
| 300 | 3 | semantic_topk | hierarchical | 0 | 81 | 79.0 | 6.2e-19 | 4.9e-18 |
| 300 | 3 | semantic_topk | hybrid | 0 | 19 | 17.1 | 3.6e-05 | 7.3e-05 |

`b=0` (semantic never wins where the other loses) is now an **empirical** result
of a decoupled agent, not a construction identity — previously the agent shared
the router's formula, which made b=0 tautological.

![recall cliff](docs/recall_cliff.svg)

> Chart + a full sample run are committed under [`docs/`](docs/); regenerate
> everything with `make bench`. Per-difficulty and hit-rate tables are in
> `summary.json` (`cells_by_difficulty`).

---

## Why the cliff is real, and how far it generalizes

- **Nested staircase**: `catalog(100) ⊂ catalog(200) ⊂ catalog(300)`. Same gold
  tools at every size; only the distractor pool grows.
- A query shares `core_share` (=8) tokens + a rare keyword with its gold tool.
  Each distractor shares `Xd ~ Uniform{4..len(query)}` tokens; under pure semantic
  similarity a distractor out-ranks gold once it shares more. Distractors are
  round-robin assigned to the queried tools, so each gold accrues ~1 distractor at
  N=100 and ~8 at N=300 → gold falls out of a small top-k → **recall collapses.**

**Is the cliff just an artifact of `core_share=8` or of bag-of-words?** Checked:

- `make sweep` varies `core_share` 6→10; the recall@1 cliff-drop stays **−0.32 to
  −0.57** across all of them (not a single hand-picked constant).
- `--embed mock_char` re-runs under a **char-trigram geometry** (not token-additive);
  the cliff survives (recall@1 0.89 → 0.38 at core_share=8).

**Honest limit:** this is still a *synthetic* catalog with *mock* embeddings. It
demonstrates the mechanism from first principles and shows it is robust to two
knobs — it does **not** prove the slope you'd see on `bge-small` or a real MCP
server corpus. That requires the `.[local]` / real-harvest paths (roadmap).

### When NOT to use semantic top-k

Small k trades recall for tokens; the trade is worst exactly when the catalog is
large and full of look-alikes — when you'd reach for a gateway. Few or
well-separated tools → plain top-k is fine. Many near-duplicate tools → top-k
drops the right one, and no downstream prompting recovers a tool that was never
exposed. Use hybrid, or raise k and pay the tokens.

---

## What is real vs mock (read before trusting a number)

| concern | offline default (used for ALL numbers here) | opt-in adapter (wired, **NOT run**) |
|---|---|---|
| embeddings | hashed bag-of-words / char-trigram (deterministic) | `bge-small-en-v1.5` (`.[local]`) |
| agent / LLM | weak Jaccard mock (decoupled from router) | Claude tool-use + LangGraph (`.[agent]`) |
| labeler | semantic+lexical, threshold cardinality | same, via Claude (`.[claude]`) |
| vector index | pure-Python cosine | pgvector HNSW (`.[pg]`) |
| token cost | JSON-schema chars/4 **heuristic** | real tokenizer (roadmap) |

- **No human labels exist.** The label report is *self-consistency* of the auto
  labeler vs the synthesized ground truth (κ ≈ 0.40), a sanity check on the
  labeling mechanism — **not** human-verified label quality, and not a selling
  point. See `labeling/labeler.py`.
- **Latency** in `results.csv` is an in-process, single-call, no-load indicator —
  non-deterministic and NOT part of the reproducibility guarantee.
- Token cost is a chars/4 heuristic; the "~100× fewer tokens than passthrough"
  ratio (3-of-N tools) is robust to the constant, the absolute count is not.

## Quickstart (zero dependencies, offline, deterministic)

Pure Python stdlib — no API keys, no network, no `pip install`. Same seed →
byte-identical recall, task-success, CIs, and κ (only wall-clock latency varies).

```bash
make bench     # PYTHONPATH=src python -m mcp_router bench run --out artifacts
make test      # 20 unittest cases incl. cliff, decoupling, and geometry checks
python -m mcp_router bench sweep --shares 6,7,8,9,10   # core_share sensitivity
python -m mcp_router bench sweep --embed mock_char     # different geometry
```

## Reproducibility envelope

Every run stamps `git_sha`, `seed`, `embed_model`, `llm_model_id`, `core_share`,
`kw_collision_ratio`, and the full config into `summary.json`. CIs use a
**gold-cluster** bootstrap (the 180 queries cluster over 30 gold tools, so an iid
bootstrap would understate them); strategy comparisons use McNemar with
Benjamini-Hochberg correction.

## Roadmap (deliberately cut from M3 to stay solo-shippable)

- Live gateway **serving** path (federation, RBAC, circuit breakers) — this is
  the evaluation half only.
- Run the **real** adapters: bge-small embeddings, Claude tool-use agent (commit
  cassettes), pgvector; replace the heuristic token count with a real tokenizer.
- Real MCP server harvesting alongside the synthetic staircase.
- CI recall-regression gate; GraphRAG routing as a 5th strategy.

## Layout

```
src/mcp_router/
  catalog/     staircase catalog (keyword-collision distractors) + query gen + Spec
  routing/     four strategies + RoutingContext
  providers/   embed (mock bow / mock_char / local) + llm (mock jaccard / claude)
  vectorstore/ cosine index (memory default; pgvector opt-in)
  labeling/    self-consistency labeler + Cohen's kappa
  bench/       decoupled ReAct agent, metrics (fractional recall, cluster bootstrap,
               McNemar+BH), runner, report
  tracing.py · cli.py (bench run | bench sweep)
tests/         20 unittest cases
```
