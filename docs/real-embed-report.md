# MCP Gateway — tool-routing benchmark (M3)

**Reproducibility envelope**

- git_sha: `5a4c01e` · seed: `1234` · embed: `bge-small-en-v1.5` · llm: `mock-jaccard-agent-v1` · vector: `memory`
- queries: 120 · staircase: [100, 200, 300] · k: [1, 3, 5, 10] · cluster-bootstrap n=300
- core_share=8 · kw_collision_ratio=0.2

> **Metric note.** `recall@k` below is *fractional* recall (|gold∩topk|/|gold|). Hit-rate (all-gold set-cover) and per-difficulty breakdowns are in `summary.json` (`cells`/`cells_by_difficulty`). For multi-tool queries at k<|gold| a full hit is structurally impossible, so difficulty is reported separately.

## Label self-consistency (NOT human-verified)

- self-consistency vs synthesized ground truth; NOT human-verified
- self-check queries: 50 · labeled pairs: 338 · disagreements: 84
- **self-consistency κ = 0.4177** · raw agreement = 0.7515 · labeler = `mock-jaccard-agent-v1`

## fractional recall@k — catalog size 100

| strategy | recall@1 | recall@3 | recall@5 | recall@10 | tokens(p95) |
|---|---|---|---|---|---|
| passthrough | 1.00 | 1.00 | 1.00 | 1.00 | 10349 |
| semantic_topk | 0.89 | 0.95 | 0.98 | 1.00 | 1049 |
| hierarchical | 0.89 | 0.95 | 0.98 | 0.98 | 1049 |
| hybrid | 0.89 | 1.00 | 1.00 | 1.00 | 1049 |

## fractional recall@k — catalog size 200

| strategy | recall@1 | recall@3 | recall@5 | recall@10 | tokens(p95) |
|---|---|---|---|---|---|
| passthrough | 1.00 | 1.00 | 1.00 | 1.00 | 20719 |
| semantic_topk | 0.78 | 0.92 | 0.92 | 0.98 | 1045 |
| hierarchical | 0.79 | 0.92 | 0.93 | 0.98 | 1048 |
| hybrid | 0.78 | 0.96 | 0.97 | 1.00 | 1046 |

## fractional recall@k — catalog size 300

| strategy | recall@1 | recall@3 | recall@5 | recall@10 | tokens(p95) |
|---|---|---|---|---|---|
| passthrough | 1.00 | 1.00 | 1.00 | 1.00 | 31084 |
| semantic_topk | 0.70 | 0.90 | 0.92 | 0.96 | 1050 |
| hierarchical | 0.71 | 0.91 | 0.92 | 0.97 | 1050 |
| hybrid | 0.70 | 0.93 | 0.97 | 1.00 | 1050 |

## by difficulty — catalog size 300

**single**

| strategy | recall@1 | recall@3 | recall@5 | recall@10 | tokens(p95) |
|---|---|---|---|---|---|
| passthrough | 1.00 | 1.00 | 1.00 | 1.00 | 31084 |
| semantic_topk | 0.77 | 1.00 | 1.00 | 1.00 | 1051 |
| hierarchical | 0.77 | 1.00 | 1.00 | 1.00 | 1051 |
| hybrid | 0.77 | 1.00 | 1.00 | 1.00 | 1051 |

**multi**

| strategy | recall@1 | recall@3 | recall@5 | recall@10 | tokens(p95) |
|---|---|---|---|---|---|
| passthrough | 1.00 | 1.00 | 1.00 | 1.00 | 31084 |
| semantic_topk | 0.64 | 0.77 | 0.77 | 0.86 | 1046 |
| hierarchical | 0.64 | 0.80 | 0.80 | 0.88 | 1047 |
| hybrid | 0.64 | 0.88 | 0.99 | 1.00 | 1046 |

**ambiguous**

| strategy | recall@1 | recall@3 | recall@5 | recall@10 | tokens(p95) |
|---|---|---|---|---|---|
| passthrough | 1.00 | 1.00 | 1.00 | 1.00 | 31084 |
| semantic_topk | 0.50 | 0.69 | 0.81 | 1.00 | 1046 |
| hierarchical | 0.56 | 0.69 | 0.81 | 1.00 | 1045 |
| hybrid | 0.50 | 0.69 | 0.81 | 1.00 | 1046 |

## task-success (SECONDARY, weak offline agent)

phi(recall_hit, task_success) = **0.2625** — low magnitude means task_success is a genuinely separate signal from recall, not a re-labeling. Absolute task-success values are low because the offline agent is a deliberately weak Jaccard proxy that is NOT told how many tools to pick; only the ordering and the recall→ceiling relationship are meaningful. `task_success`/`hit_rate` per cell are in `summary.json`.

## Recall cliff

Detected **193** cliff events (gold exists but ranked past the exposed top-k under semantic routing). See `traces.jsonl`.

Example traces:

- q#8 [multi] semantic_topk@100 k=1: gold ranks [2, 4] · task_success=False · `bcba5be8aba1d974`
- q#8 [multi] semantic_topk@100 k=3: gold ranks [2, 4] · task_success=False · `0e0d9a1e32fd4b02`
- q#25 [multi] semantic_topk@100 k=1: gold ranks [6, 1] · task_success=False · `79f0189d0c314344`
- q#25 [multi] semantic_topk@100 k=3: gold ranks [6, 1] · task_success=False · `55798dbe9737c5a8`
- q#25 [multi] semantic_topk@100 k=5: gold ranks [6, 1] · task_success=False · `2fc7f4e297a77b39`
- q#34 [multi] semantic_topk@100 k=1: gold ranks [6, 3, 1] · task_success=False · `c9489c5f7f80fe7c`

## McNemar (recall_hit, paired by query, BH-corrected)

Computed on **recall_hit** (the routing question), not the weak task_success. `b` = A hit / B miss; `c` = B hit / A miss. `q` = Benjamini-Hochberg adjusted p. hierarchical-vs-hybrid is genuinely two-sided (keyword collisions).

| size | k | A | B | b | c | χ² | p | q |
|---|---|---|---|---|---|---|---|---|
| 100 | 1 | semantic_topk | hybrid | 0 | 0 | 0.0 | 1.0000 | 1.0000 |
| 100 | 1 | semantic_topk | hierarchical | 0 | 0 | 0.0 | 1.0000 | 1.0000 |
| 100 | 1 | hierarchical | hybrid | 0 | 0 | 0.0 | 1.0000 | 1.0000 |
| 100 | 3 | semantic_topk | hybrid | 0 | 11 | 9.0909 | 0.0026 | 0.0154 |
| 100 | 3 | semantic_topk | hierarchical | 0 | 0 | 0.0 | 1.0000 | 1.0000 |
| 100 | 3 | hierarchical | hybrid | 0 | 11 | 9.0909 | 0.0026 | 0.0154 |
| 200 | 1 | semantic_topk | hybrid | 0 | 0 | 0.0 | 1.0000 | 1.0000 |
| 200 | 1 | semantic_topk | hierarchical | 0 | 1 | 0.0 | 1.0000 | 1.0000 |
| 200 | 1 | hierarchical | hybrid | 1 | 0 | 0.0 | 1.0000 | 1.0000 |
| 200 | 3 | semantic_topk | hybrid | 0 | 9 | 7.1111 | 0.0077 | 0.0345 |
| 200 | 3 | semantic_topk | hierarchical | 0 | 1 | 0.0 | 1.0000 | 1.0000 |
| 200 | 3 | hierarchical | hybrid | 0 | 8 | 6.125 | 0.0133 | 0.0480 |
| 300 | 1 | semantic_topk | hybrid | 0 | 0 | 0.0 | 1.0000 | 1.0000 |
| 300 | 1 | semantic_topk | hierarchical | 0 | 1 | 0.0 | 1.0000 | 1.0000 |
| 300 | 1 | hierarchical | hybrid | 1 | 0 | 0.0 | 1.0000 | 1.0000 |
| 300 | 3 | semantic_topk | hybrid | 0 | 6 | 4.1667 | 0.0412 | 0.1237 |
| 300 | 3 | semantic_topk | hierarchical | 0 | 1 | 0.0 | 1.0000 | 1.0000 |
| 300 | 3 | hierarchical | hybrid | 0 | 5 | 3.2 | 0.0736 | 0.1657 |

![recall cliff](recall_cliff.svg)
