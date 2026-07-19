# MCP Gateway — tool-routing benchmark (M3)

**Reproducibility envelope**

- git_sha: `a6aed70` · seed: `1234` · embed: `mock-bow-hash-v1` · llm: `mock-jaccard-agent-v1` · vector: `memory`
- queries: 180 · staircase: [100, 200, 300] · k: [1, 3, 5, 10] · cluster-bootstrap n=1000
- core_share=8 · kw_collision_ratio=0.2

> **Metric note.** `recall@k` below is *fractional* recall (|gold∩topk|/|gold|). Hit-rate (all-gold set-cover) and per-difficulty breakdowns are in `summary.json` (`cells`/`cells_by_difficulty`). For multi-tool queries at k<|gold| a full hit is structurally impossible, so difficulty is reported separately.

## Label self-consistency (NOT human-verified)

- self-consistency vs synthesized ground truth; NOT human-verified
- self-check queries: 50 · labeled pairs: 339 · disagreements: 86
- **self-consistency κ = 0.4044** · raw agreement = 0.7463 · labeler = `mock-jaccard-agent-v1`

## fractional recall@k — catalog size 100

| strategy | recall@1 | recall@3 | recall@5 | recall@10 | tokens(p95) |
|---|---|---|---|---|---|
| passthrough | 1.00 | 1.00 | 1.00 | 1.00 | 10349 |
| semantic_topk | 0.70 | 0.99 | 1.00 | 1.00 | 1049 |
| hierarchical | 0.80 | 1.00 | 1.00 | 1.00 | 1051 |
| hybrid | 0.90 | 1.00 | 1.00 | 1.00 | 1049 |

## fractional recall@k — catalog size 200

| strategy | recall@1 | recall@3 | recall@5 | recall@10 | tokens(p95) |
|---|---|---|---|---|---|
| passthrough | 1.00 | 1.00 | 1.00 | 1.00 | 20719 |
| semantic_topk | 0.28 | 0.87 | 0.99 | 1.00 | 1047 |
| hierarchical | 0.59 | 0.89 | 0.99 | 1.00 | 1052 |
| hybrid | 0.59 | 0.97 | 1.00 | 1.00 | 1047 |

## fractional recall@k — catalog size 300

| strategy | recall@1 | recall@3 | recall@5 | recall@10 | tokens(p95) |
|---|---|---|---|---|---|
| passthrough | 1.00 | 1.00 | 1.00 | 1.00 | 31084 |
| semantic_topk | 0.13 | 0.62 | 0.93 | 1.00 | 1048 |
| hierarchical | 0.52 | 0.70 | 0.95 | 1.00 | 1050 |
| hybrid | 0.45 | 0.92 | 1.00 | 1.00 | 1048 |

## by difficulty — catalog size 300

**single**

| strategy | recall@1 | recall@3 | recall@5 | recall@10 | tokens(p95) |
|---|---|---|---|---|---|
| passthrough | 1.00 | 1.00 | 1.00 | 1.00 | 31084 |
| semantic_topk | 0.16 | 0.70 | 1.00 | 1.00 | 1048 |
| hierarchical | 0.57 | 0.76 | 1.00 | 1.00 | 1050 |
| hybrid | 0.54 | 1.00 | 1.00 | 1.00 | 1048 |

**multi**

| strategy | recall@1 | recall@3 | recall@5 | recall@10 | tokens(p95) |
|---|---|---|---|---|---|
| passthrough | 1.00 | 1.00 | 1.00 | 1.00 | 31084 |
| semantic_topk | 0.08 | 0.48 | 0.72 | 0.99 | 1048 |
| hierarchical | 0.34 | 0.56 | 0.83 | 0.99 | 1050 |
| hybrid | 0.43 | 0.92 | 1.00 | 1.00 | 1048 |

**ambiguous**

| strategy | recall@1 | recall@3 | recall@5 | recall@10 | tokens(p95) |
|---|---|---|---|---|---|
| passthrough | 1.00 | 1.00 | 1.00 | 1.00 | 31084 |
| semantic_topk | 0.08 | 0.54 | 1.00 | 1.00 | 1046 |
| hierarchical | 0.62 | 0.71 | 1.00 | 1.00 | 1050 |
| hybrid | 0.08 | 0.54 | 1.00 | 1.00 | 1046 |

## task-success (SECONDARY, weak offline agent)

phi(recall_hit, task_success) = **0.2887** — low magnitude means task_success is a genuinely separate signal from recall, not a re-labeling. Absolute task-success values are low because the offline agent is a deliberately weak Jaccard proxy that is NOT told how many tools to pick; only the ordering and the recall→ceiling relationship are meaningful. `task_success`/`hit_rate` per cell are in `summary.json`.

## Recall cliff

Detected **464** cliff events (gold exists but ranked past the exposed top-k under semantic routing). See `traces.jsonl`.

Example traces:

- q#1 [single] semantic_topk@100 k=1: gold ranks [2] · task_success=False · `1d854d0bc44f8e5b`
- q#3 [single] semantic_topk@100 k=1: gold ranks [2] · task_success=False · `f829f4221f264fa5`
- q#8 [multi] semantic_topk@100 k=1: gold ranks [2, 3] · task_success=False · `bcba5be8aba1d974`
- q#10 [single] semantic_topk@100 k=1: gold ranks [2] · task_success=False · `7a8a8fb7150030dd`
- q#11 [single] semantic_topk@100 k=1: gold ranks [2] · task_success=False · `41e621da8e0d6c30`
- q#17 [single] semantic_topk@100 k=1: gold ranks [2] · task_success=False · `71af26bc0751f212`

## McNemar (recall_hit, paired by query, BH-corrected)

Computed on **recall_hit** (the routing question), not the weak task_success. `b` = A hit / B miss; `c` = B hit / A miss. `q` = Benjamini-Hochberg adjusted p. The hierarchical-vs-hybrid pair can be two-sided (keyword collisions let hybrid lose recall) — read b and c from the table below for this run.

| size | k | A | B | b | c | χ² | p | q |
|---|---|---|---|---|---|---|---|---|
| 100 | 1 | semantic_topk | hybrid | 0 | 31 | 29.0323 | 7.12e-08 | 3.66e-07 |
| 100 | 1 | semantic_topk | hierarchical | 0 | 16 | 14.0625 | 0.0002 | 0.0006 |
| 100 | 1 | hierarchical | hybrid | 3 | 18 | 9.3333 | 0.0023 | 0.0062 |
| 100 | 3 | semantic_topk | hybrid | 0 | 2 | 0.5 | 0.4795 | 0.7193 |
| 100 | 3 | semantic_topk | hierarchical | 1 | 1 | 0.5 | 0.4795 | 0.7193 |
| 100 | 3 | hierarchical | hybrid | 0 | 2 | 0.5 | 0.4795 | 0.7193 |
| 200 | 1 | semantic_topk | hybrid | 0 | 53 | 51.0189 | 9.15e-13 | 8.23e-12 |
| 200 | 1 | semantic_topk | hierarchical | 0 | 55 | 53.0182 | 3.30e-13 | 3.97e-12 |
| 200 | 1 | hierarchical | hybrid | 29 | 27 | 0.0179 | 0.8937 | 1.0000 |
| 200 | 3 | semantic_topk | hybrid | 0 | 19 | 17.0526 | 3.64e-05 | 0.0002 |
| 200 | 3 | semantic_topk | hierarchical | 0 | 3 | 1.3333 | 0.2482 | 0.4703 |
| 200 | 3 | hierarchical | hybrid | 0 | 16 | 14.0625 | 0.0002 | 0.0006 |
| 300 | 1 | semantic_topk | hybrid | 0 | 56 | 54.0179 | 1.99e-13 | 3.58e-12 |
| 300 | 1 | semantic_topk | hierarchical | 0 | 69 | 67.0145 | 2.70e-16 | 9.70e-15 |
| 300 | 1 | hierarchical | hybrid | 34 | 21 | 2.6182 | 0.1056 | 0.2377 |
| 300 | 3 | semantic_topk | hybrid | 0 | 52 | 50.0192 | 1.52e-12 | 1.10e-11 |
| 300 | 3 | semantic_topk | hierarchical | 0 | 13 | 11.0769 | 0.0009 | 0.0026 |
| 300 | 3 | hierarchical | hybrid | 5 | 44 | 29.4694 | 5.68e-08 | 3.41e-07 |

![recall cliff](recall_cliff.svg)
