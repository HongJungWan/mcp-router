# MCP Gateway — tool-routing benchmark (M3)

**Reproducibility envelope**

- git_sha: `64e61db` · seed: `1234` · embed: `mock-bow-hash-v1` · llm: `mock-jaccard-agent-v1` · vector: `memory`
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

## Recall cliff

Detected **994** cliff events (gold exists but ranked past the exposed top-k under semantic routing). See `traces.jsonl`.

Example traces:

- q#1 [single] semantic_topk@100 k=1: gold ranks [2] · task_success=False · `1d854d0bc44f8e5b`
- q#1 [single] hierarchical@100 k=1: gold ranks [2] · task_success=False · `03baa2ea3c81bc33`
- q#3 [single] semantic_topk@100 k=1: gold ranks [2] · task_success=False · `f829f4221f264fa5`
- q#8 [multi] semantic_topk@100 k=1: gold ranks [2, 3] · task_success=False · `bcba5be8aba1d974`
- q#8 [multi] hierarchical@100 k=1: gold ranks [2, 3] · task_success=False · `ff19ee495b9020b5`
- q#8 [multi] hybrid@100 k=1: gold ranks [2, 3] · task_success=False · `28363af73f3f6ee0`

## McNemar (task success, paired by query, BH-corrected)

`b` = semantic wins / other loses; `c` = other wins / semantic loses. `q` = Benjamini-Hochberg adjusted p across all comparisons.

| size | k | A | B | b | c | χ² | p | q |
|---|---|---|---|---|---|---|---|---|
| 100 | 1 | semantic_topk | hybrid | 0 | 31 | 29.0323 | 7.12e-08 | 1.55e-07 |
| 100 | 1 | semantic_topk | hierarchical | 0 | 16 | 14.0625 | 0.0002 | 0.0003 |
| 100 | 3 | semantic_topk | hybrid | 0 | 0 | 0.0 | 1.0000 | 1.0000 |
| 100 | 3 | semantic_topk | hierarchical | 1 | 17 | 12.5 | 0.0004 | 0.0006 |
| 200 | 1 | semantic_topk | hybrid | 0 | 53 | 51.0189 | 9.15e-13 | 2.20e-12 |
| 200 | 1 | semantic_topk | hierarchical | 0 | 55 | 53.0182 | 3.30e-13 | 8.81e-13 |
| 200 | 3 | semantic_topk | hybrid | 0 | 3 | 1.3333 | 0.2482 | 0.3504 |
| 200 | 3 | semantic_topk | hierarchical | 0 | 56 | 54.0179 | 1.99e-13 | 5.96e-13 |
| 300 | 1 | semantic_topk | hybrid | 0 | 56 | 54.0179 | 1.99e-13 | 5.96e-13 |
| 300 | 1 | semantic_topk | hierarchical | 0 | 69 | 67.0145 | 2.70e-16 | 1.62e-15 |
| 300 | 3 | semantic_topk | hybrid | 0 | 19 | 17.0526 | 3.64e-05 | 7.27e-05 |
| 300 | 3 | semantic_topk | hierarchical | 0 | 81 | 79.0123 | 6.17e-19 | 4.94e-18 |

## Latency (non-deterministic indicator — NOT a headline metric)

In-process, single-call routing latency with **no load generation**; varies run-to-run and is not part of the reproducibility guarantee. Real numbers need a load harness (roadmap). p50/p95 per cell are in `results.csv`.

![recall cliff](recall_cliff.svg)
