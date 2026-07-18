# MCP Gateway — tool-routing benchmark (M3)

**Reproducibility envelope**

- git_sha: `unknown`  · seed: `1234`  · embed: `mock-bow-hash-v1`  · llm: `mock-react-v1`  · vector: `memory`
- queries: 180  · catalog staircase: [100, 200, 300]  · k sweep: [1, 3, 5, 10]
- bootstrap resamples: 1000

## Label quality (human-verified subset)

- human-verified queries: 50  · labeled pairs: 261  · disagreements: 10
- **Cohen's κ = 0.8863**  · raw agreement = 0.9617  · labeler = `mock-react-v1`

## recall@k — catalog size 100

| strategy | recall@1 | recall@3 | recall@5 | recall@10 | tokens(p95) |
|---|---|---|---|---|---|
| passthrough | 1.00 | 1.00 | 1.00 | 1.00 | 8600 |
| semantic_topk | 0.56 | 0.98 | 1.00 | 1.00 | 860 |
| hierarchical | 0.66 | 0.97 | 0.99 | 0.99 | 860 |
| hybrid | 0.85 | 1.00 | 1.00 | 1.00 | 860 |

## recall@k — catalog size 200

| strategy | recall@1 | recall@3 | recall@5 | recall@10 | tokens(p95) |
|---|---|---|---|---|---|
| passthrough | 1.00 | 1.00 | 1.00 | 1.00 | 17200 |
| semantic_topk | 0.22 | 0.83 | 0.96 | 1.00 | 860 |
| hierarchical | 0.53 | 0.84 | 0.97 | 0.99 | 860 |
| hybrid | 0.78 | 0.98 | 1.00 | 1.00 | 860 |

## recall@k — catalog size 300

| strategy | recall@1 | recall@3 | recall@5 | recall@10 | tokens(p95) |
|---|---|---|---|---|---|
| passthrough | 1.00 | 1.00 | 1.00 | 1.00 | 25800 |
| semantic_topk | 0.17 | 0.60 | 0.84 | 0.99 | 860 |
| hierarchical | 0.52 | 0.66 | 0.86 | 0.99 | 860 |
| hybrid | 0.77 | 0.93 | 0.98 | 1.00 | 860 |

## Recall cliff

Detected **1002** cliff events (gold tool exists but was ranked past the exposed top-k under semantic routing). See `traces.jsonl`.

Example cliff traces:

- q#1 [single] semantic_topk@100 k=1: gold ranks [2] (dropped past k) · task_success=False · trace `1d854d0bc44f8e5b`
- q#1 [single] hierarchical@100 k=1: gold ranks [2] (dropped past k) · task_success=False · trace `03baa2ea3c81bc33`
- q#4 [single] semantic_topk@100 k=1: gold ranks [2] (dropped past k) · task_success=False · trace `0d4a6cc36cc7bbbf`
- q#8 [multi] semantic_topk@100 k=1: gold ranks [2, 3] (dropped past k) · task_success=False · trace `bcba5be8aba1d974`
- q#8 [multi] hierarchical@100 k=1: gold ranks [2, 3] (dropped past k) · task_success=False · trace `ff19ee495b9020b5`
- q#8 [multi] hybrid@100 k=1: gold ranks [2, 3] (dropped past k) · task_success=False · trace `28363af73f3f6ee0`

## McNemar (task success, paired by query)

A beats B on `c` queries where B fails and A succeeds; `b` is the reverse.

| size | k | A | B | b | c | χ² | p |
|---|---|---|---|---|---|---|---|
| 100 | 1 | semantic_topk | hybrid | 0 | 53 | 51.0189 | 0.0 |
| 100 | 1 | semantic_topk | hierarchical | 0 | 18 | 16.0556 | 6e-05 |
| 100 | 3 | semantic_topk | hybrid | 0 | 4 | 2.25 | 0.13361 |
| 100 | 3 | semantic_topk | hierarchical | 1 | 5 | 1.5 | 0.22067 |
| 200 | 1 | semantic_topk | hybrid | 0 | 100 | 98.01 | 0.0 |
| 200 | 1 | semantic_topk | hierarchical | 0 | 56 | 54.0179 | 0.0 |
| 200 | 3 | semantic_topk | hybrid | 0 | 27 | 25.037 | 0.0 |
| 200 | 3 | semantic_topk | hierarchical | 0 | 14 | 12.0714 | 0.00051 |
| 300 | 1 | semantic_topk | hybrid | 0 | 109 | 107.0092 | 0.0 |
| 300 | 1 | semantic_topk | hierarchical | 0 | 63 | 61.0159 | 0.0 |
| 300 | 3 | semantic_topk | hybrid | 0 | 60 | 58.0167 | 0.0 |
| 300 | 3 | semantic_topk | hierarchical | 0 | 20 | 18.05 | 2e-05 |

![recall cliff](recall_cliff.svg)
