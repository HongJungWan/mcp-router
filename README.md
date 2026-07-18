# MCP tool-routing recall benchmark

A benchmark harness that asks one question and then **checks its own answer
against a real embedding model**:

> When an agent's MCP tool catalog grows to hundreds of near-duplicate tools,
> does exposing only a semantic top-k of them silently drop the right tool — and
> is a hybrid (vector + lexical) router actually needed to fix it?

**The honest, calibrated answer (this is the interesting part):**

- Under a **naive lexical-overlap similarity** (a hashed bag-of-words mock), the
  cliff is dramatic — recall@1 falls **0.70 → 0.13** as the catalog grows
  100→300, and hybrid routing looks essential.
- Under a **real dense embedding (`bge-small-en-v1.5`)**, the *same catalog and
  queries* produce a **much milder** cliff — recall@1 falls only **0.89 → 0.70**,
  and hybrid's advantage shrinks to **marginal** (recall@3 0.90 → 0.93). A good
  embedding already "sees through" most of the near-duplicate crowding.

So the dramatic version of the cliff is largely a property of *weak, lexical*
similarity, **not** an inevitability of large catalogs. Running the real model
partially **falsified the mock headline** — and that's the point: the benchmark
tells you *which regime you're in* rather than selling a single scary number.

> Status: **M3 — an offline evaluation harness** (not a running gateway; no
> serving/protocol/RBAC code — that's roadmap). The mock path is pure-stdlib and
> byte-for-byte reproducible; the `bge-small` numbers were produced by actually
> running the model (`--embed local`) and are float-reproducible, not bit-exact.

---

## Results: mock (BoW) vs real (bge-small)

**The cliff — semantic_topk fractional recall@1 across the staircase:**

| catalog size | mock BoW | bge-small (real) |
|---|---|---|
| 100 | 0.70 | 0.89 |
| 200 | 0.28 | 0.78 |
| 300 | **0.13** | **0.70** |
| drop 100→300 | **−0.58** | **−0.19** |

**Strategies at catalog=300 (fractional recall@k):**

| strategy | mock r@1 / r@3 | **bge r@1 / r@3** | tokens@3 |
|---|---|---|---|
| passthrough (expose all) | 1.00 / 1.00 | 1.00 / 1.00 | ~31,000 |
| semantic_topk | 0.13 / 0.62 | **0.70 / 0.90** | ~316 |
| hierarchical | 0.52 / 0.70 | 0.71 / 0.91 | ~316 |
| hybrid | 0.45 / 0.92 | 0.70 / **0.93** | ~316 |

Takeaways a reviewer can trust:
- The cliff **direction** is real under both (recall drops as the catalog grows;
  hybrid still helps at higher k: bge recall@5 0.92→0.97, recall@10 0.96→1.00).
- The cliff **magnitude** and hybrid's necessity are **much smaller with a real
  embedding** — on this corpus, plain semantic top-k at k≥3 is already ~0.90.
- Exposing *everything* costs ~100× the tokens (≈31k vs ≈316 for k=3) for a recall
  gain that a decent embedding at k=3–5 nearly matches. That token/recall trade,
  not "hybrid magic", is the durable finding.

![mock vs real cliff](docs/recall_cliff.svg)

Committed evidence in [`docs/`](docs/): `recall_cliff.svg` (mock),
`recall_cliff_bge.svg` (real), and full `sample-report.md` / `real-embed-report.md`
+ CSVs. Regenerate: `make bench` (mock) and `make bench-real` (bge-small, needs
`pip install .[local]`).

---

## Why the cliff exists, and its limits

- **Nested staircase**: `catalog(100) ⊂ catalog(200) ⊂ catalog(300)`; the same
  gold tools exist at every size, only the near-duplicate distractor pool grows.
- Distractors are built by sampling the query's own tokens (`synth.py`), so under
  a lexical similarity they crowd the gold tool out of a small top-k. A dense
  embedding captures meaning beyond token overlap, which is exactly why it
  resists the effect — as the bge run shows.
- **Robustness checks** (that the *mock* cliff isn't a single-constant artifact):
  `make sweep` varies `core_share` 6→10 (cliff-drop stays −0.32…−0.57); a
  char-trigram embedder (`--embed mock_char`) is a subword-smoothed variant that
  still shows it. **Neither proves generalization** — the bge run is the real
  external check, and it says: milder than the mock implies.

### When routing strategy actually matters
Hybrid/hierarchical routing earns its keep when your similarity signal is weak,
your tools are lexically near-duplicate, or k is very small. With a strong
embedding and a moderate catalog, plain semantic top-k at k≥3 is often enough.
This harness exists to measure which case you're in — with numbers, on your data.

---

## What is real vs mock

| concern | offline default (mock numbers) | real path (bge numbers here) |
|---|---|---|
| embeddings | hashed bag-of-words / char-trigram | **bge-small-en-v1.5 (run)** |
| agent / task-success | weak decoupled Jaccard mock | Claude tool-use (`.[claude]`, unrun) |
| vector index | pure-Python exact cosine | same (pgvector intentionally not built) |

- **Task-success is a weak SECONDARY signal.** A deliberately weak Jaccard agent,
  NOT told how many tools to pick, selects from the exposed set. We report
  **phi(recall_hit, task_success) ≈ 0.27** so you can see it is *decoupled* from
  recall, not a re-labeling of it. Absolute task-success values are low by design;
  only the ordering matters. The routing significance test (McNemar, BH-corrected)
  is computed on **recall_hit**, not task-success.
- **No human labels exist.** The label report is auto-labeler *self-consistency*
  vs synthesized ground truth (κ ≈ 0.42) — a mechanism sanity check, **not**
  human-verified label quality.
- Token cost is a JSON-schema chars/4 **heuristic**; the ~100× ratio (3-of-N tools)
  is robust to the constant, the absolute count is not.

### Deliberately NOT built (scope discipline, not gaps)
pgvector/HNSW (exact brute-force is faster/exact for a few-hundred vectors),
docker-compose, a separate LangGraph agent (the Claude adapter already does
tool-use), an OpenTelemetry emitter (JSONL suffices for a single-process bench),
and a latency pipeline (in-process timings without load are meaningless). These
were removed rather than shipped unused.

## Quickstart

```bash
make bench      # offline, pure-stdlib, byte-identical across runs
make test       # 21 unittest cases (cliff, agent-decoupling, geometry, report e2e)
make sweep      # core_share sensitivity
make bench-real # bge-small embeddings (pip install .[local]); float-reproducible
```

## Methodology notes
Fractional recall@k (primary) + hit-rate (secondary); per-difficulty
stratification; **gold-cluster bootstrap** CIs (queries cluster over 30 gold
tools, so an iid bootstrap would understate them); McNemar on recall_hit with
Benjamini-Hochberg correction (hierarchical-vs-hybrid is genuinely two-sided —
b=34/c=21, p=0.11 at k=1). Every knob is stamped into `summary.json`.

## Roadmap
Real MCP server corpus (harvest 20-30 real tool defs; measure their actual
pairwise similarity — the true external-validity anchor); a running gateway
(federation/RBAC/circuit-breakers); Claude tool-use agent run with committed
cassettes; a real tokenizer for token cost.

## Layout
```
src/mcp_router/
  catalog/     staircase catalog (keyword-collision distractors) + query gen + Spec
  routing/     four strategies + RoutingContext
  providers/   embed (mock / mock_char / bge-small) + llm (mock jaccard / claude)
  vectorstore/ pure-Python exact cosine index
  labeling/    self-consistency labeler + Cohen's kappa
  bench/       decoupled agent, metrics (fractional recall, cluster bootstrap,
               McNemar+BH, phi), runner, report
  tracing.py · cli.py (bench run | bench sweep)
tests/         21 unittest cases
```
