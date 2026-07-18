# Real MCP tools vs. the synthetic catalog — pairwise similarity

Embedding model: `BAAI/bge-small-en-v1.5`. Real corpus: **70 tools / 6 servers** (filesystem, github, slack, git, memory, gdrive), harvested from public modelcontextprotocol servers. Synthetic: `build_catalog(300)`.

**The question:** do real tool catalogs actually contain the near-duplicate crowding the benchmark assumes? The nearest-neighbour cosine per tool is the load-bearing number — a high value means the tool has a near-twin that can displace it from a small top-k.

| nearest-neighbour cosine | real MCP tools | synthetic catalog |
|---|---|---|
| mean | 0.823 | 0.934 |
| median | 0.831 | 0.947 |
| p90 | 0.902 | 0.971 |
| max | 0.943 | 0.99 |
| fraction with NN > 0.80 | 0.671 | 0.963 |
| fraction with NN > 0.90 | 0.114 | 0.88 |

Real, within-server mean cosine = **0.672**, cross-server = **0.524** (tools from the same server are the near-duplicates, as expected).

**Top real near-duplicate pairs (bge cosine):**

| cosine | tool A | tool B | server |
|---|---|---|---|
| 0.943 | delete_entities | delete_relations | memory |
| 0.930 | get_pull_request_comments | get_pull_request_reviews | github |
| 0.926 | list_directory | list_directory_with_sizes | filesystem |
| 0.902 | create_issue | update_issue | github |
| 0.900 | search_repositories | search_code | github |
| 0.897 | create_pull_request_review | get_pull_request_reviews | github |
| 0.893 | delete_entities | delete_observations | memory |
| 0.891 | git_diff_unstaged | git_diff_staged | git |
| 0.883 | create_pull_request_review | get_pull_request_comments | github |
| 0.876 | slack_reply_to_thread | slack_get_thread_replies | slack |
| 0.875 | create_repository | create_branch | github |
| 0.864 | delete_observations | delete_relations | memory |

## Reading

Real MCP tools carry genuine near-duplicates: median nearest-neighbour cosine 0.831, with 67% of tools having a neighbour above 0.80. So the crowding the benchmark studies is NOT a synthetic invention — it exists in real catalogs, concentrated within a server (within 0.672 vs cross 0.524).

Versus the synthetic catalog, the real corpus is **milder than** the synthetic one on median NN cosine (real 0.831 vs synthetic 0.947). Caveat: 70 tools / 6 servers is a modest sample, and a real deployment mixes many more servers — this is an anchor, not a population estimate.