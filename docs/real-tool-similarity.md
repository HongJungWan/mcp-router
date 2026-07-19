# Real MCP tools vs. the synthetic catalog — pairwise similarity

Embedding model: `BAAI/bge-small-en-v1.5`. Real corpus: **247 tools / 22 servers** (aws-kb, brave-search, everart, fetch, filesystem, firecrawl, gdrive, git, github, gitlab, google-maps, memory, mongodb, playwright, postgres, puppeteer, sentry, sequentialthinking, slack, sqlite, tavily, time), harvested from public MCP servers (modelcontextprotocol reference/archived + firecrawl/tavily/playwright/mongodb). Synthetic: `build_catalog(300)`.

**The question:** do real tool catalogs actually contain the near-duplicate crowding the benchmark assumes? The nearest-neighbour cosine per tool is the load-bearing number — a high value means the tool has a near-twin that can displace it from a small top-k.

| nearest-neighbour cosine | real MCP tools | synthetic catalog |
|---|---|---|
| mean | 0.833 | 0.928 |
| median | 0.841 | 0.941 |
| p90 | 0.91 | 0.971 |
| max | 0.955 | 0.99 |
| fraction with NN > 0.80 | 0.7 | 0.955 |
| fraction with NN > 0.90 | 0.13 | 0.842 |

Real, within-server mean cosine = **0.637**, cross-server = **0.514** (tools from the same server are the near-duplicates, as expected).

**Top real near-duplicate pairs (bge cosine):**

| cosine | tool A | tool B | server |
|---|---|---|---|
| 0.955 | aggregate | aggregate-db | mongodb |
| 0.948 | maps_geocode | maps_reverse_geocode | google-maps |
| 0.947 | atlas-create-access-list | atlas-inspect-access-list | mongodb |
| 0.946 | browser_verify_element_visible | browser_verify_text_visible | playwright |
| 0.943 | delete_entities | delete_relations | memory |
| 0.935 | browser_mouse_drag_xy | browser_mouse_move_xy | playwright |
| 0.930 | get_pull_request_comments | get_pull_request_reviews | github |
| 0.927 | browser_verify_element_visible | browser_verify_list_visible | playwright |
| 0.926 | list_directory | list_directory_with_sizes | filesystem |
| 0.923 | atlas-local-connect-deployment | atlas-local-create-deployment | mongodb |
| 0.923 | browser_mouse_click_xy | browser_mouse_move_xy | playwright |
| 0.922 | browser_set_storage_state | browser_storage_state | playwright |

## Reading

Real MCP tools carry genuine near-duplicates: median nearest-neighbour cosine 0.841, with 70% of tools having a neighbour above 0.80. So the crowding the benchmark studies is NOT a synthetic invention — it exists in real catalogs, concentrated within a server (within 0.637 vs cross 0.514).

Versus the size-matched synthetic catalog, the real corpus is **milder than** it on median NN cosine (real 0.841 vs synthetic 0.941). Read this as **directional, not a controlled magnitude**: the synthetic NN level is partly an output of the crowding knobs (core_share, kw_collision), and the synthetic embed_text is templated (filler + numbered fake-protocol suffixes) which inflates its bge cosine relative to the real tools' natural-language descriptions — so the two NN columns are not strictly apples-to-apples.

Two more caveats. (1) NN cosine is a tool↔tool geometry metric; recall@k depends on query↔tool ranking, so a near-duplicate is closer to a *necessary* than a *sufficient* condition for the cliff. (2) 247 tools / 22 servers is a real but non-exhaustive sample — an anchor, not a population estimate.