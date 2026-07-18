# MCP 툴 라우팅 recall 벤치마크

에이전트가 쓰는 MCP 툴이 수백 개로 불어나면, 그중 semantic top-k만 노출하는 라우팅이
정작 필요한 툴을 조용히 떨어뜨리는가? 그리고 그걸 막으려고 hybrid(벡터+렉시컬) 라우터까지
정말 필요한가?

이 저장소는 그 질문을 합성 카탈로그로 재현한 다음, **실제 임베딩 모델로 자기 답을 검증한다.**

결론부터 말하면 이렇다.

- 토큰 중첩만 보는 약한 유사도(bag-of-words mock)에서는 절벽이 가파르다. 카탈로그가
  100→300으로 커지는 동안 recall@1이 **0.70 → 0.13**까지 무너지고, hybrid 라우팅이 필수처럼 보인다.
- 그런데 **같은 카탈로그·같은 쿼리**를 실제 dense 임베딩(`bge-small-en-v1.5`)으로 돌리면 절벽이
  훨씬 완만하다. recall@1이 **0.89 → 0.70**로만 내려가고, hybrid의 이점도 k=3 기준 0.90 → 0.93로
  거의 사라진다.

즉 가파른 절벽은 대체로 *약한 렉시컬 유사도*의 성질이지, 카탈로그가 크다고 반드시 생기는 게
아니다. 실제 모델을 돌려보니 처음의 자극적인 헤드라인이 **부분적으로 뒤집혔다.** 이 벤치의
목적은 겁주는 숫자 하나를 파는 게 아니라, 지금 내가 어느 쪽 상황에 있는지를 데이터로 알려주는
것이다.

> 상태: **M3 — 오프라인 평가 하네스**다. 돌아가는 게이트웨이가 아니다(서빙·프로토콜·RBAC 코드는
> 없고 로드맵). mock 경로는 순수 stdlib이고 같은 seed면 바이트 단위로 재현된다. `bge-small`
> 수치는 모델을 실제로 돌려서(`--embed local`) 얻었고, float 수준에서 재현된다(비트 단위는 아님).

---

## 결과: mock(BoW) vs 실제(bge-small)

**절벽 — semantic_topk 의 fractional recall@1, 카탈로그 크기별:**

| 카탈로그 | mock BoW | bge-small (실제) |
|---|---|---|
| 100 | 0.70 | 0.89 |
| 200 | 0.28 | 0.78 |
| 300 | **0.13** | **0.70** |
| 100→300 낙폭 | **−0.58** | **−0.19** |

**카탈로그 300에서 전략별 recall@k:**

| 전략 | mock r@1 / r@3 | **bge r@1 / r@3** | tokens@3 |
|---|---|---|---|
| passthrough (전부 노출) | 1.00 / 1.00 | 1.00 / 1.00 | ~31,000 |
| semantic_topk | 0.13 / 0.62 | **0.70 / 0.90** | ~316 |
| hierarchical | 0.52 / 0.70 | 0.71 / 0.91 | ~316 |
| hybrid | 0.45 / 0.92 | 0.70 / **0.93** | ~316 |

읽는 사람이 믿어도 되는 지점만 추리면:

- 방향은 두 경우 다 맞다. 카탈로그가 커지면 recall이 떨어지고, hybrid는 큰 k에서 여전히 도움이
  된다(bge 기준 recall@5 0.92→0.97, recall@10 0.96→1.00).
- 낙폭의 크기와 "hybrid가 꼭 필요한가"는 실제 임베딩에서 훨씬 작다. 이 코퍼스에선 semantic
  top-k도 k≥3이면 이미 ~0.90이다.
- 전부 노출하면 토큰이 약 100배 든다(k=3에서 ~31k vs ~316). 그 대가로 얻는 recall은 괜찮은
  임베딩을 k=3~5로 쓰면 거의 따라잡힌다. "hybrid 마법"이 아니라 이 **토큰/recall 트레이드오프**가
  오래 남는 발견이다.

![mock vs 실제 절벽](docs/recall_cliff.svg)

`docs/`에 증거를 커밋해 뒀다 — `recall_cliff.svg`(mock), `recall_cliff_bge.svg`(실제), 그리고
전체 리포트/CSV. 직접 재현하려면 `make bench`(mock), `make bench-real`(bge-small,
`pip install .[local]` 필요).

---

## 시스템 구조

```mermaid
%%{init: {"theme":"base","themeVariables":{"fontFamily":"Pretendard, Segoe UI, sans-serif","primaryColor":"#F0EEE6","primaryTextColor":"#29261B","primaryBorderColor":"#C6714F","lineColor":"#B5654A","clusterBkg":"#FAF9F5","clusterBorder":"#DED7C9"}}}%%
flowchart TB
    CLI["CLI · mcp_router bench run | sweep"]

    subgraph GEN["데이터 생성 (합성·결정적)"]
        CAT["Catalog<br/>계단식 카탈로그 100/200/300<br/>키워드 충돌 distractor"]
        QRY["Query 생성<br/>single · multi · ambiguous"]
    end

    subgraph CORE["라우팅 코어"]
        EMB["Embedder<br/>mock BoW · mock_char · bge-small"]
        IDX["VectorIndex<br/>순수 파이썬 exact cosine"]
        RTR["Router (4 전략)<br/>passthrough · semantic_topk<br/>hierarchical · hybrid"]
    end

    subgraph EVAL["평가"]
        AGT["ReAct Agent<br/>노출된 툴만 보고 선택<br/>(Jaccard mock / Claude)"]
        LBL["Labeler<br/>self-consistency κ"]
        MET["Metrics<br/>fractional recall · cluster bootstrap<br/>McNemar+BH · phi"]
    end

    OUT["Report<br/>report.md · results.csv<br/>summary.json · recall_cliff.svg · traces.jsonl"]

    CLI --> GEN --> CORE
    EMB --> IDX --> RTR
    RTR --> AGT --> MET
    CAT -.-> LBL --> MET
    RTR --> OUT
    MET --> OUT

    classDef gen fill:#EADFCB,stroke:#CBB897,color:#29261B;
    classDef core fill:#D9A883,stroke:#BE7C57,color:#2B2013;
    classDef eval fill:#C6714F,stroke:#A8543A,color:#FCF9F3;
    class CAT,QRY gen;
    class EMB,IDX,RTR core;
    class AGT,LBL,MET eval;
```

## 벤치 프로세스

```mermaid
%%{init: {"theme":"base","themeVariables":{"fontFamily":"Pretendard, Segoe UI, sans-serif","primaryColor":"#F0EEE6","primaryTextColor":"#29261B","primaryBorderColor":"#C6714F","lineColor":"#B5654A"}}}%%
flowchart LR
    S(["카탈로그 크기<br/>100 → 200 → 300"]) --> Q["쿼리마다"]
    Q --> R["4개 전략 × k=1,3,5,10<br/>로 라우팅"]
    R --> E["노출된 top-k 집합"]
    E --> RC{"gold 툴이<br/>노출됐나?"}
    RC -->|"예"| REC["recall@k 기록"]
    RC -->|"아니오 (절벽)"| TR["cliff 트레이스<br/>gold rank &gt; k 기록"]
    E --> AG["에이전트가<br/>노출 집합에서 선택"]
    AG --> TS["task_success<br/>(phi로 recall과 독립성 확인)"]
    REC --> AGG["집계<br/>난이도 층화 · cluster bootstrap"]
    TS --> AGG
    AGG --> MC["McNemar (recall_hit) + BH 보정"]
    MC --> RPT(["report.md · summary.json · svg"])
    TR --> RPT

    classDef n fill:#F6F3EA,stroke:#DED7C9,color:#29261B;
    classDef hot fill:#C6714F,stroke:#A8543A,color:#FCF9F3;
    class Q,R,E,REC,AG,TS,AGG,MC n;
    class TR hot;
```

---

## 절벽이 왜 생기고, 어디까지 믿을 수 있나

- **계단식 카탈로그**: `catalog(100) ⊂ catalog(200) ⊂ catalog(300)`. gold 툴은 어느 크기에서나
  그대로 있고, 근접 중복 distractor 풀만 커진다.
- distractor는 쿼리의 토큰을 그대로 샘플링해서 만든다(`synth.py`). 그래서 렉시컬 유사도에서는
  이들이 gold를 작은 top-k 밖으로 밀어낸다. dense 임베딩은 토큰 중첩 너머의 의미를 잡기 때문에
  이 효과에 강한데, bge 실행이 정확히 그걸 보여준다.
- **강건성 점검**(mock 절벽이 상수 하나의 산물이 아님을 확인): `make sweep`으로 `core_share`를
  6→10까지 흔들어도 낙폭이 −0.32~−0.57로 유지되고, char-trigram 임베더(`--embed mock_char`)에서도
  나타난다. 다만 이건 **일반화 증명이 아니다.** 진짜 외부 검증은 bge 실행이고, 그 답은 "mock이
  말하는 것보다 완만하다"였다.

### 라우팅 전략이 실제로 중요해지는 때
hybrid·hierarchical은 유사도 신호가 약하거나, 툴들이 렉시컬로 거의 판박이거나, k가 아주 작을
때 값을 한다. 임베딩이 좋고 카탈로그가 적당하면 semantic top-k를 k≥3으로 쓰는 걸로 충분한 경우가
많다. 이 하네스는 그중 어느 경우인지를 내 데이터 위에서 숫자로 재보라고 있는 것이다.

## 무엇이 실제고 무엇이 mock인가

| 항목 | 오프라인 기본 (mock 수치) | 실제 경로 (여기 bge 수치) |
|---|---|---|
| 임베딩 | hashed bag-of-words / char-trigram | **bge-small-en-v1.5 (실행함)** |
| 에이전트 / task_success | 일부러 약하게 만든 Jaccard mock | Claude tool-use (`.[claude]`, 미실행) |
| 벡터 인덱스 | 순수 파이썬 exact cosine | 동일 (pgvector는 일부러 안 만듦) |

- **task_success는 약한 보조 지표다.** 정답 개수를 알려주지 않은 약한 Jaccard 에이전트가 노출
  집합에서 고른다. recall과 얼마나 독립인지 보라고 **phi(recall_hit, task_success) ≈ 0.27**을 같이
  적어 둔다(낮으면 recall을 이름만 바꾼 게 아니라는 뜻). 절대값 자체는 낮게 설계돼 있고 순서만
  의미가 있다. 유의성 검정(McNemar, BH 보정)은 task_success가 아니라 **recall_hit** 위에서 돌린다.
- **사람이 붙인 라벨은 없다.** 라벨 리포트는 자동 라벨러가 합성 정답을 얼마나 되찾는지의
  self-consistency(κ ≈ 0.42)일 뿐, 사람 검증 라벨 품질이 아니다.
- 토큰 비용은 JSON 스키마 chars/4 **휴리스틱**이다. ~100배 비율(3-of-N)은 상수에 강건하지만 절대
  토큰 수는 아니다.

### 일부러 안 만든 것 (스코프 규율)
pgvector/HNSW(수백 개 벡터엔 exact brute-force가 더 빠르고 정확), docker-compose, 별도 LangGraph
에이전트(Claude 어댑터가 이미 tool-use를 함), OpenTelemetry 익스포터(단일 프로세스 벤치엔 JSONL로
충분), latency 파이프라인(부하 없는 in-process 계측은 의미 없음). 안 쓸 걸 만들어 두는 대신 뺐다.

## 실행

```bash
make bench      # 오프라인·순수 stdlib·같은 seed면 바이트 단위 재현
make test       # unittest 21케이스 (절벽·에이전트 분리·기하·리포트 e2e)
make sweep      # core_share 민감도
make bench-real # bge-small 임베딩 (pip install .[local]); float 재현
```

## 방법론 메모
primary는 fractional recall@k, secondary는 hit-rate. 난이도(single/multi/ambiguous)별로 나눠 본다.
CI는 **gold-cluster bootstrap**이다(쿼리 180개가 gold 툴 30개 위에 뭉쳐 있어서 iid 부트스트랩은
CI를 과소추정한다). McNemar는 recall_hit 위에서 돌리고 Benjamini-Hochberg로 다중비교를 보정한다
(hierarchical vs hybrid는 실제로 양방향이다 — k=1에서 b=34/c=21, p=0.11). 모든 knob은
`summary.json`에 박힌다.

## 로드맵
실제 MCP 서버 코퍼스(공개 서버 20~30개 툴 정의를 덤프해 실제 pairwise 유사도 분포를 재는 것 —
합성 코퍼스의 진짜 외적 타당성 앵커), 돌아가는 게이트웨이(federation·RBAC·서킷브레이커), cassette를
커밋한 Claude tool-use 에이전트 실행, 토큰 비용용 실제 토크나이저.

## 구조
```
src/mcp_router/
  catalog/     계단식 카탈로그(키워드 충돌 distractor) + 쿼리 생성 + Spec
  routing/     4개 전략 + RoutingContext
  providers/   임베딩(mock / mock_char / bge-small) + llm(mock jaccard / claude)
  vectorstore/ 순수 파이썬 exact cosine 인덱스
  labeling/    self-consistency 라벨러 + Cohen's kappa
  bench/       분리된 에이전트, metrics(fractional recall·cluster bootstrap·McNemar+BH·phi),
               runner, report
  tracing.py · cli.py (bench run | bench sweep)
tests/         unittest 21케이스
```
