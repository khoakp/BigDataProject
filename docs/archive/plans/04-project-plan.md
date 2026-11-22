# 04 — Kế hoạch triển khai BigDataProject trong 21 ngày

> **Hồ sơ lịch sử GOAL 03.5:** giữ để truy nguồn quyết định/kết quả cũ. [Phạm vi v2 hiện hành](../../current/scope-and-decisions.md) · [Tiến độ v2](../../current/progress-and-evidence.md).

> **Lưu lịch sử D01–D13:** Từ 21/11/2026, định hướng và thứ tự công việc mới nằm ở [Project Direction v1.0](10-direction-update-v1.md) và [sprint S01–S14](11-sprint14-plan.md). Các mốc D01–D21 bên dưới là kế hoạch cũ để đối chiếu bằng chứng, không còn là critical path đang thực hiện. Không đổi nhãn các EXP đã chạy.

> **Phiên bản kế hoạch lịch sử:** 0.2 · **Bản gốc:** 10/11/2026 · **Trạng thái:** không còn là lịch hiện hành; xem [S01–S14](11-sprint14-plan.md).
> **D01 bắt đầu:** 19/11/2026; hạn nộp/giờ nộp chính thức **CHƯA XÁC NHẬN**. D01–D21 là thứ tự công việc, không mặc định 21 buổi rảnh.
> **Bản trước:** [Kế hoạch 7 tuần (archived)](04-project-plan-7weeks-archived.md). Các W1D1…W7D7 và D01…D21 là các hệ nhãn lịch sử; công việc mới dùng S01…S14.

## 1. Đích 21 ngày và phạm vi

**Tiến độ thực tế 21/11/2026:** D01–D13 DONE. G1 đạt trên toy/100k; G2 đạt sau D07 CUDA smoke đến 1M; G3 MVP đạt sau clean rerun D10. D12 EXP-01 có năm repetitions trên sáu cặp H/Δ; D13 EXP-02 so R=1/R=5 trên cùng 100k events và common-boundary equality PASS. [D12](../../research/d12-exp01-results.md), [D13](../../research/d13-exp02-results.md). Bảng bên dưới giữ tiêu chí kế hoạch.

**Critical path:** Verify MovieLens 32M → deterministic canonical events → Full/Incremental graph state correctness → sparse normalization correctness → LightGCN MVP → EXP-01 → EXP-02 (nếu đủ thời gian) → báo cáo/demo.

**Movie metadata:** `movies.csv` là lookup theo `movieId`, **LEFT JOIN sau khi Top-K ranking** để gắn `title`/`genres` và phân tích mô tả. Không thêm Movie–Genre edges hoặc genre features vào LightGCN. RetailRocket **không** là dataset dùng trong hạn 21 ngày. Dataset của model là chronological prefix từ MovieLens 32M sau khi sort, không đánh đồng với file MovieLens 1M độc lập.

**Các mốc bắt buộc:**

| Gate | Ngày | Điều kiện chấp nhận | Nếu thất bại |
|---|---|---|---|
| G1 — Canonical & graph | D05 | E2E canonical state trên toy/100k, exact Full/Incremental edge/degree equivalence, idempotency pass | D06–D08 sửa correctness, giảm scale |
| G2 — Graph for model | D07 | Normalization toy/reference đúng; GPU/LightGCN smoke đã được thử và có log | Fallback CPU toy / GPU nhỏ hoặc Kaggle, không giấu OOM |
| **G3 — MVP** | **D10** | `ratings.csv` → events → versioned graph → normalized graph → LightGCN → raw Top-K + metadata Top-K, temporal eval tối thiểu | Cắt extras và EXP-03; giữ tối thiểu 100k |
| G4 — Results | D14 | EXP-01 raw logs với correctness pass; EXP-02 theo ngân sách, có ghi thiếu/blocked | D15 hoàn thiện EXP-01 trước |
| **G5 — Feature freeze** | **D16** | Tests pass, config/commits/checkpoints/figures được chốt | Chỉ bugfix/blocker/report |
| G6 — Nộp | D21 hoặc hạn thật sớm hơn | README, code, tests, raw logs, report, slides, demo | Nộp theo hạn chính thức, không giả định 29/11 |

### Phân cấp ưu tiên

- **P0 bắt buộc:** deterministic replay, Full & Incremental equivalence, dedup/degree/idempotency, model-ready normalized graph, LightGCN end-to-end ở safe scale, temporal metrics, EXP-01, README/report/demo.
- **P1 ưu tiên cao:** EXP-02 `R=1` vs `R=5`; metadata title/genres Top-K có tests; genre distribution plot/coverage (AN-01), recovery tối thiểu.
- **P2 / stretch:** EXP-03 stale/propagation-only/retrain, R=20, MovieLens >5M training, extra methods, RetailRocket, UI/cloud orchestration/Kafka/Spark.

## 2. Lịch chi tiết 21 ngày

### Tuần 1 — D01–D07: Data, graph correctness và GPU smoke

| Ngày | Ưu tiên | Việc làm | Định nghĩa Done (bằng chứng) |
|---|---|---|---|
| **D01** | P0 | Kiểm tra CPU/RAM/SSD/CUDA và công cụ DuckDB/PyTorch; tải và đọc license/README MovieLens 32M, hash `ratings.csv` và `movies.csv`; tạo repo | `docs/research/environment.md`, manifest nguồn, torch smoke log |
| **D02** | P0 | Profile `ratings`: schema, null, rating range, duplicates/timestamp ties; canonical schema, `source_row_id`, event-time sort có tie-break; Parquet prefix 100k/1M | Deterministic ordering/hash test, số rows khớp nguồn |
| **D03** | P0 | Chốt positive rule đề xuất `rating >= 4.0`; tạo Full Rebuild reference, graph node registry/edge set/degree trên toy và 100k | Toy edge counts + invariant `sum user degree = edges = sum movie degree` |
| **D04** | P0 | Stateful `apply_batch()` với anti-join, ID ổn định, transaction + `applied_batches` ledger | Replay cùng `(batch_id,hash)` là no-op; conflict ID khác hash bị reject |
| **D05** | P0 | So sánh Full/Incremental trên toy/100k với batch segmentation khác nhau | Edge/registry/degree equivalence pass; G1 |
| **D06** | P0 | Sparse symmetric normalized adjacency; tách `canonical_version`/`model_graph_version`; metadata schema validation riêng | Normalization toy test T-05, version test T-06, `movies.movieId` unique |
| **D07** | P0 | LightGCN/PyG CUDA sparse forward/backward; thử scale 100k rồi 1M nếu đủ; lưu memory profile | `gpu_probe.csv` có status/VRAM thực, model smoke pass hoặc fallback ghi rõ; G2 |

### Tuần 2 — D08–D14: MVP và thí nghiệm

| Ngày | Ưu tiên | Việc làm | Định nghĩa Done (bằng chứng) |
|---|---|---|---|
| **D08** | P0 | Tạo temporal cutoff + strict train graph, train LightGCN baseline; evaluator raw movie IDs, chunked Top-K | Không leak future data; metrics trên warm cohort, log exclusions |
| **D09** | P0/P1 | Kết nối events→graph→normalized→LightGCN; LEFT JOIN Top-K với `movies.csv`; kiểm tra metadata missing/row count/rank/score | E2E script, model checkpoint, T-10/11/12 metadata invariance pass |
| **D10** | P0 | End-to-end clean rerun, save manifests/checkpoint/model IDs; 1 demo 3–5 phút | **MVP accepted**, 100k minimum; G3 |
| **D11** | P0 | EXP-01 harness, cùng data/output/CPU/cache policy; pilot H={100k,500k,1M}, D={1k,10k} nếu đủ | Correctness + raw timing/RSS/bytes, warmup/repetition metadata |
| **D12** | P0 | EXP-01 repetitions và phân tích speedup/crossover, scale >1M nếu ổn | `results/raw/exp01_full_vs_incremental.csv`, plots từ dữ liệu thật |
| **D13** | P1 | EXP-02 cùng replay batches, `R=1` và `R=5`; cost + staleness; common-boundary normalized equality | `results/raw/exp02_normalization.csv`, correctness/timing logs |
| **D14** | P1 | Tổng hợp EXP-01/02; diễn giải kết quả âm tính; AN-01 coverage và genre counts đơn giản nếu thuận lợi | Research results draft + G4; nếu EXP-02 trễ, ưu tiên EXP-01 |

### Tuần 3 — D15–D21: ổn định, báo cáo và bảo vệ

| Ngày | Ưu tiên | Việc làm | Định nghĩa Done (bằng chứng) |
|---|---|---|---|
| **D15** | P0 | Fix blockers, review tất cả versions & tests; chỉ thử mini EXP-03 nếu G1–G4 đều pass | ADR xác nhận làm/không làm EXP-03 và phạm vi báo cáo |
| **D16** | P0 | **FEATURE FREEZE**; pin Python/PyTorch/PyG, source/model hashes, reproducibility run | Git tag/config/requirements/checkpoints; G5 |
| **D17** | P0 | Viết report methods/architecture, data contracts; hoàn thiện AN-01 và hình metadata Top-K | Phần methods + metadata analysis có source/hash, không bịa số |
| **D18** | P0 | Report results: tests, EXP-01/02, model temporal evaluation; limitations/cost-quality | Claim → evidence mapping đầy đủ |
| **D19** | P0 | Slides 10–12 trang từ raw results, demo script 5–8 phút | Slide/rehearsal lần 1, genre chỉ là minh họa |
| **D20** | P0 | Chạy lại từ README sạch, xét failure/empty metadata/cold users, chạy demo thử | README validated, artifacts backed up |
| **D21** | P0 | Buffer/QA/nộp theo deadline chính thức | Report + slides + source/tests/logs/configs/checkpoint, G6 |

**Timebox gợi ý:** 20–35 phút học đúng nội dung task, 2–3 giờ code, 20–30 phút test/log. Nếu chỉ có 2 giờ/ngày, ưu tiên P0; không thay đổi chính sách dữ liệu để lấy kết quả nhanh hơn.

## 3. Testing bắt buộc

- T-01 deterministic event ordering/checksum từ cùng input+config; timestamps ties xử lý theo source row id.
- T-02 Full vs Incremental cùng cutoff cho cùng edge set, degree và stable node IDs theo chính sách cấp ID đã chốt.
- T-03 idempotency: reapply same batch không tăng degree; conflicting batch hash reject.
- T-04 checkpoint/restart + replay deltas == direct state, nếu persistence được triển khai.
- T-05 normalized matrix toy/reference equality tại cùng version và tolerance được định nghĩa.
- T-06 canonical/model graph/checkpoint versions có provenance và không bị lẫn.
- T-07 no future leakage, train graph và negatives chỉ đến cutoff; warm-only coverage được công bố.
- T-08 chunked Top-K = tiny full-score reference với masking giống nhau.
- T-09 GPU OOM/failure được ghi status/scale, không giả lập thành PASS.
- **T-10** `movies.csv` `movieId` unique/non-null; missing title/genres/IDs được báo cáo.
- **T-11** LEFT JOIN giữ đúng số dòng Top-K và `(user,item,rank,score)`; không loại movie thiếu metadata.
- **T-12** Recall@K/NDCG@K độc lập metadata enrich; genre labels không đi vào model input.

## 4. EXP và metadata analyses

| Mã | Mục tiêu | Ưu tiên | Data & metric |
|---|---|---|---|
| EXP-01 | Full Rebuild vs Incremental cost, correctness | P0 | `ratings.csv`, H/Δ, wall seconds/CPU/RSS/I/O + equality flag |
| EXP-02 | `R=1` vs `R=5` normalized graph refresh | P1 | same replay, cumulative seconds/bytes, stale batches, equality at common boundary |
| EXP-03 | Stale vs propagation-only vs training refresh | P2 | same temporal labels, only warm nodes, time/VRAM/Recall–NDCG |
| **AN-01** | Movie metadata coverage, Top-K genre distribution | P1 nhỏ | `movies.csv` joined post-ranking; metadata match rate, multi-label genre counts; **descriptive only** |

**Nguyên tắc fairness:** cùng input/checksum/config, máy cho EXP-01/02, lịch sử, cutoff và output contract; ghi prebuild state cost riêng. Không trộn model timing từ Kaggle với pipeline timings trên laptop để tạo ra một speedup tổng. Không đưa metadata enrichment vào thời gian Full/Incremental trừ khi cả hai baseline đều phải thực hiện cùng bước đó.

## 5. Sơ đồ repository và đầu ra

```text
BigDataProject/
  README.md
  requirements.txt
  configs/
  data/raw/{ratings.csv,movies.csv}              # ignore from git, giữ metadata về license/hash
  data/processed/canonical_events/*.parquet
  src/ingest/{validate,canonicalize,replay}.py
  src/graph/{full_builder,incremental,registry,degree,normalize,snapshot}.py
  src/models/lightgcn.py
  src/evaluation/{temporal,topk}.py
  src/metadata/movie_lookup.py                 # post-ranking only
  src/benchmarking/runner.py
  tests/{test_graph_equivalence,test_idempotency,test_normalization,test_temporal_leakage,test_metadata_enrichment}.py
  results/raw/{exp01_*,exp02_*,an01_*}.csv
  results/figures/
  artifacts/{checkpoints,manifests}/
  docs/{environment,final-report,defense-slides}
```

## 6. Rủi ro và fallback

1. **Dữ liệu không vừa RAM/SSD**: dùng DuckDB external sort, dữ liệu prefix nhỏ; 32M không bắt buộc phải chạy train.
2. **Incremental state sai**: dừng tối ưu và EXP-01 timing, sửa correctness cho toy/100k.
3. **GPU/PyG không ổn**: giảm embeddings/layers, thử GPU Kaggle nhưng lưu đầy đủ software versions/mappings; CPU toy fallback nếu cloud không sẵn.
4. **Metadata join sai**: bypass hiển thị metadata, không ảnh hưởng raw model results/evaluation, sửa T-10/11/12 sau.
5. **Không kịp D10**: thu nhỏ MVP 100k, bỏ EXP-03, cắt UI/feature engineering/genre reranking.
6. **Deadline thật sớm**: rút ngắn D17–D21, tuyệt đối khóa tính năng sớm hơn; xác minh ngày nộp với giảng viên.

## 7. Cách theo dõi

- D01…D21 là ngày tương đối, không phải ngày dự án thực sự chạy; cập nhật start date + official due date ở đầu file ngay khi biết.
- Mỗi ngày ghi task/test/log/commit vào [05 — Work Log](../legacy-reports/05-work-log.md), quyết định mới vào [06 — Decisions & Risks](../legacy-reports/06-decisions-and-risks.md), raw metrics vào [07 — Experiment Log](../../research/experiment-log.md).
- Chỉ đánh dấu DONE khi thực sự có artifacts/tests/logs. Không tạo synthetic measurements để điền các bảng report.
