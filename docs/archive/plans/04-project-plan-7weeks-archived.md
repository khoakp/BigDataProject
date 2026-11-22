# 04 — Kế hoạch 7 tuần, deadline chi tiết và nghiệm thu

> **Kế hoạch có 49 ngày tương đối (W1D1…W7D7).** Đây là lịch triển khai dự kiến, không phải nhật ký công việc đã hoàn thành.  
> **Ngày bắt đầu:** `YYYY-MM-DD` (CHƯA ĐƯỢC CUNG CẤP) · **Ngày nộp chính thức:** `YYYY-MM-DD` (CHƯA ĐƯỢC CUNG CẤP)  
> **Timezone:** Asia/Ho_Chi_Minh · **Chu kỳ:** Day 1 = ngày bắt đầu; `WkDj` ứng với ngày thứ `7(k−1)+j`.  
> **Timebox:** thời gian tập trung tham khảo, điều chỉnh theo lịch học thực tế. Không giả định bạn dành trọn 7 ngày/tuần.

## 1. Mục tiêu điều hành và nguyên tắc

**Critical path:** Setup + dữ liệu → deterministic replay → Full/Incremental correctness → normalized correctness → GPU LightGCN MVP → experiments → report.

**Mỗi task đạt Done khi:** (1) code/tests hoặc bằng chứng xuất hiện, (2) có log/commit/config liên quan, (3) tiêu chí nghiệm thu cột cuối được thỏa, (4) cập nhật [05 — Work Log](../legacy-reports/05-work-log.md). Nếu thiếu một điều kiện, ghi `Blocked` hoặc `Partial`, không đánh dấu Done.

**Trạng thái được phép:** `TODO`, `DOING`, `BLOCKED`, `DONE`, `CANCELLED`. Mỗi task chỉ có một trạng thái tại một thời điểm.

## 2. Milestones và hard gates

| Deadline tương đối | Milestone | Điều kiện chấp nhận tối thiểu | Go / No-Go |
|---|---|---|---|
| **W1D7** | M1 — Feasibility | CUDA sparse forward/backward, dataset/checksum, xác định GNN safe scale sơ bộ | Nếu không chạy 1M: giảm embedding_dim/representation hoặc CPU toy fallback; không đòi train 32M |
| **W2D7** | M2 — Canonical state | Full vs Incremental equivalence + idempotency pass | Chưa pass → dừng thêm model features |
| **W3D7** | M3 — Model-ready graph | Normalization/reference equality, versioning, recovery pass | Chưa pass → ưu tiên fix correctness, bỏ extras |
| **W4D7** | **M4 — MVP end-to-end** | events → graph → normalized adjacency → LightGCN → temporal metrics chạy được | Bắt buộc chốt MVP; không thêm Kafka/UI |
| **W5D7** | M5 — DE results | EXP-01/02 raw CSV, graphs, equality assertions | Không có 32M nếu thiếu RAM thì ghi giới hạn, không bịa số |
| **W6D6** | **M6 — FEATURE FREEZE** | EXP-03, quality/cost analysis, code+plots finalized | Chỉ sửa bugs/blockers sau mốc này |
| **W7D7** | M7 — Submission | Report, repo, README, demo, reproducibility evidence | Deadline thực theo giảng viên có ưu tiên hơn W7D7 |

## 3. Task mapping/ownership

- **Owner mặc định:** một sinh viên; không tạo nhiệm vụ song song phụ thuộc tài nguyên GPU nếu chỉ có một laptop.
- **P0**: correctness + MVP + evidence + submission; **P1**: plot polish, repetitions bổ sung; **P2**: UI, compaction benchmark, full32M GNN.
- **Phụ thuộc:** task không bắt đầu khi task prerequisite có trạng thái `BLOCKED`, trừ khi ghi lý do rõ ràng.
- **Bằng chứng:** results/raw, pytest output, screenshot CUDA, commit hash, file versions, written analysis.

## 4. Lịch công việc đến từng ngày


### Tuần 1: Chuẩn bị + GPU feasibility

| Ngày / task | Timebox | Công việc | Đầu ra / Điều kiện nghiệm thu | Phụ thuộc |
|---|---:|---|---|---|
| **W1D1 · P-001** | 2–3h | Kiểm tra OS/CPU/RAM/SSD, nvidia-smi, Python/CUDA, tạo repo và venv | hardware.md; torch CUDA tensor op PASS | — |
| **W1D2 · P-002** | 3–4h | Lấy MovieLens 32M, đọc usage/README, checksum; tạo data manifest | source verified; input hash và license ghi nhận | P-001 |
| **W1D3 · P-003** | 3–4h | Audit fields/null/ties/duplicates; sort events và xuất 1M Parquet | 1M replay deterministic; query counts | P-002 |
| **W1D4 · P-004** | 4h | CUDA/PyG sparse test LightGCN forward, backward, short train ở 1M | log peak allocated/reserved; không có silent OOM | P-001,P-003 |
| **W1D5 · P-005** | 4h | Scale ladder 5M→10M→20M nếu an toàn; GPU model scale gate | MODEL_MAX_SAFE_SCALE; failures logged | P-004 |
| **W1D6 · P-006** | 3h | Prototype Full vs Incremental trên toy/1M; freeze edge semantics/threshold | toy outputs same; quyết định dataset/model cutoff | P-003 |
| **W1D7 · P-007** | 2h | Buffer + tổng kết và rà soát phạm vi, chốt W2 task order | W1 review có blockers và điều chỉnh scope | P-001…P-006 |


### Tuần 2: Canonical graph correctness

| Ngày / task | Timebox | Công việc | Đầu ra / Điều kiện nghiệm thu | Phụ thuộc |
|---|---:|---|---|---|
| **W2D1 · P-008** | 3h | Khóa data contracts và state schemas; viết tests toy cases | schema/constraints, tests green | P-006 |
| **W2D2 · P-009** | 3–4h | Implement stable NodeRegistry dùng monotonic IDs | mapping không đổi khi thêm node mới | P-008 |
| **W2D3 · P-010** | 4h | Implement Full Rebuild reference trên prefix | reference edges/degrees deterministically reproducible | P-008,P-009 |
| **W2D4 · P-011** | 4h | Implement transactional Incremental Builder + batch ledger | batch apply và duplicate rejection hoạt động | P-009,P-010 |
| **W2D5 · P-012** | 3–4h | Implement DegreeState / edge dedup / positive rule | degree sums == edge count; duplicate edges no-op | P-011 |
| **W2D6 · P-013** | 4h | Test equivalence, idempotency, random batch segmentation | T-02/T-03 PASS trên toy+1M | P-010,P-012 |
| **W2D7 · P-014** | 2h | Buffer + refactor, ghi DEC và chuẩn bị Week 3 | canonical graph MVP ổn định | P-013 |


### Tuần 3: Normalized snapshot + versioning

| Ngày / task | Timebox | Công việc | Đầu ra / Điều kiện nghiệm thu | Phụ thuộc |
|---|---:|---|---|---|
| **W3D1 · P-015** | 3h | Toy normalized adjacency reference theo đúng LightGCN | expected weights manual+unit tests | P-013 |
| **W3D2 · P-016** | 4h | Materialize sparse normalized graph, snapshot model index mapping | T-05 pass; graph format load được trên GPU scale nhỏ | P-015 |
| **W3D3 · P-017** | 3h | Manifest version, checksums, source/config/git provenance | canonical_version và model_graph_version độc lập | P-016 |
| **W3D4 · P-018** | 3h | Refresh policy R=1/5/20 + ghi staleness | model_graph materializes đúng lịch R | P-017 |
| **W3D5 · P-019** | 3–4h | Recovery from checkpoint + deltas; crash simulation toy | T-04 pass, no conflicting batch state | P-017 |
| **W3D6 · P-020** | 4h | Equivalence normalized Full vs Incremental ở common version | T-05/T-06 pass with specified tolerance | P-018,P-019 |
| **W3D7 · P-021** | 2h | Buffer; chuẩn hóa API graph-to-model | model-ready artifact contract stable | P-020 |


### Tuần 4: MVP end-to-end

| Ngày / task | Timebox | Công việc | Đầu ra / Điều kiện nghiệm thu | Phụ thuộc |
|---|---:|---|---|---|
| **W4D1 · P-022** | 3h | Temporal train/val/test cutoff + warm-only evaluation cohort | T-07 pass; cohort counts and exclusions logged | P-021 |
| **W4D2 · P-023** | 3h | Popularity baseline + top-K evaluator reference | Recall/NDCG small toy pass | P-022 |
| **W4D3 · P-024** | 4h | BPR-MF train and evaluate (fixed budget) | checkpoint + config/seed + metrics | P-023 |
| **W4D4 · P-025** | 4–5h | LightGCN baseline train theo safe scale đã chốt W1 | short/full fit logs, no OOM, checkpoint version | P-024,P-016 |
| **W4D5 · P-026** | 4h | Chunked scoring, filter seen items, validate against toy full scores | T-08 pass; no dense all-user/all-movie matrix | P-025 |
| **W4D6 · P-027** | 4h | End-to-end CLI: events→versioned graph→LightGCN→metrics | MVP command works from clean state | P-026 |
| **W4D7 · P-028** | 2–3h | MVP buffer / gate; giảm scope nếu lỗi nghiêm trọng | **MVP ACCEPTED by W4 end** or fallback recorded | P-027 |


### Tuần 5: EXP-01/02 về Data Engineering

| Ngày / task | Timebox | Công việc | Đầu ra / Điều kiện nghiệm thu | Phụ thuộc |
|---|---:|---|---|---|
| **W5D1 · P-029** | 3h | Create benchmark harness, warmup, isolated repeats, system metrics | run metadata schema; 1 warm-up + n repeats | P-028 |
| **W5D2 · P-030** | 4h | EXP-01 small/medium size × delta pilot; correctness first | raw rows + consistent checksum across methods | P-029 |
| **W5D3 · P-031** | 4–5h | EXP-01 10M/20M/32M where feasible; inspect crossover | history-v-delta plots raw CSV | P-030 |
| **W5D4 · P-032** | 4h | EXP-02 R=1/5/20 on same event replay, fixed refresh points | cumulative cost & staleness logs | P-029,P-020 |
| **W5D5 · P-033** | 4h | Repeat/validate CPU benchmark fairness, normalization equality | outliers documented; equality PASS | P-031,P-032 |
| **W5D6 · P-034** | 3h | Generate plots and concise results for RQ1/RQ2 | plots from CSV, 1-page interpretation each | P-033 |
| **W5D7 · P-035** | 2h | Buffer, checkpoint outputs, risk review | EXP-01 and EXP-02 artifact set complete | P-034 |


### Tuần 6: EXP-03 + đóng băng tính năng

| Ngày / task | Timebox | Công việc | Đầu ra / Điều kiện nghiệm thu | Phụ thuộc |
|---|---:|---|---|---|
| **W6D1 · P-036** | 3h | Freeze temporal evaluation cutoffs, warm cohort, methods | EXP-03 protocol logged before running | P-035 |
| **W6D2 · P-037** | 4h | EXP-03 stale vs propagation-only vs trained refresh timings | GPU allocated/reserved, synchronized timings | P-036 |
| **W6D3 · P-038** | 4h | EXP-03 Recall/NDCG across 3 seeds if budget permits | raw per-seed evaluation, same future target | P-037 |
| **W6D4 · P-039** | 3h | Interpret quality–cost trade-off; check no leakage | RQ3 plot, missingness, caveats | P-038 |
| **W6D5 · P-040** | 3h | Fresh reproducibility run; check hashes/env/hardware metadata | representative run reproducible | P-039 |
| **W6D6 · P-041** | 3h | **Feature freeze**, tag code/docs; collect final figures | no new features; report-ready data/plots | P-040 |
| **W6D7 · P-042** | 2h | Buffer; freeze residual decisions and outline conclusion | W6 review+technical debt for W7 | P-041 |


### Tuần 7: Báo cáo và bảo vệ

| Ngày / task | Timebox | Công việc | Đầu ra / Điều kiện nghiệm thu | Phụ thuộc |
|---|---:|---|---|---|
| **W7D1 · P-043** | 4h | Write introduction/related work/dataset/methods and architecture | report draft with all citations | P-041 |
| **W7D2 · P-044** | 4h | Write results, discussion, limits, validity threats | every claim supported by EXP CSV/graph tests | P-043 |
| **W7D3 · P-045** | 3h | Create defense slides 10–12, charts direct from source | slides complete, no fabricated results | P-044 |
| **W7D4 · P-046** | 3h | Build reliable fast demo with saved checkpoints/data versions | demo < 10 min; fallback screenshots/logs | P-045 |
| **W7D5 · P-047** | 3h | Fresh clone/setup test on clean env if possible | README commands, dependencies, no secrets | P-046 |
| **W7D6 · P-048** | 2–3h | Mock defense, run through likely objections | answers to 10 questions, time checked | P-047 |
| **W7D7 · P-049** | 2–3h | Submission buffer; package report/code/evidence | **Submit by official deadline; verify archive** | P-048 |


## 5. Quy đổi sang ngày tháng thật

Khi có ngày bắt đầu `START_DATE`, deadline tương đối của task ngày thứ `n` là:

```text
calendar_date(WkDj) = START_DATE + (7 * (k - 1) + (j - 1)) days
```

Không tự chọn giờ nộp hoặc hứa rằng ngày nghỉ cũng rảnh; nếu phải nghỉ, chèn ngày buffer và điều chỉnh lại toàn bộ lịch theo hạn thật.

### Bảng thông tin lịch cần điền

| Tham số | Giá trị hiện tại |
|---|---|
| START_DATE | `[chưa cung cấp]` |
| OFFICIAL_SUBMISSION_DATE | `[chưa cung cấp]` |
| Số giờ/ngày các ngày thường | `[chưa cung cấp]` |
| Ngày không thể làm (lịch học/thi) | `[chưa cung cấp]` |
| Daily check-in | Sau buổi làm việc (không cố định giờ) |
| Weekly review | Ngày cuối tuần, 20–30 phút |

## 6. Chu trình vận hành hàng ngày (15 phút quản lý)

**Trước khi làm (5 phút):** xem task ID → định nghĩa kết quả mong đợi → kiểm tra dependency, timebox.  
**Trong lúc làm:** commit code, lưu CSV/log vào experiment ID nếu có.  
**Sau khi làm (10 phút):** ghi [05 — Work Log](../legacy-reports/05-work-log.md), mark DONE/BLOCKED, nêu vấn đề. Cần thay thiết kế → [06 — DEC/Risk](../legacy-reports/06-decisions-and-risks.md); có benchmark → [07 — Experiment Log](../../research/experiment-log.md).

### Mẫu weekly review

```markdown
## Weekly review — W?
- Kế hoạch: task P-...
- Hoàn thành: .../...; bằng chứng: commits/CSV/tests
- Milestone pass? Có/Không. Nếu Không, lỗi nào?
- Tổng thời gian tập trung: ... giờ
- 3 blockers lớn nhất:
- Một việc sẽ cắt để bảo vệ MVP/deadline:
- Task chuyển sang tuần sau:
- Cần hỏi giảng viên/mentor điều gì?
```

## 7. Quy tắc quản lý trễ tiến độ

- Trễ **1 ngày**: ưu tiên sửa blocker, dùng task buffer (`WkD7`), ghi nguyên nhân.
- Trễ **2–3 ngày**: loại P2 (UI, compaction, extra baselines), giảm grid kích thước/seed nhưng giữ công bằng và số đo trung thực.
- **Cuối W2 chưa correctness:** chỉ làm pipeline/test, không triển khai thêm model refresh.
- **Cuối W4 chưa MVP:** thu nhỏ model 100k/1M, tách VRAM/CPU issues, ưu tiên end-to-end một lần chạy.
- **Cuối W6:** không mở rộng scope, chỉ bug fixes, repro, báo cáo.
- Nếu hạn nộp thực tế sớm hơn 49 ngày: tái baseline kế hoạch, đặt milestone theo hạn thật; không im lặng trượt deadline.

## 8. Lộ trình ưu tiên khi thời gian thực sự chỉ còn ít

1. **Không được cắt:** deterministic data → graph correctness → normalized correctness → end-to-end LightGCN → EXP-01 → luận giải hạn chế.
2. **Giảm phạm vi có điều kiện:** EXP-02 chỉ R=1 và R=5, EXP-03 một cutoff và ít seeds (công bố hạn chế rõ).
3. **Cắt đầu tiên:** web demo, training full32M, Kafka/Spark/Neo4j, compaction ablation, hyperparameter search lớn.

**Ghi chú:** timeboxes không phải số liệu thời gian đã thực hiện. Cập nhật thời gian thực trong [05](../legacy-reports/05-work-log.md).
