# 06 — Nhật ký quyết định kiến trúc (ADR) và quản lý rủi ro

> **Hướng mới 21/11/2026:** DEC-016 ghi định hướng v1.0 đã thống nhất; DEC-017 ghi giao thức REC-01 đã được Project Owner duyệt. DEC-001/013/014 và risk cũ bên dưới được giữ như lịch sử, không còn quyết định ưu tiên sprint S01–S14. Xem [định hướng](10-direction-update-v1.md) và [sprint mới](11-sprint14-plan.md).

### DEC-016 — Hai trụ cột và sprint S01–S14

- **Ngày / trạng thái:** 21/11/2026, ACCEPTED theo Project Direction Update v1.0 do Project Owner cung cấp.
- **Lựa chọn:** Big Data Engineering và LightGCN recommendation đều là P0; thêm Popularity, Recall/NDCG@20, checkpoint validation selection và Streamlit MVP. Kaggle GPU là môi trường training chính; laptop dùng ETL/benchmark/app và là fallback có ghi nhãn khi Kaggle không khả dụng. EXP-01 giữ là benchmark bắt buộc; D13 EXP-02 đã chạy được giữ làm phần mở rộng, không nằm trên critical path mới. Các milestone D01–D13 và số hiệu EXP cũ giữ nguyên; công việc mới dùng S01–S14, REC-01 và REC-02.
- **Lý do:** hướng mới cần chứng minh cả correctness/chi phí dữ liệu lẫn chất lượng và trải nghiệm gợi ý, trong 14 ngày dự kiến.
- **Hậu quả:** README, overview, plan, report outline cần dẫn kế hoạch mới; S01 audit trước khi thay code. Không khẳng định graph update làm thay đổi model parameters hay recommendations nếu chưa refresh thật.
- **Bằng chứng:** [bản định hướng](10-direction-update-v1.md), [kế hoạch sprint](11-sprint14-plan.md), [S01 audit](s01-evaluation-spec.md).

### DEC-017 — REC-01 evaluation protocol

- **Ngày / trạng thái:** 21/11/2026, ACCEPTED; Project Owner duyệt trong chat trước final training/evaluation.
- **Đề xuất:** primary canonical prefix 1M với cutoffs đã đo, warm train-catalog candidates, Recall/NDCG@20, Popularity cùng cohort, validation selection từ steps 100/300/500, test chỉ sau chọn. Chi tiết và giới hạn tại [S01](s01-evaluation-spec.md).
- **Điều kiện khóa:** dataset/cutoffs/candidate/selection rule đã được Project Owner xác nhận. Lưu config/hash và không đổi theo test metric. Scale bổ sung có protocol/run ID riêng.

### DEC-018 — Feedback graph branch và model freshness S10

- **Ngày / trạng thái:** 21/11/2026, ACCEPTED trong phạm vi S10 Streamlit demo.
- **Lựa chọn:** sao chép sealed REC-01 train graph sang state demo riêng; rating mới theo batch dùng D04 ledger, không sửa source/checkpoint gốc. Rating `>=4.0` tạo binary positive edge theo DEC-005; rating thấp hơn chỉ tạo event. Không ghi đè cặp đã rating trong train hoặc demo branch.
- **Version:** `canonical_graph_version` là version hiện tại của branch; `model_graph_version` và checkpoint step luôn là version/step REC-01 đã train cho đến khi có refresh thực sự. Giao diện ghi `staleness_batches = canonical - model`; Top-20 vẫn từ checkpoint cũ.
- **Lý do:** bảo toàn kết quả REC-01 đã seal, không để thao tác demo gây temporal leakage hoặc tạo claim sai rằng graph update đã train lại model. Intent JSON và batch ledger giúp retry/reopen idempotent trong demo một tiến trình.
- **Giới hạn:** chưa có multi-process writer, power-loss durability, model refresh, quality/latency post-feedback hoặc deployment production. Nếu triển khai REC-02, phải có protocol và artifact riêng, không dùng nhánh demo làm test set.
- **Bằng chứng:** [S10](s10-rating-graph-refresh.md), [store](../src/app/feedback.py), [integration/UI tests](../tests/test_s10_feedback.py).

> **Ngày khởi tạo:** 20/10/2026 · **Cập nhật:** 21/11/2026. **Phân biệt:** `PROPOSED` = lựa chọn đề xuất, `ACCEPTED` = đã chốt, `SUPERSEDED` = đã thay thế, `REJECTED` = không chọn.
> Mọi thay đổi lớn cần viết một ADR riêng, liên kết [01](01-project-overview.md), [03](03-system-design.md), [04](04-project-plan.md), [07](07-experiment-log.md).

## A. Decision Log ban đầu

| ID | Quyết định | Trạng thái ban đầu | Căn cứ và điều kiện xét lại | Liên quan |
|---|---|---|---|---|
| **DEC-001** | Trọng tâm DE, LightGCN là downstream validator chứ không nghiên cứu kiến trúc mới | ACCEPTED | Ràng buộc môn + deadline ngắn; xét lại chỉ khi giảng viên bắt buộc novel model | Charter |
| **DEC-002** | ML32M làm DE scale, LightGCN bắt đầu 100k/1M rồi scale sau GPU probe | ACCEPTED (scale lớn chưa chốt) | RTX 4050 6GB; Kaggle GPU tùy chọn; phải đo VRAM để xác định safe scale | D01/D07 |
| **DEC-003** | Canonical edge+registry+degree khác normalized model graph | ACCEPTED | Cơ chế LightGCN degree normalization | T-05/EXP-02 |
| **DEC-004** | Dùng event-time historical replay, không gọi production streaming | ACCEPTED | Dữ liệu MovieLens là lịch sử có timestamp | P-003 |
| **DEC-005** | Positive graph = unique pair có ít nhất một rating >= 4.0 | ACCEPTED cho D03, 19/11/2026 | Profile rating toàn bộ nguồn; rule nghiên cứu, chưa chứng minh tối ưu Recall/NDCG. Nếu đổi phải chạy lại graph/benchmark | [D03](d03-full-reference.md) |
| **DEC-006** | So sánh stale/propagation-only/trained refresh trên warm nodes | PROPOSED | Tránh vấn đề strict cold-start embedding; đo EXP-03 | P-036 |
| **DEC-007** | Một máy cho DE benchmark, DuckDB/Parquet/PyTorch; Kaggle GPU tùy chọn | ACCEPTED | 21 ngày, không mặc định Spark/Kafka/Neo4j; DE benchmark phải cùng phần cứng | D01/D07 |
| **DEC-008** | Chỉ công bố lợi ích deferred normalization nếu EXP-02 chứng minh | ACCEPTED | Tránh confirmation bias; âm tính vẫn hợp lệ | EXP-02 |
| **DEC-009** | Full vs Incremental chỉ benchmark tốc độ sau khi graph equivalence pass | ACCEPTED | Sự đúng đắn quan trọng hơn tốc độ | P-013 |
| **DEC-010** | Hard gates: MVP cuối W4, feature freeze W6D6 | SUPERSEDED | Thay bởi kế hoạch 21 ngày | DEC-013 |
| **DEC-011** | MovieLens 32M là dataset chính; RetailRocket hoãn sau MVP | ACCEPTED | MovieLens phù hợp graph + temporal LightGCN; RetailRocket vẫn hữu ích học ETL nhưng không trong critical path | Charter, D01 |
| **DEC-012** | `movies.csv` enrichment **sau** Top-K; không dùng genre/title trong LightGCN MVP | ACCEPTED | Giữ bipartite graph semantics và benchmark comparability; cho demo/genre profiling không đổi metric | Design 2.4, T-10/11/12 |
| **DEC-013** | Rebaseline còn ~21 ngày: D10 MVP, D16 feature freeze; EXP-01 P0, EXP-02 P1, EXP-03 stretch | ACCEPTED (deadline thật chưa xác nhận) | Phải ưu tiên tính đúng đắn và sản phẩm nộp trước scope mở rộng | Plan 21 ngày |
| **DEC-014** | Kaggle GPU được phép dùng LightGCN, benchmark CPU Full/Incremental vẫn cùng môi trường | ACCEPTED (tùy tài nguyên) | Tránh so sánh thời gian giữa hai loại máy; log software/hardware/config riêng | D07–D14 |

### DEC-015 — Representation và version contract của normalized snapshot D06

- **Ngày / trạng thái:** 19/11/2026, ACCEPTED trong phạm vi triển khai D06 theo thiết kế mục 6–7.
- **Lựa chọn:** materialize sparse COO float64, int64 node indices, không self loops; normalize một lần tại graph materializer. Tolerance kiểm chứng absolute `1e-12`, relative 0. Không thêm dependency tensor nặng cho kiểm tra CPU D06.
- **Version:** `model_graph_version` là canonical source version của snapshot, không phải số thứ tự lần refresh. Current canonical version được báo riêng; node map/shape/weights snapshot cũ giữ nguyên. `model_checkpoint_version` là null đến khi thực sự có checkpoint.
- **Persistence:** manifest/node map/COO trong cùng DuckDB transaction; export file dùng manifest cuối cùng làm completion marker. Không nhận là checkpoint recovery hoặc publish atomic xuyên DB/filesystem.
- **Phương án khác:** normalize on-the-fly lúc propagation có chi phí khác và chưa được đo ở đây. Chuyển sang float32/tensor ở D07 phải kiểm tra tolerance và tránh normalize hai lần.
- **Bằng chứng / hạn chế:** [D06](d06-normalization.md), dense toy reference, snapshot/version tests và integration probe. Không kết luận về speedup hoặc GPU compatibility từ lựa chọn này.

### Mẫu ADR mới

```markdown
### DEC-XXX — Tên quyết định
- Ngày / Owner / Trạng thái:
- Bối cảnh và vấn đề kỹ thuật:
- Các lựa chọn đã cân nhắc:
- Bằng chứng (paper, issue, log hoặc benchmark):
- Phương án chọn:
- Ưu/nhược và hậu quả:
- Ảnh hưởng tới tasks/tests/experiments:
- Trigger phải xem xét lại:
- Ngày review:
```

## B. Risk register

Quy tắc chấm: **Probability (P) 1–5 × Impact (I) 1–5**. Điểm **ước lượng kế hoạch**, không phải xác suất đã đo. Review mỗi cuối tuần và ngay khi có blocker.

| ID | Rủi ro | P×I | Trigger | Biện pháp phòng ngừa / fallback | Thời điểm kiểm tra |
|---|---|---:|---|---|---|
| R-01 | Scope creep: bổ sung Kafka/UI/online retrain | 4×5=20 | Nhiệm vụ ngoài RQ chiếm >1 ngày | Freeze MVP, cắt P2, chỉ dùng tech khi có evidence | Hằng tuần |
| R-02 | GPU 6GB OOM với LightGCN | 3×4=12 | 1M forward/backward/training không ổn định | Giảm d=64→32/16, số layers, batch, dùng sparse SpMM/CPU toy; không train 32M | D07 |
| R-03 | RAM/SSD không đủ scan/sort 32M | 3×4=12 | OOM/disk spill quá nhiều | DuckDB external processing, giảm grid, scale 20M ghi giới hạn | D01–D02 |
| R-04 | CUDA/PyG version incompatibility | 3×4=12 | Install/import/grad errors | Pin wheels/version, sparse PyTorch native fallback | D07 |
| R-05 | Graph logic sai bởi edge duplicate/degree | 2×5=10 | T-02/T-03 fail | Toy cases + reference full rebuild, suspend optimization | D05 |
| R-06 | Temporal data leakage | 2×5=10 | future edges in training/degrees/candidates | Strict cutoff tests, train-only graph & warm cohort | D08 |
| R-07 | Deferred normalization không có cải thiện chi phí | 3×3=9 | EXP-02 little gain / worse | Trình bày result âm; phân tích sensitivity, IO cost và staleness | D13 |
| R-08 | Incremental thua Full ở scale thấp | 3×2=6 | EXP-01 small histories | Tìm crossover, report overhead tradeoff; không sửa số đo | D12 |
| R-09 | Reapply batch / crash làm corrupt manifest | 2×4=8 | Recovery mismatch | Applied batches ledger + two-stage publish/reconcile | D09–D16 |
| R-10 | Benchmark không công bằng vì OS cache/tuning | 3×3=9 | large run-to-run variance | Randomized order, warm-up, isolated runs, same device output & config | D11 |
| R-11 | Đến D10 chưa có end-to-end LightGCN | 3×5=15 | W4D5 còn chưa train/eval | Cắt P2, giảm model scale, ưu tiên 100k/1M smoke-end-to-end | D10 |
| R-12 | 21 ngày kế hoạch không khớp deadline thật | 4×4=16 | Lịch thi/ngày nộp sớm hơn | Xác nhận deadline, sắp D01–D21 theo ngày thật, giữ D21 buffer | D01 |
| R-13 | Metadata JOIN làm nhân đôi/mất Top-K hoặc thay đổi ranking metrics | 3×4=12 | Duplicate `movieId`, inner join bỏ mất item, score/order đổi | Validate unique; LEFT JOIN; test số rows/rank/score và metrics isolation | D09–D10 |
| R-14 | Mất notebook session / GPU cloud thay đổi | 3×3=9 | quota hết, checkpoint không lưu, khác version PyTorch | Lưu local checkpoint/config/mappings; laptop CPU/GPU fallback; pin environment | D07–D10 |
| R-15 | Thêm genres trực tiếp vào model gây scope creep | 4×4=16 | bắt đầu thêm Movie–Genre edges/feature learning trước MVP | Đóng băng giới hạn hậu xử lý, đưa thay đổi thành future work | Hằng tuần |

### Mẫu risk mới

```markdown
### R-XX — Tên
- Ngày phát hiện / Owner:
- Probability 1–5:
- Impact 1–5:
- Trigger / early indicators:
- Mitigation / fallback:
- Deadline quyết định:
- Status: Open / Mitigating / Closed
- Evidence / related tasks:
```

## C. Gates & hành động nếu thất bại

1. **D02:** ETL/chonological replay sai → dừng mở rộng quy mô; kiểm tra input hash, stable tie-break và schema.
2. **D05:** canonical graph Full/Incremental mismatch → dừng performance experiments, sửa invariants/idempotency.
3. **D06:** normalized graph mismatch → không đưa output sai vào LightGCN.
4. **D07–D10:** GPU/LightGCN OOM → giảm quy mô, dùng Kaggle nếu sẵn có, hoặc smoke train trên subset; không nới deadline.
5. **D09–D10:** metadata JOIN làm thay ranking/metrics → bỏ enrichment khỏi eval path đến khi T-10/11/12 pass.
6. **D10:** chưa có MVP → cắt EXP-03 và extras, thu nhỏ về 100k, cứu end-to-end.
7. **D16:** feature freeze; chỉ bugfix, reproducibility, report và demo.

## D. Decision review checklist

- [ ] Dựa trên nguồn/benchmark nào?
- [ ] Là quyết định kỹ thuật hay kết quả đã đo?
- [ ] Có thay đổi data contract, graph semantics hay evaluation cohort không?
- [ ] Có cần chạy lại thí nghiệm cũ để so sánh công bằng không?
- [ ] Đã cập nhật phần liên quan trong Overview / Design / Plan chưa?
