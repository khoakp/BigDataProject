# 06 — Nhật ký quyết định kiến trúc (ADR) và quản lý rủi ro

> **Ngày khởi tạo:** 20/10/2026 · **Cập nhật:** 10/11/2026. **Phân biệt:** `PROPOSED` = lựa chọn đề xuất, `ACCEPTED` = đã chốt, `SUPERSEDED` = đã thay thế, `REJECTED` = không chọn.  
> Mọi thay đổi lớn cần viết một ADR riêng, liên kết [01](01-project-overview.md), [03](03-system-design.md), [04](04-project-plan.md), [07](07-experiment-log.md).

## A. Decision Log ban đầu

| ID | Quyết định | Trạng thái ban đầu | Căn cứ và điều kiện xét lại | Liên quan |
|---|---|---|---|---|
| **DEC-001** | Trọng tâm DE, LightGCN là downstream validator chứ không nghiên cứu kiến trúc mới | ACCEPTED | Ràng buộc môn + deadline ngắn; xét lại chỉ khi giảng viên bắt buộc novel model | Charter |
| **DEC-002** | ML32M làm DE scale, LightGCN bắt đầu 100k/1M rồi scale sau GPU probe | ACCEPTED (scale lớn chưa chốt) | RTX 4050 6GB; Kaggle GPU tùy chọn; phải đo VRAM để xác định safe scale | D01/D07 |
| **DEC-003** | Canonical edge+registry+degree khác normalized model graph | ACCEPTED | Cơ chế LightGCN degree normalization | T-05/EXP-02 |
| **DEC-004** | Dùng event-time historical replay, không gọi production streaming | ACCEPTED | Dữ liệu MovieLens là lịch sử có timestamp | P-003 |
| **DEC-005** | Positive graph = unique pair có ít nhất một rating >= 4.0 | PROPOSED | Policy nghiên cứu, inspect duplicates và freeze D03 | D03 |
| **DEC-006** | So sánh stale/propagation-only/trained refresh trên warm nodes | PROPOSED | Tránh vấn đề strict cold-start embedding; đo EXP-03 | P-036 |
| **DEC-007** | Một máy cho DE benchmark, DuckDB/Parquet/PyTorch; Kaggle GPU tùy chọn | ACCEPTED | 21 ngày, không mặc định Spark/Kafka/Neo4j; DE benchmark phải cùng phần cứng | D01/D07 |
| **DEC-008** | Chỉ công bố lợi ích deferred normalization nếu EXP-02 chứng minh | ACCEPTED | Tránh confirmation bias; âm tính vẫn hợp lệ | EXP-02 |
| **DEC-009** | Full vs Incremental chỉ benchmark tốc độ sau khi graph equivalence pass | ACCEPTED | Sự đúng đắn quan trọng hơn tốc độ | P-013 |
| **DEC-010** | Hard gates: MVP cuối W4, feature freeze W6D6 | SUPERSEDED | Thay bởi kế hoạch 21 ngày | DEC-013 |
| **DEC-011** | MovieLens 32M là dataset chính; RetailRocket hoãn sau MVP | ACCEPTED | MovieLens phù hợp graph + temporal LightGCN; RetailRocket vẫn hữu ích học ETL nhưng không trong critical path | Charter, D01 |
| **DEC-012** | `movies.csv` enrichment **sau** Top-K; không dùng genre/title trong LightGCN MVP | ACCEPTED | Giữ bipartite graph semantics và benchmark comparability; cho demo/genre profiling không đổi metric | Design 2.4, T-10/11/12 |
| **DEC-013** | Rebaseline còn ~21 ngày: D10 MVP, D16 feature freeze; EXP-01 P0, EXP-02 P1, EXP-03 stretch | ACCEPTED (deadline thật chưa xác nhận) | Phải ưu tiên tính đúng đắn và sản phẩm nộp trước scope mở rộng | Plan 21 ngày |
| **DEC-014** | Kaggle GPU được phép dùng LightGCN, benchmark CPU Full/Incremental vẫn cùng môi trường | ACCEPTED (tùy tài nguyên) | Tránh so sánh thời gian giữa hai loại máy; log software/hardware/config riêng | D07–D14 |

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
