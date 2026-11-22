# D05 — Exact Full/Incremental equivalence và G1

> **Hồ sơ lịch sử GOAL 03.5:** giữ để truy nguồn quyết định/kết quả cũ. [Phạm vi v2 hiện hành](../../../current/scope-and-decisions.md) · [Tiến độ v2](../../../current/progress-and-evidence.md).

**Kết quả: D05 DONE, G1 PASS trong phạm vi toy và canonical MovieLens 32M prefix 100k đã kiểm tra.** Không suy rộng sang full 32M, normalized graph, model hoặc hiệu năng. Ngày thực hiện: 19/11/2026 (Asia/Saigon).

## Research Objective và lý thuyết kiểm chứng

Với cùng chuỗi sự kiện đến cutoff, Full Rebuild đọc lại prefix và Incremental áp dụng lần lượt các batch phải tạo cùng logical graph. Số cạnh hoặc tổng degree bằng nhau chỉ là điều kiện cần: hai graph khác cạnh có thể cùng số lượng. D05 vì vậy so sánh toàn bộ các dòng state đã sắp thứ tự, đồng thời ghi SHA-256 để truy vết. Đây là kiểm chứng thực nghiệm hữu hạn, không phải chứng minh cho mọi chuỗi sự kiện có thể có.

## Methodology và implementation

Harness [d05_probe.py](../../../../src/graph/d05_probe.py) cắt cùng input thành nhiều segmentation. Sau **mỗi** batch, script tạo lại canonical prefix đúng cutoff, gọi [Full Rebuild](../../../../src/graph/full_builder.py) từ đầu và so với [Incremental](../../../../src/graph/incremental.py). Module [equivalence.py](../../../../src/graph/equivalence.py) đọc output Full và query Incremental theo projection sau:

| Thành phần | Các trường đối chiếu exact |
|---|---|
| Registry | `node_id, node_type, raw_id, first_event_id` |
| Edge set | `user_node_id, movie_node_id, first_event_id` |
| Degree | `node_id, degree` |

Edge endpoint IDs được đối chiếu cùng toàn bộ registry, nên mapping raw IDs cũng phải khớp. `created_version`, `updated_version`, batch ledger và batch hashes mô tả lịch sử xử lý và có thể khác giữa segmentation; chúng không thuộc logical graph projection. Harness kiểm tra riêng version hợp lệ, số ledger entries, watermark, tổng events/inserted edges và node IDs cũ giữ nguyên. Sơ đồ: [d05-equivalence-flow.mmd](../../../report-assets/d05-equivalence-flow.mmd).

Sau mỗi batch, replay ngay phải trả `NO_OP` và full snapshot (gồm ledger/version metadata) giữ nguyên. Sau khi xử lý toàn bộ input, replay các batch cũ theo thứ tự ngược cũng phải giữ nguyên snapshot. Hash conflict và rollback vẫn do unit tests D04 kiểm tra. Comparator có mutation test: giữ số dòng nhưng đổi registry raw ID, edge endpoint hoặc degree đều bị phát hiện.

## Experimental Setup và tái lập

- Run ID: `D05-equivalence`; bắt đầu `2026-11-19T16:20:13.499173+00:00` (23:20:13 ngày 19/11/2026 Asia/Saigon).
- Python 3.10.11, DuckDB 1.5.6, một thread; máy theo [environment.md](../../../research/environment.md). Rule `rating >= 4.0` theo DEC-005. Segmentation cố định, không sampling; seed không áp dụng.
- Input thật: `data/processed/canonical_events/canonical_100000.parquet`; SHA-256 `8a01bc6f132362f44ab5a206575fb16ab7f34ce44c6bd224b418ea076c7bf519`; nguồn canonical theo [D02](d02-data-profile.md).
- Toy 8 events được tạo trực tiếp trong harness: prefix toàn rating âm; timestamp ties; raw ID giống nhau ở hai node types; duplicate positive pair; positive cũ có rating âm về sau; node chỉ có rating âm. Toy này dùng để kiểm tra hành vi, không đại diện phân phối MovieLens.
- Mỗi segmentation có DB tạm riêng, không dùng state của segmentation trước. Không tạo train/validation split vì chỉ kiểm tra graph prefix và không đánh giá mô hình.
- Git HEAD lúc chạy `735fe20c6b8f1a508b1b2a4e40aa57400ae4230b`; code D05 chưa commit. Report có SHA-256 của cả bốn source files để nhận diện working tree đã chạy.

```powershell
..\venv\Scripts\python.exe -m unittest discover -s tests -v
..\venv\Scripts\python.exe -m src.graph.d05_probe
```

Report từng cutoff, hashes, provenance: `data/processed/d05_probe/report.json` ngoài Git. Script ghi `status=FAIL` và lỗi nếu probe thất bại; kiểm tra status/exception khi tái lập. DB và Parquet tạm được đóng/dọn sau run.

## Results and Discussion

| Dataset | Kích thước các batch | Số boundary đối chiếu | Kết quả |
|---|---|---:|---|
| Toy 8 events | 8 × 1 | 8 | PASS |
| Toy | 4 × 2 | 4 | PASS |
| Toy | 3, 1, 4 | 3 | PASS |
| Toy | 8 | 1 | PASS |
| MovieLens prefix 100k | 10 × 10.000 | 10 | PASS |
| MovieLens prefix 100k | 4 × 25.000 | 4 | PASS |
| MovieLens prefix 100k | 33.333, 33.333, 33.333, 1 | 4 | PASS |
| MovieLens prefix 100k | 100.000 | 1 | PASS |

Tổng cộng 35 boundary checks (có cutoff trùng nhau giữa segmentation): 105 component comparisons của registry/edge/degree đều exact. 35 immediate replays và 35 historical replays đều `NO_OP`, snapshot giữ nguyên. Final logical-state fingerprints bằng nhau giữa 4 segmentation của từng dataset; hash là thông tin bổ trợ cho phép so sánh từng dòng tại boundary.

Toy cuối có 4 cạnh, 4 node. Prefix 100k cuối có **57.925 cạnh, 2.819 node**. Final fingerprint của logical state toy: `a420db1b5fbcb07dda257eee9c834bf0cf21fd3df37d4349c1336eb016e54e65`; prefix 100k: `45ef2c054e11ed69c30ce64af424800c92a46df323c9be75b7e55edc9e08d378`. Khác fingerprint D04 là bình thường vì D04 hash full snapshot gồm cả ledger/version metadata; D05 hash logical projection.

Unit suite: **7 tests PASS** (D02–D05). D05 phát hiện và sửa lỗi Full Rebuild trên prefix chưa có positive edge: SQL `SUM` của tập rỗng trả NULL thay vì 0; chuyển sang `coalesce(sum(...),0)`. Regression test `test_full_rebuild_all_negative_prefix` thất bại trước khi sửa và PASS sau sửa. Full Rebuild cũng được đóng connection bằng context manager để chạy nhiều cutoff không giữ tài nguyên thừa. Không phát hiện mismatch Full/Incremental trong probe đã chạy sau sửa.

## Report-ready Content và giới hạn

> Tính tương đương của cập nhật gia tăng được kiểm chứng bằng cách xây dựng lại toàn bộ graph tại từng boundary và so sánh chính xác node registry, edge set và degree. Bốn cách chia batch được sử dụng cho dữ liệu toy và bốn cách cho prefix MovieLens 100.000 sự kiện. Cả 35 lần đối chiếu đều đạt; 70 lượt phát lại batch không thay đổi state. Kết quả hỗ trợ tính đúng đắn của implementation trên các trường hợp đã kiểm tra và hoàn thành gate G1. Kết luận này chưa bao gồm biểu diễn normalized, chất lượng gợi ý hoặc lợi ích về thời gian và bộ nhớ.

Chưa kiểm tra equivalence ở 1M/full32M, segmentation ngẫu nhiên, multi-writer, OS crash, checkpoint+delta recovery, normalized matrix hoặc temporal evaluation. Chưa đo EXP-01/02 và chưa có speedup để báo cáo. Bước theo plan tiếp theo là D06 normalization và version semantics; chỉ bắt đầu khi được giao milestone.
