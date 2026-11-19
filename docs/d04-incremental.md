# D04 — Stateful apply_batch và batch ledger

Ngày thực hiện: 19/11/2026 (Asia/Saigon). Mục tiêu: cập nhật canonical graph theo batch, giữ node IDs và ngăn batch replay làm tăng cạnh/degree. Chưa thực hiện kiểm chứng Full/Incremental với nhiều batch segmentation (D05), normalization hoặc benchmark.

## Phương pháp và implementation

Code [incremental.py](../src/graph/incremental.py) cung cấp `IncrementalGraph(database)` và `apply_batch(batch_id, events: Path)`. Input là canonical Parquet với sáu cột của D02; threshold cố định `rating >= 4.0` theo DEC-005. Class dùng một DuckDB connection, một thread và một writer. Dùng context manager để đóng connection và giải phóng khóa DB.

Luồng xử lý: đọc batch → validate các trường/thứ tự → hash payload → begin transaction → tra ledger → lọc positive và anti-join các cạnh đã tồn tại → dedup trong batch → cấp ID cho endpoint chưa thấy → thêm cạnh → tăng degree từ cạnh vừa thêm → ghi ledger → commit. Nếu có lỗi trước commit, rollback cả registry, edge, degree và ledger. Sơ đồ nguồn cho báo cáo ở [d04-batch-flow.mmd](report-assets/d04-batch-flow.mmd).

Hash là SHA-256 của từng row `[event_id,event_ts,user_raw_id,movie_raw_id,rating,source_row_id]`, JSON compact một dòng/row, sắp theo `event_id`; rating được chuẩn hóa sang float. Hash bao gồm cả events không positive. Cùng `batch_id` và payload hash đã lưu trả `NO_OP`, không đổi persistent state. `graph_version` trong kết quả NO_OP là version lần batch đó được áp dụng ban đầu, không phải current version. Cùng ID khác hash bị reject.

Batch mới phải không rỗng, event IDs liên tục từ watermark + 1 (khởi đầu là 0), order key `(event_ts,user_raw_id,movie_raw_id,source_row_id)` tăng trong batch và qua boundary. Batch chỉ có events âm vẫn tăng version/watermark và ghi ledger nhưng không cấp node mới. Empty batch bị reject. Batch IDs chỉ cần không âm và duy nhất; không bắt buộc liên tiếp vì thứ tự xử lý được kiểm bằng event watermark.

Registry dùng một không gian node_id, cấp tiếp từ max hiện tại, theo first positive event; user trước movie nếu cùng event. Degree tăng đúng bằng số cạnh mới incident: một duplicate positive event không tăng degree. DB là nguồn version chính qua `applied_batches`; D04 chưa xuất snapshot manifest bên ngoài DB, nên chưa có bài toán reconcile DB/file publish hoặc checkpoint recovery T-04 đầy đủ. Mở lại DB ở D04 chỉ kiểm tra persistence của transaction đã commit.

## Experimental Setup

- Run: `D04-100k-10k`, thời điểm bắt đầu `2026-11-19T16:05:21.081564+00:00` (23:05:21 ngày 19/11/2026 Asia/Saigon).
- Input: canonical 100k MovieLens 32M, SHA-256 `8a01bc6f132362f44ab5a206575fb16ab7f34ce44c6bd224b418ea076c7bf519`.
- Python 3.10.11, DuckDB 1.5.6, 1 thread; môi trường máy ở [environment.md](environment.md). Không có sampling/randomization nên seed không áp dụng.
- 10 batches × 10.000 events; DB tạm cô lập mỗi run. Script [d04_probe.py](../src/graph/d04_probe.py) lưu report ở `data/processed/d04_probe/report.json` (ngoài Git).
- Git HEAD `735fe20c6b8f1a508b1b2a4e40aa57400ae4230b`; code trong working tree chưa commit. HEAD không phải commit chứa implementation D04.

## Results and Discussion

| Kiểm chứng | Kết quả thực chạy |
|---|---|
| Unit suite (D02, D03 và 3 tests D04) | 5 tests PASS |
| Cạnh sau 100k events / nodes | 57.925 / 2.819 |
| Tổng degree user / movie | 57.925 / 57.925 |
| Replay từng batch 100k | 10/10 NO_OP; full snapshot trước/sau bằng nhau |
| Node IDs cũ sau mỗi batch | Giữ nguyên |
| Đếm lại degree từng node từ edges | PASS |
| Đóng/mở DB và replay batch cuối | State bằng trước; NO_OP |
| Toy: duplicate positive trong/qua batch | Không tăng degree cho cạnh cũ |
| Toy: batch ID cũ đổi payload | Reject, persistent state bằng trước |
| Toy: null, empty, event IDs trùng/gap, sai order trong/qua batch | Reject, persistent state bằng trước |
| Toy: chỉ event âm | Ledger tiến; không thêm node/cạnh |
| Inject lỗi trước ledger sau khi đã ghi edge/degree | Rollback về snapshot trước; retry sau đó thành công |

Full-state fingerprint của integration probe (gồm ledger và version metadata): `18c1678cbdb3406aec597c9ac540cc091c0f198bc60f01b9423fa81e75fa5563`. Đây không phải hash Parquet hay bằng chứng Full/Incremental equality; số cạnh trùng với D03 chưa đủ chứng minh hai trạng thái giống nhau.

Lần unit run đầu phát hiện lỗi 10 SQL placeholders cho 9 ledger values; đã sửa và rerun suite PASS. Fault injection kiểm chứng exception rollback trước commit, chưa mô phỏng OS crash hoặc concurrent writers. Chưa có số đo timing/RAM/I/O hoặc metric mô hình.

## Tái lập

Từ root repo:

```powershell
..\venv\Scripts\python.exe -m unittest discover -s tests -v
..\venv\Scripts\python.exe -m src.graph.d04_probe
```

Integration probe dùng `assert`; chạy Python bình thường, không dùng `-O`. Output JSON được tạo lại ở `data/processed/d04_probe/report.json`; DB tạm được đóng và dọn sau probe. Test code ở [test_incremental.py](../tests/test_incremental.py) chứa dữ liệu toy và điểm inject failure.

## Report-ready Content

> Cập nhật gia tăng được triển khai bằng giao dịch DuckDB để ghi đồng bộ registry, cạnh, degree và batch ledger. Mỗi batch được nhận diện bằng ID và SHA-256 của payload canonical. Các positive pairs được khử trùng lặp và đối chiếu với trạng thái cạnh hiện tại trước khi thêm; degree chỉ tăng từ cạnh mới. Trên prefix 100.000 sự kiện chia thành 10 batches, hệ thống tạo 57.925 cạnh và giữ nguyên state khi replay từng batch. Thử nghiệm inject lỗi trước commit cho thấy transaction rollback bảo toàn state trước batch. Kết quả xác nhận các hành vi đã kiểm tra của apply_batch; tính tương đương đầy đủ với Full Rebuild và hiệu năng cập nhật còn cần được đánh giá riêng.
