# D03 — Positive rule và Full Rebuild reference

> **Hồ sơ lịch sử GOAL 03.5:** giữ để truy nguồn quyết định/kết quả cũ. [Phạm vi v2 hiện hành](current/scope-and-decisions.md) · [Tiến độ v2](current/progress-and-evidence.md).

Ngày chạy: 19/11/2026. Input: canonical 100k prefix của MovieLens 32M, SHA-256 `8a01bc6f132362f44ab5a206575fb16ab7f34ce44c6bd224b418ea076c7bf519` ([D02](d02-data-profile.md)). Python 3.10.11, DuckDB 1.5.6, 1 thread. Code [full_builder.py](../src/graph/full_builder.py), test [test_full_builder.py](../tests/test_full_builder.py). Repo HEAD trước thay đổi `735fe20c6b8f1a508b1b2a4e40aa57400ae4230b`; D03 chưa commit.

## Research Objective và phương pháp

Mục tiêu D03 là định nghĩa cạnh positive và tạo baseline **đúng về trạng thái** cho các phép so sánh incremental ở D05. Rule đã chọn theo DEC-005: `rating >= 4.0`. Một `(user_raw_id,movie_raw_id)` có ít nhất một event dương tạo một cạnh binary, `first_event_id` là event dương đầu tiên. Không xóa cạnh khi có event khác; không gán trọng số rating cho cạnh. Node chỉ được cấp ID khi lần đầu tham gia cạnh positive; tại cùng event, cấp user trước movie. Policy này là lựa chọn nghiên cứu, **không phải kết quả tối ưu hóa chất lượng gợi ý**.

Profile toàn bộ 32.000.204 ratings: các mức rating 0.5/1/1.5/2/2.5/3/3.5/4/4.5/5 lần lượt có `525132 / 946675 / 531063 / 2028622 / 1685386 / 6054990 / 4290105 / 8367654 / 2974000 / 4596577` events. Rule `>=4.0` chọn 15.938.231 events, chiếm khoảng 49,81% toàn bộ ratings. Kết quả này mô tả **mật độ theo ngưỡng**, chưa đo Recall/NDCG hoặc lợi ích downstream. Ngưỡng 3.5 sẽ tạo graph dày hơn; 4.5 sẽ thưa hơn. Giữ 4.0 đúng theo policy đã đề xuất, đánh giá độ nhạy ở giai đoạn sau nếu còn thời gian.

## Implementation và setup

`rebuild()` đọc canonical Parquet, lọc positive, `GROUP BY` raw pair để lấy first event, lập registry từ first appearance của mỗi endpoint, join thành `edges`, rồi đếm `degrees` từ chính các cạnh đã insert. Output `registry.parquet`, `edges.parquet`, `degrees.parquet`, `manifest.json` ở `data/processed/full_reference_100000/` (ngoài Git). Chưa có batch state, transaction, idempotency hoặc normalized graph; các phần đó thuộc D04–D06.

## Results and Discussion (đã kiểm chứng)

| Metric, prefix 100k | Kết quả |
|---|---:|
| Canonical events | 100.000 |
| Positive events | 57.925 |
| Unique positive edges | 57.925 |
| User nodes / movie nodes | 2.251 / 568 |
| Tổng node | 2.819 |
| Tổng degree user / movie | 57.925 / 57.925 |
| Duplicate edges | 0 |
| Degree mismatches khi đếm lại từ edges | 0 |
| Registry first-event mismatches với canonical positive input | 0 |

Toy test có 5 events, gồm một rating dưới ngưỡng và một duplicate positive pair: 4 positive events tạo 3 cạnh, 4 node; degree user và movie đều 3. Node IDs theo lần xuất hiện đầu là `user 10 → 0`, `movie 50 → 1`, `user 20 → 2`, `movie 60 → 3`. `unittest discover` chạy 2 tests (D02 + D03), đều PASS. Kiểm tra 100k là đối chiếu lại file state với distinct positive pairs, degree recomputation và first event từ canonical prefix; chưa phải kiểm tra Full vs Incremental.

Hashes output của run 100k: registry `fe7f6984f586b1a8b9e4d088c1ba5ead62e13ac8a8c4275d6bd0ae17f79522df`; edges `6c704b483ed4fdbe8a7985b770897bb0f2a7ffdc21167bf3df0511e7bd5dcf65`; degrees `799688bc63d85a2ab13fa5aeb66703103fabb69298f079678d82587f19a13042`. Đây là fingerprint của run trong môi trường hiện tại, không phải benchmark tốc độ.

## Report-ready Content

> Đồ thị được định nghĩa từ các rating có giá trị tối thiểu 4,0 trên thang 5 điểm. Mỗi cặp user–movie chỉ tạo một cạnh binary, bất kể số sự kiện positive, nhằm biểu diễn việc đã từng có tương tác dương. Full rebuild từ canonical prefix 100.000 sự kiện tạo 57.925 cạnh và 2.819 node; tổng bậc hai phía đều bằng số cạnh. Các đối chiếu với input và cạnh đã xuất không phát hiện cạnh trùng, sai degree hoặc sai lần xuất hiện đầu. Kết quả này xác nhận baseline full rebuild trên prefix đã kiểm tra; chưa cho phép kết luận tính tương đương hay hiệu năng của cập nhật gia tăng.

Tái lập: `..\venv\Scripts\python.exe -m unittest discover -s tests -v` và `..\venv\Scripts\python.exe src\graph\full_builder.py` từ root repo. Cần Khoa xác nhận rule 4.0 phù hợp yêu cầu giảng viên trước khi khóa benchmark cuối; thay rule sau đó đòi chạy lại graph và các phép so sánh.
