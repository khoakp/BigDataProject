# S10 — Rating và cập nhật graph demo

**Trạng thái 21/11/2026:** DONE trong phạm vi demo local một tiến trình. Ứng dụng Streamlit cho phép xếp tối đa 20 rating vào hàng chờ rồi áp dụng thành một batch lên bản sao của train graph REC-01. Graph demo tăng version; checkpoint LightGCN và Top-20 vẫn ở train graph version 81. [DEC-018](06-decisions-and-risks.md) khóa cách hiểu version.

## What và Why

[`FeedbackStore`](../src/app/feedback.py) kiểm report `PASS`, run ID, SHA-256 của graph đã seal và canonical source; sao chép `graph_build/graph.duckdb` sang thư mục state riêng dưới `data/processed/s10_feedback/`. Graph và checkpoint tải từ Kaggle không bị sửa. Rating mới dùng user/movie raw ID, nửa sao từ 0,5 đến 5,0, không ghi đè cặp user–movie đã có trong train hoặc nhánh demo. Phim được chọn từ 773 train candidates, vì đây là catalog của demo REC-01; rating negative vẫn tạo event nhưng không tạo positive edge.

Khi bấm áp dụng, store ghi intent JSON theo `request_id` UUID trước, tạo Parquet canonical với event IDs liên tiếp, rồi gọi [`IncrementalGraph.apply_batch`](../src/graph/incremental.py). Batch ledger trong DuckDB bảo đảm gửi lại cùng request trả `NO_OP` và không cộng degree hai lần. Khi mở lại, store replay intents; nếu Parquet mất sau khi intent đã ghi, nó tạo lại file từ intent. Đây là phục hồi gián đoạn tiến trình ở demo, chưa có bảo đảm durability khi mất điện hay nhiều server cùng ghi.

App hiển thị `Graph demo version`, `Model graph version`, checkpoint step và số batch kể từ checkpoint. Ví dụ sau một batch: graph 82, model graph 81, checkpoint 300. `model.recommend` luôn dùng embeddings/checkpoint đã chọn từ REC-01; graph mới chưa được materialize hoặc train lại. Warm user vẫn nhận LightGCN cũ; user mới dùng Popularity trên train graph. Đây là chính sách refresh **stale model** được hiển thị trung thực, chưa là REC-02.

## Chạy demo

Từ root repo, chạy lệnh Streamlit trong [README](../README.md). App dùng `REC01_RUN_DIR`, `REC01_EVENTS_PARQUET`, `MOVIELENS_MOVIES_CSV` như S09; `REC01_FEEDBACK_DIR` tùy chọn trỏ tới state demo. Mặc định state ở `data/processed/s10_feedback/REC01-Kaggle-1M/` và nằm ngoài Git. Chọn warm user hoặc nhập user ID mới, thêm rating, áp dụng batch, xem version mới và bảng rating đã ghi. Không dùng thư mục state này làm nguồn train/evaluation REC-01.

## Verification và giới hạn

- [`test_s10_feedback.py`](../tests/test_s10_feedback.py) chạy trên sealed Kaggle artifact thật: batch hai ratings (5,0 và 2,5) tăng graph 81→82, thêm đúng một positive edge, gửi lại UUID trả `NO_OP`, từ chối request conflict/cặp đã đánh giá, mở lại và phục hồi Parquet từ intent. SHA-256 graph Kaggle gốc và Top-20 từ checkpoint không đổi.
- AppTest thao tác form thêm rating rồi áp dụng: UI đổi graph 81→82, model graph giữ 81, checkpoint giữ 300, stale batches 0→1. Toàn bộ 57 local tests PASS ngày 21/11/2026.
- Chỉ có single-process writer lock; chưa kiểm thử multi-process hoặc concurrent server. Không đo latency/throughput của UI và không suy ra quality sau feedback. Rating mới là event thời điểm demo, không được chèn ngược vào temporal test.

**Bản nháp báo cáo:** Hệ thống nhận rating mới theo batch, áp dụng lên canonical positive graph bằng ledger idempotent và hiển thị version tăng. Trong thử nghiệm tích hợp với artifact REC-01, batch gồm một rating positive và một rating negative tăng graph version từ 81 lên 82 nhưng chỉ thêm một cạnh. Checkpoint LightGCN vẫn gắn với graph version 81, vì vậy việc cập nhật graph không được trình bày như cập nhật tham số mô hình hay cải thiện chất lượng gợi ý.
