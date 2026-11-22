# BigDataProject — Tóm tắt cập nhật tài liệu (10/11/2026)

## Phạm vi đã chốt trong cuộc trao đổi

1. **MovieLens 32M là nguồn dữ liệu chính** cho canonical events, incremental user–movie graph, LightGCN và các phép thử; RetailRocket hoãn làm hướng ETL mở rộng.
2. `ratings.csv` dùng các cột `userId`, `movieId`, `rating`, `timestamp`. **Đề xuất** `rating >= 4.0` tạo historical-positive binary edge; cần profile rồi chốt bằng ADR trước khi chạy benchmark.
3. `movies.csv` chứa `movieId`, `title`, `genres` được kiểm tra riêng và JOIN **sau** khi mô hình sinh Top-K movie IDs. Tên phim/thể loại chỉ dùng hiển thị và phân tích mô tả. Không dùng genre làm node/edge/model feature trong MVP 21 ngày.
4. Metadata enrichment không được làm thay row count, score/rank, raw Recall@K/NDCG@K; dùng LEFT JOIN, validate `movieId` unique, báo missing/coverage.
5. **Kế hoạch hoạt động 21 ngày tương đối:** D10 MVP; D16 feature freeze; EXP-01 P0, EXP-02 P1, EXP-03 stretch. Hạn nộp/giờ nộp chính thức chưa được xác nhận.
6. GPU laptop hoặc Kaggle chỉ cho model nếu cần; Full vs Incremental benchmark phải chạy trong cùng môi trường kiểm soát. Không khai báo kết quả chưa đo.

## File được cập nhật

- `01-project-overview.md`: định hướng, phạm vi metadata, tiêu chí hoàn thành, kế hoạch ba tuần.
- `02-research-journal.md`: journal về metadata / lựa chọn dataset, reading references, các research questions mới.
- `03-system-design.md`: nhánh `movies.csv` hậu xử lý trên sơ đồ, data contract, test T-10/11/12, module metadata.
- `04-project-plan.md`: **được thay bằng lịch 21 ngày hiện hành**, bao gồm hàng ngày và gates.
- `05-work-log.md`: task IDs D01–D21, log metadata và checklist theo ba tuần.
- `06-decisions-and-risks.md`: DEC-011/12/13/14, rủi ro metadata/Kaggle và gates mới; DEC-010 cũ SUPERSEDED.
- `07-experiment-log.md`: EXP-01/02 rút gọn, EXP-03 stretch, AN-01 genre profiling không phải thí nghiệm chứng minh cải thiện model.
- `08-final-report-outline.md`: bổ sung metadata trong methods/results, checklist bằng chứng và demo Top-K có tên/thể loại.

## Tài liệu bổ sung

- `04-project-plan-21days.md`: bản sao kế hoạch 21 ngày, thuận tiện tham chiếu trực tiếp.
- `04-project-plan-7weeks-archived.md`: bản kế hoạch 7 tuần cũ để đối chiếu; **không** dùng làm lịch chính thức.

## Những gì chưa được xác nhận

- Chưa có code, unit test, checkpoint hay benchmark nào được xác nhận hoàn thành bởi việc cập nhật tài liệu này.
- Chưa biết deadline chính thức, CPU/RAM/SSD và GPU scale an toàn; không suy đoán.
- Quy tắc `rating >= 4.0` chưa chốt cuối cùng trước profiling.
- RQ3/EXP-03 là phạm vi tùy chọn phải ghi rõ trước báo cáo cuối kỳ.
