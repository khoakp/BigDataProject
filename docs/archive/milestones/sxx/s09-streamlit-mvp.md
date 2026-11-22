# S09 — Streamlit recommendation MVP

> **Hồ sơ lịch sử GOAL 03.5:** giữ để truy nguồn quyết định/kết quả cũ. [Phạm vi v2 hiện hành](../../../current/scope-and-decisions.md) · [Tiến độ v2](../../../current/progress-and-evidence.md).

**Trạng thái 21/11/2026:** ứng dụng Streamlit đã chạy local với artifact Kaggle REC-01 Version 3, kiểm tra bằng AppTest và trong trình duyệt. Nó phục vụ snapshot **chỉ đọc**: chọn warm user, xem Top-20 LightGCN, lịch sử rating train và train catalog; user mới nhận Popularity fallback. Ghi rating mới và cập nhật graph thuộc S10.

## What và Why

[`streamlit_app.py`](../../../../src/app/streamlit_app.py) dùng [`Rec01Inference`](../../../../src/models/rec01_inference.py) đã kiểm hash để phục vụ checkpoint được chọn ở bước 300. Không train lại khi người dùng chuyển trang hoặc chọn user. [`rec01_view.py`](../../../../src/app/rec01_view.py) chỉ lấy ratings có `event_ts <= train_end_ts` từ canonical source đã đối chiếu SHA-256 với training report; title/genres được bổ sung sau ranking. Lịch sử hiển thị cả rating dưới ngưỡng positive, còn train graph và candidate universe dùng `rating >= 4.0` theo REC-01.

UI cho biết graph version 81, checkpoint step 300, 773 phim ứng viên và 392.626 cạnh positive train. Thông báo snapshot cố định ngăn hiểu nhầm rằng rating/graph mới tự đổi checkpoint. User mới không có embedding nên dùng train-only Popularity.

## Cách chạy

Từ root repo `BigDataProject`, dùng venv hiện có với Python 3.10, PyTorch và DuckDB. Cài Streamlit đã ghim phiên bản trong [`requirements-app.txt`](../../../../requirements-app.txt):

```powershell
..\venv\Scripts\python.exe -m pip install -r requirements-app.txt
..\venv\Scripts\python.exe -m streamlit run src/app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Mở `http://127.0.0.1:8501`. Mặc định app tìm run ở `data/processed/rec01_kaggle_runs/version_3/rec01_project/REC01-Kaggle-1M`, metadata ở `../ml-32m/movies.csv`, canonical events ở `data/processed/canonical_events/canonical_1000000.parquet`. Có thể đổi bằng `REC01_RUN_DIR`, `MOVIELENS_MOVIES_CSV`, `REC01_EVENTS_PARQUET`; artifact và MovieLens data nằm ngoài Git. Trước khi chạy, dùng [S03 verifier](../../../research/s03-kaggle-workflow.md) để đối chiếu output tải từ Kaggle.

## Verification

- Streamlit 1.54.0, Python 3.10, PyTorch 2.8.0+cu126 và DuckDB 1.5.6 trên venv local. HTTP `/_stcore/health` trả `200 ok` khi server chạy local.
- [`test_s09_view.py`](../../../../tests/test_s09_view.py) kiểm cutoff không lộ future ratings, thứ tự lịch sử, metadata và catalog sorting. [`test_s09_app.py`](../../../../tests/test_s09_app.py) chạy AppTest trên artifact thật: warm default, đổi sang warm user 25062, rồi cold fallback; không có exception/error, mỗi luồng có 20 gợi ý và catalog có 773 phim. Toàn bộ 55 local tests PASS.
- Trình duyệt local: warm user 173074 hiển thị LightGCN Top-20, 667 train ratings (113 positive), version 81/checkpoint 300; user mới hiển thị Popularity Top-20, lịch sử train rỗng và catalog tab có các phim ứng viên. Title/genres xuất hiện trong bảng; metadata coverage 20/20 ở hai luồng.

**Giới hạn:** chưa có rating form hoặc graph refresh; mô hình luôn dùng checkpoint trên train snapshot cũ. Không dùng giao diện này để khẳng định chất lượng trên cold users hoặc toàn bộ 32M. [S04](../../../research/s04-rec01-results.md) là nơi ghi quality trên warm evaluation cohort.
