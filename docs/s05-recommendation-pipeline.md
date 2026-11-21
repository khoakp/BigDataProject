# S05 — Top-20 inference từ REC-01 checkpoint

**Trạng thái 21/11/2026:** loader và CLI đã chạy trên artifact Kaggle REC-01 Version 3; 52 local unit tests PASS. S05 chỉ phục vụ checkpoint và train graph đã niêm phong. Streamlit, rating mới và graph refresh thuộc S09–S10.

## What và Why

[`src/models/rec01_inference.py`](../src/models/rec01_inference.py) tải đúng checkpoint LightGCN được chọn bằng validation, model-index ↔ raw-ID mapping và normalized train graph. Nó xác minh hashes, graph version, source/config/protocol và graph Full/Incremental equivalence trước khi phục vụ. Như vậy gợi ý có thể tái lập từ artifact đã xuất mà không cần mở lại training database hoặc truy cập validation/test labels. [`src/models/rec01_demo.py`](../src/models/rec01_demo.py) là CLI JSON để gọi cùng loader; một lần load có thể phục vụ nhiều raw user IDs.

## How

1. Với warm user có embedding trong train graph, LightGCN propagation tạo embedding từ **train graph**; chấm điểm các phim thuộc train catalog và mask phim user đã có positive train edge. Top-20 sắp theo score giảm, raw movie ID tăng khi bằng điểm.
2. Với cold user không có embedding, trả Top-20 Popularity đếm unique train-positive users theo movie, rồi raw movie ID tăng khi hòa. Fallback không giả định embedding mới đã được học.
3. `movies.csv` chỉ cung cấp title/genres **sau ranking**; thiếu metadata không làm đổi thứ tự/score. Kết quả ghi method, checkpoint step nếu là warm, graph version, candidate count và metadata coverage.

Ví dụ từ root repo `BigDataProject`:

```powershell
..\venv\Scripts\python.exe -m src.models.rec01_demo `
  --run data/processed/rec01_kaggle_runs/version_3/rec01_project/REC01-Kaggle-1M `
  --movies ../ml-32m/movies.csv --user 25062 --user -1
```

CLI in một JSON array gồm hai kết quả; dùng `--device cuda` khi muốn inference trên GPU và môi trường có CUDA. Mặc định dùng CPU. Run dir chứa `report.json`, `model_ids.json`, `model_graph/` và checkpoint được chọn; toàn bộ artifact nằm ngoài Git. [S03](s03-kaggle-workflow.md) hướng dẫn tải và kiểm hash trước khi dùng.

## Verification và giới hạn

- Chạy CLI trên artifact Kaggle thật: user `25062` → `lightgcn`, 20 items, 20/20 metadata; user `-1` → `popularity`, 20 items, 20/20 metadata. Cả hai báo `model_graph_version=81`.
- 8 tests ở [`test_rec01_inference.py`](../tests/test_rec01_inference.py) kiểm checkpoint/graph/code hashes, version và ID mapping, chọn checkpoint, warm train-seen mask, cold fallback, tie handling và metadata. Toàn bộ 52 local tests PASS.
- CLI là bề mặt kiểm tra kỹ thuật, chưa là Streamlit UI. Recommendations dùng checkpoint đã train ở train cutoff; thêm rating hoặc cập nhật graph sẽ không tự làm mới embedding hay Top-20 hiện tại. Quality và giới hạn warm cohort nằm ở [S04](s04-rec01-results.md).
