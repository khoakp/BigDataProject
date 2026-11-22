# BigDataProject v2.0

Đề tài: **Thiết kế và đánh giá hệ thống cập nhật đồ thị tương tác tăng dần và chiến lược làm mới mô hình LightGCN trên dữ liệu MovieLens 32M.**

- **RQ1:** Điều kiện để Incremental Graph Update tương đương Full Rebuild nhưng có chi phí thấp hơn.
- **RQ2:** Đánh đổi giữa Stale Model, Propagation-only và Fine-tuning/Retraining về chi phí, độ mới và chất lượng.
- **Engineering study:** Chu kỳ tạo normalized graph snapshot ảnh hưởng thế nào đến chi phí và độ trễ dữ liệu.

Graph là cumulative historical-positive user–movie, ngưỡng `rating >= 4.0`. Project Owner đã chốt **01/01/2020 00:00 UTC** làm cutoff sau temporal profiling: toàn bộ history trước mốc này là base, phần sau là replay. MovieLens 32M là dataset chính; quy mô graph benchmark và LightGCN training được quyết định riêng. Streamlit là Research Dashboard. Chưa có measured v2 Model Freshness result.

Luồng chính: MovieLens CSV → canonical events → Full Rebuild/Incremental Graph → normalized snapshot → LightGCN → temporal evaluation/inference → Research Dashboard.

## Tài liệu hiện hành

- [Phạm vi và quyết định v2](docs/current/scope-and-decisions.md)
- [Kiến trúc và version contract](docs/current/architecture.md)
- [Protocol thực nghiệm](docs/current/experiment-protocol.md)
- [Tiến độ và bằng chứng](docs/current/progress-and-evidence.md)
- [Architecture Walkthrough tiếng Việt](docs/current/architecture-walkthrough-vi.md)
- [Bảo trì tài liệu](docs/current/documentation-maintenance.md)
- [Chỉ mục nghiên cứu](docs/research/README.md)
- [Hồ sơ lưu trữ](docs/archive/README.md)
- [AGENTS.md](AGENTS.md) — quy tắc triển khai hiện hành.

Measured reports và protocol đã khóa nằm trong research index. Kế hoạch, milestone và thiết kế cũ nằm trong archive; chúng không điều khiển mục tiêu v2.

## Chạy pipeline nhỏ, có thể kiểm tra

Từ root repo, dùng venv có sẵn. Hai toy tests dưới đây tạo dữ liệu trong thư mục tạm và chạy luồng canonical events → Full Rebuild / Incremental replay → equivalence → normalized snapshot/export; chúng không ghi đè dataset hay artifact. Lệnh Full Rebuild sau đó là ví dụ riêng trên canonical 100k, cần chọn thư mục `--out` mới.

```powershell
..\venv\Scripts\python.exe -m unittest tests.test_exp01 tests.test_exp02 -v
..\venv\Scripts\python.exe -m unittest tests.test_rec01_train tests.test_rec01_inference -v
..\venv\Scripts\python.exe -m src.graph.full_builder --events data/processed/canonical_events/canonical_100000.parquet --out data/processed/v2_toy_full
..\venv\Scripts\python.exe -m unittest discover -s tests -v
```

For a fresh canonical prefix from `../ml-32m/ratings.csv`, run `python -m src.ingest.d02_profile --source ... --out <fresh-directory>` only after choosing a fresh output directory; it scans the full CSV. Existing graph benchmark commands are `python -m src.benchmarking.exp01 --help` and `python -m src.benchmarking.exp02 --help`; both require explicit resource planning before large runs. Fixed REC-01 training is `python -m src.models.rec01_train --help`; the local demo is `python -m streamlit run src/app/streamlit_app.py`. These are current runnable entrypoints, while a unified v2 refresh pipeline remains planned.

The sealed REC-01 artifact at `data/processed/rec01_kaggle_runs/version_3/rec01_project/REC01-Kaggle-1M` is verified by checkpoint, graph, mapping and config contracts. Historical source-code hashes remain provenance. The app shows stale model recommendations when feedback advances the graph.
