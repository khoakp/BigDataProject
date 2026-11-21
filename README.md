# BigDataProject

**Định hướng hiện hành (21/11/2026):** [Project Direction v1.0](docs/10-direction-update-v1.md), [kế hoạch sprint S01–S14](docs/11-sprint14-plan.md), [giao thức S01/REC-01](docs/s01-evaluation-spec.md) và [kết quả REC-01 Kaggle](docs/s04-rec01-results.md). README và mô tả D01–D13 bên dưới phản ánh các mốc lịch sử.

Đồ án Introduction to Big Data về hệ thống gợi ý phim cá nhân hóa trên MovieLens 32M. Hai trụ cột là pipeline dữ liệu/đồ thị tăng dần có kiểm chứng và mô hình LightGCN tạo gợi ý Top-K có đánh giá temporal. Popularity baseline và Top-20 REC-01 đã chạy trên canonical prefix 1M; [Streamlit S09–S10](docs/s10-rating-graph-refresh.md) phục vụ gợi ý, lịch sử train, rating mới và graph demo theo batch. [S04](docs/s04-rec01-results.md) ghi số đo và giới hạn.

| Tài liệu | Nội dung |
| --- | --- |
| [Tổng quan](docs/01-project-overview.md) | Phạm vi, câu hỏi nghiên cứu và tiêu chí hoàn thành |
| [Thiết kế](docs/03-system-design.md) | Schema, NodeRegistry, batch replay và snapshot |
| [Kế hoạch 21 ngày](docs/04-project-plan.md) | Công việc D01–D21 và tiêu chí nghiệm thu |
| [Quyết định và rủi ro](docs/06-decisions-and-risks.md) | ADR và phương án dự phòng |
| [Nhật ký thực nghiệm](docs/07-experiment-log.md) | Protocol và mẫu ghi kết quả |
| [Ghi chú báo cáo](docs/08-report-notes.md) | Bản nháp học thuật theo trạng thái bằng chứng |
| [Ma trận bằng chứng](docs/09-report-evidence.md) | Claim gắn với tài liệu, dữ liệu, code và test |
| [Dàn ý báo cáo](docs/08-final-report-outline.md) | Bố cục báo cáo và câu hỏi bảo vệ |
| [Báo cáo IUH S12](docs/Bao-cao-MovieGraph-IUH.docx) | Bản DOCX theo mẫu đã cung cấp; Project Owner tự điền thông tin cá nhân và xác nhận nhóm |
| [Báo cáo PDF](docs/Bao-cao-MovieGraph-IUH.pdf) | Bản xem trước 18 trang đã kiểm dàn trang |
| [Báo cáo kỹ thuật](docs/final-report.md) | Bản nội dung kỹ thuật và links tới evidence |
| [Reproducibility S12](docs/s12-reproducibility.md) | Môi trường local/Kaggle và hash manifest của source/artifacts |
| [Slide và Q&A S13](docs/s13-defense-qa.md) | Deck Spaces, PDF 12 trang, câu hỏi bảo vệ và kịch bản demo |
| [Gói nộp S14](docs/s14-submission.md) | ZIP private, manifest SHA-256 và kết quả kiểm tra từ bản giải nén |
| [AGENTS.md](AGENTS.md) | Hướng dẫn làm việc với coding agent |

## Bắt đầu

Để mở giao diện local sau khi đã [tải và xác minh REC-01 artifact](docs/s03-kaggle-workflow.md):

```powershell
..\venv\Scripts\python.exe -m pip install -r requirements-app.txt
..\venv\Scripts\python.exe -m streamlit run src/app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Mở `http://127.0.0.1:8501`; [hướng dẫn S10](docs/s10-rating-graph-refresh.md) ghi vị trí dữ liệu và biến môi trường. Rating mới chỉ cập nhật bản graph demo; checkpoint LightGCN vẫn ở version 81 cho đến khi có model refresh thực sự.

Repo đã hoàn thành D01–D13. G1/G2/G3 đạt trong phạm vi đã ghi. D12 EXP-01 có năm repetitions trên sáu cặp H/Δ và run riêng H=2M; D13 EXP-02 so `R=1`/`R=5` có cost, staleness và normalized equality ở v5/v10. Venv dùng PyTorch 2.8.0+cu126; MovieLens 32M ở `../ml-32m/`. [D10 và demo](docs/d10-mvp.md), [D12 EXP-01](docs/d12-exp01-results.md), [D13 EXP-02](docs/d13-exp02-results.md).

Chạy unit tests và các probe từ root repo.

```powershell
..\venv\Scripts\python.exe -m unittest discover -s tests -v
..\venv\Scripts\python.exe -m src.graph.d04_probe
..\venv\Scripts\python.exe -m src.graph.d05_probe
..\venv\Scripts\python.exe -m src.graph.d06_probe
..\venv\Scripts\python.exe -m src.models.d07_probe
..\venv\Scripts\python.exe -m src.models.d08_baseline --steps 100 --device cuda
..\venv\Scripts\python.exe -m src.models.d09_pipeline --device cuda --steps 100
..\venv\Scripts\python.exe -m src.models.d10_mvp --device cuda --steps 100 --out data/processed/d10_mvp/my-clean-run
..\venv\Scripts\python.exe -m src.models.d10_demo data/processed/d10_mvp/my-clean-run
..\venv\Scripts\python.exe -m src.benchmarking.exp01 pilot --ratings ..\ml-32m\ratings.csv --out data/processed/d11_exp01/my-pilot --csv results/raw/my-exp01-pilot.csv
..\venv\Scripts\python.exe -m src.benchmarking.exp01 repeat --out data/processed/d12_exp01/my-repeat --csv results/raw/my-exp01-repeat.csv
..\venv\Scripts\python.exe -m src.benchmarking.exp02 run --out data/processed/d13_exp02/my-run --csv results/raw/my-exp02.csv
```

Prompt mở đầu cho Codex:

> Đọc AGENTS.md và milestone hiện hành trong docs/04-project-plan.md. Đối chiếu code, tài liệu và bằng chứng; triển khai trong phạm vi đã duyệt, kiểm thử, cập nhật docs/05, docs/07, docs/08 và docs/09, rồi giải thích kiến thức và giới hạn cho mình. Nếu cần ví dụ, dùng dữ liệu toy.
