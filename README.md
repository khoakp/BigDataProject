# BigDataProject

Đồ án Introduction to Big Data về cập nhật đồ thị user–movie gia tăng trên MovieLens 32M. Mục tiêu chính là kiểm chứng đồ thị cập nhật theo batch tương đương full rebuild và đo chi phí của hai cách trong cùng môi trường. LightGCN dùng snapshot đồ thị để tạo gợi ý phim.

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
| [AGENTS.md](AGENTS.md) | Hướng dẫn làm việc với coding agent |

## Bắt đầu

Repo đã hoàn thành D01–D07. G1 đạt trên toy/100k và G2 đạt sau CUDA smoke trên toy/100k/1M. Venv dùng PyTorch 2.8.0+cu126. Dữ liệu MovieLens 32M ở `../ml-32m/`. Báo cáo ghi rõ các điều kiện chạy, bộ nhớ và phạm vi kiểm tra. [D05](docs/d05-equivalence.md), [D06](docs/d06-normalization.md), [D07](docs/d07-lightgcn-smoke.md), [môi trường](docs/environment.md).

Chạy unit tests và các probe từ root repo.

```powershell
..\venv\Scripts\python.exe -m unittest discover -s tests -v
..\venv\Scripts\python.exe -m src.graph.d04_probe
..\venv\Scripts\python.exe -m src.graph.d05_probe
..\venv\Scripts\python.exe -m src.graph.d06_probe
..\venv\Scripts\python.exe -m src.models.d07_probe
```

Prompt mở đầu cho Codex:

> Đọc AGENTS.md và milestone hiện hành trong docs/04-project-plan.md. Đối chiếu code, tài liệu và bằng chứng; triển khai trong phạm vi đã duyệt, kiểm thử, cập nhật docs/05, docs/07, docs/08 và docs/09, rồi giải thích kiến thức và giới hạn cho mình. Nếu cần ví dụ, dùng dữ liệu toy.
