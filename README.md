# BigDataProject

Đồ án Introduction to Big Data về cập nhật đồ thị user–movie gia tăng trên MovieLens 32M. Mục tiêu chính là kiểm chứng đồ thị cập nhật theo batch tương đương full rebuild và đo chi phí của hai cách trong cùng môi trường. LightGCN dùng snapshot đồ thị để tạo gợi ý phim.

| Tài liệu | Nội dung |
| --- | --- |
| [Tổng quan](docs/01-project-overview.md) | Phạm vi, câu hỏi nghiên cứu và tiêu chí hoàn thành |
| [Thiết kế](docs/03-system-design.md) | Schema, NodeRegistry, batch replay và snapshot |
| [Kế hoạch 21 ngày](docs/04-project-plan.md) | Công việc D01–D21 và tiêu chí nghiệm thu |
| [Quyết định và rủi ro](docs/06-decisions-and-risks.md) | ADR và phương án dự phòng |
| [Nhật ký thực nghiệm](docs/07-experiment-log.md) | Protocol và mẫu ghi kết quả |
| [AGENTS.md](AGENTS.md) | Hướng dẫn làm việc với coding agent |

## Bắt đầu

Commit khởi tạo chứa tài liệu và quy ước làm việc. Đối chiếu code hiện có trên máy với D01–D02 trước khi triển khai thêm. Đặt `ratings.csv` và `movies.csv` trong `data/raw/` khi chạy cục bộ, rồi ghi thông tin môi trường và checksum theo D01.

Prompt mở đầu cho Codex:

> Đọc AGENTS.md và phần D01–D02 trong docs/04-project-plan.md. Đối chiếu với các file hiện có. Đề xuất bước nhỏ tiếp theo và tiêu chí nghiệm thu để mình tự triển khai. Nếu cần ví dụ, dùng dữ liệu toy.
