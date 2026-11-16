# BigDataProject

- Dùng `docs/04-project-plan.md` để xác định milestone và tiêu chí nghiệm thu. Đọc phần liên quan trong `docs/03-system-design.md` và kiểm tra ADR trong `docs/06-decisions-and-risks.md`.
- Khoa xây dự án theo hướng vừa học vừa làm. Với yêu cầu hướng dẫn, giải thích lựa chọn, làm rõ input/output và dùng ví dụ nhỏ để Khoa tự triển khai. Khi được giao sửa code, giữ thay đổi theo phạm vi milestone.
- Giữ event ordering xác định, node ID ổn định và batch replay idempotent. Dùng full rebuild làm đối chứng cho edge set, registry và degree trên dữ liệu toy trước khi đo hiệu năng.
- `rating >= 4.0` là ngưỡng đang đề xuất. Chốt bằng profiling và cập nhật ADR trước benchmark. Dùng metadata phim sau khi xếp hạng.
- Ghi kết quả đo thực tế vào `docs/07-experiment-log.md`, kèm cấu hình, dữ liệu và commit. Dùng `docs/05-work-log.md` để lưu tiến độ và bước tiếp theo khi kết thúc milestone.
- Giữ dữ liệu MovieLens, checkpoint, file môi trường chứa secret và kết quả thô dung lượng lớn ngoài Git.
