# 05 — Nhật ký làm việc hằng ngày

> **Vai trò:** nhật ký thực thi, không phải nhật ký học thuật ([02](02-research-journal.md)), không thay sổ thực nghiệm ([07](07-experiment-log.md)).  
> **Trạng thái:** mẫu cập nhật 10/11/2026; **chưa có bản ghi triển khai code** vì chưa có log từ bạn.

## 1. Quy tắc 5–10 phút mỗi buổi

- Ghi cả việc chưa xong, blockers, lỗi, chạy thất bại.
- Có `Task ID` D01–D21 theo [04 — Plan 21 ngày](04-project-plan.md) và `commit SHA` nếu đã commit.
- Không ghi con số thời gian/VRAM giả định; nếu chưa đo, ghi `Chưa đo`.
- Research question → [02]; quyết định design → [06]; benchmark → [07].
- Trong ngày không làm dự án: có thể ghi `Ngày nghỉ/thi` để phản ánh timeline thực.

## 2. Mẫu copy cho ngày làm việc

```markdown
## YYYY-MM-DD — D?? (Work session #)

### Mục tiêu trước khi làm
- Task ID / priority:
- Dự kiến tập trung: ___ phút
- Definition of Done (định nghĩa xong):

### Việc đã thực hiện
- Thời gian thực tế: ___ phút
- Hoàn thành: [ ]
- Partial / chưa xong:
- File thay đổi / commit:
- Command / test đã chạy:

### Bằng chứng và liên kết
- Test report/path:
- Dataset/graph version:
- `ratings_source_hash` / `movies_source_hash` nếu làm metadata:
- `metadata_found` coverage / Top-K invariance test (nếu liên quan):
- EXP-xx / result CSV (nếu có):
- Screenshot/log:

### Blocker, lỗi và hướng xử lý
- Vấn đề:
- Mức ảnh hưởng P0/P1/P2:
- Đã thử:
- Hướng tiếp theo:
- Tạo/đổi Risk/DEC?:

### Chốt ngày
- Trạng thái task: TODO / DOING / BLOCKED / DONE / CANCELLED
- Việc ưu tiên ngày mai:
- Cần hỏi giảng viên/mentor:
```

## 3. Mẫu phiên ngắn cho ngày bận

```text
Ngày: ____  |  Task: D___  |  Focused: ___ phút
Đã làm: ____
Bằng chứng: ____
Blocker: ____
Ngày mai: ____
```

## 4. Bản ghi khởi tạo — LẬP KẾ HOẠCH, KHÔNG PHẢI ĐÃ TRIỂN KHAI

### INIT — 20/10/2026

- **Loại:** ghi nhận bối cảnh đã bàn luận trong chat, không ghi nhận coding đã hoàn thành.
- **Đã thống nhất về hướng:** Versioned Incremental Interaction Graph + Deferred Normalization + Controlled LightGCN Refresh.
- **Bổ sung quyết định 10/11/2026:** MovieLens 32M làm dữ liệu chính; `movies.csv` phục vụ metadata post-ranking; 21 ngày là lịch rút gọn, không được coi là tiến độ đã hoàn thành.
- **Thời lượng thực tế:** chưa có dữ liệu.
- **Chưa kiểm tra:** CPU/RAM, CUDA/PyG compatibility, MovieLens raw checksum, data pipeline, VRAM safe scale.
- **Công việc tiếp theo khi bắt đầu:** `D01` kiểm tra phần cứng, OS và môi trường.
- **Bằng chứng:** [01 — Overview](01-project-overview.md), [04 — Plan](04-project-plan.md).

## 5. Bảng theo dõi 3 tuần (điền dần)

| Tuần | Task DONE/plan | Giờ thực tế | Blocker chính | Milestone | Review link |
|---|---:|---:|---|---|---|
| W1 (D01–D07) | Chưa xác nhận | — | — | Canonical + graph + GPU smoke: chờ | — |
| W2 (D08–D14) | Chưa xác nhận | — | — | D10 MVP; EXP-01/02: chờ | — |
| W3 (D15–D21) | Chưa xác nhận | — | — | D16 freeze; submission: chờ | — |

## 6. Điều cần ghi trong commit/message

Gợi ý format `D04: add idempotent incremental batch ledger` hoặc `EXP-01: record 1M/10k warm-cache repeats`; metadata: `D09: post-ranking movie lookup invariance tests`. Mỗi log nêu rõ file/commit và trạng thái; không cần chép nguyên toàn bộ stacktrace vào Markdown — lưu log raw vào `results/logs/`.
