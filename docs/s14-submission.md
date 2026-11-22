# S14 — Final QA và gói nộp

> **Hồ sơ lịch sử GOAL 03.5:** giữ để truy nguồn quyết định/kết quả cũ. [Phạm vi v2 hiện hành](current/scope-and-decisions.md) · [Tiến độ v2](current/progress-and-evidence.md).

`scripts/s14_package.py` tạo `data/processed/s14_submission/MovieGraph-submission-S14.zip` bằng danh sách chọn tường minh, kèm `SUBMISSION-MANIFEST.json` có SHA-256/byte size từng thành viên. ZIP là gói private local, không upload và không commit. Có source, tests, notebook, requirements, DOCX/PDF báo cáo, PDF/PNG slide, Q&A, canonical 1M, MovieLens `movies.csv`, Kaggle input bundle và sealed run, raw EXP-01/02 CSV, S11 report. Không lấy raw `ratings.csv` 32M, credentials, cache hoặc feedback branch tạm.

```powershell
..\venv\Scripts\python.exe -m scripts.s14_package build
..\venv\Scripts\python.exe -m scripts.s14_package verify --archive data/processed/s14_submission/MovieGraph-submission-S14.zip
```

Đã giải nén gói vào thư mục kiểm tra riêng, chạy `scripts.rec01_kaggle_prep verify-output` trên chính file trong gói: **PASS**, source SHA-256 `f17048bd7b12bea613387c169c65db6ce67442aa7341314f5e18fba39efcfbae`. Lần kiểm đầu có 3 integration tests bị skip vì tests tìm metadata ở `../ml-32m/movies.csv`; package đặt metadata ở `external/ml-32m/movies.csv`. Tests và RUNME đã được cập nhật để dùng biến `MOVIELENS_MOVIES_CSV`. Lần giải nén thứ hai chạy đủ **60 tests PASS, không skip**; repo gốc cũng đạt 60 tests PASS trước khi đóng gói.

Gói có `RUNME.md` giải thích thiết lập môi trường, verify REC-01, chạy tests và mở Streamlit. Slide ngoại tuyến là PDF 12 trang; deck gốc có thể chỉnh sửa nằm trong Spaces. Lần export PPTX qua artifact session chưa trả tệp nên không đánh dấu là đã có PPTX local.

Project Owner sẽ tự điền họ tên/MSSV/lớp, giảng viên, lời cảm ơn, cam đoan ký tên, số buổi/giờ làm việc nhóm thực tế và nơi/hạn nộp nếu có. Các ô này được đánh dấu trong DOCX; không lấy danh tính ví dụ từ mẫu. Sau khi điền, cần xuất lại PDF và rebuild ZIP nếu dùng bản nộp chính thức. Repo vẫn dirty và không tự commit/push theo `AGENTS.md`.
