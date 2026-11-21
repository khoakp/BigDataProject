# S12 — Báo cáo, environment và artifact freeze candidate

**Trạng thái 21/11/2026:** [bản báo cáo kỹ thuật](final-report.md) và [báo cáo theo mẫu IUH](Bao-cao-MovieGraph-IUH.docx) ([PDF](Bao-cao-MovieGraph-IUH.pdf)) đã được viết từ logs thật. [`s12-freeze-manifest.json`](s12-freeze-manifest.json) ghi SHA-256/byte size của source/report/figure files và 10 evidence artifacts hiện có. Đây là **freeze candidate của worktree local**: Git HEAD `9cc6d69e6386358b12f0a6b620d8b9d10eb41ba6` chưa chứa các thay đổi S01–S14 (`git_worktree_dirty=true`). Chưa có commit/tag/release. Mẫu đã được cung cấp; họ tên/MSSV/lớp, giảng viên, cam đoan và thông tin nhóm thực tế còn phải được người thực hiện điền.

## Môi trường đã dùng

| Thành phần | Local app/tests | Kaggle REC-01 |
|---|---|---|
| Python | 3.10.11 | 3.13.16 |
| DuckDB | 1.5.6 | 1.5.6 |
| PyTorch | 2.8.0+cu126 | 2.11.0+cu130 |
| Streamlit | 1.54.0 | không dùng |
| GPU | RTX 4050 Laptop cho D07–D10; app/tests CPU | Tesla T4 trong training report |
| Nguồn | canonical 1M và MovieLens metadata local | verified private bundle canonical 1M |

Các package khác của local venv được ghi trong [manifest](s12-freeze-manifest.json): NumPy 2.2.6, pandas 2.3.3, PyArrow 25.0.1, psutil 7.2.2. [`requirements-local-repro.txt`](../requirements-local-repro.txt) pin các package đó; PyTorch cần chọn wheel CPU/CUDA theo máy. Bản local đã kiểm là CUDA 12.6; [trang phiên bản PyTorch chính thức](https://pytorch.org/get-started/previous-versions/) có lệnh cho 2.8.0 `--index-url https://download.pytorch.org/whl/cu126`. Đây là hướng dẫn tái tạo môi trường local, không thay thế environment Kaggle đã ghi trong S03 report.

```powershell
# Từ root repo BigDataProject, trong Python 3.10 venv mới:
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-local-repro.txt
python -m unittest discover -s tests -q
```

Nếu máy không có CUDA phù hợp, dùng wheel CPU chính thức; REC-01 inference và tests có thể chạy CPU. Training Kaggle cần môi trường riêng và [Notebook/bundle S03](s03-kaggle-workflow.md), không áp đặt PyTorch local lên Notebook.

## Freeze manifest và phạm vi kiểm

Tạo lại manifest sau mọi thay đổi code/report/figures hoặc artifact:

```powershell
..\venv\Scripts\python.exe -m scripts.s12_freeze_manifest --out docs/s12-freeze-manifest.json
```

Script fail nếu thiếu một trong 10 artifact bắt buộc: canonical source 1M, Kaggle input manifest, output ZIP/report/seal/selected checkpoint, hai EXP-01 raw CSVs, EXP-02 raw CSV và S11 report. `source_files` gồm mã `src/`, `scripts/`, `tests/`, requirements, báo cáo và PNG figures. Manifest chứa hash của file đang có, **không** tự chứng minh một người khác đã có file ignored ngoài Git; S14 phải kiểm submission package thực tế. S03 `verify-output` mới là bộ kiểm sâu của sealed REC-01 artifact; S06 auditor mới kiểm từng paired EXP-01 output.

## Đã rà và phần cần hoàn tất

- Báo cáo có đủ: câu hỏi nghiên cứu, nguồn, lý thuyết, architecture, temporal REC-01, EXP-01/02 methods, bảng kết quả, negative result Popularity, app S10/S11, validity threats và links tới raw evidence. 43 Markdown links đã kiểm; toàn bộ internal targets tồn tại tại thời điểm kiểm. Nội dung không nhận REC-02 là kết quả đã chạy.
- Code/test S11 có 60 tests PASS; S11 probe exact Full/Incremental trên 1M+20 PASS. Những evidence này được dẫn trong [S11](s11-integration-scaling.md) và [ma trận 09](09-report-evidence.md).
- DOCX đã được dựng trên mẫu `MauBaocaoBigData.docx`, giữ hai bìa/logo và các yêu cầu cấu trúc 1.1–1.6. Word đã cập nhật mục lục, xuất PDF 18 trang và rà hình mọi trang. Biểu đồ và bảng dùng số liệu thật; bảy tài liệu tham khảo đã kiểm. Cần Project Owner điền tên, MSSV, lớp, giảng viên, lời cảm ơn, cam đoan, thông tin nhóm và deadline thực nếu có.
- [S13](s13-defense-qa.md) có slide, PDF ngoại tuyến và Q&A. [S14](s14-submission.md) kiểm package source, raw logs và checkpoint thực tế. Chưa có commit/tag vì [AGENTS.md](../AGENTS.md) cấm tự commit/push.
