# 05 — Nhật ký làm việc hằng ngày

> **Vai trò:** nhật ký thực thi, không phải nhật ký học thuật ([02](02-research-journal.md)), không thay sổ thực nghiệm ([07](07-experiment-log.md)).  
> **Trạng thái:** D01–D07 hoàn thành tính đến 20/11/2026. G1 và G2 đạt.

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

### 19/11/2026 — Kiểm kê tài liệu và dữ liệu (chưa hoàn thành D01)

- **Việc đã làm:** đọc `AGENTS.md` và hệ thống docs; xác nhận repo chưa có source code, tests, results; tạo [08 — Report Notes](08-report-notes.md), [09 — Evidence](09-report-evidence.md), `report-assets/README.md`.
- **Kiểm tra đã chạy:** `Get-FileHash -Algorithm MD5` cho `../ml-32m/ratings.csv` và `movies.csv`; cả hai khớp `../ml-32m/checksums.txt`. Đọc header và README đi kèm. Chi tiết lệnh/hash ở [09, E-003/E-004](09-report-evidence.md).
- **Trạng thái:** kiểm kê tài liệu/dữ liệu đã hoàn tất; **D01 vẫn DOING/CHƯA NGHIỆM THU** vì chưa ghi CPU/RAM/GPU, phần mềm, smoke log và manifest nguồn. Profiling là D02. Không có benchmark hoặc model run.
- **Commit:** chưa commit các thay đổi phiên này; HEAD khi kiểm kê `735fe20c6b8f1a508b1b2a4e40aa57400ae4230b`.
- **Bước tiếp theo:** thực hiện các kiểm tra môi trường và dataset còn thiếu của D01 theo [04](04-project-plan.md), rồi cập nhật evidence tương ứng.

### 19/11/2026 — D01: kiểm tra môi trường và nguồn dữ liệu

- **Hoàn thành:** ghi CPU/RAM/SSD/GPU, Python/DuckDB/PyTorch, đọc README/license MovieLens; hash hai input; tạo [environment.md](environment.md) và [source-manifest.md](source-manifest.md).
- **Smoke:** CPU PyTorch autograd PASS (`loss=93.5`, gradient shape `4×4`); CUDA smoke **UNAVAILABLE** vì system PyTorch `2.8.0+cpu`; `../venv` chưa có torch/PyG. Xem [environment.md](environment.md).
- **Lỗi đã gặp:** CIM không truy cập được; dùng `psutil` lấy CPU/RAM/disk. CPU smoke lần đầu đọc gradient của non-leaf tensor, đã sửa và rerun PASS.
- **Trạng thái D01:** DONE cho kiểm kê và smoke trong môi trường hiện hữu; **GPU training chưa khả thi**, cần cài bản PyTorch CUDA/PyG và xác nhận ở D07. Chưa có benchmark hoặc model run. Không có commit mới.
- **Bước tiếp theo:** D02 profiling `ratings.csv`, chốt canonical ordering và kiểm tra deterministic output; không tự chuyển milestone khi chưa có phản hồi.

### 19/11/2026 — D02: profile và canonical prefix

- **Đã làm:** viết `src/ingest/d02_profile.py`, profile toàn bộ 32,000,204 ratings, sort theo hợp đồng [03](03-system-design.md), xuất Parquet prefix 100k/1M và manifest ngoài Git. Kết quả chi tiết ở [d02-data-profile.md](d02-data-profile.md).
- **Kiểm chứng:** hai full-data runs cho cùng SHA-256 của cả hai prefix; toy T-01 PASS; 1M prefix có 0 order violations, 1M source row IDs phân biệt và event IDs 0–999999.
- **Quyết định/giới hạn:** ngưỡng positive vẫn PROPOSED, nên chưa ghi `is_positive`; chưa làm graph D03 hoặc model. File Parquet được ignore theo `.gitignore`.
- **Trạng thái:** D02 DONE theo tiêu chí plan. Không có commit mới. Chờ phản hồi trước khi sang D03.

### 19/11/2026 — D03: positive rule và Full Rebuild reference

- **Đã làm:** profile phân phối rating toàn nguồn; chốt DEC-005 `>=4.0` cho graph; viết `src/graph/full_builder.py` và toy test; xuất registry/edges/degrees cho canonical 100k. Số đo và hash ở [d03-full-reference.md](d03-full-reference.md).
- **Kiểm chứng:** 2 unit tests PASS; toy có duplicate positive pair cho 3 unique edges; trên 100k có 57.925 edges và `sum user degree = edges = sum movie degree`; đối chiếu lại output có 0 duplicate edges, 0 degree/registry mismatches.
- **Giới hạn:** chưa có incremental updater, idempotency, performance result hoặc model metric. Không có commit mới.
- **Trạng thái:** D03 DONE theo tiêu chí plan. Chờ phản hồi trước D04.

### 19/11/2026 — D04: transactional incremental apply_batch

- **Đã làm:** thêm `src/graph/incremental.py`, tests và integration probe; registry IDs ổn định, anti-join/dedup cạnh, degree delta và batch ledger trong một DuckDB transaction.
- **Kiểm chứng:** 5 unit tests PASS; prefix 100k / batches 10k tạo 57.925 edges, 2.819 nodes; 10 replay NO_OP; từng node có degree đúng khi đếm lại; reopen DB giữ state. Toy kiểm tra hash conflict, invalid batches và rollback khi inject lỗi.
- **Lỗi đã sửa:** số SQL placeholders ghi ledger không khớp số values ở unit run đầu; rerun PASS. Kết quả và cách tái lập tại [D04](d04-incremental.md).
- **Trạng thái:** D04 DONE theo tiêu chí plan; chưa có Full/Incremental equivalence đa segmentation, benchmark hoặc checkpoint recovery đầy đủ. Chưa commit/push. Chờ phản hồi trước D05.

## 5. Bảng theo dõi 3 tuần (điền dần)

### 19/11/2026 — D06: sparse normalization, snapshot versions, movies validation

- **Đã làm:** thêm `normalize.py`, `snapshot.py`, metadata validator, integration probe và 3 test modules; ghi COO/node map/manifest theo source canonical version, export Parquet kèm hashes.
- **Kiểm chứng:** 15 unit tests PASS. Run `D06-20261119T163958346261Z` PASS; ở v1/v5/v10, shape/coordinates exact và max_abs_diff so Full SQL reference đều 0; symmetric, 0 self loops, export/load/reopen và old-snapshot immutability PASS. `movies.csv` 87.585 unique/non-null IDs, 7.080 unknown genre labels; graph isolation PASS.
- **Failure ghi nhận:** run `D06-20261119T163832513684Z` FAIL ở SQL reference do CREATE VIEW không nhận prepared parameter; chuyển sang CREATE TEMP TABLE và rerun PASS. Giữ failure report ngoài Git. Chi tiết setup/artifacts/lệnh ở [D06](d06-normalization.md).
- **Trạng thái:** D06 DONE, T-05/T-06 PASS trong phạm vi toy/100k; G2 vẫn cần D07 GPU/LightGCN smoke. Không có benchmark hoặc model results, chưa commit/push. Chờ được giao D07.

### 19/11/2026 — D05: exact graph equivalence, G1

- **Đã làm:** thêm exact comparator, multi-segmentation harness và regression tests; so Full/Incremental sau mỗi batch trên toy và canonical prefix 100k, mỗi dataset 4 cách chia batch.
- **Kiểm chứng:** 35 boundary checks, mỗi lần gồm registry/edge/degree exact; 70 replays không đổi full snapshot; stable node IDs và ledger checks PASS. Unit suite 7 tests PASS. Report/config/hashes và lệnh tại [D05](d05-equivalence.md), raw `data/processed/d05_probe/report.json`.
- **Lỗi đã sửa:** Full Rebuild trả NULL degree sum với graph rỗng, regression test ban đầu FAIL; đã dùng coalesce về 0, rerun unit và integration PASS. Đóng full-builder connection theo context để chạy nhiều cutoff.
- **Trạng thái:** D05 DONE, G1 PASS trên toy/100k. Chưa có normalization, performance benchmark hoặc model results. Chưa commit/push. Bước kế tiếp D06 theo plan, chờ được giao.

### 20/11/2026 — D07: LightGCN sparse CUDA smoke

- Cài PyTorch 2.8.0+cu126 vào venv. Thêm propagation LightGCN bằng PyTorch native, loss BPR và probe ghi peak CUDA allocated/reserved từng phase.
- 18 unit tests đạt. Run `D07-20261119T172548657263Z` đạt trên toy, 100k và 1M. Peak allocated/reserved của 1M là 126,26/148 MiB. D07 hoàn thành, G2 đạt.
- Đã lưu hai dòng hoàn tất từ lần thử đầu, kết quả của run chính và diễn giải phương án chuẩn bị 1M trong [hồ sơ D07](d07-lightgcn-smoke.md). D08 sẽ kiểm tra temporal split và baseline khi được giao.

| Tuần | Task DONE/plan | Giờ thực tế | Blocker chính | Milestone | Review link |
|---|---:|---:|---|---|---|
| W1 (D01–D07) | D01–D07 DONE (7/7) | Chưa ghi | — | G1/G2 PASS | [D07](d07-lightgcn-smoke.md) |
| W2 (D08–D14) | Chưa xác nhận | — | — | D10 MVP; EXP-01/02: chờ | — |
| W3 (D15–D21) | Chưa xác nhận | — | — | D16 freeze; submission: chờ | — |

## 6. Điều cần ghi trong commit/message

Gợi ý format `D04: add idempotent incremental batch ledger` hoặc `EXP-01: record 1M/10k warm-cache repeats`; metadata: `D09: post-ranking movie lookup invariance tests`. Mỗi log nêu rõ file/commit và trạng thái; không cần chép nguyên toàn bộ stacktrace vào Markdown — lưu log raw vào `results/logs/`.
