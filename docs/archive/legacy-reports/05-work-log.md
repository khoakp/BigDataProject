# 05 — Nhật ký làm việc hằng ngày

### 21/11/2026 — S01: audit và đề xuất REC-01

- Đọc và lưu [Project Direction Update v1.0](../plans/10-direction-update-v1.md), mở [kế hoạch S01–S14](../plans/11-sprint14-plan.md) mà không đổi lịch sử D01–D13. Đối chiếu code, D08/D10 reports, tests và artifacts với yêu cầu Top-20, Popularity, validation selection, Kaggle, Streamlit.
- [Đặc tả S01](../../research/s01-evaluation-spec.md) xác định primary prefix 1M/cutoffs, train-only candidates, warm exclusions, Top-20 metrics, Popularity và rule chọn checkpoint. Project Owner đã duyệt trong chat; S03–S05 đã chạy theo giao thức này, UI chưa xây.
- Ghi DEC-016 về hướng đã thống nhất và DEC-017 về protocol đề xuất. Giữ EXP-01/02 đã chạy theo tên cũ. Không commit/push trong phiên này.
- S02 xây train-only Popularity, so Top-20 cùng cohort/candidates/seen mask với LightGCN và chọn checkpoint chỉ từ validation metrics. Toy tests kiểm ranking tie, exclusions, leak guard và selection; các thành phần này đã dùng trong REC-01 S04.
- S02 validation-only probe trên D08 1M checkpoint cũ PASS: LightGCN Recall/NDCG@20 0,11949/0,08200; Popularity 0,12070/0,08534 trên cùng 482 warm users/773 candidates. [Phương pháp và giới hạn](../milestones/sxx/s02-validation-smoke.md); không dùng test và không gọi đây là REC-01 final run.
- S03 runner `src/models/rec01_train.py` khóa protocol 1M, lưu ba checkpoints, chọn validation rồi mới query test, xuất versioned model graph/ID map và raw results. Bundle canonical 1M + code đã kiểm hash tại `data/processed/rec01_kaggle_bundle_s03/`; Dataset private đã upload. Kaggle [Version 3](https://www.kaggle.com/code/khoakp/rec01-movielens-1m-gpu-validation?scriptVersionId=372012165) chạy `REC01_KAGGLE_PASS`, `seal-output PASS`; tải ZIP về và local `verify-output PASS`. [Workflow, hash và lỗi V1/V2](../../research/s03-kaggle-workflow.md).
- S04 [REC-01](../../research/s04-rec01-results.md): chọn LightGCN checkpoint 300 bằng validation NDCG@20; test trên 403 warm users/773 train movies đạt Recall/NDCG@20 0,128000/0,085161, so Popularity 0,128099/0,086897. Popularity nhỉnh hơn; báo cáo rõ cold exclusions và chỉ prefix 1M.
- S05 [loader và JSON CLI](../milestones/sxx/s05-recommendation-pipeline.md) kiểm hash code/artifact, load checkpoint đã chọn, Top-20 LightGCN warm-user với train-seen mask, metadata enrichment sau xếp hạng, Popularity train-only cho cold user. Tám toy tests và 52 tests local PASS; CLI smoke trên artifact Kaggle thật: warm user `25062` và cold user `-1` đều có 20/20 metadata, route đúng method và graph version 81. S05 hoàn tất trong phạm vi inference hiện tại; Streamlit thuộc sprint sau.
- S06 [audit EXP-01](../../research/s06-exp01-audit.md): tái dùng raw D12 thay vì chạy lại. Script kiểm 72+24 CSV rows, 36+12 paired repetitions, provenance, summary, logical hashes, từng Parquet SHA-256 và export byte count; cả hai run PASS. `events_per_s` có mẫu số khác nhau giữa Full và Incremental nên không dùng làm speedup trực tiếp.
- S07 [Engineering conclusions](../../research/s07-exp01-engineering-conclusions.md): tổng hợp raw D12 và audit S06, đối chiếu 8 cases H/Δ với 5 repetitions/case. Core H≤1M Full nhanh hơn toàn bộ paired runs; run 2M riêng Δ=1k cho Incremental ratio 1,156 (5/5 wins), Δ=10k gần hòa (2/5). Tái sinh summary trùng SHA-256 và 5 PNG trùng byte với report assets; xem trực quan các hình. Gate DE PASS trong phạm vi benchmark H≤2M, không suy rộng 32M.
- S09 [Streamlit MVP](../milestones/sxx/s09-streamlit-mvp.md): cài Streamlit 1.54.0 cho venv Python 3.10, app đọc REC-01 artifact đã xác minh, chọn warm user hoặc new-user Popularity, hiển thị Top-20 title/genres, lịch sử rating trước train cutoff, train catalog và graph/checkpoint version. Hai tests mới và AppTest trên artifact thật PASS; 55 local tests PASS. HTTP health `200 ok`; trình duyệt xác nhận warm/cold UI và catalog. App vẫn là snapshot chỉ đọc, ghi rating/graph refresh thuộc S10.
- S10 [rating và graph refresh demo](../milestones/sxx/s10-rating-graph-refresh.md): thêm `FeedbackStore`, bản sao sealed graph tách riêng, intent JSON theo UUID, Parquet batch và D04 ledger; UI xếp tối đa 20 rating rồi áp dụng. Trên artifact thật batch gồm rating 5,0 và 2,5 tăng graph version 81→82, thêm một cạnh; retry `NO_OP`, phục hồi sau gián đoạn và checkpoint/Top-20 bất biến. AppTest xác nhận graph 82/model graph 81/checkpoint 300; 57 local tests PASS. DEC-018 ghi refresh policy stale model. Chưa commit/push.
- S11 [integration và failure paths](../../research/s11-integration-scaling.md): probe mới trên canonical 1M + batch 20 ratings so từng dòng Full/Incremental; registry 14.388, edges 392.636, degrees 14.388 đều exact PASS. Graph 81→82, model graph 81/checkpoint 300; 10 cạnh positive mới và Top-20 giữ nguyên. Bổ sung tests gián đoạn sau intent, payload/timestamp sai, intent bị sửa và hai threads cùng tiến trình; toàn bộ 60 tests PASS. Một lượt apply 0,183 s chỉ là diagnostic, không phải benchmark. Raw report ngoài Git tại `data/processed/s11_integration/rec01_1m_batch20_v1/report.json`. Chưa commit/push.
- S12 [báo cáo và freeze candidate](../../s12-reproducibility.md): viết [bản báo cáo kỹ thuật](../../final-report.md) gồm REC-01, EXP-01/02, app và limitations; pin môi trường, ghi manifest 71 source/report/figure files + 10 evidence artifacts. Từ mẫu IUH do Project Owner cung cấp, dựng [DOCX](../../Bao-cao-MovieGraph-IUH.docx) và [PDF 18 trang](../../Bao-cao-MovieGraph-IUH.pdf), cập nhật mục lục và rà hình từng trang. Các ô nhân thân/cam đoan/nhóm giữ trống để Project Owner tự điền. Worktree dirty, chưa commit/tag.
- S13 [slide và Q&A](../../s13-defense-qa.md): deck Spaces 12 slide với chart REC-01/EXP-01 có thể sửa; render/rà từng slide 1280×720; xuất [PDF 12 trang](../../MovieGraph-defense.pdf) và PNG ngoại tuyến, chuẩn bị kịch bản demo và câu hỏi bảo vệ. Artifact session chưa tạo được PPTX cục bộ, nên không ghi nhận có file PPTX.
- S14 [gói nộp và QA](../milestones/sxx/s14-submission.md): ZIP private gồm source, docs, 1M canonical, REC-01 bundle/sealed run, metadata phim, raw EXP-01/02 CSV và S11 report. Manifest SHA-256 từng thành viên, ZIP verifier PASS. Giải nén riêng rồi chạy `verify-output PASS` và 60 tests PASS không skip với metadata path trong gói. Project Owner sẽ tự điền thông tin cá nhân trước khi nộp; không commit/push hoặc upload gói.

> **Vai trò:** nhật ký thực thi, không phải nhật ký học thuật ([02](02-research-journal.md)), không thay sổ thực nghiệm ([07](../../research/experiment-log.md)).
> **Trạng thái:** D01–D13 hoàn thành tính đến 21/11/2026. G1/G2/G3 đạt trong phạm vi đã ghi.

## 1. Quy tắc 5–10 phút mỗi buổi

- Ghi cả việc chưa xong, blockers, lỗi, chạy thất bại.
- Có `Task ID` D01–D21 theo [04 — Plan 21 ngày](../plans/04-project-plan.md) và `commit SHA` nếu đã commit.
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
- **Bằng chứng:** [01 — Overview](../plans/01-project-overview.md), [04 — Plan](../plans/04-project-plan.md).

### 19/11/2026 — Kiểm kê tài liệu và dữ liệu (chưa hoàn thành D01)

- **Việc đã làm:** đọc `AGENTS.md` và hệ thống docs; xác nhận repo chưa có source code, tests, results; tạo [08 — Report Notes](08-report-notes.md), [09 — Evidence](../../research/evidence-matrix.md), `report-assets/README.md`.
- **Kiểm tra đã chạy:** `Get-FileHash -Algorithm MD5` cho `../ml-32m/ratings.csv` và `movies.csv`; cả hai khớp `../ml-32m/checksums.txt`. Đọc header và README đi kèm. Chi tiết lệnh/hash ở [09, E-003/E-004](../../research/evidence-matrix.md).
- **Trạng thái:** kiểm kê tài liệu/dữ liệu đã hoàn tất; **D01 vẫn DOING/CHƯA NGHIỆM THU** vì chưa ghi CPU/RAM/GPU, phần mềm, smoke log và manifest nguồn. Profiling là D02. Không có benchmark hoặc model run.
- **Commit:** chưa commit các thay đổi phiên này; HEAD khi kiểm kê `735fe20c6b8f1a508b1b2a4e40aa57400ae4230b`.
- **Bước tiếp theo:** thực hiện các kiểm tra môi trường và dataset còn thiếu của D01 theo [04](../plans/04-project-plan.md), rồi cập nhật evidence tương ứng.

### 19/11/2026 — D01: kiểm tra môi trường và nguồn dữ liệu

- **Hoàn thành:** ghi CPU/RAM/SSD/GPU, Python/DuckDB/PyTorch, đọc README/license MovieLens; hash hai input; tạo [environment.md](../../research/environment.md) và [source-manifest.md](../../source-manifest.md).
- **Smoke:** CPU PyTorch autograd PASS (`loss=93.5`, gradient shape `4×4`); CUDA smoke **UNAVAILABLE** vì system PyTorch `2.8.0+cpu`; `../venv` chưa có torch/PyG. Xem [environment.md](../../research/environment.md).
- **Lỗi đã gặp:** CIM không truy cập được; dùng `psutil` lấy CPU/RAM/disk. CPU smoke lần đầu đọc gradient của non-leaf tensor, đã sửa và rerun PASS.
- **Trạng thái D01:** DONE cho kiểm kê và smoke trong môi trường hiện hữu; **GPU training chưa khả thi**, cần cài bản PyTorch CUDA/PyG và xác nhận ở D07. Chưa có benchmark hoặc model run. Không có commit mới.
- **Bước tiếp theo:** D02 profiling `ratings.csv`, chốt canonical ordering và kiểm tra deterministic output; không tự chuyển milestone khi chưa có phản hồi.

### 19/11/2026 — D02: profile và canonical prefix

- **Đã làm:** viết `src/ingest/d02_profile.py`, profile toàn bộ 32,000,204 ratings, sort theo hợp đồng [03](03-system-design.md), xuất Parquet prefix 100k/1M và manifest ngoài Git. Kết quả chi tiết ở [d02-data-profile.md](../milestones/dxx/d02-data-profile.md).
- **Kiểm chứng:** hai full-data runs cho cùng SHA-256 của cả hai prefix; toy T-01 PASS; 1M prefix có 0 order violations, 1M source row IDs phân biệt và event IDs 0–999999.
- **Quyết định/giới hạn:** ngưỡng positive vẫn PROPOSED, nên chưa ghi `is_positive`; chưa làm graph D03 hoặc model. File Parquet được ignore theo `.gitignore`.
- **Trạng thái:** D02 DONE theo tiêu chí plan. Không có commit mới. Chờ phản hồi trước khi sang D03.

### 19/11/2026 — D03: positive rule và Full Rebuild reference

- **Đã làm:** profile phân phối rating toàn nguồn; chốt DEC-005 `>=4.0` cho graph; viết `src/graph/full_builder.py` và toy test; xuất registry/edges/degrees cho canonical 100k. Số đo và hash ở [d03-full-reference.md](../milestones/dxx/d03-full-reference.md).
- **Kiểm chứng:** 2 unit tests PASS; toy có duplicate positive pair cho 3 unique edges; trên 100k có 57.925 edges và `sum user degree = edges = sum movie degree`; đối chiếu lại output có 0 duplicate edges, 0 degree/registry mismatches.
- **Giới hạn:** chưa có incremental updater, idempotency, performance result hoặc model metric. Không có commit mới.
- **Trạng thái:** D03 DONE theo tiêu chí plan. Chờ phản hồi trước D04.

### 19/11/2026 — D04: transactional incremental apply_batch

- **Đã làm:** thêm `src/graph/incremental.py`, tests và integration probe; registry IDs ổn định, anti-join/dedup cạnh, degree delta và batch ledger trong một DuckDB transaction.
- **Kiểm chứng:** 5 unit tests PASS; prefix 100k / batches 10k tạo 57.925 edges, 2.819 nodes; 10 replay NO_OP; từng node có degree đúng khi đếm lại; reopen DB giữ state. Toy kiểm tra hash conflict, invalid batches và rollback khi inject lỗi.
- **Lỗi đã sửa:** số SQL placeholders ghi ledger không khớp số values ở unit run đầu; rerun PASS. Kết quả và cách tái lập tại [D04](../milestones/dxx/d04-incremental.md).
- **Trạng thái:** D04 DONE theo tiêu chí plan; chưa có Full/Incremental equivalence đa segmentation, benchmark hoặc checkpoint recovery đầy đủ. Chưa commit/push. Chờ phản hồi trước D05.

## 5. Bảng theo dõi 3 tuần (điền dần)

### 19/11/2026 — D06: sparse normalization, snapshot versions, movies validation

- **Đã làm:** thêm `normalize.py`, `snapshot.py`, metadata validator, integration probe và 3 test modules; ghi COO/node map/manifest theo source canonical version, export Parquet kèm hashes.
- **Kiểm chứng:** 15 unit tests PASS. Run `D06-20261119T163958346261Z` PASS; ở v1/v5/v10, shape/coordinates exact và max_abs_diff so Full SQL reference đều 0; symmetric, 0 self loops, export/load/reopen và old-snapshot immutability PASS. `movies.csv` 87.585 unique/non-null IDs, 7.080 unknown genre labels; graph isolation PASS.
- **Failure ghi nhận:** run `D06-20261119T163832513684Z` FAIL ở SQL reference do CREATE VIEW không nhận prepared parameter; chuyển sang CREATE TEMP TABLE và rerun PASS. Giữ failure report ngoài Git. Chi tiết setup/artifacts/lệnh ở [D06](../milestones/dxx/d06-normalization.md).
- **Trạng thái:** D06 DONE, T-05/T-06 PASS trong phạm vi toy/100k; G2 vẫn cần D07 GPU/LightGCN smoke. Không có benchmark hoặc model results, chưa commit/push. Chờ được giao D07.

### 19/11/2026 — D05: exact graph equivalence, G1

- **Đã làm:** thêm exact comparator, multi-segmentation harness và regression tests; so Full/Incremental sau mỗi batch trên toy và canonical prefix 100k, mỗi dataset 4 cách chia batch.
- **Kiểm chứng:** 35 boundary checks, mỗi lần gồm registry/edge/degree exact; 70 replays không đổi full snapshot; stable node IDs và ledger checks PASS. Unit suite 7 tests PASS. Report/config/hashes và lệnh tại [D05](../milestones/dxx/d05-equivalence.md), raw `data/processed/d05_probe/report.json`.
- **Lỗi đã sửa:** Full Rebuild trả NULL degree sum với graph rỗng, regression test ban đầu FAIL; đã dùng coalesce về 0, rerun unit và integration PASS. Đóng full-builder connection theo context để chạy nhiều cutoff.
- **Trạng thái:** D05 DONE, G1 PASS trên toy/100k. Chưa có normalization, performance benchmark hoặc model results. Chưa commit/push. Bước kế tiếp D06 theo plan, chờ được giao.

### 21/11/2026 — D13: EXP-02 cadence và staleness

- Thêm harness chạy chung 10 batch × 10k events cho R=1/R=5, một warm-up và ba measured repetitions/process mới, log riêng canonical update/materialize/export, bytes/RSS/OS I/O và graph version. Tại v5/v10 so normalized exports.
- Hai lượt chẩn đoán ban đầu bị dừng do ghi DuckDB snapshot bằng Python `unnest` mất ~75 giây ngay v1. Đổi sang `INSERT ... SELECT` trực tiếp từ canonical tables, chạy lại tests và thí nghiệm trong thư mục mới. Chỉ run `D13-100k-r1-r5-final` dùng làm kết quả.
- Kết quả: 80/80 rows và 8/8 common-boundary checks PASS, median tổng 5,283 s (R=1) vs 1,648 s (R=5); R=5 cũ tối đa 4 batches/40k events. 29 unit tests PASS. [Hồ sơ D13](../../research/d13-exp02-results.md), raw `results/raw/exp02_normalization.csv`.
- D13 DONE; D14 sẽ tổng hợp EXP-01/02 và AN-01 theo kế hoạch. Chưa commit/push.

### 20/11/2026 — D12: EXP-01 repetitions và crossover

- Mở rộng harness thành năm measured repetitions/cặp với thứ tự Full/Incremental shuffle và process riêng, giữ exact correctness ở từng lượt. Core grid 72/72 rows PASS; run H=2M riêng 24/24 PASS. CSV raw, bảng median/range và năm biểu đồ ở [D12](../../research/d12-exp01-results.md).
- Đến H=1M, Incremental không thắng cặp nào; H=2M, Δ=1k thắng 5/5 và paired ratio median 1,156; Δ=10k gần hòa. Không xác định ngưỡng crossover chính xác hoặc suy rộng sang 32M. D13 tiếp tục EXP-02.

### 20/11/2026 — D11: EXP-01 pilot harness

- Thêm harness Full vs Incremental cùng canonical input và ba bảng output; tách prebuild H khỏi timer, chạy process riêng cho warm-up/measured, log wall/CPU/sampled RSS/app bytes/OS I/O. Pilot sáu cặp H={100k,500k,1M}, Δ={1k,10k}: 24/24 rows correctness PASS. Raw CSV và điều kiện đo ở [D11](../milestones/dxx/d11-exp01-pilot.md).
- D12 sẽ chạy repetitions và phân tích spread/speedup; không kết luận nhanh/chậm từ một lượt đo.

### 20/11/2026 — D10: clean MVP rerun, G3

- Từ `ratings.csv` gốc tạo lại canonical 100k trong thư mục mới, chạy D09 CUDA end-to-end, xuất manifest hashes/config/code provenance, checkpoint và 2.562 model IDs. Run `D10-clean-100k` PASS, graph version 9, 300/300 enriched Top-K rows; demo CLI xác minh 12 hashes và hiển thị gợi ý mẫu. 27 tests PASS. [Hồ sơ D10 và kịch bản demo 4 phút](../milestones/dxx/d10-mvp.md).
- G3 PASS trong phạm vi tối thiểu 100k. D11 sẽ bắt đầu EXP-01; không coi thời gian D10 là benchmark Full/Incremental.

### 20/11/2026 — D09: versioned graph đến enriched Top-K

- E2E script D09 apply train batches qua D04 ledger, kiểm chứng exact Full equivalence, normalize, train/evaluate LightGCN rồi enrich `movies.csv` theo raw movie ID. Runs CUDA 100k/1M PASS; 25 unit tests PASS. 1M graph version 81, 8.850/8.850 Top-K rows có metadata, ranking và metrics không đổi. [Hồ sơ D09](../milestones/dxx/d09-end-to-end.md).
- D10 tiếp tục clean rerun, đóng gói manifests/checkpoint/model IDs và demo.

### 20/11/2026 — D08: temporal LightGCN baseline

- Dựng train graph từ event-time cutoff, chỉ train positive edges và train negatives; đánh giá raw Top-10 trên warm cohort, lưu checkpoint và exclusion counts. 100k và 1M CUDA runs PASS; 22 unit tests PASS. Chi tiết và số liệu ở [D08](../milestones/dxx/d08-temporal-baseline.md).
- 1M valid/test Recall@10 = 0,06390/0,07154 trên 482/403 warm users; phần lớn future pairs thuộc cold users nên chưa đại diện toàn bộ dữ liệu. D09 tiếp nối integration và metadata enrichment.

### 20/11/2026 — D07: LightGCN sparse CUDA smoke

- Cài PyTorch 2.8.0+cu126 vào venv. Thêm propagation LightGCN bằng PyTorch native, loss BPR và probe ghi peak CUDA allocated/reserved từng phase.
- 18 unit tests đạt. Run `D07-20261119T172548657263Z` đạt trên toy, 100k và 1M. Peak allocated/reserved của 1M là 126,26/148 MiB. D07 hoàn thành, G2 đạt.
- Đã lưu hai dòng hoàn tất từ lần thử đầu, kết quả của run chính và diễn giải phương án chuẩn bị 1M trong [hồ sơ D07](../milestones/dxx/d07-lightgcn-smoke.md). D08 sẽ kiểm tra temporal split và baseline khi được giao.

| Tuần | Task DONE/plan | Giờ thực tế | Blocker chính | Milestone | Review link |
|---|---:|---:|---|---|---|
| W1 (D01–D07) | D01–D07 DONE (7/7) | Chưa ghi | — | G1/G2 PASS | [D07](../milestones/dxx/d07-lightgcn-smoke.md) |
| W2 (D08–D14) | D08–D13 DONE (6/7) | Chưa ghi | Cold cohort lớn; AN-01 chờ | G3 PASS; EXP-01/02 correctness PASS | [D13](../../research/d13-exp02-results.md) |
| W3 (D15–D21) | Chưa xác nhận | — | — | D16 freeze; submission: chờ | — |

## 6. Điều cần ghi trong commit/message

Gợi ý format `D04: add idempotent incremental batch ledger` hoặc `EXP-01: record 1M/10k warm-cache repeats`; metadata: `D09: post-ranking movie lookup invariance tests`. Mỗi log nêu rõ file/commit và trạng thái; không cần chép nguyên toàn bộ stacktrace vào Markdown — lưu log raw vào `results/logs/`.
