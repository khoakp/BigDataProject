# GOAL 04 — C5 preparation và Phase C budget audit

Ngày 24/11/2026. **Local preparation PASS; Owner đã thu hẹp budget accounting vào C3 v3 và C4 v4.** [Assessment theo quyết định Owner](../../results/goal04_c5/budget_audit/assessment-owner-scope.json) tính 2.369,918 s đến log cuối, còn 12.030,082 s trong trần 4 giờ. [C5 package local](../../data/processed/goal04_c5_owner_scope_v1/package-manifest.json) đã gắn assessment SHA-256 và trạng thái `READY`. Đây là trạng thái preparation trước khi Owner duyệt submission; kết quả official C5 v1 và lỗi wrapper được ghi ở cuối tài liệu.

## Budget audit và quyết định phạm vi của Owner

[Kaggle API evidence](../../results/goal04_c5/budget_audit/api-evidence.json) lưu version-specific status, log, file creation dates và metadata; [assessment](../../results/goal04_c5/budget_audit/assessment.json) tách thời gian Kaggle khỏi thời gian Codex chuẩn bị. `KaggleApi.kernels_status`, `kernels_logs`, `kernels_list_files` được gọi read-only. Thời gian Codex viết code, tải artifacts và chạy tests local **không** tính vào trần Kaggle bốn giờ.

| Phiên | Trạng thái API | Bằng chứng thời gian | Start/end chính xác |
| --- | --- | --- | --- |
| C3 v1 | CANCEL_ACKNOWLEDGED | Không log/file | Unknown |
| C3 v2 | ERROR | Log timestamp cuối 26,90 s; report file tạo 17:19 UTC 23/11 (độ phân giải phút) | Unknown |
| C3 v3 | COMPLETE | Runner elapsed 379,03 s; log timestamp cuối 388,13 s; report file tạo 17:23 UTC 23/11 | Unknown |
| C4 v1 | CANCEL_ACKNOWLEDGED | Không log/file | Unknown |
| C4 v2 | CANCEL_ACKNOWLEDGED | Không log/file | Unknown |
| C4 v3 | CANCEL_ACKNOWLEDGED | Không log/file | Unknown |
| C4 v4 | COMPLETE | Runner elapsed 1.968,11 s; log timestamp cuối 1.981,79 s; report file tạo 18:07 UTC 23/11 | Unknown |

`last_run_time` trong version-specific `get_kernel` metadata báo **21/11** cho cả các phiên có report file được Kaggle tạo **23/11**; truy vấn lại C4 v4 thay đổi từ `04:29:50` sang `04:40:43` cùng ngày 21/11. Nó mâu thuẫn với artifacts nên không được dùng làm start/end. [Latest kernel list](../../results/goal04_c5/budget_audit/latest-kernels-list.csv) có `lastRunTime` C3 v3 `2026-11-23 17:23:00.593` và C4 v4 `2026-11-23 18:07:54.373`, phù hợp phút tạo file nhưng không chứng minh terminal end. Creation date của file cũng không phải thời điểm kết thúc job. [Assessment ban đầu](../../results/goal04_c5/budget_audit/assessment.json) giữ nguyên kết quả kiểm toán cả bảy phiên: thời lượng terminal chính xác không có đủ evidence.

Owner ngày 24/11 chỉ định **chỉ tính C3 v3 và C4 v4** vì các phiên trước là quá trình queued/cancel/save version. Theo phạm vi này, runner reports cộng **2.347,138 s**; timestamp log cuối đo **388,131 s** cho C3 v3 và **1.981,787 s** cho C4 v4, cộng **2.369,918 s** (39 phút 29,918 giây). Charging theo log cuối gồm notebook rendering, nhiều hơn runner subtotal **22,780 s**. Remaining budget theo Owner scope là **12.030,082 s** (3 giờ 20 phút 30,082 giây). `lastRunTime` và timestamp log cho thấy mốc quan sát tới dòng log cuối khoảng 17:29:28,724 UTC (C3) và 18:40:56,160 UTC (C4); đây là mốc suy từ hai nguồn, **không** phải terminal end do Kaggle xác nhận. Teardown sau log cuối chưa đo được. Thời gian Codex chuẩn bị local không tính vào Kaggle budget.

Raw evidence vẫn ghi C3 v2 `ERROR` với log 26,90 s. Phiên này được **loại theo quyết định phạm vi của Owner**, không được mô tả lại là 0 giây thực thi. Không sửa historical C3/C4 artifacts.

Không tự nới trần 14.400 s. [Package nháp cũ](../../data/processed/goal04_c5_draft_v2/package-manifest.json) vẫn mang `BLOCKED_BUDGET_AUDIT`; package builder chỉ tạo bản runnable từ audit `VERIFIED` hoặc quyết định accounting được Owner ghi rõ, kèm SHA-256 của audit. Assessment ban đầu `INCOMPLETE` bị từ chối; assessment Owner scope tạo package `READY`. C5 GPU job vẫn chưa được submit trong task chuẩn bị này.

## C5 local preparation

- [Input integrity](../../results/goal04_c5/input-integrity.json) **PASS**: strict reload/hash năm C4 candidates tại steps 0/500/1500/3000/5000, canonical SHA-256, C3 G0 snapshot manifest, fixed mapping, 90.253 nodes, 4.999.981 edges và đúng 40.666 movie nodes. Kiểm tra local không nạp toàn bộ G0 vào RAM vì máy còn khoảng 3 GiB RAM khả dụng; C3 đã strict export/reload G0 trên Kaggle.
- `scripts/kaggle_c5_runner.py` khi được mở budget sẽ xác minh lại full C4 report/steps/checkpoints qua `scripts/verify_kaggle_c4.py`, canonical/cohort bằng verifier C3 và nạp G0 qua `load_g0`. DuckDB chỉ truy vấn events có `event_ts` trong `[2020-01-01, 2021-01-01)`; 12 half-open monthly windows dùng `validation_windows` và so eligible pairs/users với [approved monthly profile](../../results/goal04_protocol/monthly.csv). Chỉ 2020 validation được phép chọn checkpoint.
- `score_batched` trong `src/evaluation/c0_validation.py` chấm toàn bộ fixed catalog, che observed-positive history trước boundary, dùng stable score-descending/raw-movie-ID-ascending tie break, exact Top-20 rồi `evaluate_topk` và macro-window aggregation hiện hành. Nó xử lý batch users để tránh 40.666 lần Python score list trên mỗi user. Không thay đổi `score_chunks` của REC-01.
- Với từng checkpoint: strict load → **một** LightGCN G0 propagation trên `cuda:0` → cache vector detached → copy sang `cuda:1`. Numerical gate dùng cùng scorer trên hai T4 và so Top-20 IDs/scores cho 16 eligible users; pass thì hai workers chấm tháng 1–6 và 7–12, fail thì toàn bộ 12 tháng chấm trên `cuda:0`. Không optimizer update, không propagation forward bổ sung cho gate/fallback. C5 chỉ xuất đủ 5×12 kết quả; chọn C0 thuộc cổng C6 riêng.
- [Toy tests](../../tests/test_kaggle_c5.py) kiểm mask, stable tie, batched/reference Top-20, dual partition bằng cùng code trên hai tensor CPU, fallback, nonfinite rejection, 2020-only query và profile mismatch. Đây **chưa** phải GPU numerical-equivalence test; gate thật chỉ chạy sau khi budget được xác minh.

Full regression sau thay đổi: **96 PASS, 0 FAIL** trong [log](../../results/goal04_c5/regression.log); `git diff --check` PASS. Draft wrapper được chạy thử local và dừng bằng `C5 submission blocked: Phase C Kaggle elapsed not audited` trước khi giải nén source hoặc dùng GPU.

## Tái kiểm local

```powershell
..\venv\Scripts\python.exe -m results.goal04_c5.check_local_inputs
..\venv\Scripts\python.exe -m unittest tests.test_kaggle_c5 -v
..\venv\Scripts\python.exe -m scripts.prepare_kaggle_c5 --output data/processed/goal04_c5_new_draft
```

Input Kaggle C5 dự kiến: hai private datasets đã duyệt (canonical/manifest và cohort), C3 kernel output chứa G0, C4 kernel output chứa năm candidates, và source bundle trong kernel script. Không cần upload raw MovieLens hoặc thêm dataset mới. C5 package chỉ được tạo bản runnable và submit sau quyết định budget; chưa có measured C5 quality hay thời gian GPU.

## Giải thích cho sinh viên mới

Training tạo năm bản tham số. C5 giữ từng bản cố định, lan truyền một lần trên G0 rồi dùng cùng vector để hỏi: với lịch sử đã biết trước đầu mỗi tháng, phim nào ở Top-20, và bao nhiêu phim được user đánh giá dương trong tháng đó được tìm thấy? Ví dụ user đã thích phim A trước tháng 3 thì A bị che ở tháng 3; phim B được thích trong tháng 3 là label, chưa được đưa vào graph để dự đoán chính tháng 3. Recall@20 và NDCG@20 được tính cho từng user, trung bình theo tháng rồi trung bình 12 tháng.

Ba câu tự kiểm tra: (1) Vì sao hai GPU chỉ nhận vector đã propagation thay vì mỗi GPU tự forward? (2) Vì sao một phiên Kaggle bị hủy không thể mặc định tính là 0 giây? (3) Vì sao phải chấm đủ năm checkpoints trên cả 12 tháng trước khi chọn C0?

## Official C5 v1: INCOMPLETE

Sau khi Owner duyệt một submission, package [official v1](../../data/processed/goal04_c5_official_v1/package-manifest.json) được submit đúng một lần thành `khoakp/moviegraph-v2-c5-2020-validation/1`. [Kaggle status](../../results/goal04_c5/run_01/status.txt) là `ERROR`; [log](../../results/goal04_c5/run_01/kaggle.log) chỉ ra wrapper gán `float` vào `os.environ` tại dòng 5 và nhận `TypeError: str expected, not float`. Lỗi xảy ra trước giải nén source, nạp C3 v3/C4 v4 hoặc chạy C5 runner. [Assessment](../../results/goal04_c5/run_01/assessment.json) ghi `INCOMPLETE`: 0/60 matrix records, không có scores, C0 hoặc GPU metrics. Dòng log cuối ở 6,824 s; terminal elapsed không có nên chưa cộng một số chính xác vào Owner-scope budget.

Đã sửa package builder để ghi budget value dạng chuỗi và thêm AST check trong pre-submit gate. Package [corrected review v2](../../data/processed/goal04_c5_corrected_review_v2/package-manifest.json) được tạo ở thư mục mới, **chưa submit**. Gate mới PASS với v2 và từ chối chính package v1 đã lỗi; 7 targeted tests và [full regression 99 PASS](../../results/goal04_c5/postfailure-regression.log). Chỉ C3 **version 3** và C4 **version 4** được dùng làm nguồn; các phiên trước giữ nguyên như bằng chứng lịch sử, không đưa vào tính budget. Dừng chờ Owner review trước mọi submission khác hoặc C6.

## Official C5 v2: COMPLETE_VALIDATION

Owner ngày 24/11/2026 cho phép thêm **một** submission sau lỗi v1. Lần gọi đầu nhận `409 Conflict` trước khi tạo version vì metadata id không khớp slug thật; đã sửa id và thêm gate. [Package cuối](../../data/processed/goal04_c5_retry_official_v4/package-manifest.json) dùng cùng source bundle, bổ sung [accounting](../../results/goal04_c5/budget_audit/assessment-retry-scope.json) cho 6,824 s log cuối của C5 v1. Gate PASS, [regression 99 PASS](../../results/goal04_c5/retry-regression.log), C3 v3/C4 v4 strict verification PASS. Kaggle tạo đúng [C5 version 2](../../results/goal04_c5/run_02/submission.json), timeout 11.800 s; [status](../../results/goal04_c5/run_02/status.txt) `COMPLETE`.

[Verifier độc lập](../../results/goal04_c5/run_02/verification.json) **PASS** trên [output](../../results/goal04_c5/run_02/download/goal04_c5/report.json): 60/60 records = năm checkpoints × 12 monthly windows 2020; fixed 40.666 movie catalog, 5 G0 propagation forwards, 0 optimizer updates, không có C0 selection. Cả năm candidates dùng T4×2; numerical gate PASS với 16 users mỗi candidate, Top-20 giống nhau và max score diff 0 trong mẫu. 2020 labels/users từng tháng khớp approved profile; không chấm 2021–2023.

| Checkpoint step | Macro Recall@20 | Macro NDCG@20 |
| ---: | ---: | ---: |
| 0 | 0,000578651 | 0,000395025 |
| 500 | 0,038282769 | 0,028192684 |
| 1500 | 0,038658790 | 0,028306432 |
| 3000 | 0,038405946 | 0,028085449 |
| 5000 | 0,038323185 | 0,028011263 |

[Matrix đầy đủ](../../results/goal04_c5/run_02/download/goal04_c5/validation-matrix.json) và [bảng so sánh](../../results/goal04_c5/run_02/download/goal04_c5/candidate-comparison.csv) lưu scores gốc, eligible users/labels, boundary/mask/catalog provenance và GPU mode. Bảng này chỉ so sánh; **không chọn C0 trong C5**.

Runner end-to-end 147,304 s; input verification 11,829 s, G0 load 73,061 s, window build 22,260 s. Peak process-tree RSS 5.762.551.808 bytes; peak CUDA allocation 639.998.976 bytes trên `cuda:0`, 124.751.872 bytes trên `cuda:1`. Peak output disk được monitor trong riêng thư mục `goal04_c5` là 109.659 bytes, không bao gồm source extraction. Kaggle dùng Python 3.13.16, PyTorch 2.11.0+cu130, DuckDB 1.3.2. [Log cuối](../../results/goal04_c5/run_02/download/moviegraph-v2-c5-2020-validation.log) ở 152,504 s gồm notebook rendering; theo cùng phương pháp Owner-scope, tổng C3 v3 + C4 v4 + C5 v1 + C5 v2 đến log cuối là 2.529,247 s, còn 11.870,753 s. Terminal teardown chưa đo được nên không gọi đây là exact elapsed.

C5 sẵn sàng để Owner nghiệm thu và quyết định C6. Numerical gate chỉ lấy 16 users, validation chỉ áp dụng cho fixed warm cohort; C0 sau này sẽ có checkpoint-selection bias vì chọn bằng stale-G0 validation. Chưa có kết luận S0/S1/S2 hoặc 2023 holdout, không chạy C6, Phase D hoặc commit/push/merge.
