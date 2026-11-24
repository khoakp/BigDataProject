# Bản thảo báo cáo nghiên cứu v2

**Trạng thái:** báo cáo đang phát triển, chỉ chứa kết quả đã có evidence. [Bản đồ claim](claim-evidence-map.md) nối mỗi số liệu với run/hạn chế. [Báo cáo lịch sử frozen](../final-report.md) và REC-01 1M giữ vai trò lịch sử, không được gộp vào kết quả v2 5M.

## 1. Giới thiệu và câu hỏi nghiên cứu

Hệ gợi ý dựa trên graph tương tác cần cập nhật khi ratings mới đến. RQ1 hỏi khi nào Incremental Graph Maintenance giữ đúng trạng thái của Full Rebuild mà giảm chi phí. RQ2 hỏi chi phí và chất lượng thay đổi thế nào nếu giữ model cũ (S0), chỉ tính lại propagation (S1), hoặc dùng popularity từ lịch sử quan sát (S2). Phase C mới tạo và chọn checkpoint nền C0; phép so S0/S1/S2 chưa diễn ra.

## 2. Cơ sở lý thuyết

MovieLens 32M được canonicalize theo `(event_ts,user_raw_id,movie_raw_id,source_row_id)`. Một cặp user–movie trở thành positive edge khi có rating ≥4; giữ first positive, rating thấp sau đó không xóa edge. Graph hai phía nối users với movies. Normalization đối xứng dùng `1/sqrt(degree(user)×degree(movie))`. LightGCN truyền embedding qua graph rồi lấy trung bình các lớp; BPR tối ưu thứ tự positive so với sampled negative. BPR training loss đo mục tiêu train, không phải Recall@20/NDCG@20. [Protocol](../current/experiment-protocol.md) định nghĩa metric, windows và exclusion order.

## 3. Phương pháp và kiến trúc

Luồng: raw CSV → canonical Parquet → fixed-cohort projection/versioned normalized G0 → một LightGCN training trajectory → C4 checkpoints → C5 monthly validation → C6 C0 selection. Base cutoff là 01/01/2020 UTC; 2020 là validation, 2021–2022 development và Jan–Sep 2023 final holdout chưa mở. Model cohort cố định **49.587 users, 40.666 movies, 4.999.981 base positive edges, 90.253 nodes**. G0 chỉ có events `< cutoff`; mọi strategy sau này phải dùng cùng candidate catalog, observed-positive mask và evaluation users. [Architecture](../current/architecture.md), [C3](../research/goal04-phase-c3.md), [Phase C](../research/goal04-phase-c.md).

## 4. Thiết kế thực nghiệm

RQ1 dùng exact graph/degree equivalence trước khi so chi phí Full/Incremental; paired runs và snapshot cadence được mô tả trong [D12](../research/d12-exp01-results.md) và [D13](../research/d13-exp02-results.md). RQ2 dùng một seed 7, LightGCN dim64/layers3, BPR batch1024, Adam LR0,001, tối đa 5.000 steps; checkpoints 0/500/1500/3000/5000. C5 chấm full catalog và rolling mask trên 12 monthly windows 2020, chọn bằng macro NDCG@20, tie Recall@20 rồi step nhỏ nhất. Offline quality khác serving-like ranking cost; protocol ấn định ba repetitions cho Phase D, chưa được chạy.

## 5. Kết quả và thảo luận

**RQ1:** D12/S07 ghi nhận exact equivalence trong phạm vi đo H≤2M; tại Δ=1k và H=2M Incremental thắng 5/5 với median paired ratio 1,156. D13 cho thấy ít snapshot hơn khi R=5 với staleness tăng, normalized exports khớp ở common versions. Đây là kết quả prefix/config cụ thể, không chứng minh speedup trên 32M. [Claim R1-01–03](claim-evidence-map.md).

**RQ2 Phase C:** C3 v3 preflight G0 PASS trên dữ liệu thật. C4 v4 chạy một trajectory 5.000 steps, optimizer elapsed 1.847,123 s; năm checkpoint strict reload PASS. C5 v2 hoàn thành 60/60 quality records trên T4×2. Macro validation:

| Step | Recall@20 | NDCG@20 |
| ---: | ---: | ---: |
| 0 | 0,000579 | 0,000395 |
| 500 | 0,038283 | 0,028193 |
| **1500** | **0,038659** | **0,028306** |
| 3000 | 0,038406 | 0,028085 |
| 5000 | 0,038323 | 0,028011 |

C6 chọn **C0 step 1500**, SHA `2bf073f3…e66`; NDCG hơn step 500 khoảng 0,000113748. Step 3000 có loss thấp hơn nhưng không thắng validation NDCG. C5 runner mất 147,304 s, peak CPU process-tree RSS 5.762.551.808 bytes. [C0 selection](../../results/goal04_training/selection.json), [C5 matrix](../../results/goal04_c5/run_02/download/goal04_c5/validation-matrix.json).

**[CHƯA CÓ DỮ LIỆU]** Bảng paired S1−S0, S2 reference, coverage/CI theo 2021–2022 và cost repetitions. Cần Phase D implementation, toy gates, real preflight và Owner phê duyệt GPU budget; không suy kết quả từ Phase C. **[CHƯA CÓ DỮ LIỆU]** Final holdout Jan–Sep 2023; chỉ chấm sau khi khóa policy và được phép.

## 6. Kết luận và hạn chế

Đã khóa protocol, xác minh G0 và có một C0 reproducible cho nghiên cứu freshness. C0 selection dựa stale-G0 validation nên có thể ưu ái S0 khi so S1. Một seed, fixed warm cohort và chênh NDCG nhỏ không cho phép tuyên bố significance hay khái quát toàn MovieLens 32M. C3 single-boundary timing không chứng minh Phase D đáp ứng ngân sách. RQ2 cost–quality chưa có kết luận.

## 7. Tài liệu tham khảo và phụ lục

**[CẦN HOÀN THIỆN]** Danh mục paper gốc về LightGCN, BPR, recommender evaluation và MovieLens theo chuẩn trích dẫn của trường; không tự chế DOI/trang. Phụ lục cần protocol, hashes, raw comparison, run versions và lệnh verification từ [artifact recovery](../current/artifact-recovery.md). Các biểu đồ hiện có: [RQ1 crossover](../report-assets/exp01_crossover.svg), [snapshot tradeoff](../report-assets/exp02_cost_staleness.svg). Chưa tạo figure RQ2 S0/S1.
