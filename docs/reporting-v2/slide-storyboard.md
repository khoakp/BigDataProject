# Storyboard bảo vệ đồ án v2 — 12 slide

Đây là dàn ý, chưa phải PPTX/PDF. Nguồn claims: [claim map](claim-evidence-map.md). Dùng assets sẵn có, không lấy ảnh ngoài dự án.

| # / tiêu đề | Thông điệp và nội dung | Hình/bảng, source | Speaker notes; câu hỏi có thể gặp |
| --- | --- | --- | --- |
| 1. Đề tài | Cập nhật graph và độ mới LightGCN trên MovieLens 32M | Sơ đồ nhỏ tự vẽ theo [architecture](../current/architecture.md) | Nêu RQ1/RQ2. Hỏi: hai RQ khác nhau ở đâu? |
| 2. Bài toán freshness | Ratings mới có thể làm graph và embedding cũ lệch lịch sử | Timeline `v/g/c/p` từ [protocol](../current/experiment-protocol.md) | Nói rõ graph update khác optimizer. Hỏi: stale nghĩa là gì? |
| 3. Dữ liệu và canonical | Positive ≥4, first-positive edge, UTC cutoff 2020 | Cohort/cutoff bảng trong [C3](../research/goal04-phase-c3.md) | Chỉ ra 32M ratings không có nghĩa 32M edges. Hỏi: rating thấp có xóa edge? |
| 4. Kiến trúc | Canonical → projection → snapshot → LightGCN → validation | Flowchart trong [architecture](../current/architecture.md) | Phân biệt full graph và fixed model cohort. Hỏi: vì sao mapping cố định? |
| 5. Protocol công bằng | 2020 validation; 2021–22 development; 2023 holdout khóa | Timeline [protocol](../current/experiment-protocol.md) | Nhấn mạnh mask `< boundary`, không dùng future. Hỏi: event đúng boundary thuộc đâu? |
| 6. RQ1 Full/Incremental | Exact equivalence trước chi phí; crossover chỉ trong phạm vi đã đo | [EXP-01 crossover](../report-assets/exp01_crossover.svg), [D12/S07](../research/s07-exp01-engineering-conclusions.md) | H=2M, Δ=1k thắng 5/5, median ratio 1,156. Hỏi: có suy ra 32M không? |
| 7. Snapshot cadence | R=5 giảm số export, đổi lấy staleness | [EXP-02 cost–staleness](../report-assets/exp02_cost_staleness.svg), [D13](../research/d13-exp02-results.md) | Common normalized versions khớp. Hỏi: staleness có phải quality? |
| 8. G0 preflight | C3 v3 kiểm 4.999.981 edges/90.253 nodes, hashes và no-future | Bảng G0 [C3](../research/goal04-phase-c3.md) | Không gọi preflight là model quality. Hỏi: vì sao G0 không chứa 2020? |
| 9. C4 trajectory | Một seed, một optimizer, năm checkpoints | Timeline steps [C4](../research/goal04-phase-c4.md) | Loss là train signal. Hỏi: step 3000 loss thấp thì sao? |
| 10. C5/C6 chọn C0 | 60 validation rows; step 1500 macro NDCG cao nhất | Bảng năm steps [Phase C](../research/goal04-phase-c.md), [matrix](../../results/goal04_c5/run_02/download/goal04_c5/validation-matrix.json) | NDCG hơn step 500 chỉ 0,000113748; chưa significance. Hỏi: vì sao chọn theo validation? |
| 11. D tiếp theo | S0 stale, S1 propagation-only, S2 popularity; chưa có số đo | Sơ đồ đề xuất [D0](../research/goal04-phase-d0.md) | Đây là plan, không trình bày như result. Hỏi: fairness giữa strategies? |
| 12. Kết luận/giới hạn | RQ1 có evidence theo phạm vi; RQ2 mới khóa C0 | Claim map và placeholders [report draft](report-draft.md) | Một seed/warm cohort/stale-G0 bias; final holdout chưa chạy. Hỏi: bước nghiệm thu kế tiếp? |

Khi tạo slides chính thức sau D0: kiểm số và mẫu số từng slide theo claim ID, đọc speaker notes thành tiếng, rồi mới chọn layout. Không dùng biểu đồ REC-01 1M để minh họa kết quả 5M nếu thiếu nhãn rõ.
