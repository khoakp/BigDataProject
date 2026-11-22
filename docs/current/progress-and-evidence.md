# Tiến độ và bằng chứng v2

Đã có: temporal profiling và cutoff 2020 được Owner chốt; graph Full/Incremental, equivalence, normalization và snapshot; workload 5M/64d/3 layers đo resource trên một Kaggle T4; REC-01 historical checkpoint/inference; Streamlit research demo. [Temporal evidence](../research/temporal-profile-v2.md), [GPU evidence](../research/gpu-feasibility-v2.md), [research evidence index](../research/README.md).

GOAL 03.5 đã triển khai M0–M6 trên `refactor/v2-cleanup`. [Audit/baseline](../archive/legacy-reports/goal035-audit-plan.md), [implementation report và dependency graph](../archive/legacy-reports/goal035-implementation-report.md), [Markdown classification map](../archive/legacy-reports/goal035-documentation-map.md). Bằng chứng baseline và checkpoint trước/sau refactor nằm trong `results/goal035_baseline/`; full regression đạt **74 PASS, 0 FAIL, 0 SKIP**.

GOAL 03.6 đã chuyển nội dung lịch sử vào archive và measured reports vào research, giữ compatibility paths cho frozen report/scripts. Xem [documentation maintenance](documentation-maintenance.md). Thay đổi đang chờ Project Owner nghiệm thu; chưa commit/push/merge.

Chưa có Model Refresh Engine, persisted propagated embeddings v2 hoặc kết quả freshness RQ2. GOAL 04 bắt đầu sau khi refactor được nghiệm thu và policy cho new nodes, candidate universe, refresh budget được chốt.
