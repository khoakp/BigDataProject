# Phạm vi và quyết định v2 hiện hành

Đề tài: thiết kế và đánh giá hệ thống cập nhật đồ thị tương tác tăng dần và chiến lược làm mới mô hình LightGCN trên MovieLens 32M. **RQ1** kiểm tra correctness, efficiency và scalability của Incremental Graph Maintenance so với Full Rebuild. **RQ2** sẽ so Stale Model, Propagation-only và Fine-tuning/Retraining trong GOAL 04. Snapshot cadence là nghiên cứu kỹ thuật hỗ trợ, không phải RQ thứ ba.

Các quyết định Project Owner đã chốt: graph cumulative historical-positive; một cạnh duy nhất cho `(user,movie)` sau bất kỳ rating `>= 4.0`; base `< 2020-01-01 00:00:00 UTC` (`1577836800`), replay `>=` mốc đó. Dữ liệu gốc MovieLens 32M và protocol đánh giá không được tự đổi. Model workload được chọn từ [GPU feasibility](../research/gpu-feasibility-v2.md): **4.999.981 positive edges, 49.587 users, 40.666 movies**, LightGCN 64 chiều, 3 propagation layers, một Kaggle Tesla T4. Đây là kết quả resource feasibility, chưa phải model freshness quality result.

[Temporal profile](../research/temporal-profile-v2.md) là bằng chứng cho cutoff. [Cohort manifest](../../results/model_cohort_v2_manifest.json) ghi selection và coverage. REC-01 1M và cutoffs riêng là bằng chứng lịch sử, không được biến thành protocol v2. [ADR cũ](../archive/legacy-reports/06-decisions-and-risks.md) còn hữu ích để truy nguồn quyết định, nhưng trang này sở hữu trạng thái đã duyệt mới nhất.

GOAL 03.5 refactor các APIs hiện có, bảo toàn graph correctness và checkpoint REC-01. Mandatory runtime source-code SHA-256 equality được Owner cho phép bỏ; report vẫn giữ source hashes lịch sử. Loader phải kiểm checkpoint integrity, graph version/normalization, node mapping, config và tensor shapes. Model Refresh Engine chưa có; không ghi kết quả RQ2 trước implementation/measurement.
