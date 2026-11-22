# S04 — REC-01: LightGCN và Popularity trên MovieLens 1M

> **Hồ sơ lịch sử GOAL 03.5:** giữ để truy nguồn quyết định/kết quả cũ. [Phạm vi v2 hiện hành](../current/scope-and-decisions.md) · [Tiến độ v2](../current/progress-and-evidence.md).

**Run:** `REC01-Kaggle-1M`, [Kaggle Notebook Version 3](https://www.kaggle.com/code/khoakp/rec01-movielens-1m-gpu-validation?scriptVersionId=372012165), GPU Tesla T4. Dataset là canonical prefix 1.000.000 sự kiện từ MovieLens 32M, SHA-256 `f17048bd7b12bea613387c169c65db6ce67442aa7341314f5e18fba39efcfbae`. Artifact local ở `data/processed/rec01_kaggle_runs/version_3/rec01_project/REC01-Kaggle-1M/`; `verify-output PASS` với bundle S03. Đây là kết quả REC-01 trên prefix 1M, không phải toàn bộ 32M.

## Giao thức

Theo [S01 đã duyệt](s01-evaluation-spec.md): chia theo 80%/90% **distinct timestamps** (`train_end=840100781`, `valid_end=841267862`), tương ứng 802.988/102.046/94.966 sự kiện train/validation/test. Candidate universe gồm 773 phim trong train catalog; loại train-seen items khi xếp hạng, đánh giá Top-20 trên cùng warm cohort cho hai phương pháp. Positive là `rating >= 4.0`. LightGCN embedding 64 chiều, 3 lớp, BPR batch 1.024, learning rate 0,001, seed 7, tối đa 500 optimizer steps. Chọn trong checkpoint 100/300/500 bằng validation NDCG@20, sau đó reload checkpoint đã chọn rồi mới đánh giá test. Popularity đếm positive interactions chỉ từ train. Cấu hình SHA-256 `8b8223681cf181d0ae9bd2b794d0e0f8159b3c01443622fabffff27820b8b8e8`.

## Kết quả thực đo

| Split | Phương pháp | Checkpoint | Warm users | Warm positive pairs | Recall@20 | NDCG@20 |
|---|---|---:|---:|---:|---:|---:|
| Validation | LightGCN | 100 | 482 | 3.333 | 0,119490 | 0,082000 |
| Validation | LightGCN | 300 | 482 | 3.333 | 0,120954 | 0,083543 |
| Validation | LightGCN | 500 | 482 | 3.333 | 0,119489 | 0,082913 |
| Validation | Popularity | — | 482 | 3.333 | 0,120696 | 0,085337 |
| Test | LightGCN | **300** | 403 | 2.069 | 0,128000 | 0,085161 |
| Test | Popularity | — | 403 | 2.069 | **0,128099** | **0,086897** |

Checkpoint 300 có validation NDCG@20 cao nhất trong ba checkpoint LightGCN; SHA-256 `2c1d1e5e61b069776204be751635ee950cea7ccf27062107eedb53411688d9b7`. `selected_checkpoint_reloaded=true`. Trên test, Popularity cao hơn LightGCN khoảng 0,000099 Recall@20 và 0,001736 NDCG@20. Không có kiểm định độ tin cậy thống kê; không kết luận mô hình nào thắng một cách tổng quát. Validation Popularity cũng cao hơn checkpoint LightGCN được chọn, nhưng quy tắc S01 chọn checkpoint **trong các checkpoint LightGCN**, không chọn mô hình theo validation.

## Cohort và giới hạn

| Split | Future positive pairs | Warm users | Warm pairs | Cold-user exclusions | Cold-movie exclusions | No-candidate users |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 47.300 | 482 | 3.333 | 43.723 | 244 | 0 |
| Test | 44.865 | 403 | 2.069 | 42.176 | 620 | 0 |

Warm cohort rất nhỏ so với future positive pairs vì chỉ 773 train movies trong prefix này. Hai phương pháp dùng cùng cohort, candidates và train-seen mask, nên so sánh nội bộ là nhất quán; con số không đại diện cho cold-start hay toàn bộ 32M. Chưa thử hyperparameter khác hoặc nhiều seed. Không đổi protocol sau khi xem test. `movies.csv` chỉ enrich title/genres sau ranking, không là đầu vào graph hay model.

## Kiểm chứng và provenance

- `report.json`, `rec01_quality.csv`, `output_manifest.json`, ba checkpoint và `graph_build/` nằm trong run dir nêu trên. ZIP gốc `data/processed/rec01_kaggle_runs/version_3/results.zip` có SHA-256 `25b3f97606aa330fc5892195e7afea7b825f48818c45b31b8b947d3f6172fe49`.
- `verify-output PASS`: kiểm danh sách/hash file, source/config/code, checkpoint/reload, cohorts, sáu CSV rows, graph Full/Incremental và ID map. Graph version 81, 14.388 nodes, 392.626 positive train edges; normalized adjacency có 785.252 nonzeros; Full/Incremental registry/edges/degrees hashes khớp.
- `report.json` ghi graph build 22,11 giây và training trajectory 17,08 giây; trajectory gồm optimizer, checkpoint save và validation scoring. Kaggle hiển thị tổng runtime 1 phút 35 giây. Không đồng nhất ba thời gian này hoặc xem trajectory là thời gian train thuần.
- Kiểm thử inference thật ở [S05 loader](../../src/models/rec01_inference.py): warm user `25062` trả 20 phim theo LightGCN; cold user `-1` trả 20 phim theo Popularity; cả hai có metadata 20/20. Đây là smoke inference, không phải thêm một phép đo quality.
