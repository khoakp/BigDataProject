# GOAL 04 — Phase C hoàn thành: chọn C0

Ngày 24/11/2026. **PHASE_C_COMPLETE; dừng trước Phase D.** Owner đã nghiệm thu C5 và cho phép C6 chọn C0 từ đúng 12 tháng validation năm 2020. [Selection manifest](../../results/goal04_training/selection.json) khóa **checkpoint step 1500** bằng tham chiếu đến bytes C4 gốc; [re-verification](../../results/goal04_training/verification.json) PASS. Không train lại, không rewrite checkpoint, không tạo propagated embeddings P0 hoặc chấm final holdout.

## Input và preflight

Protocol dùng MovieLens 32M canonical, positive `rating >= 4.0`, base graph chỉ chứa events trước `2020-01-01T00:00:00Z`. Fixed cohort gồm 49.587 users, 40.666 movies, 4.999.981 positive edges và 90.253 nodes. [Local real-5M preflight](goal04-phase-c-preflight.md) run_02 đã export/reload G0 và replay một monthly boundary; [Kaggle C3 v3](goal04-phase-c3.md) kiểm nguồn canonical, cohort, fixed mapping, edge/degree/COO hashes, normalization và snapshot export/reload trên dữ liệu thật. Không có future edge trong G0. C3 v3 chỉ preflight, không optimizer hoặc quality evaluation.

Input C6 là [C4 candidate manifest](../../results/goal04_c4/run_04/goal04_c4/candidates/report.json), [C5 matrix gốc](../../results/goal04_c5/run_02/download/goal04_c5/validation-matrix.json), [comparison CSV](../../results/goal04_c5/run_02/download/goal04_c5/candidate-comparison.csv) và [C5 verification](../../results/goal04_c5/run_02/verification.json). Không lấy GitHub remote làm nguồn thay thế cho local worktree đang có các thay đổi chưa commit.

## Training và validation

[C4 v4](goal04-phase-c4.md) chạy một trajectory seed 7 trên T4 `cuda:0`: LightGCN 64 chiều, 3 layers, BPR batch 1.024, Adam LR 0,001, 5.000 optimizer steps. Checkpoints tại 0/500/1500/3000/5000. Optimizer elapsed **1.847,123 s**, runner end-to-end **1.968,106 s**; peak process-tree RSS **4.348.219.392 bytes**, peak CUDA allocated **1.001.759.744 bytes**. Training loss là sampled BPR loss, không dùng để chọn C0.

[C5 v2](goal04-phase-c5-preparation.md) chấm đủ 60/60 `(step, monthly window)` records trên 12 tháng 2020 với fixed 40.666-movie catalog, rolling observed-positive mask và exact Top-20. Mỗi checkpoint propagation một lần trên G0; hai T4 chia tháng 1–6/7–12 sau numerical gate. Cả năm gate đều PASS: 16 sampled users/checkpoint, cùng Top-20 và max score difference 0 trong mẫu; tổng **5 forwards, 0 optimizer updates**. C5 runner end-to-end **147,304 s**; peak process-tree RSS **5.762.551.808 bytes**, peak CUDA allocated **639.998.976 bytes** (`cuda:0`) và **124.751.872 bytes** (`cuda:1`).

| Step | Training loss tại checkpoint | Macro Recall@20 | Macro NDCG@20 |
| ---: | ---: | ---: | ---: |
| 0 | — | 0,000578651 | 0,000395025 |
| 500 | 0,076977 | 0,038282769 | 0,028192684 |
| **1500** | 0,072429 | **0,038658790** | **0,028306432** |
| 3000 | 0,060620 | 0,038405946 | 0,028085449 |
| 5000 | 0,070978 | 0,038323185 | 0,028011263 |

`scripts/finalize_goal04_c0.py` kiểm SHA-256 matrix gốc `e88170c851dd1658fd31491a83eb11dcac8b6276e7e87cf4904928df5bceb138`, status C5 COMPLETE và verifier PASS; kiểm 60 unique records, đúng 12 UTC windows, cùng users/labels **counts** và mask/catalog/source provenance giữa candidates mỗi tháng; tính lại macro metrics từ từng monthly row và so official comparison. `src/evaluation/checkpoint_selection.py` chọn NDCG@20 cao nhất, nếu bằng thì Recall@20 cao nhất, cuối cùng step nhỏ nhất. Kết quả độc lập là **1500**, khớp dự kiến. NDCG hơn step 500 khoảng **0,000113748**; không thay tie tolerance và không chọn theo loss. Matrix không chứa raw user-ID/label-list hashes, nên đối chiếu trực tiếp từng danh sách user/label là giới hạn còn lại; C5 runner dùng chung một windows object cho mọi candidate.

## C0 artifact và provenance

C0 tham chiếu [checkpoint C4 step 1500](../../results/goal04_c4/run_04/goal04_c4/candidates/candidate_1500.pt) đúng **23.107.674 bytes**, SHA-256 `2bf073f328551561bf21941ee9fa7f3d239080139922de348f86426862a25e66`. Strict `load_candidate` xác minh schema `lightgcn_v2_c0_candidate_v1`, config, payload step 1500, embedding shape **90.253 × 64**, 3 layers, tensor dtypes/shapes, full checkpoint bytes và provenance. G0 snapshot manifest SHA-256 `0feaa2c521b0adb364fa78468a86f8a39fac979d630053fe0cb8aff39910ebdb`; fixed mapping `996848022d92e20ef63b498cfedb38b97545fb006da4c00fef9ed35d40dc90f9`; canonical source `494a1e70e67a2d750297872020b3919da9e557dd68cb631f53d2c5cb9570f73e`; cohort manifest `10411c083ec8076bf831e4cbf5c8180518fcd557feaa43898142547ced77929e`. Training Git HEAD `1745739744f6f4aae0d97ddf83d03a7a80eacbb9`, source bundle `4884205b49063cd12a9b277678f5d121161b33506cb323ffc7dc6be1f04c81a2`. Per-file/evidence hashes nằm trong selection manifest. HEAD không bao gồm local uncommitted changes; bundle hash xác định code Kaggle đã chạy.

GitHub lưu source, reports và manifests; Parquet snapshots cùng checkpoint `.pt` vẫn là artifacts local/private Kaggle và không nằm trong Git. Muốn chạy strict verification trên checkout mới, cần lấy đúng bytes C3 v3/C4 v4 đã duyệt rồi đối chiếu SHA-256 trong manifest.

## Budget và giới hạn diễn giải

Owner-scope accounting dùng log cuối của C3 v3, C4 v4, C5 v1 và C5 v2: **2.529,247/14.400 s**, còn **11.870,753 s**. Đây không phải exact terminal elapsed; teardown sau log cuối chưa đo được. C6 chỉ xử lý local, không thêm Kaggle job. [C5 assessment](../../results/goal04_c5/run_02/assessment.json) giữ raw timing/resource evidence.

C0 được chọn bằng **validation**, không phải independent holdout performance. Đây là single-seed result; chênh NDCG 1500–500 nhỏ và chưa có kiểm định ý nghĩa thống kê. Metrics đại diện fixed warm cohort, không suy ra toàn MovieLens 32M. Chọn checkpoint trên stale-G0 validation có thể ưu ái S0 khi sau này so Propagation-only S1. Chưa có S0/S1/S2 comparison hoặc final holdout quality; không tuyên bố chiến lược nào vượt trội. Numerical gate chỉ lấy mẫu 16 users/checkpoint, không chứng minh mọi user có identical scores trên hai GPU.

## Tái lập và readiness Phase D

`src/models/c0.py::load_candidate` là strict checkpoint API; `src/evaluation/checkpoint_selection.py::select_checkpoint` áp approved ordering; `scripts/finalize_goal04_c0.py::recompute_matrix` và `verify_selection` tái kiểm matrix và manifest. Dòng dữ liệu: C3 G0 + C4 candidates → C5 monthly quality → C6 recompute/selection → C0 manifest tham chiếu checkpoint C4. C0 hiện đã khóa theo hash và sẵn sàng để Owner review Phase D. Phase D cần approval riêng; P0 và model refresh engine chưa tạo.

[C6 targeted tests](../../tests/test_goal04_c6.py) kiểm deterministic NDCG/Recall/step tie breaks, thiếu monthly row, sai mapping, sai selected step và checkpoint bytes bị đổi. [Full regression](../../results/goal04_training/regression.log) **103 PASS**; post-write `verify_selection` PASS, `git diff --check` PASS. Không sửa raw evidence hoặc historical artifacts.

```powershell
..\venv\Scripts\python.exe -m scripts.finalize_goal04_c0 --verify --output results/goal04_training/selection.json
..\venv\Scripts\python.exe -m unittest discover -s tests -p test_goal04_c6.py -v
..\venv\Scripts\python.exe -m unittest discover -s tests -v
```

Ví dụ cho sinh viên mới: step 3000 có training loss thấp hơn step 1500, nhưng NDCG validation thấp hơn. Vì luật đã chốt dùng NDCG trước, C0 là step 1500. Recall cho biết tỷ lệ phim relevant xuất hiện trong Top-20; NDCG còn thưởng phim relevant xuất hiện gần đầu danh sách. Final holdout sau này mới kiểm khả năng khái quát ngoài năm chọn checkpoint.

Ba câu tự kiểm tra: (1) Vì sao không chọn step 3000 theo training loss? (2) Vì sao chênh NDCG 0,000113748 chưa đủ để khẳng định ý nghĩa thống kê? (3) Vì sao C0 selection trên G0 có thể gây bias khi so S0 với S1?
