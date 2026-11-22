# S01 — Kiểm kê và giao thức đánh giá REC-01

> **Hồ sơ lịch sử GOAL 03.5:** giữ để truy nguồn quyết định/kết quả cũ. [Phạm vi v2 hiện hành](current/scope-and-decisions.md) · [Tiến độ v2](current/progress-and-evidence.md).

**Ngày khóa:** 21/11/2026. **Trạng thái tại thời điểm khóa:** Project Owner đã duyệt giao thức REC-01 trong chat; khi đó chưa có kết quả REC-01. [Định hướng v1.0](10-direction-update-v1.md) là nguồn phạm vi. Các run D08–D10 là kiểm chứng MVP cũ, không được đổi nhãn thành REC-01. **Cập nhật thực thi:** [S03 Kaggle](s03-kaggle-workflow.md) và [S04 REC-01](s04-rec01-results.md) đã hoàn tất theo giao thức này; các mục “chưa có” bên dưới là kiểm kê lịch sử trước run.

## 1. Kiểm kê code và bằng chứng tại thời điểm khóa S01

| Yêu cầu mới | Trạng thái hiện tại | Bằng chứng / việc còn thiếu |
|---|---|---|
| Train graph chỉ dùng quá khứ, split theo event-time | Đã có trên 100k/1M | `src/evaluation/temporal.py`, `src/models/d08_baseline.py`, `tests/test_temporal_leakage.py`; cần cố định một dataset/cutoff cho REC-01 |
| LightGCN sparse, BPR, negative sampling, checkpoint | Có MVP 100 bước trên laptop CUDA | `src/models/lightgcn.py`, `src/models/d08_baseline.py`, D10 manifest; chưa có workflow Kaggle hoặc checkpoint selection bằng validation |
| Recall@20, NDCG@20 và Popularity baseline | Utilities S02 đã thêm; **chưa có run REC-01** | `src/evaluation/baselines.py`, `src/evaluation/rec01.py`, toy tests; các run cũ vẫn là Top-10, chưa có comparison thật |
| Seen-item masking và exclusions | Có cho warm train catalog | `score_chunks` và `cutoff_cohort`; phải công bố candidate count, tỷ lệ warm/cold trong REC-01 |
| EXP-01 Full/Incremental correctness và cost | Đã có D11–D12 | Giữ kết quả lịch sử, không trộn với quality evaluation |
| EXP-02 deferred normalization | Đã có D13 | Giữ mã EXP-02, chuyển khỏi critical path mới |
| Streamlit, rating mới, graph/model refresh | **Chưa có** | D10 chỉ là CLI đọc Top-10 tính sẵn; S09–S10 phải triển khai và kiểm tra version semantics |

Run D08 100k và 1M dùng các cutoff tuyệt đối khác nhau vì tính tỷ lệ trên từng prefix. Không dùng hai bộ Recall@10 đó làm đường chất lượng theo scale. D10 100k chỉ có 20 user validation và 10 user test đủ điều kiện; vì vậy run đó là smoke MVP, không đại diện MovieLens 32M.

## 2. Dataset, thời gian và nhãn của REC-01 chính

- Nguồn chính để mở gate S04: **canonical prefix 1.000.000 sự kiện** của MovieLens 32M, SHA-256 `f17048bd7b12bea613387c169c65db6ce67442aa7341314f5e18fba39efcfbae`. Raw ratings vẫn được giữ để Data Engineering. Scale khác nếu chạy sẽ là run bổ sung có ID/config/cutoff riêng; không ghép metric với run chính như cùng một cohort.
- Canonical ordering theo D02: `timestamp, userId, movieId, source_row_id`. Dùng distinct event timestamps để chọn tỷ lệ thời gian 80% train, 10% validation, 10% test; giữ nguyên tất cả events cùng timestamp một phía. Kiểm tra trực tiếp file prefix 1M cho 478.972 timestamps phân biệt, `train_end_ts=840100781`, `valid_end_ts=841267862`, số events train/valid/test = `802988 / 102046 / 94966`. S03 phải assert lại các giá trị và input hash trước khi train.
- Graph và mọi tham số của model chỉ học từ train events `event_ts <= train_end_ts`. Positive interaction là unique user–movie pair với `rating >= 4.0`; đây là lựa chọn thiết kế DEC-005, chưa chứng minh ngưỡng tối ưu. Validation labels ở `(train_end_ts, valid_end_ts]`, test labels ở `> valid_end_ts`. Không dùng validation/test để tạo node, degree, negative sampling pool hay Popularity counts.
- Candidate universe chính: **mọi movie node có positive train edge**; mask những movie user đã có positive train edge. Không coi mọi movie trong `movies.csv` là candidate. Future positive chỉ đủ điều kiện khi user và movie đều thuộc train graph và movie chưa nằm trong train-seen set của user. User có ít nhất một future positive đủ điều kiện mới vào macro metrics. Công bố `future_positive_pairs`, `warm_pairs`, `warm_users`, `cold_user`, `cold_movie`, `train_seen_repeat`, số candidate movies và tỷ lệ được bao phủ.
- Kiểm kê trên run D08 1M cũ: train graph có 14.388 nodes, 392.626 edges, 773 candidate movies; valid 482 warm users/3.333 warm pairs từ 47.300 future positives, test 403/2.069 từ 44.865. Đây là **profile cohort trước REC-01**, không phải kết quả đánh giá Top-20. Hạn chế cold user phải đi cùng mọi kết luận về quality.

## 3. Models, checkpoint và metrics

- **LightGCN:** sparse normalized train graph, 64 chiều, 3 propagation layers, BPR với train-positive và train-catalog negative chưa có positive edge của user; Adam `lr=0.001`, batch 1.024 triples, seed 7. Chạy tối đa 500 optimizer steps trên một train graph. Candidate checkpoints ở bước **100, 300, 500** cùng một trajectory. Ghi rõ objective implementation hiện tại là `softplus(-margin)` **chưa có L2 regularization**; không mô tả nó như full objective nếu chưa triển khai regularization.
- **Chọn checkpoint:** sau mỗi candidate step, tính validation NDCG@20 trên cohort cố định. Chọn NDCG@20 cao nhất; nếu bằng nhau, ưu tiên Recall@20 cao hơn, rồi bước sớm hơn. Ghi cả ba validation scores. Chỉ sau khi đã chốt checkpoint, load đúng file checkpoint đó và tính test **một lần**; không chọn theo test. Nếu run thất bại/OOM, ghi status và artifact của attempt, không thay đổi tập candidate steps sau khi nhìn test.
- **Popularity baseline bắt buộc:** đếm số unique train-positive users cho từng movie từ cùng train graph (tương ứng degree phía movie). Xếp theo count giảm, `movie_raw_id` tăng để phá hòa; áp dụng cùng candidate universe, seen mask, eligible users và nhãn. Không dùng future popularity. Utility S02 đã có tại thời điểm khóa; [S04](s04-rec01-results.md) ghi run 1M sau đó.
- **Metric chính:** macro Recall@20 và NDCG@20 trên eligible users. Với mỗi user, Recall là số relevant items trong Top-20 chia số relevant eligible items; NDCG dùng binary relevance, `DCG = Σ hit_r/log2(r+1)` cho rank một-based và `IDCG` theo `min(20, |relevant|)`. Trung bình theo user, không theo event. Nếu cohort rỗng, ghi `UNAVAILABLE`, không điền 0. So LightGCN/Popularity theo cùng validation và test cohort; công bố số user và candidate/exclusion counts cạnh metric.
- Không tune hyperparameter, cutoff, candidate set, sampling policy hoặc train scale theo test. Số liệu D08/D10 Top-10 chỉ là lịch sử; run REC-01 Top-20 với validation selection được ghi riêng ở [S04](s04-rec01-results.md).

## 4. Artifact, kiểm thử và gate

REC-01 raw artifact yêu cầu: quality CSV (mỗi model/checkpoint/split một dòng), config JSON với input hash và cutoffs, train graph manifest/version, từng candidate checkpoint có SHA-256, model-index ↔ raw-ID mapping, environment/GPU/Kaggle provenance, cohort/exclusion report, per-user ranking hoặc các hash để audit. Artifact thực tế được giữ ngoài Git ở `data/processed/rec01_kaggle_runs/version_3/rec01_project/REC01-Kaggle-1M/`; [S04](s04-rec01-results.md) dẫn file và hash. `model_graph_version`, `checkpoint_train_cutoff`, `parameter/checkpoint_version` và thời điểm propagation phải tách biệt; graph update đơn thuần không làm checkpoint mới.

S02 cần unit/integration tests cho: timestamp tie không leak; test labels không đi vào graph/negative pool; Popularity dùng train-only counts; hai models cùng candidate/seen mask/cohort; Top-20 tie determinism và metric toy reference; checkpoint selection không đọc test. S03 cần chứng minh checkpoint load lại cho cùng Top-K trên warm users. S04 chỉ PASS khi REC-01 raw results cho LightGCN **và** Popularity, Recall/NDCG@20 valid/test, selection log, hashes, exclusions và test results đạt.

Lệnh cũ để đối chiếu (chưa phải lệnh REC-01):

```powershell
..\venv\Scripts\python.exe -m unittest discover -s tests
..\venv\Scripts\python.exe -m src.models.d08_baseline --source data/processed/canonical_events/canonical_1000000.parquet --out data/processed/d08_baseline_1m --steps 100 --device cuda
```

**Project Owner đã duyệt:** primary prefix 1M, cutoff 80%/90% distinct timestamps, warm train-catalog candidates, checkpoint steps 100/300/500 và quy tắc tie như trên. Kaggle Dataset private, Notebook Version 3 và artifact local đã được kiểm chứng ở [S03](s03-kaggle-workflow.md). Thời hạn nộp chính thức vẫn cần xác nhận cho S14.
