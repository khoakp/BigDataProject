# GPU feasibility và LightGCN workload selection (v2)

**Trạng thái:** cutoff **01/01/2020 00:00 UTC đã được Project Owner duyệt**. Cohort selection, probe local, private Kaggle T4 sweep và trajectory 500 bước đều đã chạy. Sau khi Project Owner cho phép upload private, hai kernel versions trên Kaggle đo được workload **5M edges / 64d / 3 layers** và ngân sách **500 optimizer steps** trên `cuda:0`. Đây là quyết định về **khả thi tài nguyên**, chưa phải kết quả quality hay Model Refresh Engine.

## Nguồn dữ liệu và định nghĩa cold

Theo [temporal profile](temporal-profile-v2.md) của `ratings.csv` SHA-256 `91159850e41ee59c86231165a688709647e2726cab2e7ba9faf04001bd5261ee`, base `< 1577836800` gồm **27.364.974 ratings**, **13.644.889 positive edges**, **179.192 positive user nodes** và **41.515 positive movie nodes**. Replay `>= 1577836800` gồm **4.635.230 ratings**, **2.293.342 positive pairs**. Ngưỡng `rating >= 4.0` và cumulative historical-positive semantics giữ nguyên.

Con số **73,4%** là **21.529/29.346 distinct users có bất kỳ rating nào trong replay** mà rating đầu tiên của họ ở hoặc sau cutoff. Nó không phải 73,4% của rating rows, positive pairs, hay users ngoài model cohort. Tương tự, **24.480/70.027 replay-active movies = 35,0%** có rating đầu tiên sau cutoff; đây không phải ngày phim được phát hành. Với positive graph, 21.534/29.181 positive-active replay users (73,8%) và 13.659/40.181 positive-active replay movies (34,0%) lần đầu có positive edge sau cutoff. Trong **2.293.342 future positive pairs**, **1.948.688 pairs** không có user node trong base positive graph, khoảng 85,0% theo trọng số pair; tập này còn có thể gồm user đã rating trước cutoff nhưng chưa có positive edge. Base có 59.952 movies từng được rating nhưng chỉ 41.515 positive-graph movie nodes. Các mẫu số rating, graph node và pair phải giữ riêng.

## Bottleneck audit của pipeline hiện có

| Pha | Code hiện tại | Hệ quả khi tăng graph |
| --- | --- | --- |
| CPU graph state | `src/graph/equivalence.py:incremental_state` dùng `fetchall()` cho registry/edges/degrees. `src/graph/normalize.py:normalize_state` tạo set dedup, Counter, hai Python tuples cho mỗi edge rồi sort `2E` entries. | Peak CPU RAM lớn hơn nhiều so với payload COO `40E` bytes; O(E log E) sort. Đây là gate trước khi vào GPU. |
| Snapshot | `src/graph/snapshot.py:materialize_normalized` tạo state và adjacency Python, lưu COO vào DB; `load_export` fetch toàn bộ lần nữa. `src/models/rec01_train.py:242-255` còn giữ/reference và so sánh nhiều bản graph. | Nhiều bản sao graph/sparse COO đồng thời; snapshot build, export, reload phải đo riêng. |
| GPU tensor | `src/models/lightgcn.py:to_sparse_tensor` dùng `zip(*adjacency.entries)`, tạo `int64` indices và `float32` weights rồi `.coalesce()`. | Các tuple/copy trung gian CPU và GPU có thể vượt xa COO payload 20 bytes/entry. |
| Training | `LightGCN.forward` chạy sparse matrix multiplication trên **toàn graph** ở mỗi layer; `src/models/rec01_train.py:281-291` gọi forward/backward trên toàn graph cho mỗi optimizer step. | Mini-batch BPR không làm chi phí propagation theo graph giảm. Adam cấp phát moment ở step đầu; phải đo first step riêng. |
| Checkpoint/evaluation | `rec01_train.py:_save_checkpoint` lưu parameter **và registry**; `src/evaluation/temporal.py:score_chunks` xếp hạng trên train catalog, `.tolist()` đồng bộ GPU→CPU mỗi chunk. | Checkpoint bytes, warm-user × catalog scoring và validation checkpoint selection phải tách khỏi pure optimizer timing. REC-01 `trajectory_s` đã gồm save và validation, không dùng làm training-step latency. |

REC-01 Kaggle T4 lịch sử chỉ có 14.388 nodes, 392.626 edges; checkpoint ~4,16 MB. Không ngoại suy thời gian REC-01 sang graph v2 có triệu edges.

Ước lượng lý thuyết cho graph **5M**: 9.999.962 directed COO entries × (2 `int64` indices + 1 `float32` weight) = **190,7 MiB** payload; 90.253 nodes × 64d × 16 bytes cho parameter + gradient + hai Adam moments = **88,1 MiB**. Hai khoản này chưa gồm Python tuples, coalesce copies, activations và CUDA allocator. Số đo thực tế bên dưới là **3,42 GiB process RSS** và **1.124 MiB GPU reserved**; không dùng payload lý thuyết thay cho peak.

## Cohort protocol và độ phủ

`src/evaluation/model_cohort.py:prepare_cohorts` đọc positive edges **trước cutoff** để xếp users theo last positive timestamp giảm dần, sau đó degree giảm dần và raw user ID tăng dần; chọn prefix users có tổng edges không vượt budget và giữ **tất cả** positive edges trước cutoff của users đó. Graph này là cumulative historical-positive **đầy đủ trong selected-user cohort**, với node IDs riêng, xác định. Future positive pairs chỉ được đọc **sau khi selection cố định** để đo coverage; không dùng future labels để chọn graph. Việc so sánh policies theo future coverage ở đây là **phân tích khám phá**. Trước EXP-03 phải đóng băng policy/budget và tách một evaluation window chưa dùng cho quyết định quality; các tỷ lệ coverage trên toàn replay không phải ước lượng quality độc lập.

| Cohort | Users | Movies | Positive edges | Candidate catalog / base movies | Evaluable future pairs / 2.293.342 | Evaluable warm users | Evaluable warm movies |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| User-recency 0,5M | 1.950 | 24.616 | 499.694 | 59,3% | 85.184 (**3,71%**) | 1.622 | 12.014 |
| User-recency 2M | 13.540 | 36.468 | 1.999.995 | 87,8% | 213.629 (**9,32%**) | 5.398 | 17.051 |
| **User-recency 5M** | **49.587** | **40.666** | **4.999.981** | **98,0%** | **268.965 (11,73%)** | **7.152** | **18.349** |
| Full pre-2020 base | 179.192 | 41.515 | 13.644.889 | 100% | 291.748 (**12,72%**) | 7.557 | 18.732 |

Ở 2M, **2.031.453** future pairs bị loại vì user ngoài model: **1.948.688 true base-cold** và **82.765 pre-cutoff user bị cohort bỏ qua**. Trong phần còn lại, **48.260** pairs có movie ngoài model: **47.556 true base-cold movie** và **704 pre-cutoff movie bị cohort bỏ qua**. Các loại trừ là phân hoạch có thứ tự: xét user trước, rồi movie, rồi train-seen. Dataset này không có future positive pair lặp với train-positive pair, nên `train_seen_repeat_pairs=0`; mask vẫn là yêu cầu giao thức.

Ở workload **5M đã chọn**, **1.972.244** future pairs có user ngoài model, gồm **1.948.688 base-cold** và **23.556 warm-base users bị cohort bỏ qua**. Sau khi lọc user, **52.133** pairs có movie ngoài model, gồm **52.049 base-cold movies** và **84 warm-base movies bị bỏ qua**; còn **268.965** pairs đánh giá được. Các counts là số **pairs** theo thứ tự loại trừ, khác với first-observed **distinct entities** ở phần đầu.

[Policy comparison CSV](../results/model_cohort_v2_policy_comparison.csv) so top-movie, top-user, user-recency, chronological prefix và deterministic hash ở cùng budget. Hash edge sampling 2M cho tỷ lệ future-pair cao hơn, nhưng bỏ một phần historical-positive edges giữa selected nodes, nên **không hợp lệ** cho model graph cumulative trong nghiên cứu này. Top-movie 2M chỉ còn 49 candidate movies và 0,58% evaluable pairs. User-recency có bias về users hoạt động gần cutoff, vì vậy báo cáo quality sau này chỉ áp dụng cho cohort đã chọn, không đại diện toàn bộ users.

Các graph Parquet và per-cohort manifest nằm local ở `results/model_cohort_v2_final/` (khoảng 208 MB, được Git ignore để tránh đưa dữ liệu lớn vào repo). Compact [cohort manifest](../results/model_cohort_v2_manifest.json) và policy CSV ghi counts, source hash, cutoff, selection và coverage. Mỗi graph có `nodes.parquet` (`node_id,node_type,raw_id`) và `edges.parquet` (`user_node_id,movie_node_id,first_event_ts`), SHA-256 trong manifest; tất cả edges có timestamp trước cutoff.

## Probe local thực đo, không phải T4

`scripts/gpu_feasibility_probe.py` đo từng pha bằng đồng bộ CUDA, sampled process RSS (chu kỳ 10 ms, có thể bỏ sót transient ngắn), peak allocated/reserved VRAM và checkpoint bytes. Ba run dùng **RTX 4050 Laptop GPU 6 GB**, PyTorch 2.8.0+cu126, `cuda:0`, dimension 64, 3 layers, BPR batch 1.024, Adam LR 0,001, seed 7, **5 steps**, ranking 100 users có node ID nhỏ nhất. Ranking là smoke cost, không đại diện latency trên mọi warm users. Mỗi dòng là **một run**, không có dispersion/CI; thời gian giữa GPU khác nhau không được ngoại suy.

| Cohort | CPU peak RSS | GPU peak reserved | Python normalize | COO conversion | First backward+Adam step | 4 steps tiếp | Rank 100 users | Checkpoint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0,5M | 1,38 GiB | 162 MiB | 0,72 s | 0,92 s | 0,33 s | 0,08 s | 1,37 s | 7,28 MiB |
| 2M | 2,11 GiB | 456 MiB | 2,88 s | 3,35 s | 0,17 s | 0,32 s | 1,90 s | 13,69 MiB |
| 5M | 3,61 GiB | 1.124 MiB | 7,47 s | 10,49 s | 0,48 s | 0,83 s | 2,15 s | 24,74 MiB |

Phase-level run reports: [0,5M](../results/gpu_feasibility_local/500k.json), [2M](../results/gpu_feasibility_local/2m.json), [5M](../results/gpu_feasibility_local/5m.json), [summary.csv](../results/gpu_feasibility_local/summary.csv). Chỉ trường `cohort` trong bản Git được đổi thành đường dẫn tương đối; số đo và hashes giữ nguyên. First-step timing chịu khởi tạo Adam/CUDA nên không diễn giải như steady-state throughput. CPU RSS gồm Python/PyTorch baseline; graph construction làm nó tăng rõ rệt. Đây là preflight local, không phải kết quả Kaggle T4 bên dưới.

## Một T4 16 GB: protocol và số đo Kaggle

[Kaggle kernel private](https://www.kaggle.com/code/khoakp/moviegraph-v2-gpu-feasibility) dùng [dataset private](https://www.kaggle.com/datasets/khoakp/moviegraph-v2-feasibility-input) đã được Project Owner cho phép upload. Dataset chứa **derived pre-2020 cohorts** và mã probe; không chứa `ratings.csv` gốc. Kaggle tự giải nén ZIP, nên runner xác minh SHA-256 **từng file sau giải nén** từ `input_manifest.json`, rồi chạy mỗi cohort trong process riêng. Kernel có hai T4 hiển thị nhưng probe chỉ dùng **`cuda:0`**; VRAM thiết bị đó được báo là **14,56 GiB**, không gộp hai GPU. Runtime: Python 3.13.16, PyTorch 2.11.0+cu130, DuckDB 1.3.2, CPU cgroup limit 30 GiB.

Version 1 chạy 0,5M → 2M → 5M, mỗi mức 5 BPR steps; **cả ba PASS**. Phép đo có CUDA synchronize quanh mỗi pha, peak PyTorch allocated/reserved, process RSS sampled 10 ms, host/cgroup available RAM sampled 10 ms, GPU free trước/sau pha, loss finite và checkpoint bytes. Runner dừng khi lỗi/OOM, GPU effective headroom <30% hoặc CPU available <2 GiB; không có ngưỡng nào bị chạm.

| Cohort | CPU peak RSS | CPU available min | GPU peak reserved | GPU effective headroom | Checkpoint |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0,5M | 1,40 GiB | 28,03 GiB | 162 MiB | 98,0% | 7,28 MiB |
| 2M | 2,10 GiB | 27,17 GiB | 456 MiB | 96,1% | 13,69 MiB |
| **5M** | **3,42 GiB** | **25,80 GiB** | **1.124 MiB** | **91,6%** | **24,74 MiB** |

| Cohort | Load Python graph | Normalize Python | Sampler | COO conversion | Forward | First + 4 BPR steps | Rank 100 users |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0,5M | 2,49 s | 1,41 s | 0,26 s | 0,98 s | 0,16 s | 0,76 s | 2,38 s |
| 2M | 3,64 s | 6,22 s | 1,35 s | 4,01 s | 0,10 s | 1,01 s | 3,32 s |
| **5M** | **8,52 s** | **15,49 s** | **3,22 s** | **10,75 s** | **0,24 s** | **2,00 s** | **3,71 s** |

Nguồn pha và cấu hình: [0,5M](../results/gpu_feasibility_t4/500k.json), [2M](../results/gpu_feasibility_t4/2m.json), [5M](../results/gpu_feasibility_t4/5m.json), [summary](../results/gpu_feasibility_t4/summary.json), [kernel log v1](../results/gpu_feasibility_t4/kernel_v1.log). Graph node/edge hashes khớp [cohort manifest](../results/model_cohort_v2_manifest.json); 36 source Python files trong output khớp per-file SHA của input manifest. Mỗi cohort chỉ có **một run**; không có CI/dispersion. CPU sampling có thể bỏ sót transient dưới 10 ms; GPU free chỉ quan sát ở ranh giới pha, còn peak reserved là bộ nhớ PyTorch quản lý.

Version 2 dùng **cùng graph 5M, 64d, 3 layers, batch 1.024, seed 7** và chạy **500 BPR optimizer steps** (`cuda:0`). Nó PASS: bước đầu 0,75 s, 499 bước còn lại **184,16 s**, tổng optimizer **184,91 s**; end-to-end từ start đến finish khoảng **236 s** gồm graph prep, checkpoint và rank 100 users. Python graph load 10,01 s, normalization 15,60 s, COO conversion 10,54 s; peak RSS **3,43 GiB**, CPU available min **25,98 GiB**, GPU peak reserved **1.124 MiB**, GPU effective headroom **91,6%**. BPR *training loss* ở bước 1/100/300/500 là **0,693/0,387/0,123/0,077**. Đây là loss trên sampled train triples, **không phải Recall/NDCG** hoặc bằng chứng quality trên future labels. 500 × 1.024 = 512.000 sampled triples, khoảng 10,2% số train edges nếu so về lượng mẫu, nhưng sampler có lặp; không phải một epoch đầy đủ. [Trajectory report](../results/gpu_feasibility_t4/trajectory_500.json), [summary](../results/gpu_feasibility_t4/trajectory_summary.json), [kernel log v2](../results/gpu_feasibility_t4/kernel_v2.log), [plan](../results/gpu_feasibility_t4/trajectory_plan.json).

**Workload cố định đề xuất cho EXP-03:** user-recency **4.999.981 positive edges / 90.253 nodes** (49.587 users, 40.666 movies), **64d, 3 layers, BPR batch 1.024, Adam LR 0,001, seed 7**. Quy tắc chọn user chỉ dùng pre-cutoff recency; 5M giữ **97,95%** base positive movie catalog và là mức lớn nhất **đã đo** với headroom lớn trên một T4. Mức 10M và full 13,64M chưa thử, nên không kết luận 5M là tối đa T4. Future positive coverage 11,73% là hậu kiểm mô tả; không dùng để chọn policy theo future labels.

**Ngân sách tính toán mỗi refresh boundary để EXP-03 kiểm chứng:** Stale Model dùng checkpoint/embeddings cũ, **0 forward/0 optimizer step**. Propagation-only dùng **1 full-graph forward, 0 optimizer step**; trên 5M forward riêng lẻ đo **0,10–0,34 s** qua hai runs, nhưng nếu phải đọc và normalize graph mới thì graph setup riêng đã đo khoảng **36 s**. Fine-tuning reference dùng **500 optimizer steps** với giới hạn tài nguyên trên; version 2 đo **184,91 s optimizer** và khoảng **236 s end-to-end** với graph prep, checkpoint và ranking smoke test. Đây là **compute cap cố định**, không phải cam kết quality. EXP-03 phải đánh giá Recall/NDCG trên window tách khỏi bước lựa chọn cohort, giữ graph/checkpoint versions và candidate mask nhất quán; chu kỳ refresh nào mang lại lợi ích quality là câu hỏi thí nghiệm tiếp theo. Không triển khai Refresh Engine trong task này.

## Tái lập và giới hạn kết luận

```powershell
..\venv\Scripts\python.exe -m unittest tests.test_model_cohort tests.test_gpu_feasibility_probe tests.test_prepare_kaggle_feasibility -v
..\venv\Scripts\python.exe -m src.evaluation.model_cohort --source ..\ml-32m\ratings.csv --out results\model_cohort_v2_repro --strategy user_recency --memory-limit 1GB --threads 2
..\venv\Scripts\python.exe -m scripts.gpu_feasibility_probe --cohort results\model_cohort_v2_final\edges_5000000 --out data\processed\gpu_probe_new --device cuda:0 --dim 64 --layers 3 --steps 500
..\venv\Scripts\python.exe -m scripts.prepare_kaggle_feasibility --cohort results\model_cohort_v2_final\edges_500000 --cohort results\model_cohort_v2_final\edges_2000000 --cohort results\model_cohort_v2_final\edges_5000000 --plan results\gpu_feasibility_t4\trajectory_plan.json --out data\processed\kaggle_feasibility_repro --username khoakp
```

Chọn output directory mới cho mọi lần chạy. File Parquet lớn ở `results/model_cohort_v2_final/` và Kaggle bundle local ở `data/processed/` được Git ignore; report/manifest gọn được lưu trong repo. Nếu tái chạy trên Kaggle, dùng Kaggle CLI `datasets version` cho private dataset `khoakp/moviegraph-v2-feasibility-input` khi input files thay đổi, rồi `kernels push` với thư mục `kernel/` vừa sinh. CLI chỉ upload `code_file` của kernel, vì vậy `--plan` nhúng cấu hình 500 bước vào chính `probe_runner.py`; không trông đợi sidecar `probe_plan.json` xuất hiện trong runtime. Submitted version 2 runner SHA-256 là `ae5aff35217d268e48f0576f342b0460d7c4a5c63d3ac1cb7dfc2adf0b2545fa`; lệnh prepare ở trên tái tạo đúng hash này từ cùng source và plan. Không có dataset gốc, sealed REC-01 artifact hay multi-GPU training bị sửa.

Budget được chốt ở **quy mô đã đo**, không ngoại suy lên 10M hoặc full 13,64M edges. 100 users có node ID nhỏ nhất chỉ là smoke evaluation cost; full warm-user ranking, quality Recall/NDCG, độ nhạy theo window và hiệu quả refresh thực tế còn thuộc EXP-03. Kết quả trong tài liệu này đủ để Project Owner review model graph và compute cap trước khi triển khai Model Refresh Engine.

**Tự kiểm tra:** (1) Vì sao 73,4% distinct replay users không bằng 85,0% future positive pairs bị base-cold user loại? (2) Vì sao 5M user-recency hợp lệ về cumulative graph còn hash edge subset không? (3) Vì sao T4 500-step training loss giảm chưa chứng minh Propagation-only/Fine-tuning cải thiện Recall/NDCG?
