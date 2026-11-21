# GPU feasibility và LightGCN workload selection (v2)

**Trạng thái:** cutoff **01/01/2020 00:00 UTC đã được Project Owner duyệt**. Cohort selection và probe local đã chạy. **Chưa có số đo Kaggle T4**; workload EXP-03 và refresh budget vì vậy là **đề xuất có điều kiện**, chưa chốt. Auto-review đã từ chối upload gói cohort/source lên Kaggle vì chưa có chấp thuận cụ thể cho payload gửi tới dịch vụ bên ngoài. Không có dataset hay notebook Kaggle mới được tạo trong task này.

## Nguồn dữ liệu và định nghĩa cold

Theo [temporal profile](temporal-profile-v2.md) của `ratings.csv` SHA-256 `91159850e41ee59c86231165a688709647e2726cab2e7ba9faf04001bd5261ee`, base `< 1577836800` gồm **27.364.974 ratings**, **13.644.889 positive edges**, **179.192 positive user nodes** và **41.515 positive movie nodes**. Replay `>= 1577836800` gồm **4.635.230 ratings**, **2.293.342 positive pairs**. Ngưỡng `rating >= 4.0` và cumulative historical-positive semantics giữ nguyên.

Con số **73,4%** là **21.529/29.346 distinct users có bất kỳ rating nào trong replay** mà rating đầu tiên của họ ở hoặc sau cutoff. Nó không phải 73,4% của rating rows, positive pairs, hay users ngoài model cohort. Với positive graph, 21.534/29.181 positive-active replay users lần đầu có positive edge sau cutoff (73,8%). Trong **2.293.342 future positive pairs**, **1.948.688 pairs** không có user node trong base positive graph, khoảng 85,0% theo trọng số pair; tập này còn có thể gồm user đã rating trước cutoff nhưng chưa có positive edge. Các mẫu số khác nhau phải giữ riêng.

## Bottleneck audit của pipeline hiện có

| Pha | Code hiện tại | Hệ quả khi tăng graph |
| --- | --- | --- |
| CPU graph state | `src/graph/equivalence.py:incremental_state` dùng `fetchall()` cho registry/edges/degrees. `src/graph/normalize.py:normalize_state` tạo set dedup, Counter, hai Python tuples cho mỗi edge rồi sort `2E` entries. | Peak CPU RAM lớn hơn nhiều so với payload COO `40E` bytes; O(E log E) sort. Đây là gate trước khi vào GPU. |
| Snapshot | `src/graph/snapshot.py:materialize_normalized` tạo state và adjacency Python, lưu COO vào DB; `load_export` fetch toàn bộ lần nữa. `src/models/rec01_train.py:242-255` còn giữ/reference và so sánh nhiều bản graph. | Nhiều bản sao graph/sparse COO đồng thời; snapshot build, export, reload phải đo riêng. |
| GPU tensor | `src/models/lightgcn.py:to_sparse_tensor` dùng `zip(*adjacency.entries)`, tạo `int64` indices và `float32` weights rồi `.coalesce()`. | Các tuple/copy trung gian CPU và GPU có thể vượt xa COO payload 20 bytes/entry. |
| Training | `LightGCN.forward` chạy sparse matrix multiplication trên **toàn graph** ở mỗi layer; `src/models/rec01_train.py:281-291` gọi forward/backward trên toàn graph cho mỗi optimizer step. | Mini-batch BPR không làm chi phí propagation theo graph giảm. Adam cấp phát moment ở step đầu; phải đo first step riêng. |
| Checkpoint/evaluation | `rec01_train.py:_save_checkpoint` lưu parameter **và registry**; `src/evaluation/temporal.py:score_chunks` xếp hạng trên train catalog, `.tolist()` đồng bộ GPU→CPU mỗi chunk. | Checkpoint bytes, warm-user × catalog scoring và validation checkpoint selection phải tách khỏi pure optimizer timing. REC-01 `trajectory_s` đã gồm save và validation, không dùng làm training-step latency. |

REC-01 Kaggle T4 lịch sử chỉ có 14.388 nodes, 392.626 edges; checkpoint ~4,16 MB. Không ngoại suy thời gian REC-01 sang graph v2 có triệu edges.

## Cohort protocol và độ phủ

`src/evaluation/model_cohort.py:prepare_cohorts` đọc positive edges **trước cutoff** để xếp users theo last positive timestamp giảm dần, sau đó degree giảm dần và raw user ID tăng dần; chọn prefix users có tổng edges không vượt budget và giữ **tất cả** positive edges trước cutoff của users đó. Graph này là cumulative historical-positive **đầy đủ trong selected-user cohort**, với node IDs riêng, xác định. Future positive pairs chỉ được đọc **sau khi selection cố định** để đo coverage; không dùng future labels để chọn graph. Việc so sánh policies theo future coverage ở đây là **phân tích khám phá**. Trước EXP-03 phải đóng băng policy/budget và tách một evaluation window chưa dùng cho quyết định quality; các tỷ lệ coverage trên toàn replay không phải ước lượng quality độc lập.

| Cohort | Users | Movies | Positive edges | Candidate catalog / base movies | Evaluable future pairs / 2.293.342 | Evaluable warm users |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| User-recency 0,5M | 1.950 | 24.616 | 499.694 | 59,3% | 85.184 (**3,71%**) | 1.622 |
| **User-recency 2M** | **13.540** | **36.468** | **1.999.995** | **87,8%** | **213.629 (9,32%)** | **5.398** |
| User-recency 5M | 49.587 | 40.666 | 4.999.981 | 98,0% | 268.965 (**11,73%**) | 7.152 |
| Full pre-2020 base | 179.192 | 41.515 | 13.644.889 | 100% | 291.748 (**12,72%**) | 7.557 |

Ở 2M, **2.031.453** future pairs bị loại vì user ngoài model: **1.948.688 true base-cold** và **82.765 pre-cutoff user bị cohort bỏ qua**. Trong phần còn lại, **48.260** pairs có movie ngoài model: **47.556 true base-cold movie** và **704 pre-cutoff movie bị cohort bỏ qua**. Các loại trừ là phân hoạch có thứ tự: xét user trước, rồi movie, rồi train-seen. Dataset này không có future positive pair lặp với train-positive pair, nên `train_seen_repeat_pairs=0`; mask vẫn là yêu cầu giao thức.

[Policy comparison CSV](../results/model_cohort_v2_policy_comparison.csv) so top-movie, top-user, user-recency, chronological prefix và deterministic hash ở cùng budget. Hash edge sampling 2M cho tỷ lệ future-pair cao hơn, nhưng bỏ một phần historical-positive edges giữa selected nodes, nên **không hợp lệ** cho model graph cumulative trong nghiên cứu này. Top-movie 2M chỉ còn 49 candidate movies và 0,58% evaluable pairs. User-recency có bias về users hoạt động gần cutoff, vì vậy báo cáo quality sau này chỉ áp dụng cho cohort đã chọn, không đại diện toàn bộ users.

Các graph Parquet và per-cohort manifest nằm local ở `results/model_cohort_v2_final/` (khoảng 208 MB, được Git ignore để tránh đưa dữ liệu lớn vào repo). Compact [cohort manifest](../results/model_cohort_v2_manifest.json) và policy CSV ghi counts, source hash, cutoff, selection và coverage. Mỗi graph có `nodes.parquet` (`node_id,node_type,raw_id`) và `edges.parquet` (`user_node_id,movie_node_id,first_event_ts`), SHA-256 trong manifest; tất cả edges có timestamp trước cutoff.

## Probe local thực đo, không phải T4

`scripts/gpu_feasibility_probe.py` đo từng pha bằng đồng bộ CUDA, sampled process RSS (chu kỳ 10 ms, có thể bỏ sót transient ngắn), peak allocated/reserved VRAM và checkpoint bytes. Ba run dùng **RTX 4050 Laptop GPU 6 GB**, PyTorch 2.8.0+cu126, `cuda:0`, dimension 64, 3 layers, BPR batch 1.024, Adam LR 0,001, seed 7, **5 steps**, ranking 100 users có node ID nhỏ nhất. Ranking là smoke cost, không đại diện latency trên mọi warm users. Mỗi dòng là **một run**, không có dispersion/CI; thời gian giữa GPU khác nhau không được ngoại suy.

| Cohort | CPU peak RSS | GPU peak reserved | Python normalize | COO conversion | First backward+Adam step | 4 steps tiếp | Rank 100 users | Checkpoint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0,5M | 1,38 GiB | 162 MiB | 0,72 s | 0,92 s | 0,33 s | 0,08 s | 1,37 s | 7,28 MiB |
| 2M | 2,11 GiB | 456 MiB | 2,88 s | 3,35 s | 0,17 s | 0,32 s | 1,90 s | 13,69 MiB |
| 5M | 3,61 GiB | 1.124 MiB | 7,47 s | 10,49 s | 0,48 s | 0,83 s | 2,15 s | 24,74 MiB |

Phase-level run reports: [0,5M](../results/gpu_feasibility_local/500k.json), [2M](../results/gpu_feasibility_local/2m.json), [5M](../results/gpu_feasibility_local/5m.json), [summary.csv](../results/gpu_feasibility_local/summary.csv). Chỉ trường `cohort` trong bản Git được đổi thành đường dẫn tương đối; số đo và hashes giữ nguyên. First-step timing chịu khởi tạo Adam/CUDA nên không diễn giải như steady-state throughput. CPU RSS gồm Python/PyTorch baseline; graph construction làm nó tăng rõ rệt. Không có phép đo T4, không có quality metric, không có full training.

## Một T4 16 GB: protocol cần chạy và quyết định có điều kiện

Kaggle CLI [kernel protocol](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md) hỗ trợ accelerator `NvidiaTeslaT4`, môi trường có thể hiển thị **hai T4 riêng biệt**. Probe chỉ dùng `cuda:0` và ghi tổng/free memory của **thiết bị đó**; không cộng thành 32 GB. `scripts/prepare_kaggle_feasibility.py` tạo private dataset bundle của ba cohort 0,5M → 2M → 5M và private script kernel theo [metadata contract](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md); `scripts/kaggle_feasibility_runner.py` xác minh SHA-256 rồi chạy mỗi cohort trong process riêng.

Mỗi run phải đo: load Parquet → Python normalize → sampler build → sparse tensor/coalesce → forward → first backward+Adam step → bốn steps nữa → checkpoint → rank 100 users. Log Python/PyTorch/CUDA/GPU name, CPU peak RSS, GPU peak allocated/reserved/free, latency từng pha, loss finite, checkpoint size. Dừng progression khi lỗi/OOM hoặc **effective GPU headroom <30%**; cũng xem CPU RAM và thời gian thực tế trước khi tăng graph. T4 run phải dùng **derived pre-2020 cohort** và cùng source/config hashes; không dùng synthetic graph để suy quality coverage.

**Workload EXP-03 đề xuất để T4 xác nhận:** user-recency **2M positive edges / 50.008 nodes**, 64d, 3 layers, batch 1.024, seed 7. Quy tắc chọn user dùng pre-cutoff recency; budget 2M giữ 87,8% base positive movie catalog trong khi local CPU/GPU probe vẫn chạy được. Hậu kiểm cho thấy nó bao phủ 9,32% future positives; 5M tăng thêm 2,41 điểm phần trăm nhưng cần 2,5× edges và local CPU normalization 2,6×. Các future percentages chỉ là mô tả, không dùng để tối ưu policy hoặc kết luận quality. Candidate refresh budget: Stale = 0 propagation; Propagation-only = **1 full forward** mỗi selected monthly snapshot; fine-tuning reference chỉ xem xét sau khi T4 đo 5-step probe và chốt số steps cố định theo thời gian/VRAM. Chưa chốt training trajectory hay refresh interval vì chưa có T4 timing và quality evidence.

## Tái lập và điểm cần Project Owner cho phép

```powershell
..\venv\Scripts\python.exe -m unittest tests.test_model_cohort tests.test_gpu_feasibility_probe tests.test_prepare_kaggle_feasibility -v
..\venv\Scripts\python.exe -m src.evaluation.model_cohort --source ..\ml-32m\ratings.csv --out results\model_cohort_v2_repro --strategy user_recency --memory-limit 1GB --threads 2
..\venv\Scripts\python.exe -m scripts.gpu_feasibility_probe --cohort results\model_cohort_v2_final\edges_2000000 --out data\processed\gpu_probe_new --device cuda:0 --dim 64 --layers 3 --steps 5
```

Chọn output directory mới cho mọi lần chạy. REC-01 sealed source/artifacts không bị sửa. Gói Kaggle đã chuẩn bị local ở `data/processed/kaggle_feasibility_v2/` (private dataset gồm derived cohort Parquet + source Python cần cho probe, private notebook/script dùng `cuda:0`). **Auto-review đã từ chối upload** vì quyền gửi payload cụ thể này tới Kaggle chưa rõ. Cần Project Owner phê duyệt rõ việc upload **private** gói đó vào tài khoản Kaggle `khoakp` và chạy private T4 probe; sau đó mới có thể chốt workload/refresh budget từ T4 evidence.

**Tự kiểm tra:** (1) Vì sao 73,4% distinct replay users không bằng 85,0% future positive pairs bị base-cold user loại? (2) Vì sao 2M user-recency hợp lệ về cumulative graph còn 2M hash edge subset không? (3) Vì sao local 456 MiB VRAM ở 2M chưa đủ để khẳng định T4 training trajectory khả thi?
