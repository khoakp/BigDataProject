# 07 — Sổ thực nghiệm, protocol và kết quả benchmark

> **Trạng thái:** PRE-REGISTERED PROTOCOL / CHƯA CÓ BENCHMARK RESULTS · Cập nhật 10/11/2026 (MovieLens + metadata).  
> Đây là nơi ghi **kết quả đo thực** và failures. Không điền số hoặc đồ thị suy đoán.  
> Nguồn số liệu: `results/raw/*.csv` và `results/logs/*`; plots sinh từ CSV, không vẽ thủ công.

**Kiểm tra dữ liệu đầu vào, không phải benchmark (19/11/2026):** MD5 của `../ml-32m/ratings.csv` và `movies.csv` khớp `checksums.txt` đi kèm. Lệnh, kích thước và hash được ghi tại [09, E-003](09-report-evidence.md). Chưa profile dữ liệu hay chạy EXP-01/02/03; trạng thái protocol bên dưới không thay đổi.

**D01 feasibility (19/11/2026):** CPU autograd smoke PASS; `torch 2.8.0+cpu` không có CUDA, nên CUDA smoke `UNAVAILABLE`, không phải GPU benchmark. Máy nhận RTX 4050 qua `nvidia-smi`. Cấu hình và lệnh ở [environment.md](environment.md).

**D02 data profile (19/11/2026, không phải EXP-01/02):** profile và deterministic replay kiểm chứng trên MovieLens ratings; 32,000,204 rows, 0 null ở bốn cột, 0 duplicate user–movie pairs, 6,023,681 rows vượt số timestamp phân biệt; hai prefix 100k/1M có hash giống nhau qua hai runs. Raw profile, hash, config và lệnh ở [d02-data-profile.md](d02-data-profile.md). Chưa đo hiệu năng Full/Incremental.

**D03 graph correctness (19/11/2026, không phải speed benchmark):** full reference trên 100k canonical events tạo 57.925 positive edges; degree sum mỗi phía bằng 57.925. Toy dedup test và 100k state validation PASS; [d03-full-reference.md](d03-full-reference.md) có input/output hashes, phiên bản phần mềm, code và lệnh. EXP-01 vẫn CHƯA CHẠY vì chưa có Incremental path/equivalence.

## 1. Quy định áp dụng chung

**D06 normalization/version correctness (19/11/2026):** run `D06-20261119T163958346261Z` PASS, 100k prefix / 10 batches; Full SQL reference tại v1/v5/v10 có cùng normalized shape/coordinates và max_abs_diff=0 (float64 atol 1e-12). Stale snapshots, export/reopen và metadata isolation PASS; unit suite 15 tests PASS. Full catalog validation: 87.585 movie IDs unique/non-null. Raw report ở `data/processed/d06_probe/report.json`; setup/hashes và run FAIL ban đầu được ghi ở [D06](d06-normalization.md). Chưa chạy EXP-02 cost/freshness hoặc LightGCN.

**D05 exact equivalence (19/11/2026):** run `D05-equivalence` PASS trên toy và prefix 100k, mỗi dataset 4 segmentation. Registry/edge/degree bằng nhau tại 35 boundary checks; 70 immediate/historical replays giữ nguyên snapshot. Unit suite 7 tests PASS sau khi sửa Full Rebuild empty-positive SUM; raw report ở `data/processed/d05_probe/report.json`, input/code hashes và setup ở [D05](d05-equivalence.md). G1 PASS trong phạm vi đã kiểm tra; EXP-01/02 vẫn chưa chạy.

**D04 correctness (19/11/2026):** run `D04-100k-10k` đạt với 57.925 edges, 2.819 nodes; degree sum mỗi phía 57.925; 10/10 batch replay NO_OP; stable IDs, degree recount và reopen PASS. Unit suite 5 tests PASS gồm hash conflict và rollback injection. Cấu hình, lệnh, state fingerprint, lỗi ban đầu và giới hạn ở [d04-incremental.md](d04-incremental.md); report thô tại `data/processed/d04_probe/report.json`. Đây chưa phải EXP-01 hay kết quả Full/Incremental equivalence D05.

**Dataset:** MovieLens 32M là nguồn chính. `ratings.csv` tạo graph/benchmark; `movies.csv` là metadata dùng hậu xử lý. Bản ghi EXP-01/02 không được bao gồm metadata JOIN trong thời gian đo graph core, trừ khi tạo một phép đo enrichment riêng có nhãn.

**Mỗi run phải lưu:** `run_id`, `experiment`, `timestamp`, `dataset_id`, `source_hash`, `history_events`, `delta_events`, `canonical_version`, `model_graph_version`, `git_commit`, `config_hash`, `seed`, `python_version`, `torch_version`, `cuda_runtime`, `cpu`, `RAM`, `GPU`, `OS`, `status`, `failure_reason`.

**CPU benchmarks:** cùng máy, cùng output contract, cùng data source, 1 warm-up + 5 measured repetitions **nếu có thời gian**; mỗi repetition bằng process mới; alternation/randomization Full vs Incremental; công bố median và spread. `warm-cache` benchmark nếu không có cách reset OS page cache có thể kiểm chứng; không gắn nhãn cold-cache sai.

**Timing:** `time.perf_counter()` around same semantics; `time.process_time()` supplemental; peak RSS từ sampler `psutil` ghi sampling interval (peak RSS sample ≠ tuyệt đối); bytes đọc/ghi đo application payload + OS I/O nếu có, không tráo hai loại. Tách cost one-time ingest khỏi per-batch update nếu câu hỏi là update cost.

**GPU benchmarks:** 1 warm-up + measured runs, `torch.cuda.synchronize()` trước/sau timed section, reset peak stats, log both `max_memory_allocated()` và `max_memory_reserved()`; separate startup/training/propagation/evaluation. OOM là row `status=OOM`, không âm thầm xóa.

## 2. Hardware feasibility gate — sớm nhất có thể (D01–D07)

### Checklist hệ thống

```bash
nvidia-smi
python --version
python -c "import torch; print('torch', torch.__version__, 'cuda build', torch.version.cuda, 'available', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

**Windows PowerShell — ghi CPU/RAM:**

```powershell
Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors
Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory
Get-PSDrive -PSProvider FileSystem
```

**Tiny CUDA peak check (chưa phải LightGCN benchmark):**

```python
import torch
assert torch.cuda.is_available(), 'CUDA not available'
torch.cuda.reset_peak_memory_stats()
x = torch.randn(2048, 64, device='cuda', requires_grad=True)
y = x.square().mean(); y.backward()
torch.cuda.synchronize()
print('allocated_MB', torch.cuda.max_memory_allocated()/2**20)
print('reserved_MB', torch.cuda.max_memory_reserved()/2**20)
```

### Scale ladder (chỉ tiến khi mốc trước chạy ổn)

| Size / thử nghiệm | CUDA forward | Backward | Short train | Peak allocated | Peak reserved | Kết luận |
|---|---|---|---|---|---|---|
| Toy | PASS | PASS | 4 bước | 10,20 MiB | 24 MiB | Dense reference đạt |
| 100k | PASS | PASS | 4 bước | 23,65 MiB | 34 MiB | Snapshot D06 v10 |
| 1M | PASS | PASS | 4 bước | 126,26 MiB | 148 MiB | Full Rebuild + normalization D06 |
| 5M | CHƯA CHẠY | — | — | — | — | — |
| 10M | CHƯA CHẠY | — | — | — | — | — |
| 20M | CHƯA CHẠY | — | — | — | — | — |
| 32M | **STRETCH ONLY** | — | — | — | — | Không là tiêu chí hoàn thành |

**Lưu ý:** 1M rating events != 1M positive graph edges; phải ghi cả nodes, edges, embedding dim/layers, graph representation. Chọn model scale có **headroom**, không chọn size chạy đúng một lần sát ngưỡng OOM.

Run `D07-20261119T172548657263Z` ngày 20/11/2026 dùng RTX 4050, PyTorch 2.8.0+cu126, COO float32, 64 chiều, 3 lớp và 2.048 bộ ba mỗi bước. Có một bước warm-up và ba bước đo. Setup, CUDA timings, memory theo phase và T-09 có trong [hồ sơ D07](d07-lightgcn-smoke.md). EXP-01/02/03 vẫn chưa chạy.

## 3. EXP-01 — Full Rebuild vs Stateful Incremental Update

**RQ1:** thời gian/RAM/I/O của hai phương án khi history `H` và delta `D` thay đổi, outputs tương đương?

- **Baseline:** Full Rebuild đọc lại lịch sử đến `H+D`, tạo canonical state.
- **Treatment:** Incremental apply `D` lên snapshot chứa `H` events; state prebuild cost ghi riêng (không gian lận bằng giấu prebuild).
- **Grid pilot kế hoạch 21 ngày:** `H ∈ {100k,500k,1M}` và `D ∈ {1k,10k}`; `H=5M` hoặc lớn hơn chỉ khi đúng/đủ RAM/thời gian. Không chọn `H+D > total events`.
- **Control:** cùng positive rule, dedup semantics, node mapping, CPU, storage state/output, warm-cache policy, thread count.
- **Correctness gates:** edge set, stable IDs, degree arrays, checksums, counts; chỉ tính speedup nếu pass.
- **Metrics:** wall seconds, process CPU seconds, sampled peak RSS bytes, application bytes read/written, events/sec, inserted positive edges, copy/export cost.

**Schema raw CSV đề xuất:**

```csv
run_id,history_events,delta_events,method,wall_s,cpu_s,peak_rss_bytes,app_bytes_read,app_bytes_written,events_per_s,positive_edges_inserted,edge_checksum,degree_checksum,correctness_pass,status,git_commit,config_hash
```

**Kết quả:** CHƯA CHẠY — `results/raw/exp01_full_vs_incremental.csv` chưa tồn tại. Grid rút gọn 21 ngày: H=100k/500k/1M, D=1k/10k; chỉ thử 5M+ khi correctness/resource budget pass.

**Đồ thị phải vẽ:** latency vs H (fix D), peak RSS vs H, speedup vs D/H, bytes written, crossover conditions. Báo cả trường hợp incremental chậm hơn.

## 4. EXP-02 — Eager vs Deferred Normalization

**RQ2:** khác biệt chi phí tích lũy khi materialize sau mỗi `R` batch (R=1/5/20) trên **cùng** replay batch sequence?

- **Baseline:** `R=1` normalize/model-ready snapshot sau mỗi batch.
- **Treatments:** `R=5` là P1; `R=20` stretch; chỉ materialize ở refresh boundary, canonical state vẫn cập nhật sau mọi batch.
- **Cost:** canonical update seconds, normalization seconds, cumulative pipeline seconds, bytes output, peak RSS, number of materializations.
- **Freshness:** staleness in batches, event counts, event-time lag if meaningful; lag có thể khác wall-clock lag.
- **Correctness:** tại `v` là common refresh boundary, normalized sparse matrices phải cùng shape/index ordering và values gần bằng theo tolerance float; ví dụ `max_abs_diff <= 1e-6` **phải xác nhận phù hợp dtype và phép tính trong W3**.
- **Cảnh báo:** nếu normalization on-the-fly rẻ và không materialize full weights, dự án phải nêu đúng dạng chi phí đã benchmark; so sánh một output mỗi batch với output mỗi 20 batch đương nhiên khác cadence/semantics, không tuyên bố công bằng về freshness nếu thiếu staleness chart.

```csv
run_id,refresh_every_batches,canonical_version,model_graph_version,canonical_update_s,normalization_s,cumulative_s,app_bytes_written,peak_rss_bytes,staleness_batches,staleness_events,normalized_equivalence_pass,status,git_commit,config_hash
```

**Kết quả:** CHƯA CHẠY — `results/raw/exp02_normalization.csv` chưa tồn tại. Ưu tiên `R=1` và `R=5`; `R=20` stretch.

**Đồ thị:** cumulative cost vs batch version, cost vs R, cost vs staleness, normalized snapshot equality summary.

## 5. EXP-03 — Freshness vs refresh cost/quality (**stretch trong kế hoạch 21 ngày**)

**RQ3:** trên warm cohort, stale vs propagation-only vs periodic trained reference khác nhau về latency/VRAM và Recall/NDCG?

**Ground truth protocol:** cố định cutoff(s) và future evaluation window(s), chỉ graph tương tác `<= t` được sử dụng tại cutoff `t`; tập positive tương lai `(t,t+h]` chỉ làm labels. Ghi warm-user/item coverage và candidate universe, mask train positives.

| Regime | Base embeddings | Propagation graph | Gradient retraining | Ghi chú |
|---|---|---|---|---|
| `stale` | checkpoint cũ | snapshot cũ | Không | re-use embeddings cũ |
| `prop_only` | **frozen** checkpoint cũ | snapshot canonical mới được normalize | Không | warm nodes only |
| `train_ref` | checkpoint được cập nhật với dữ liệu đến t | graph mới | Có, fixed budget | expensive periodic reference |

**Baselines tĩnh:** Popularity / BPR-MF / LightGCN. Lưu `seed`, epochs, embedding_dim, layers, LR, graph version, metric-specific masking.

**GPU metrics:** timed propagation/training, peak allocated/reserved, model checkpoint bytes; quality Recall@10, Recall@20, NDCG@10, NDCG@20, optionally 3 seeds nếu budget có.

```csv
run_id,cutoff,seed,regime,model_graph_version,checkpoint_version,warm_users,warm_items,refresh_s,peak_allocated_bytes,peak_reserved_bytes,recall_10,recall_20,ndcg_10,ndcg_20,status,git_commit,config_hash
```

**Kết quả:** CHƯA CHẠY — `results/raw/exp03_refresh.csv` chưa tồn tại.

**Đồ thị:** Recall@20 vs refresh seconds; NDCG@20 vs lag; peak VRAM by regime; freshness quality trade-off.

## 5A. AN-01 — Metadata Enrichment & Genre Profiling (phân tích mô tả, không phải experiment causal)

- **Mục đích:** làm đầu ra Top-K dễ hiểu bằng `title`/`genres`, kiểm tra metadata coverage và mô tả phân phối thể loại; không đưa metadata vào input graph/model.
- **Nguồn:** MovieLens 32M `movies.csv`, LEFT JOIN theo `movieId` với Top-K `movie_raw_id` sau ranking.
- **Kiểm tra bắt buộc:** `movieId` non-null/unique; count Top-K rows trước = sau JOIN; user/item/rank/score version giữ nguyên; báo `metadata_found` coverage; Recall/NDCG tính trên raw movie IDs, trước JOIN.
- **Analysis candidates:** unique genres và số/lượt xuất hiện genres trong Top-K, tỷ lệ có `(no genres listed)`, catalog-vs-recommendation genre distribution. Genre là **đa nhãn**, một phim có thể đóng góp cho nhiều nhãn; nêu mẫu số rõ trước khi vẽ.
- **Không kết luận:** LightGCN học trực tiếp thể loại, tính giải thích có quan hệ nhân quả, hoặc genres cải thiện chất lượng gợi ý.
- **Provenance:** `ratings_source_hash`, `movies_source_hash`, code/config version, `model_graph_version`, `checkpoint_version`, `topk_run_id`, cutoff.
- **Đầu ra dự kiến:** `results/raw/an01_metadata_coverage.csv`, `results/raw/an01_genre_counts.csv`, `results/figures/genre_distribution.png`; trạng thái **CHƯA THỰC HIỆN**.

## 6. Experiment entry template — chỉ điền khi thực sự chạy

```markdown
### RUN-YYYYMMDD-XXX — EXP-0X
- Câu hỏi / giả thuyết:
- Start/end date:
- Dataset/version/source checksum:
- CPU/RAM/GPU/OS:
- Graph state version / model graph version:
- Config / seed / git commit:
- Input sizes, batch sizes:
- Baseline / treatment:
- Các metrics đã đo:
- Log/raw CSV path:
- Correctness PASS/FAIL và test report:
- Failure/OOM/outlier:
- Quan sát:
- Điều KHÔNG thể kết luận:
- Quyết định hoặc next run:
```

## 7. Miniature *test cases* phải có trước khi benchmark

1. Toy graph 2 users–3 items, thêm edge làm thay đổi normalized weights của edge cũ.
2. Replay cùng batch hai lần không tăng degrees.
3. Có hai identical positive events nhưng chỉ một graph edge.
4. New user/new movie không làm thay đổi stable node IDs đã có.
5. Random batch segmentation vs Full rebuild ra cùng canonical graph.
6. Temporal split không leak future nodes/edges/degrees vào train snapshot.
7. Chunked top-K trùng full-scoring trên toy dataset.
8. Restart/checkpoint và direct replay cùng version cho ra cùng hashes.
9. `movies.csv` movieId uniqueness; LEFT JOIN không thay Top-K ranks/scores/count.
10. Ranking metrics không thay dù metadata missing hoặc genres đổi.

## 8. Cách báo cáo kết quả trung thực

- Không điền các bảng trên bằng ví dụ đo giả.
- Không dùng số milliseconds từ phản biện AI như máy bạn đã benchmark.
- Báo trung vị, n, dispersion và failures; không chỉ best run.
- Phân biệt cost of canonical ingestion, state initialization, incremental update, normalized materialization, GPU propagation và parameter training.
- Khi experimental result không hỗ trợ giả thuyết, giữ kết luận âm tính và giải thích giới hạn.
