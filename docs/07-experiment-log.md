# 07 — Sổ thực nghiệm, protocol và kết quả benchmark

**S01/REC-01 (21/11/2026):** [giao thức Top-20 + Popularity](s01-evaluation-spec.md) đã được Project Owner duyệt và dùng trong [Kaggle run S03–S04](s04-rec01-results.md). Không đổi nhãn D08/D10 Top-10 thành REC-01. D11–D12 EXP-01 và D13 EXP-02 giữ đúng tên và số liệu lịch sử.

**S03–S04 REC-01 (21/11/2026):** [Kaggle Notebook Version 3](https://www.kaggle.com/code/khoakp/rec01-movielens-1m-gpu-validation?scriptVersionId=372012165), run `REC01-Kaggle-1M`, Tesla T4, source canonical 1M SHA-256 `f17048bd7b12bea613387c169c65db6ce67442aa7341314f5e18fba39efcfbae`, config SHA-256 `8b8223681cf181d0ae9bd2b794d0e0f8159b3c01443622fabffff27820b8b8e8`. `seal-output PASS`, local `verify-output PASS`. Checkpoint bước 300 chọn theo validation NDCG@20 0,083543; test LightGCN Recall/NDCG@20 0,128000/0,085161 và Popularity 0,128099/0,086897 trên cùng 403 warm users, 2.069 warm pairs, 773 train movies. Popularity nhỉnh hơn; không có kiểm định thống kê. Raw `data/processed/rec01_kaggle_runs/version_3/rec01_project/REC01-Kaggle-1M/rec01_quality.csv`, report/manifest cùng thư mục, ZIP SHA-256 `25b3f97606aa330fc5892195e7afea7b825f48818c45b31b8b947d3f6172fe49`. [Bảng đầy đủ, exclusions và giới hạn](s04-rec01-results.md).

**S06 EXP-01 audit (21/11/2026):** tái kiểm tra raw D12 `D12-repeat-grid` (72 rows/36 cặp) và `D12-scale-2m` (24 rows/12 cặp): CSV/report provenance, pairing, throughput denominator, summary, logical hashes, paired Parquet manifests, từng SHA-256 file và export bytes đều PASS. Không chạy benchmark mới hoặc sửa raw. [Hồ sơ audit và hash CSV](s06-exp01-audit.md). D11 vẫn chỉ là pilot.

**S07 EXP-01 synthesis (21/11/2026):** [kết luận Engineering](s07-exp01-engineering-conclusions.md) dùng 8 cases đã audit. Core H≤1M: Incremental 0/5 wins ở mỗi case; run 2M riêng tại Δ=1k có ratio median 1,156 và 5/5 wins, Δ=10k ratio 0,999 và 2/5. Tái sinh summary SHA-256 `590ff3ec77cf2621de2f9b415c7d4fa06b9b75e695346b5dc0d1d76dd35ac973` và 5 PNG trùng byte; không có phép đo mới hoặc ngưỡng crossover chính xác.

**S10 integration, không phải benchmark/REC-02 (21/11/2026):** test trên sealed run `REC01-Kaggle-1M` với bản sao train graph riêng. Hai feedback events (ratings 5,0 và 2,5) tăng graph version 81→82, positive edges 392.626→392.627; `model_graph_version=81`, checkpoint step 300 và Top-20 giữ nguyên. Replay cùng UUID `NO_OP`; phục hồi Parquet từ intent sau giả lập gián đoạn. AppTest xác nhận version hiển thị; 57 tests PASS. [S10 setup, code, giới hạn](s10-rating-graph-refresh.md). Đây là kiểm tra correctness, không phải thời gian xử lý, refresh quality hay phép đo trên dữ liệu 32M.

**S11 1M+20 integration (21/11/2026):** run `rec01_1m_batch20_v1` từ sealed REC-01 Version 3, source SHA-256 `f17048bd7b12bea613387c169c65db6ce67442aa7341314f5e18fba39efcfbae`; combined train+feedback SHA-256 `eb8d2014a76e7acf2ddb135a82d8c4b124e50467a12da6e31cac88657067de70`. Một batch 20 ratings tăng edges 392.626→392.636, graph 81→82; Full Rebuild exact registry/edges/degrees 14.388/392.636/14.388, replay `NO_OP`, Top-20 không đổi. Diagnostic một lượt: load+copy 2,621 s; apply 0,183 s; full reference+compare 0,994 s. Không dùng làm benchmark speedup. Raw `data/processed/s11_integration/rec01_1m_batch20_v1/report.json`; [method và failure paths](s11-integration-scaling.md). 60 local tests PASS.

**S02 validation-only smoke (21/11/2026):** dùng checkpoint D08 100 bước cũ trên 1M, so Top-20 LightGCN/Popularity cùng 482 warm users và 773 train movie candidates. Valid Recall@20 0,11949 vs 0,12070; NDCG@20 0,08200 vs 0,08534. Chỉ validation, không chọn checkpoint hoặc tính test; [log và giới hạn](s02-validation-smoke.md). Không phải REC-01 final result.

**D13 EXP-02 (21/11/2026):** run `D13-100k-r1-r5-final` dùng cùng 10 batch × 10k events cho R=1/R=5; 80/80 rows PASS, 8/8 common-boundary comparisons v5/v10 PASS (`max_abs_diff=0`, float64 `atol=1e-12`). Ba measured repetitions cho median cumulative apply+materialize+export 5,283 s vs 1,648 s; R=5 cũ tối đa 4 batch/40k events, trước v5 dùng graph rỗng. Raw `results/raw/exp02_normalization.csv`; [phương pháp, kết quả và giới hạn](d13-exp02-results.md).

**D12 EXP-01 repetitions (20/11/2026):** core run `D12-repeat-grid` có 72/72 rows PASS trên sáu cặp H/Δ, mỗi cặp một warm-up và năm measured repetitions/method; run H=2M riêng `D12-scale-2m` có 24/24 PASS. Raw `results/raw/exp01_full_vs_incremental.csv`, `results/raw/exp01_scale2m_d12.csv`; summary và biểu đồ sinh từ raw ở [D12](d12-exp01-results.md). Δ=1k tại H=2M có paired Full/Inc ratio median 1,156, 5/5 wins; Δ=10k gần hòa. Cache không kiểm soát, không suy rộng ngưỡng crossover chính xác.

**D11 EXP-01 pilot (20/11/2026):** run `D11-pilot-grid` trên H={100k,500k,1M}, Δ={1k,10k}, một warm-up và một measured process/method/cặp; 24/24 rows exact correctness PASS. Raw `results/raw/exp01_pilot_d11.csv` có wall/CPU/sampled RSS/app bytes/OS I/O/prebuild cost, provenance và cache policy. [D11](d11-exp01-pilot.md) giữ bảng pilot và giới hạn. EXP-01 repetitions/spread và speedup chưa thực hiện.

**D10 clean MVP rerun (20/11/2026):** run `D10-clean-100k` từ raw `ratings.csv` PASS trên CUDA; 100k canonical events, train graph version 9 với 2.562 nodes/52.014 edges, Full equivalence PASS, 300/300 enriched rows. Valid Recall@10/NDCG@10 0,03491/0,06418 trên 20 warm users. Raw manifest và 12 artifact hashes ở `data/processed/d10_mvp/D10-clean-100k/`; [D10](d10-mvp.md) ghi môi trường, source hashes, code hashes và giới hạn. Thời gian run ~56,5 giây không phải EXP-01 benchmark.

**D09 E2E integration (20/11/2026):** CUDA runs 100k/1M PASS với D04 incremental graph, exact Full state equivalence và metadata enrichment sau raw ranking. 1M có 81 train batches, 8.850 raw/enriched Top-K rows, thiếu metadata 0, ranking/metrics giữ nguyên. Raw reports/checkpoint ở `data/processed/d09_incremental_1m/`; [D09](d09-end-to-end.md) ghi input hashes/config/giới hạn. Không phải EXP-01/02 cost benchmark.

**D08 temporal baseline (20/11/2026):** CUDA train/eval 100k và 1M PASS, 100 bước BPR với seed 7, 64 chiều, 3 layers, Top-10 trên warm cohort. 1M valid/test Recall@10 0,06390/0,07154; NDCG@10 0,06567/0,07092. Raw report/checkpoint ở `data/processed/d08_baseline_1m/`; cấu hình, split, exclusions và giới hạn ở [D08](d08-temporal-baseline.md). Đây không phải EXP-01/02/03.

> **Trạng thái:** EXP-01 repetitions và EXP-02 R=1/R=5 đã chạy · Cập nhật 21/11/2026.
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

**Kết quả:** D11 pilot và D12 năm repetitions đã chạy. Core raw CSV `results/raw/exp01_full_vs_incremental.csv`, scale 2M riêng `results/raw/exp01_scale2m_d12.csv`; 96/96 rows correctness PASS. Ở Δ=1k, run H=2M cho thấy Incremental nhanh hơn tại điểm đo này; Δ=10k gần hòa. [Phương pháp, số liệu và giới hạn](d12-exp01-results.md).

**Đồ thị phải vẽ:** latency vs H (fix D), peak RSS vs H, speedup vs D/H, bytes written, crossover conditions. Báo cả trường hợp incremental chậm hơn.

## 4. EXP-02 — Eager vs Deferred Normalization

**RQ2:** khác biệt chi phí tích lũy khi materialize sau mỗi `R` batch (R=1/5/20) trên **cùng** replay batch sequence?

- **Baseline:** `R=1` normalize/model-ready snapshot sau mỗi batch.
- **Treatments:** `R=5` là P1; `R=20` stretch; chỉ materialize ở refresh boundary, canonical state vẫn cập nhật sau mọi batch.
- **Cost:** canonical update seconds, normalization seconds, cumulative pipeline seconds, bytes output, peak RSS, number of materializations.
- **Freshness:** staleness in batches, event counts, event-time lag if meaningful; lag có thể khác wall-clock lag.
- **Correctness:** tại `v` là common refresh boundary, normalized sparse matrices phải cùng shape/index ordering và values gần bằng theo tolerance float. D06 đã chốt float64 `atol=1e-12`; D13 còn kiểm COO hash exact.
- **Cảnh báo:** nếu normalization on-the-fly rẻ và không materialize full weights, dự án phải nêu đúng dạng chi phí đã benchmark; so sánh một output mỗi batch với output mỗi 20 batch đương nhiên khác cadence/semantics, không tuyên bố công bằng về freshness nếu thiếu staleness chart.

```csv
run_id,refresh_every_batches,canonical_version,model_graph_version,canonical_update_s,normalization_s,cumulative_s,app_bytes_written,peak_rss_bytes,staleness_batches,staleness_events,normalized_equivalence_pass,status,git_commit,config_hash
```

**Kết quả:** D13 đã chạy R=1/R=5 trên prefix 100k: 80/80 rows PASS, 8/8 equality checks ở v5/v10 PASS. Median cumulative cost 5,283 s vs 1,648 s, staleness tối đa 0 vs 4 batches. [D13](d13-exp02-results.md), raw `results/raw/exp02_normalization.csv`. `R=20` chưa chạy (stretch).

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
