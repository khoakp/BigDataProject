# GOAL 04 Phase C3 — Kaggle real-5M preflight

Ngày 24/11/2026. **PASS trên Kaggle, dừng trước C4.** Project Owner đã duyệt upload riêng full canonical Parquet và manifest vào dataset private. Kernel [version 3](https://www.kaggle.com/code/khoakp/moviegraph-v2-c3-real-5m-preflight) dùng hai private datasets, `enable_internet=false`, phát hiện **hai Tesla T4** (15.636.037.632 bytes VRAM mỗi card). Preflight chỉ dùng CPU; **0 optimizer steps, 0 quality evaluations, 0 holdout quality**. Tổng elapsed trong runner là **379,03 giây**; thuộc trần Phase C Kaggle 4 giờ, không phải ngân sách bổ sung.

## Input và provenance

| Input | Rows / bytes | SHA-256 và schema |
| --- | ---: | --- |
| Full canonical `canonical_32000204.parquet` | 32.000.204 / 482.544.363 | `494a1e70e67a2d750297872020b3919da9e557dd68cb631f53d2c5cb9570f73e`; `event_id,event_ts,user_raw_id,movie_raw_id,source_row_id` BIGINT; `rating` FLOAT |
| Canonical `manifest.json` | 1.253 bytes | `be1d5264abcc7ead16964d86238b6a9944c72ffe1a80493f1d57551c5528c4f9`; raw source SHA-256 `91159850e41ee59c86231165a688709647e2726cab2e7ba9faf04001bd5261ee`, transform `d02-v1`, canonical ordering `(event_ts,user_raw_id,movie_raw_id,source_row_id)` |
| Approved 5M `nodes.parquet` | 90.253 / 725.792 | `a2723792903893694cc708b65c85468180064a445dd949e321a24030f7dbb3d7`; `node_id BIGINT,node_type VARCHAR,raw_id BIGINT` |
| Approved 5M `edges.parquet` | 4.999.981 / 33.192.906 | `f697aa05eabab03db6b7bc72980229eeb157a42801c42597d92672df50e857f8`; `user_node_id,movie_node_id,first_event_ts` BIGINT |

Canonical và manifest ở private dataset `khoakp/moviegraph-v2-c3-canonical`; cohort được reuse từ private dataset đã duyệt `khoakp/moviegraph-v2-feasibility-input`. Không upload raw `ratings.csv`, sealed REC-01 bundles hay validation/holdout slice riêng. Full canonical chứa events 2020; protocol đã thống kê 12 validation months với 1.688.159 rating events, nhưng C3 không tính quality. [Input report](../../results/goal04_c3/run_03/goal04_c3/inputs_result.json) ghi mounted paths, schema và hashes; [package manifest](../../results/goal04_c3/run_03/package-manifest.json) ghi Git HEAD `1745739744f6f4aae0d97ddf83d03a7a80eacbb9`, source-file hashes và code-bundle hash. HEAD không bao gồm các thay đổi Phase A/B/C chưa commit; per-file hashes là định danh mã đã chạy.

## Kết quả graph và snapshot

| Gate | G0 tại 01/01/2020 UTC | Sau replay tháng 01, tại 01/02/2020 UTC |
| --- | --- | --- |
| Projection version / nodes / edges | `0` / 90.253 / **4.999.981** | `1` / 90.253 / **5.014.863** |
| Canonical watermark | event ID 27.364.973, timestamp 1.577.836.786 `<` cutoff 1.577.836.800 | event ID 27.490.902, timestamp 1.580.515.140 `<` boundary 1.580.515.200 |
| Monthly delta | — | 125.929 rating events, **14.882** new fixed-cohort edges |
| Fixed mapping SHA-256 | `996848022d92e20ef63b498cfedb38b97545fb006da4c00fef9ed35d40dc90f9` | cùng hash |
| Edge / degree logical hashes | `fb2c7fe7…e5392a` / `ee423151…e4b4` | `c6a90a30…fae83a` / `88080ea8…0a6677` |
| Normalized COO entries | 9.999.962 | 10.029.726 |
| SQL / Python COO | SHA-256 trùng `a5b430be…2b62d`; max abs diff **0** | SHA-256 trùng `b3535811…ea70ac`; max abs diff **0** |
| Snapshot export + reload | **PASS** | **PASS** |

`ModelGraphReplay.create` kiểm exact base projection với approved cohort và chỉ nhận event `< cutoff`; snapshot loader kiểm byte hashes, mapping, edges, degrees, tọa độ COO và trọng số chuẩn hóa. G0 watermark, first-event IDs và cutoff chứng minh không có future edge trong G0. Cả hai snapshots giữ file-byte SHA-256 trong manifests, so Python/SQL theo tolerance `atol=1e-12`; lần chạy này không có rounding difference. [Base result](../../results/goal04_c3/run_03/goal04_c3/base_result.json), [monthly result](../../results/goal04_c3/run_03/goal04_c3/monthly_result.json), [download verification](../../results/goal04_c3/run_03/verification.json). Tám Parquet snapshot đã tải về và mọi hash file khớp manifests; không thay integrity check bằng numerical tolerance.

## Tài nguyên và budget

| Pha | Wall time | Peak process-tree RSS | Min effective RAM còn trống | Peak output disk | Min disk còn trống | Sampled peak DuckDB spill |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Input verify | 12,07 s | 0,129 GiB | 28,715 GiB | ~0 MiB | 19,502 GiB | 0 |
| G0 + snapshot | 196,92 s | 1,689 GiB | 24,558 GiB | 318,46 MiB | 17,701 GiB | 18,03 MiB |
| Monthly + snapshot | 161,35 s | **1,922 GiB** | 25,220 GiB | **606,82 MiB** | 18,910 GiB | 0 |

Runner thấy cgroup RAM limit **30 GiB** và đo effective available bằng min(host, cgroup). CPU RSS/spill/disk được sample mỗi 0,2 giây nên có thể bỏ lỡ transient ngắn. `duckdb_memory()` trong stage results là số tại cuối pha, **không phải peak DuckDB buffer**; sampled spill là file tạm quan sát được. GPU không chạy kernel tính toán. Không ngoại suy các số này thành cost training/validation C4. [Resource report](../../results/goal04_c3/run_03/goal04_c3/report.json) và [Kaggle log](../../results/goal04_c3/run_03/moviegraph-v2-c3-real-5m-preflight.log) là số liệu gốc. Khi hạch toán ngân sách chung 4 giờ, cần tính cả version 1 thất bại (~16 giây theo Kaggle log) và overhead của version 3 (log kết thúc ~388 giây), tức khoảng **404 giây** đã quan sát trên hai kernel, không chỉ 379 giây timer trong runner.

## Workflow và tái lập

`scripts/prepare_kaggle_c3.py` đóng gói đúng các core modules cần thiết vào **kernel code file**; `scripts/kaggle_c3_runner.py` tìm hai dataset trong `/kaggle/input`, kiểm hashes, tạo G0, export/reload, replay và đo tài nguyên; `scripts/verify_kaggle_c3.py` kiểm reports, provenance và byte hashes của snapshots đã tải. Không tạo framework training mới. Source bundle SHA-256 version 3 là `4e3f3b830c0afd37b82770169014af987b5a517f3ed8134917c63ba849a207e1`.

```powershell
# Tái tạo kernel package trong thư mục mới; không tự submit thêm version.
..\venv\Scripts\python.exe -m scripts.prepare_kaggle_c3 --output data/processed/goal04_c3_kernel_repro

# Kiểm artifacts đã tải của version 3.
..\venv\Scripts\python.exe -m scripts.verify_kaggle_c3 `
  --root results/goal04_c3/run_03/goal04_c3 `
  --package results/goal04_c3/run_03 `
  --output results/goal04_c3/run_03/verification_repro.json
```

Version 1 của kernel bị **FAIL ở pha input**, vì runner dự đoán sai mounted path (`/kaggle/input/moviegraph-v2-c3-canonical/...`); Kaggle thật đặt dưới `/kaggle/input/datasets/khoakp/...`. Không có graph/optimizer work trong lần đó. [Failure log](../../results/goal04_c3/run_01/goal04_c3/inputs.log) được giữ nguyên. Runner sau đó tìm file trong mount và vẫn kiểm exact SHA-256. Version 3 PASS. Không có training retry hoặc model-quality selection.

`python -m unittest discover -s tests -v`: **90 PASS, 0 FAIL, 0 SKIP** sau C3. `git diff --check` PASS. Không commit/push/merge.

## Giới hạn và điểm dừng

Preflight chứng minh đường dữ liệu và snapshot trên Kaggle, **chưa** đo training, validation ranking, C0 checkpoint, hoặc 2023 holdout quality. Checkpoint-selection bias stale-G0 vẫn áp dụng cho phase sau. Phase D chưa được phép. **Dừng sau C3, chờ Project Owner nghiệm thu PASS và xác nhận trước C4.**

## Giải thích cho sinh viên mới

Preflight là bài kiểm tra đầu vào và đồ thị trước khi tốn thời gian train. *Canonical* là danh sách ratings được sắp theo thời gian xác định; *fixed mapping* gán mỗi user/phim trong cohort một node ID không đổi. G0 chỉ lấy positive interactions trước 01/01/2020. Normalization gán trọng số cạnh `1/sqrt(degree(user) × degree(movie))`; *snapshot* lưu graph và trọng số đó để nạp lại chính xác.

Ví dụ: user U đã đánh giá phim M ≥4 sao ngày 31/12/2019 thì cạnh U–M thuộc G0. Nếu rating xuất hiện ngày 15/01/2020, cạnh chỉ được thêm khi replay tới boundary 01/02/2020. Khi đọc report, kiểm input hashes trước, rồi counts/mapping, snapshot reload, normalization và cuối cùng RAM/time. `PASS` ở C3 không nói Recall@20 hay NDCG@20 tốt vì chưa xếp hạng phim.

Ba câu tự kiểm tra: (1) Vì sao cạnh ngày 15/01/2020 không được đưa vào G0? (2) Vì sao cùng 4.999.981 cạnh có thể tạo 9.999.962 COO entries? (3) Vì sao phải kiểm file hash của snapshot trước khi dùng numerical tolerance cho trọng số?
