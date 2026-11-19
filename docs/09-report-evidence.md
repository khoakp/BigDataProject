# 09 — Ma trận bằng chứng cho báo cáo

> Kiểm kê ngày 19/11/2026. Mỗi dòng phân biệt **quyết định/thiết kế**, **quan sát đã xác minh** và **kết quả thực nghiệm**. Đường dẫn `../ml-32m` tính từ root repo `BigDataProject`.

| ID | Nhận định có thể dùng | Loại và trạng thái | Bằng chứng trực tiếp | Giới hạn |
|---|---|---|---|---|
| E-001 | Mục tiêu là cập nhật graph theo batch, đối chiếu Full/Incremental rồi dùng snapshot cho LightGCN | Thiết kế đã chọn | [01, mục 3–6](01-project-overview.md), [06, DEC-001/003/009](06-decisions-and-risks.md) | Chưa có pipeline chạy |
| E-002 | Canonical replay theo event time; metadata `movies.csv` chỉ hậu xử lý Top-K | Quyết định thiết kế | [03, mục 2/8](03-system-design.md), [06, DEC-004/012](06-decisions-and-risks.md) | Chưa có test ordering/join |
| E-003 | Hai file `ratings.csv`, `movies.csv` cục bộ có MD5 trùng manifest đi kèm | Quan sát đã xác minh ngày 19/11/2026 | `../ml-32m/checksums.txt`, `../ml-32m/README.txt`; `Get-FileHash -Algorithm MD5` trên hai file | Xác nhận byte integrity so với manifest cục bộ, không xác nhận schema toàn file hoặc nguồn tải độc lập |
| E-004 | Header hai file tương ứng `userId,movieId,rating,timestamp` và `movieId,title,genres` | Quan sát đã xác minh ngày 19/11/2026 | Dòng đầu của `../ml-32m/ratings.csv` và `../ml-32m/movies.csv` | Chưa profile data rows |
| E-005 | README đi kèm công bố 32.000.204 ratings và 87.585 movies | Thông tin từ nhà phát hành, đã đọc | `../ml-32m/README.txt`, phần Summary | Chưa đếm độc lập; không gọi là số đo của dự án |
| E-006 | Ngưỡng `rating >= 4.0` | Đã chọn cho D03, chưa chứng minh tối ưu model | [06, DEC-005](06-decisions-and-risks.md), [D03 profile](d03-full-reference.md) | Thay ngưỡng đòi chạy lại graph/benchmark |
| E-007 | Công thức weight `1/sqrt(d_u d_i)` và invariant tổng bậc | Cơ sở toán học của thiết kế, implementation kiểm chứng ở D03–D06 | [03, mục 4.2/6](03-system-design.md), [D06](d06-normalization.md) | Chưa kiểm chứng framework normalization D07 |
| E-008 | EXP-01/02 có protocol và metrics | Kế hoạch đã lập | [07, mục 1/3/4](07-experiment-log.md) | Chưa có raw CSV/run |
| E-009 | D01 đã kiểm tra môi trường, source manifest và CPU autograd | Quan sát đã xác minh ngày 19/11/2026 | [environment.md](environment.md), [source-manifest.md](source-manifest.md), [05](05-work-log.md) | PyTorch CUDA unavailable; không phải GPU model smoke hoặc benchmark |
| E-010 | D02 profile 32,000,204 rows và tạo canonical prefix 100k/1M xác định trong cùng cấu hình | Kết quả đã chạy 19/11/2026 | [d02-data-profile.md](d02-data-profile.md), [code](../src/ingest/d02_profile.py), [toy test](../tests/test_d02.py), `data/processed/canonical_events/manifest.json` | Chưa có full canonical 32M Parquet; chưa kiểm chứng hash xuyên phiên bản DuckDB |
| E-011 | D03 chọn rule `rating >= 4.0` và Full Rebuild reference trên toy/100k thỏa invariant | Quyết định + kết quả đã chạy 19/11/2026 | [D03 run](d03-full-reference.md), [code](../src/graph/full_builder.py), [test](../tests/test_full_builder.py), `data/processed/full_reference_100000/manifest.json` | Chưa có Incremental equivalence, performance hoặc chất lượng model |
| E-012 | D04 replay idempotent, hash conflict bị reject, degree từ cạnh mới và rollback | Đã chạy 19/11/2026 | [D04](d04-incremental.md), [implementation](../src/graph/incremental.py), [tests](../tests/test_incremental.py), [probe](../src/graph/d04_probe.py), `data/processed/d04_probe/report.json` | Run D04 chỉ một segmentation 100k; equivalence đa segmentation bổ sung tại E-013 |
| E-013 | D05 exact Full/Incremental equivalence qua nhiều segmentation, G1 | Đã chạy 19/11/2026 | [D05](d05-equivalence.md), [comparator](../src/graph/equivalence.py), [harness](../src/graph/d05_probe.py), [regression/mutation tests](../tests/test_equivalence.py), `data/processed/d05_probe/report.json` | Toy/100k, 35 boundary checks và 70 replays; chưa có full32M/normalization/performance |
| E-014 | D06 sparse symmetric normalization và canonical/model graph version semantics | Đã chạy 19/11/2026, T-05/T-06 PASS | [D06](d06-normalization.md), [normalize](../src/graph/normalize.py), [snapshot](../src/graph/snapshot.py), [probe](../src/graph/d06_probe.py), `data/processed/d06_probe/report.json` | Dense toy và SQL Full reference tại 3 cutoff 100k; chưa có GPU/float32/model eval hoặc benchmark |
| E-015 | Full movies catalog có 87.585 IDs unique/non-null; 7.080 unknown genre labels | Đã validate 19/11/2026 | [validator](../src/metadata/movie_lookup.py), [tests](../tests/test_movie_validation.py), [D06](d06-normalization.md), raw report mục movies và source SHA-256 | Chưa có Top-K join invariance/coverage; không dùng metadata vào graph/model |
| E-016 | LightGCN CUDA forward/backward/Adam đạt toy, 100k và 1M. G2 đạt | Run `D07-20261119T172548657263Z` ngày 20/11/2026 | [D07](d07-lightgcn-smoke.md), [model](../src/models/lightgcn.py), [probe](../src/models/d07_probe.py), [tests](../tests/test_lightgcn.py), [gpu_probe.csv](../data/processed/d07_probe/D07-20261119T172548657263Z/gpu_probe.csv) | 64 chiều, 3 lớp, bốn bước, không có temporal split hay ranking metrics |

## Chi tiết kiểm tra E-003

| File | Kích thước byte tại thời điểm kiểm tra | MD5 tính bằng `Get-FileHash` | MD5 trong `checksums.txt` | Kết quả |
|---|---:|---|---|---|
| `ratings.csv` | 877076222 | `cf12b74f9ad4b94a011f079e26d4270a` | `cf12b74f9ad4b94a011f079e26d4270a` | Khớp |
| `movies.csv` | 4242926 | `0df90835c19151f9d819d0822e190797` | `0df90835c19151f9d819d0822e190797` | Khớp |

Lệnh lặp lại từ root repo:

```powershell
Get-FileHash -Algorithm MD5 ..\ml-32m\ratings.csv, ..\ml-32m\movies.csv
Get-Content ..\ml-32m\checksums.txt
```

Repo HEAD tại lần kiểm kê: `735fe20c6b8f1a508b1b2a4e40aa57400ae4230b`. Đây là commit của **tài liệu trước lần cập nhật này**, không phải commit của phép kiểm tra dữ liệu. `AGENTS.md` đang có thay đổi chưa commit tại thời điểm kiểm kê.

## Claim chưa đủ bằng chứng để đưa vào phần kết quả

| Claim cần chứng minh | Bằng chứng tối thiểu còn thiếu | Mốc dự kiến |
|---|---|---|
| Canonical ordering deterministic ngoài cấu hình hiện tại | Chạy đối chiếu thêm khi đổi DuckDB/writer settings | Sau D02 |
| Full và Incremental tương đương ở scale lớn hơn | T-02/T-03 đã PASS trên toy/100k; cần chạy lại ở scale khác trước khi suy rộng | Trước benchmark scale lớn |
| Normalization và bộ nhớ ở scale lớn hơn 1M | D06 có SQL reference đến 100k. D07 có dense reference trên toy và CUDA runs đến 1M | Khi mở rộng scale |
| Temporal eval không leakage | GPU smoke đã đạt D07. Cần split/cutoff, checkpoint, T-07/T-08 và metric logs | D08–D10 |
| Incremental có lợi hoặc bất lợi về chi phí | EXP-01 raw CSV, máy/config/cache/repetitions, correctness PASS | D11–D12 |
| Deferred normalization đánh đổi cost và staleness | EXP-02 raw CSV, equality ở common boundary | D13 |
| Metadata join không thay ranking/metrics | T-10/T-11/T-12, AN-01 coverage và source hashes | D09–D17 |

Đã có code và tests đến D07. Probe D07 lưu log ngoài Git dưới `data/processed/d07_probe/`. Model checkpoint và các run EXP-01/02/03 vẫn thuộc milestone sau. Cập nhật bảng sau mỗi milestone với commit hoặc run ID và dẫn đến log gốc.
