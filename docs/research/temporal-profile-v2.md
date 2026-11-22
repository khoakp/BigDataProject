# MovieLens 32M temporal profile và cutoff feasibility (v2)

**Trạng thái:** profiling hoàn thành; Project Owner đã duyệt cutoff **01/01/2020 00:00:00 UTC**. Chưa tạo base graph 32M hay benchmark Full/Incremental trên toàn base; [GPU feasibility](gpu-feasibility-v2.md) đã đo các model cohorts trên Kaggle T4. Các số dưới đây là thống kê dữ liệu, không phải số đo hiệu năng.

## Dữ liệu và định nghĩa

- Nguồn: `../ml-32m/ratings.csv`, 877.076.222 bytes; SHA-256 `91159850e41ee59c86231165a688709647e2726cab2e7ba9faf04001bd5261ee`. MD5 `cf12b74f9ad4b94a011f079e26d4270a` khớp `../ml-32m/checksums.txt`. Source không bị sửa.
- Phạm vi: **32.000.204** ratings, từ **09/01/1995 11:46:44** đến **13/10/2023 02:29:07 UTC**; 334 tháng có events. Positive event khi `rating >= 4.0`: **15.938.231**. Distinct `(userId,movieId)` positive edge: **15.938.231** trong bản dữ liệu này. Sự bằng nhau là kết quả đo, không là giả định của graph engine.
- Toàn bộ dữ liệu có **200.948 distinct users** và **84.432 distinct movies**. Positive graph chỉ có **200.726 user nodes** và **55.174 movie nodes**: node được đăng ký tại positive edge đầu tiên. `first_observed_movie` nghĩa là timestamp rating đầu tiên trong file, **không phải ngày phát hành phim**.
- Cutoff là nửa khoảng UTC: `base: event_ts < cutoff`; `replay: event_ts >= cutoff`. Các mốc khảo sát không có event đúng giây 00:00:00; toy test có và xác nhận event đó thuộc replay.

Raw outputs: [monthly.csv](../../results/temporal_profile_v2/monthly.csv), [yearly.csv](../../results/temporal_profile_v2/yearly.csv), [cutoffs.csv](../../results/temporal_profile_v2/cutoffs.csv), [manifest.json](../../results/temporal_profile_v2/manifest.json). Monthly có distinct entities hoạt động trong từng tháng, first-observed entities và positive graph nodes, new positive edges, các cumulative counts. Yearly có distinct users/movies trong **cả năm** (không cộng distinct theo tháng), cùng first-observed và cumulative counts. Các file CSV là thống kê raw, không chứa thông tin cá nhân ngoài counts.

## So sánh candidate cutoffs

| Base trước UTC cutoff | Base events | Base graph users + movies | Base positive edges | Replay events (% tổng) | Replay months / ngày | New users / replay-active | New movies / replay-active |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 01/01/2018 | 24.588.450 | 165.441 + 31.510 = **196.951** | 12.261.828 | 7.411.754 (23,2%) | 70 / 2.111,1 | 35.281 / 43.848 = **80,5%** | 38.884 / 79.293 = **49,0%** |
| 01/01/2019 | 25.979.507 | 172.338 + 35.946 = **208.284** | 12.955.313 | 6.020.697 (18,8%) | 58 / 1.746,1 | 28.379 / 36.263 = **78,3%** | 31.844 / 76.375 = **41,7%** |
| **01/01/2020** | **27.364.974** | **179.192 + 41.515 = 220.707** | **13.644.889** | **4.635.230 (14,5%)** | **46 / 1.381,1** | **21.529 / 29.346 = 73,4%** | **24.480 / 70.027 = 35,0%** |
| 01/01/2021 | 29.053.133 | 187.721 + 45.609 = **233.330** | 14.508.994 | 2.947.071 (9,2%) | 34 / 1.015,1 | 12.998 / 20.970 = **62,0%** | 17.682 / 62.161 = **28,4%** |

`replay-active` là distinct entity có ít nhất một rating sau cutoff. `new` nghĩa là rating đầu tiên của entity ở hoặc sau cutoff; vì vậy tỷ lệ này đo cohort mới trong replay, không phải tỷ lệ trên toàn bộ catalog. Nếu xét **positive graph** riêng, tại 2020 có 21.534 / 29.181 = **73,8%** positive-active users và 13.659 / 40.181 = **34,0%** positive-active movies lần đầu vào graph trong replay. Base 2020 có 59.952 distinct movies trong mọi rating nhưng chỉ 41.515 movie nodes có positive edge.

**Đề xuất 2020:** 4,64 triệu replay events và 46 monthly windows đủ để quan sát biến động graph/snapshot qua nhiều năm, trong khi base graph vẫn chiếm 13,64 triệu edges để thử chi phí cập nhật trên lịch sử lớn. So với 2019, replay bớt 1,39 triệu events và 12 tháng; so với 2021, có thêm 1,69 triệu events và 12 tháng. Đây là điểm cân bằng về **quy mô thí nghiệm**, không phải chứng minh tối ưu hóa.

**Rủi ro:** 73,4% replay-active users chưa từng xuất hiện trước 2020; warm-only evaluation cho RQ2 có thể rất hẹp. Movie churn cũng cao. Số ratings năm 2020 (1.688.159) cao hơn 2019 (1.385.467), nên kiểm tra độ nhạy bằng cutoff 2019 và 2021 trước khi diễn giải mô hình theo thời gian. Dataset kết thúc 10/2023, vì vậy năm 2023 không đủ 12 tháng; không so annual totals như hai năm đầy đủ. First-observed movie trong ratings không phản ánh release date hay catalog availability.

## Replay protocol đề xuất

1. Dùng cùng canonical order hiện hành `(event_ts, user_raw_id, movie_raw_id, source_row_id)` và `event_id` liên tục từ 0. Validate source IDs, rating domain và timestamp. Không đổi graph rule `rating >= 4.0` hoặc dedup cặp user–movie.
2. **Calendar batches cho RQ2:** base là mọi event trước 2020 UTC; replay mỗi tháng `month_start <= event_ts < next_month_start`, bắt đầu 01/2020. Snapshot và evaluation chỉ tại ranh giới tháng; future labels phải có `event_ts > snapshot watermark` và nằm trong window đánh giá cố định. Không dùng future events để tạo train graph, candidate mask hoặc embeddings.
3. **Event-count batches cho RQ1:** cùng canonical stream, chia replay thành các đoạn 10k và 100k events để khảo sát chi phí theo Δ; batch cuối có thể ngắn hơn. Full và Incremental phải nhận cùng prefix và so ở **cùng event_id watermark**. Tie timestamp có thể nằm ở hai count batches; không dùng trạng thái giữa tie làm temporal evaluation cutoff.
4. Ghi manifest cho từng batch: source SHA-256, `start_event_id`, `end_event_id` (exclusive), UTC min/max timestamp, batch hash, cutoff và graph version. Prebuild base state từ event 0 đến `base_events-1`; incremental replay tiếp tục với event ID `base_events`, không reset ID. Reapply cùng batch phải `NO_OP`; conflict phải bị từ chối. Calendar và count segmentation là hai cách nhìn trên **cùng canonical stream**.

Đây là **protocol đề xuất**, chưa phải pipeline replay toàn bộ đã triển khai hoặc benchmark đã chạy.

## Ước lượng tài nguyên, không thay cho GPU test

Với `N = base positive graph nodes`, `E = base unique positive edges`, LightGCN symmetric COO có `2E` entries. Giả định COO chứa hai `int64` indices và một `float32` weight: **20 bytes/entry**, tức `40E` bytes payload. Ví dụ embedding dimension `d=64`, ba propagation layers, budget tham chiếu cộng `16Nd` bytes cho parameter + gradient + Adam moments và `4(3+1)Nd` bytes cho bốn bảng layer embeddings. Đây là **ước lượng payload**, chưa tính duplicate/coalesce buffers, autograd graph, mini-batch, CUDA allocator, runtime và graph state trên CPU; VRAM thực tế có thể cao hơn nhiều.

| Base cutoff | N | E | COO entries | COO payload | Param/grad/Adam 64d | 4 layer tables 64d | Tổng ba khoản |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2018 | 196.951 | 12.261.828 | 24.523.656 | 467,8 MiB | 192,3 MiB | 192,3 MiB | 852,4 MiB |
| 2019 | 208.284 | 12.955.313 | 25.910.626 | 494,2 MiB | 203,4 MiB | 203,4 MiB | 901,0 MiB |
| 2020 | 220.707 | 13.644.889 | 27.289.778 | 520,5 MiB | 215,5 MiB | 215,5 MiB | 951,6 MiB |
| 2021 | 233.330 | 14.508.994 | 29.017.988 | 553,5 MiB | 227,9 MiB | 227,9 MiB | 1.009,2 MiB |
| Toàn bộ dữ liệu | 255.900 | 15.938.231 | 31.876.462 | 608,0 MiB | 249,9 MiB | 249,9 MiB | 1.107,8 MiB |

GPU feasibility kế tiếp nên tăng theo **positive edge count thực đo**, ví dụ khoảng 0,5M → 2M → 5M → 10M → 13,64M, dừng tại OOM hoặc thời gian vượt budget. Với mỗi mức đo peak allocated/reserved VRAM, propagation/training step time và checkpoint bytes trên một Kaggle T4 16 GB; chọn workload model từ phép đo, không suy từ bảng payload rằng full 32M sẽ vừa GPU.

## Tái lập và kiểm tra

Từ root repo với venv hiện có, chọn `--out` **chưa tồn tại**; profile tạo CSV và manifest, xóa DuckDB working database sau khi thành công:

```powershell
..\venv\Scripts\python.exe -m unittest tests.test_temporal_profile -v
..\venv\Scripts\python.exe -m src.ingest.temporal_profile --source ..\ml-32m\ratings.csv --out results\temporal_profile_v2_repro --memory-limit 1GB --threads 2
..\venv\Scripts\python.exe -m unittest discover -s tests -v
```

Profiler đối chiếu tổng monthly với toàn bộ events/positives/unique edges và đối chiếu `base + replay` cho từng cutoff. Toy tests đặt events đúng ranh giới UTC, rating âm trước positive, duplicate positive edge và entity xuất hiện trong rating trước khi trở thành graph node. Bản chạy thực tế dùng Python 3.10.11, DuckDB 1.5.6, memory limit 1GB, 2 threads; kết quả nằm trong manifest với checksum CSV. Các số 32M chưa được đối chiếu với một pipeline graph full-scale độc lập, nên đó là bước verification tiếp theo trước kết luận hiệu năng.

**Câu hỏi tự kiểm tra:** (1) Vì sao `base_distinct_movies` lớn hơn `base_graph_movies`? (2) Vì sao count batch có thể chia đôi một timestamp nhưng monthly cutoff thì không? (3) Tại sao 951,6 MiB không chứng minh graph 2020 sẽ train vừa T4?
