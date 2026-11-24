# GOAL 04 Phase C — real-5M preflight bị dừng ở resource gate

Ngày 23/11/2026. **INCOMPLETE: chưa được chạy GPU training, chưa có C0.** Phase B đã được Owner nghiệm thu. Phase C được duyệt một trajectory T4 có điều kiện preflight PASS; điều kiện này chưa đạt. Không chạy lại real-data preflight sau lần dừng, không upload, không commit/push/merge.

## 1. Kết quả thật và nguyên nhân dừng

Run `results/goal04_preflight/run_01/` dùng output mới `data/processed/goal04_preflight_01/`. Host có 16.780.570.624 bytes physical RAM (~15,63 GiB), khoảng 3,7 GiB available lúc kiểm tra ban đầu. DuckDB đặt memory limit 1GB, 1 thread; Python graph objects nằm ngoài DuckDB memory limit. Guard dừng khi host available <2 GiB, hoặc vượt wall cap (canonical preparation 1.800 s; base/monthly 300 s mỗi pha theo setup budget).

| Gate | Bằng chứng hiện tại | Kết luận |
| --- | --- | --- |
| Approved cohort/source | Raw và cohort node/edge hashes khớp approved manifest, kiểm lại sau run | PASS |
| Full canonical 32M | 32.000.204 rows, pipeline canonical hiện có, file hash xác minh lại | PASS |
| Base projection v0 | 4.999.981 edges, 90.253 nodes = 49.587 users + 40.666 movies; DB đã commit v0 | PASS |
| No future edges G0 | Base cutoff 1577836800; canonical watermark 27.364.973, max model first-event ID cùng giá trị; last timestamp 1577836786 < cutoff; constructor kiểm canonical order và exact base equality | PASS |
| Base normalized export/reload | Guard dừng trước khi có base snapshot; không có completed snapshot manifest | NOT COMPLETED |
| Monthly replay + export/reload | Không chạy vì base gate chưa qua | NOT RUN |
| Python/SQL COO comparison | Chưa tới bước có thể đối chiếu trên graph thật; không kết luận có/không rounding difference | NOT REACHED |
| Actual worker peak RSS | Bản guard đầu chỉ đo Windows venv launcher (~4 MiB), không phải worker | INVALID MEASUREMENT; cần đo lại |

Canonicalization mất **104,03 s**, artifact disk cuối pha **483.758.736 bytes** (gồm full file, prefix 100k, manifest/profile), sampled peak spill **3.401.547.776 bytes (~3,17 GiB)**. DuckDB profile ghi peak buffer memory **1.941.397.504 bytes (~1,81 GiB)** và peak temp directory **3.231.875.072 bytes**. Đây là các measurement khác nhau (DuckDB counter và filesystem sampling), không thay cho process RSS; configured memory limit không phải bằng chứng trần actual RSS.

Pha base bị dừng sau **85,12 s**, vì minimum sampled host available **2.135.994.368 bytes (~1,989 GiB)** thấp hơn ngưỡng **2.147.483.648 bytes**. Đây là **headroom guard stop**, không phải observed OOM, không phải timeout. Peak disk toàn artifact tree lúc đó **671.728.272 bytes**, sampled spill pha base **57.409.536 bytes**. V0 đã commit nhưng normalized snapshot chưa được xuất. DuckDB profile cuối base chỉ phản ánh connection/pha đã ghi được (peak buffer counter 157.109.928 bytes), không đo toàn Python normalization.

Raw measurements không bị sửa để che lỗi đo RSS: [run report](../../results/goal04_preflight/run_01/report.json), [canonical resources](../../results/goal04_preflight/run_01/canonical_resources.json), [base resources](../../results/goal04_preflight/run_01/base_resources.json), [assessment/corrections](../../results/goal04_preflight/run_01/assessment.json). Không dùng trường `peak_rss_bytes` của run_01 làm worker RSS trong báo cáo hiệu năng.

Nguyên nhân kỹ thuật: đường export hiện giữ Python edge rows, tạo normalized COO khoảng 10 triệu directed entries, rồi loader còn materialize state/COO/reference để kiểm lại. Headroom local thấp; DuckDB spill không giới hạn các Python allocations này. Điểm dừng quan sát được là base export trước khi có snapshot; chưa có valid peak RSS để lượng hóa chính xác từng bản sao. Giữ nguyên integrity checks, không bỏ kiểm normalization/hash để làm xanh preflight.

## 2. Provenance và artifacts còn dùng được

- Raw `ratings.csv` SHA-256: `91159850e41ee59c86231165a688709647e2726cab2e7ba9faf04001bd5261ee` (không đổi).
- Full canonical file: `data/processed/goal04_preflight_01/canonical/canonical_32000204.parquet`, **482.544.363 bytes**, SHA-256 `494a1e70e67a2d750297872020b3919da9e557dd68cb631f53d2c5cb9570f73e`.
- Canonical transform vẫn là `(event_ts,user_raw_id,movie_raw_id,source_row_id)`, IDs bắt đầu 0, preserve physical source row identity, threshold 4.0. `manifest.json` nối raw hash với canonical hash; không dùng quality để chọn dữ liệu.
- V0 mapping logical hash: `996848022d92e20ef63b498cfedb38b97545fb006da4c00fef9ed35d40dc90f9`.
- V0 edge logical hash: `fb2c7fe7d64e018bebcd7101df41249dbfa57e1db5369fae01d023e9c2e5392a`.
- V0 degree logical hash: `ee423151ba7070bc616b8a43475322410b759297c086b9331ba4f3318f49e2b4`.

Mapping/edge logical hashes không phải file-byte hashes của cohort Parquet: serializer/schema khác, runtime edges dùng canonical first-event ID thay cho timestamp. Constructor đối chiếu exact edge/timestamp rows với cohort trước khi commit. DB mở read-only sau khi process kết thúc đã xác nhận counts và ledger; không tự resume replay. Artifacts lớn ở `data/processed/` được Git ignore và ở local, chưa upload.

## 3. Thay đổi code và verification

- `src/ingest/d02_profile.py`: reuse pipeline cũ để tạo full prefix, thêm explicit memory limit và DuckDB profile; không đổi canonical ordering/graph semantics.
- `src/ingest/canonical.py`: đặt DuckDB validation memory limit 1GB thay vì để mặc định host-sized allocation.
- `scripts/goal04_preflight.py`: source/cohort hash verification → canonical preparation → v0 snapshot → monthly snapshot, mỗi bước có fresh outputs, resource guard và stop-on-failure. Không import LightGCN/training hoặc tính quality.
- Sửa guard sau run_01: đo RSS tổng process tree của launcher và actual worker; khi dừng, terminate descendants của chính child đó trước launcher. [Allocation smoke](../../results/goal04_preflight/rss_tree_smoke.json) dùng 8 MiB, không đọc dataset/GPU: thấy hai processes, tree RSS 22.781.952 bytes thay vì launcher 4.399.104 bytes. Fix chưa được dùng để chạy lại full preflight.
- Tests bổ sung cho success/failure/headroom stop, descendant RSS/termination; tests canonical vẫn xác nhận deterministic hashes và timestamp ties. Baseline **83 PASS**, sau thay đổi **84 PASS, 0 FAIL, 0 SKIP**: [regression](../../results/goal04_preflight/regression.log).

Lệnh đã chạy (không chạy lại vào thư mục này):

```powershell
..\venv\Scripts\python.exe -m scripts.goal04_preflight run --artifacts data/processed/goal04_preflight_01 --report results/goal04_preflight/run_01
..\venv\Scripts\python.exe -m unittest discover -s tests -v
```

## 4. GPU timeline, candidate quality và C0

GPU runs = **0**, optimizer steps = **0**, validation quality evaluations = **0**, holdout quality evaluations = **0**. Không tiêu GPU budget, không tạo pilot. Cả năm candidates steps 0/500/1500/3000/5000 đều **NOT RUN**; Recall@20/NDCG@20 là **chưa có**, không gán 0. Không có selected checkpoint hoặc checkpoint reload result; không chọn từ candidate subset. Phase C training/checkpoint code và tests tương ứng chưa triển khai vì yêu cầu chỉ tiến hành sau preflight PASS.

Module đã thực thi: canonical pipeline/validation, `model_replay.py`, normalization hiện có và integrity helpers. Input là raw CSV + approved cohort; output hợp lệ hiện có là full canonical và committed v0 projection. Base normalized snapshot, monthly snapshot, C0 vẫn thiếu.

## 5. Đề xuất xử lý và Phase D readiness

**Dừng chờ Owner quyết định** sau headroom failure. Đề xuất tối ưu việc giữ/copy Python buffers ở snapshot export/reload, bảo toàn exact file hashes và numerical comparison `atol=1e-12`, rồi chạy lại preflight có đo worker RSS đúng. Hoặc giải phóng đủ RAM/chọn máy CPU local có headroom lớn hơn. Không tự upload raw/full canonical sang máy cloud; mọi chuyển dữ liệu mới phải theo approval riêng.

Canonical file đã xác minh có thể tái sử dụng sau khi kiểm hash, nhưng lần preflight mới vẫn phải đo các bước chưa đạt bằng fresh output. Chưa tự triển khai thay đổi normalization hay rerun để vượt stop gate. Nếu lần sau thấy Python/SQL floating-point rounding khác nhau, ghi cả hashes, giữ byte integrity, so numerical tolerance và thêm regression test như Owner yêu cầu.

**Phase D NOT READY / NOT AUTHORIZED**: chưa có preflight PASS, C0, full validation scores, strict checkpoint reload hoặc Phase C report hoàn chỉnh. Không triển khai engine hoặc 33-window comparison.

## 6. Giải thích cho sinh viên mới

Preflight là thử đường dữ liệu và tài nguyên trước khi tiêu GPU budget. Canonicalization sắp ratings theo thứ tự thời gian xác định. Projection giữ đúng cohort model; normalization chuyển cạnh nhị phân thành trọng số để LightGCN truyền thông tin. Các bước này không chứng minh recommendation quality.

Ví dụ graph có 5 triệu cạnh nhưng sparse matrix hai chiều có gần 10 triệu entries. Nếu giữ đồng thời Python tuples, bản SQL export và reference để kiểm, RAM thực tế có thể lớn hơn nhiều payload số học. DuckDB có thể spill dữ liệu của nó ra disk nhưng không tự spill Python tuples. Vì vậy disk còn trống không có nghĩa RAM đủ.

Đọc report theo thứ tự: input hashes → counts/boundary → snapshot reload → resources → training/quality. V0 đúng 4.999.981 edges không đủ để kết luận toàn preflight PASS khi snapshot/reload và monthly gate chưa hoàn thành.

Ba câu tự kiểm tra: (1) Vì sao DuckDB memory limit 1GB không bảo đảm toàn process chỉ dùng 1GB? (2) Vì sao đo RSS của launcher không đo được worker thật? (3) Vì sao không được bắt đầu GPU training khi đã có đúng edge count nhưng snapshot reload chưa qua?
