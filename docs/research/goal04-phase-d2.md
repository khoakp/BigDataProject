# GOAL 04 Phase D2 — CPU toy S0/S1/S2 implementation

**Status:** `PHASE_D2_READY_FOR_OWNER_REVIEW`; 26/11/2026. Baseline `develop/v2` HEAD `f28089dbc0b69d1272d21014c17dcd4617150685`; giữ nguyên hai thay đổi tài liệu D1 chưa commit. [D1 contracts](goal04-phase-d1.md) và [protocol GOAL 04](../current/experiment-protocol.md) điều khiển triển khai. Chỉ synthetic CPU, temp artifacts; không tạo official P0, dùng C0/G0 thật, đo quality development/holdout hoặc chạy GPU.

## Module map và data flow

| File | API và trách nhiệm đã triển khai |
| --- | --- |
| `src/evaluation/refresh_boundary.py` | `freeze_boundary(replay,b,snapshot)->BoundaryHistory`, `validate_history`, `MaskRef.seen_for`, `select_serving_users(replay,history)->BenchmarkUsers`, `make_quality_bundle(replay,history,e)->QualityBundle`, `RankRequest`. Replay canonical `<b`, export/verify projection, hash catalog/mask/version; labels `[b,e)` chỉ được đọc ở evaluator path. Mask tham chiếu immutable snapshot và chỉ lấy seen pairs theo requested users, không giữ bản sao 5M pairs trong bundle. |
| `src/models/refresh_cache.py` | `cache_key(identity)->SHA256`, `publish_atomic(path,embeddings,identity,before,after)->manifest`, `load_verified(path,identity,expected_manifest_sha256)->(tensor,manifest)`, `parameter_sha256(model)`. Canonical JSON identity gồm checkpoint/G0 or refreshed snapshot/mapping/source/cohort/version/boundary/edge-degree-COO/config/device/environment/repetition. Temp directory, tensor SHA, manifest last, publish bằng rename; sai hash/shape/nonfinite/partial artifact bị từ chối. |
| `src/evaluation/refresh_strategies.py` | `VerifiedC0.load()` dùng strict `load_candidate`; `create_p0` một toy CPU setup forward; `prepare(S0|S1|S2,...)`, `get_scores`, `rank_users`, `provenance`, `counters`. S0 verified-load P0 và zero forward; S1 strict snapshot + frozen C0, inference-only một forward/repetition/boundary, hash tham số trước/sau; S2 count từ unique edges. Ranker dùng `score_batched` hoặc `popularity_topk` với cùng request/mask/catalog. |
| `src/evaluation/baselines.py` | `popularity_counts` được tách từ baseline cũ; `popularity_topk(...,counts=...)` giữ nguyên API cũ, thêm optional precomputed counts để S2 chỉ đếm một lần trong boundary. Không có popularity algorithm thứ hai. |
| `src/evaluation/refresh_results.py` | `evaluate_quality(bundle,rankings)` dùng lại `evaluate_topk` cho per-user/window; `coverage`, `aggregate_split` macro theo full nonempty windows với empty count rõ ràng; `paired_comparisons` cho `S1−S0`, `S2−S0`, `S1−S2` cùng user-window. Empty labels → null, không tạo zero score. |
| `src/benchmarking/goal04_refresh.py` | `run_toy(cohort,events,checkpoint,g0_snapshot,windows,output)->report`: fresh replay mỗi rep, một P0, đúng ba strategy orders, rep 1 quality+serving-like, reps 2/3 serving-like, CPU phase timers/RSS, `COMPLETE_TOY_CPU` hoặc `INCOMPLETE`. Giới hạn node/event/window và chặn 2023 holdout; không là official experiment runner. |
| `tests/test_goal04_refresh_toy.py` | Synthetic fixture và các gates bên dưới; mọi artifact sinh dưới `TemporaryDirectory`. |

Luồng: full canonical Parquet + fixed cohort → `ModelGraphReplay` → snapshot/`BoundaryHistory` trước `b` → S0 đọc P0, S1 propagate C0 trên snapshot, S2 đếm edges → `RankRequest` chung cho serving-like; riêng rep 1 tạo `QualityBundle` sau khi state/rank serving đã đóng băng, rồi rank offline và chấm labels. Một `QualityBundle` được dùng cho cả ba strategy; `RankRequest`/`prepare` không có field labels. Không dùng `split_events` kiểu `<=` của protocol lịch sử.

## API ví dụ và version/cache rules

```python
# Trong TemporaryDirectory, với source/cohort/checkpoint synthetic đã tạo:
report = run_toy(cohort, canonical_parquet, verified_toy_c0, toy_g0_snapshot,
                 ((jan_2021, feb_2021), (feb_2021, mar_2021)), out)
assert report['status'] == 'COMPLETE_TOY_CPU'
assert (report['p0_setup_forward_attempts'], report['s1_forward_attempts']) == (1, 6)
assert report['optimizer_updates'] == 0
```

Toy P0 identity có `repetition=null`, chỉ dùng G0 projection v0; S1 identity có repetition 1/2/3 và snapshot tại `b`. S1 cache không thể bỏ qua forward của repetition khác; cache trong một boundary dùng chung cho offline/serving rankings. `load_verified` cần expected identity và manifest SHA từ publication/manifest ngoài; kiểm file bytes mỗi lần load. `VerifiedC0` mang cả checkpoint SHA lẫn G0 manifest SHA; `create_p0` đối chiếu G0 COO/edge/cohort/base cutoff với provenance gốc. Không retry sau forward lỗi; ledger tăng trước lời gọi forward, kể cả khi fail. Snapshot không hợp version/mapping/source hoặc mask/catalog bị thay đổi bị reject. `get_scores` trả bản sao embeddings hoặc bản sao popularity counts, không cho caller sửa state đang dùng để rank.

**Toy example:** base có user 1 positive phim 10, user 2 positive phim 20 và 30. Trong 2020, user 1 positive phim 20; đến boundary 01/01/2021, S0 vẫn có P0 từ base, S1 dùng thêm cạnh 1–20 và S2 đếm phim 20 từ hai users. Cả ba mask 10 và 20 cho user 1, chỉ có phim 30 chưa seen; user 2 còn phim 10. Rating mới 1–30 đúng 01/01 là label của tháng 1, chưa vào graph/mask ở boundary đó. Sang 01/02, 1–30 trở thành edge lịch sử, mask đổi; S1 tính vector lại, S0 giữ P0, S2 cập nhật count. Rating thấp không xóa cạnh. Đây là expected behavior toy, không là real-data quality result.

## Verification evidence

- Preflight: branch/HEAD đúng baseline; D1 report local có; Torch 2.8.0+cu126, DuckDB 1.5.6, psutil 7.2.2; **CPU execution only**. Trước chỉnh mã, 22/22 relevant tests PASS; D0 ghi baseline 103 PASS.
- New toy tests bao phủ: equal timestamps/canonical ties; no future graph/mask/selector; repeated positive, low rating, fixed mapping/catalog; independent full projection/normalization; S0 zero extra forward, S1 exactly one per refresh/zero optimizer, S2 observed-only unique counts; same bundle/Top-20 ties/fewer candidates; missing/tampered/wrong-version cache, shape, NaN, interrupted publication; parameter hash, empty labels/no candidates, idempotency/rollback; three orders/timing split; failures trước/sau attempted forward và holdout gate.
- **Final commands/results:** `..\venv\Scripts\python.exe -m unittest tests.test_goal04_refresh_toy -q` — **7 PASS**; targeted model/graph/evaluation + toy — **29 PASS**; `..\venv\Scripts\python.exe -m unittest discover -s tests -q` — **110 PASS, 0 FAIL** (baseline 103 + 7 tests mới). Toy end-to-end smoke chạy riêng: `COMPLETE_TOY_CPU`, P0=1, S1=6 với 2 boundaries × 3 reps, optimizer=0. Lần full regression đầu phát hiện thứ tự error message của `popularity_topk` đã đổi; đã khôi phục contract cũ và full rerun PASS. `git diff --check`, Markdown links và file-change audit được ghi ở stop gate.

Toy tests mới chỉ xác nhận logic trên graph nhỏ. Chưa đo RAM/latency 5M edges, chưa có real-data Recall/NDCG, CI, 33-boundary cost hay GPU budget proof. C0 checkpoint, G0, raw C3/C4/C5/C6, REC-01 và protocol không bị sửa. Trong toy coordinator, `RssSampler` đo peak RSS parent process (không phải process-tree); DuckDB spill/GPU telemetry và resource cap thực thuộc D3/D4. Snapshot validation/hash được lặp nhiều lần để ưu tiên correctness; cần profiling trước real run. `run_toy` viết một JSON tổng hợp, không phải production artifact schema đầy đủ.

**Stop-gate file audit:** `git diff --check` PASS; 0 broken relative Markdown links trong bốn tài liệu D2 mới/sửa; untracked source/test/report không có trailing whitespace. Git status chỉ gồm tài liệu D1 có sẵn trước D2 và các file D2 liệt kê trên; không có protected raw/protocol/REC-01 file thay đổi. Read-only SHA verification: C0 checkpoint `2bf073f328551561bf21941ee9fa7f3d239080139922de348f86426862a25e66`, G0 manifest `0feaa2c521b0adb364fa78468a86f8a39fac979d630053fe0cb8aff39910ebdb`, đều khớp [selection manifest](../../results/goal04_training/selection.json). Không commit/push/merge.

**D1 deviations / scoped decisions:** `VerifiedC0` bổ sung G0 manifest SHA bắt buộc để P0 không dùng nhầm G0; `get_scores` trả defensive copy; S2 thêm helper counts trong baseline hiện có; coordinator chỉ nhận ≤3 monthly windows và synthetic CPU input trước 2023. Official 2.000-replicate user-cluster bootstrap chưa chạy/triển khai ở D2 vì thuộc phép phân tích real-data được duyệt về sau. Không đổi scientific estimand hay strategy. Scope wording Fine-tuning/Retraining ở README/scope vẫn chờ Owner như [D1](goal04-phase-d1.md).

## D3 readiness, RQ2 và học tập

D2 tạo correctness foundation để sau này kiểm RQ2 trên cùng user/catalog/mask, nhưng không chứng minh strategy nào cải thiện chất lượng. D3 cần adversarial audit chi tiết hơn cho cache cross-process/manifest provenance, failure injection, streaming memory, phase timing và exact reproducibility trước real preflight; mỗi thay đổi phải giữ 1 P0 + 99 S1 forwards theo [protocol](../current/experiment-protocol.md). Owner cần nghiệm thu D2 trước D3; GPU, development quality và holdout cần các gate riêng.

Cho người mới: **S0** nhận C0/G0-derived P0 và history mask, xuất Top-20 bằng vector cũ. **S1** nhận C0 cùng graph snapshot mới, xuất vector/Top-20 mới sau propagation; phép nhân graph đổi vector mà không sửa trọng số C0. **S2** nhận graph positive đã quan sát, xuất Top-20 theo độ phổ biến. Hash cache sai thì dừng, không đoán vector hay tự tính lại. P0 của S0 không đổi nhưng mask phải che thêm phim user vừa positive ở tháng đã kết thúc. Toy graph giúp thấy lỗi boundary/dedup/cache với chi phí nhỏ trước khi chạm 5M edges. Toy PASS chứng minh code xử lý fixture; verified real-data experiment còn cần artifact provenance, tài nguyên, quality và thống kê trên workload đã duyệt.

Ba câu tự kiểm tra: (1) Rating đúng 01/01 có được vào S1 graph của 01/01 không? (2) Nếu P0 file đúng tên nhưng SHA sai, S0 phải làm gì? (3) Vì sao 7 toy tests PASS chưa đủ kết luận S1 tốt hơn S0 trên MovieLens 32M?
